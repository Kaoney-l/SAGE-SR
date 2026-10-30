"""Structured semantic priors and their three search-control channels."""

from __future__ import annotations

import copy
import json
import hashlib
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
from urllib import error, request

import numpy as np


CONDITION_DIM = 64
OPERATOR_ORDER = ("+", "-", "*", "/", "sin", "cos", "exp", "log")
FAMILY_ORDER = ("periodic", "rational", "additive", "multiplicative", "power_law", "polynomial")


def _weights(values: Mapping[str, Any] | None) -> dict[str, float]:
    out = {}
    for key, raw in (values or {}).items():
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("Semantic prior weights must be finite.")
        out[str(key)] = float(np.clip(value, 0.0, 1.0))
    return out


@dataclass
class StructuredPrior:
    operator_prior: dict[str, float] = field(default_factory=dict)
    variable_relevance: dict[str, float] = field(default_factory=dict)
    structural_family: dict[str, float] = field(default_factory=dict)
    motif_seeds: list[dict[str, Any]] = field(default_factory=list)
    self_confidence: dict[str, float] = field(default_factory=dict)
    interaction_pairs: list[Any] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        self.operator_prior = _weights(self.operator_prior)
        self.variable_relevance = _weights(self.variable_relevance)
        self.structural_family = _weights(self.structural_family)
        self.self_confidence = _weights(self.self_confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StructuredPrior":
        names = cls.__dataclass_fields__
        return cls(**{key: copy.deepcopy(raw) for key, raw in value.items() if key in names})

    @classmethod
    def from_json(cls, value: str) -> "StructuredPrior":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class CompiledPrior:
    condition: np.ndarray
    token_bias: dict[str, float]
    mutation_weights: tuple[float, float, float, float]

    @classmethod
    def neutral(cls) -> "CompiledPrior":
        return cls(np.zeros(CONDITION_DIM, dtype=np.float32), {}, (1.0, 1.0, 1.0, 1.0))


def compile_prior(
    prior: StructuredPrior | None,
    feature_names: Sequence[str],
    operators: Sequence[str],
    eps: float = 0.05,
) -> CompiledPrior:
    """Compile a prior into a condition vector, unscaled token biases and GP weights.

    Token-bias keys use backend names (``R``, ``x0``, ...). The trust
    coefficient is applied later by the action-prior callback, not here.
    Mutation weights are ordered as node replacement, shrink, subtree
    replacement, and insertion. They do not control crossover.
    """
    if prior is None:
        return CompiledPrior.neutral()
    if eps <= 0:
        raise ValueError("eps must be positive.")
    condition = np.zeros(CONDITION_DIM, dtype=np.float32)
    for index, operator in enumerate(OPERATOR_ORDER):
        condition[index] = prior.operator_prior.get(operator, 0.0)
    variable_weights = {}
    for index, name in enumerate(feature_names):
        value = prior.variable_relevance.get(name, prior.variable_relevance.get(f"x{index}", 0.0))
        variable_weights[f"x{index}"] = value
        if index < 10:
            condition[8 + index] = value
    for index, family in enumerate(FAMILY_ORDER):
        condition[18 + index] = prior.structural_family.get(family, 0.0)
    condition[24] = prior.self_confidence.get("operator_prior", 0.5)
    condition[25] = prior.self_confidence.get("variable_relevance", 0.5)
    # Remaining coordinates encode candidate structures and interactions
    # without evaluating or injecting LLM-generated expression strings.
    def project(label, weight):
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        index = 32 + int.from_bytes(digest[:4], "big") % 32
        condition[index] += float(weight) * (1 if digest[4] & 1 else -1)
    motif_weights = []
    for motif in prior.motif_seeds:
        weight = float(motif.get("weight", 0.5))
        if not math.isfinite(weight):
            raise ValueError("Candidate-structure weights must be finite.")
        weight = float(np.clip(weight, 0, 1))
        motif_weights.append(weight)
        project("motif:" + str(motif.get("expr", "")), weight)
    interactions = []
    for pair in prior.interaction_pairs:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            weight = float(pair[2]) if len(pair) > 2 else 0.5
            if not math.isfinite(weight):
                raise ValueError("Interaction weights must be finite.")
            weight = float(np.clip(weight, 0, 1))
            interactions.append(weight)
            project("pair:" + str(pair[0]) + ":" + str(pair[1]), weight)
    for index, name in enumerate(feature_names):
        if index >= 10:
            project("variable:" + str(name), variable_weights[f"x{index}"])
    condition[26] = float(np.mean(motif_weights)) if motif_weights else 0
    condition[27] = len(motif_weights) / (len(motif_weights) + 1)
    condition[28] = float(np.mean(interactions)) if interactions else 0
    condition[29] = len(interactions) / (len(interactions) + 1)
    condition[32:] = np.clip(condition[32:], -1, 1)

    specified = {}
    for operator in operators:
        value = prior.operator_prior.get(operator, 0.0)
        if operator == "R":
            value = prior.operator_prior.get("R", prior.operator_prior.get("const", 0.0))
        if value > 0:
            specified[operator] = value
    specified.update({name: value for name, value in variable_weights.items() if value > 0})
    mean = float(np.mean(list(specified.values()))) if specified else 0.0
    bias = {name: math.log((weight + eps) / (mean + eps)) for name, weight in specified.items()}

    weights = [1.0, 1.0, 1.0, 1.0]
    family = prior.structural_family
    rational = max(family.get("rational", 0.0), family.get("power_law", 0.0))
    periodic, additive = family.get("periodic", 0.0), family.get("additive", 0.0)
    if rational > 0.4:
        weights[3] += rational
        weights[0] += 0.5 * rational
    if periodic > 0.4:
        weights[0] += periodic
    if additive > 0.5:
        weights[1] += 0.5 * additive
    return CompiledPrior(condition, bias, tuple(weights))


SYSTEM_PROMPT = """You are the semantic controller of a symbolic-regression search.
Infer revisable structural preferences from the task context and numerical search
evidence. The search optimizer, not this controller, generates and fits candidate
expressions. Distinguish supported hypotheses from contradicted ones and do not
treat task semantics as numerical proof. Return only one JSON object with fields:
operator_prior, variable_relevance, structural_family, motif_seeds,
self_confidence, interaction_pairs, and notes.
Weights and confidence values must lie in [0,1]. Operator names must come from
the supplied operator list. Variable keys must use supplied feature names or
their x0, x1, ... aliases. Structural families include periodic, rational,
additive, multiplicative, power_law, and polynomial. motif_seeds is a list of
objects with expr, weight, and reason; these are hypotheses, not fitted solutions.
self_confidence contains operator_prior and variable_relevance confidences.
Use search history to revise hypotheses when new evidence contradicts them.
"""


class LLMController:
    """One OpenAI-compatible chat-completions request per ``generate`` call.

    Credentials come from explicit arguments or SAGE_SR_LLM_* environment
    variables. No environment file is loaded and no credentials are logged.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SAGE_SR_LLM_BASE_URL", "")).rstrip("/")
        self._api_key = api_key if api_key is not None else os.environ.get("SAGE_SR_LLM_API_KEY", "")
        self.model = model or os.environ.get("SAGE_SR_LLM_MODEL", "")
        self.timeout, self.temperature = float(timeout), float(temperature)
        self.calls = 0
        if not self.base_url or not self.model:
            raise ValueError("Set SAGE_SR_LLM_BASE_URL and SAGE_SR_LLM_MODEL.")

    def generate(self, context: Mapping[str, Any], memory: Any, operators: Sequence[str]) -> StructuredPrior:
        content = json.dumps(
            {"task_context": context, "search_history": memory, "operators": list(operators)},
            ensure_ascii=False, allow_nan=False,
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            "temperature": self.temperature,
        }, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        req = request.Request(self.base_url + "/chat/completions", data=body, headers=headers)
        self.calls += 1
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                payload = json.load(response)
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError("The semantic-controller request failed; no automatic retry was made.") from exc
        text = payload["choices"][0]["message"]["content"]
        if not isinstance(text, str) or "{" not in text or "}" not in text:
            raise ValueError("The semantic controller did not return a JSON prior.")
        prior = StructuredPrior.from_json(text[text.index("{"):text.rindex("}") + 1])
        unknown = set(prior.operator_prior) - set(operators) - {"const"}
        if unknown:
            raise ValueError("The semantic prior contains operators outside the configured grammar.")
        return prior


class StaticController:
    """Explicit offline fixture or static-prior controller; it makes no LLM call."""

    def __init__(self, prior: StructuredPrior | Mapping[str, Any] | None = None) -> None:
        self.prior = prior if isinstance(prior, StructuredPrior) else StructuredPrior.from_dict(prior or {})
        self.calls = 0

    def generate(self, context: Mapping[str, Any], memory: Any, operators: Sequence[str]) -> StructuredPrior:
        self.calls += 1
        return StructuredPrior.from_dict(self.prior.to_dict())
