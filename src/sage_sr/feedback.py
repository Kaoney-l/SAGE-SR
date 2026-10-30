"""Ancestry-aware structural evidence and explicit-bias trust calibration."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import numpy as np

from .priors import StructuredPrior, compile_prior


def _roots(candidate):
    roots = candidate.get("root_ids") or [candidate.get("root_id", "path:" + str(candidate.get("path", [])))]
    return tuple(sorted(set(str(root) for root in roots)))


def deduplicate_by_root(candidates):
    """Keep the highest training-reward candidate per ancestry root."""
    best = {}
    for candidate in candidates:
        for root in _roots(candidate):
            if root not in best or candidate["reward"] > best[root]["reward"]:
                best[root] = {**candidate, "root_id": root, "root_ids": [root]}
    return list(best.values())


def prior_agreement(prior, candidate, feature_names, operators):
    """Classify the sign of a candidate's mean compiled token preference.

    A zero score or absence of preferred tokens supplies no comparison
    evidence. This rule uses structural information, not validation rewards.
    """
    bias = compile_prior(prior, feature_names, operators).token_bias
    values = [bias[name] for name in candidate.get("token_names", []) if name in bias]
    if not values:
        return None
    score = float(np.mean(values))
    return True if score > 1e-12 else (False if score < -1e-12 else None)


def _subtrees(names, max_nodes=12):
    """Extract complete prefix subtrees, rather than arbitrary token spans."""
    out = set()
    for start in range(len(names)):
        slots = 1
        for end in range(start, min(len(names), start + max_nodes)):
            name = names[end]
            arity = 2 if name in {"+", "-", "*", "/"} else (1 if name in {"sin", "cos", "exp", "log"} else 0)
            slots += arity - 1
            if slots == 0:
                if end > start:
                    out.add(tuple(names[start:end + 1]))
                break
    return out


def audit_structure(candidates, min_roots=2):
    """Summarize complete motifs with at most one contribution per root.

    Supported/refuted labels indicate above/below-median validation reward
    in the observed candidates; they are evidence summaries, not proofs.
    """
    if not candidates:
        return {"supported": [], "refuted": []}
    reference = float(np.median([c["validation_reward"] for c in candidates]))
    evidence = defaultdict(dict)
    for candidate in candidates:
        for motif in _subtrees(candidate.get("token_names", [])):
            for root in _roots(candidate):
                evidence[motif][root] = max(candidate["validation_reward"], evidence[motif].get(root, 0.0))
    patterns = []
    for motif, roots in evidence.items():
        if len(roots) >= min_roots:
            mean = float(np.mean(list(roots.values())))
            patterns.append({"prefix": list(motif), "independent_roots": len(roots),
                             "mean_validation_reward": mean, "relative_to_median": mean - reference})
    patterns.sort(key=lambda p: (-p["relative_to_median"], -p["independent_roots"], p["prefix"]))
    return {"supported": [p for p in patterns if p["relative_to_median"] > 0][:12],
            "refuted": [p for p in reversed(patterns) if p["relative_to_median"] < 0][:12]}


@dataclass
class AlphaCalibrator:
    initial_confidence: float = 0.5
    kappa: float = 10.0
    cap: float = 3.0
    conflict_threshold: float = -0.3
    patience: int = 2
    history: list = field(default_factory=list, init=False)
    conflicts: int = field(default=0, init=False)

    def __post_init__(self):
        if not all(math.isfinite(x) for x in [self.initial_confidence, self.kappa, self.cap, self.conflict_threshold]):
            raise ValueError("Trust calibration parameters must be finite.")
        if self.kappa <= 0 or self.cap <= 0 or self.patience < 1:
            raise ValueError("Trust sensitivity, cap and patience must be positive.")
        self.alpha = min(self.cap, 2.0 * float(np.clip(self.initial_confidence, 0.1, 1.0)))

    def update(self, agree_rewards, disagree_rewards):
        before = self.alpha
        agree = np.asarray(agree_rewards, dtype=float)
        disagree = np.asarray(disagree_rewards, dtype=float)
        if not np.all(np.isfinite(agree)) or not np.all(np.isfinite(disagree)):
            raise ValueError("Calibration rewards must be finite.")
        if len(agree) == 0 or len(disagree) == 0:
            self.conflicts = 0
            entry = {"delta": None, "evidence": None, "reason": "missing_comparison_group"}
        else:
            delta = float(agree.mean() - disagree.mean())
            evidence = float(np.clip(self.kappa * delta, -1.0, 1.0))
            self.conflicts = self.conflicts + 1 if evidence < self.conflict_threshold else 0
            self.alpha = float(np.clip(self.alpha * math.exp(evidence), 0.0, self.cap))
            if self.conflicts >= self.patience:
                self.alpha = 0.0
            entry = {"delta": delta, "evidence": evidence, "reason": "permanently_disabled" if self.alpha == 0 else "calibrated"}
        self.history.append({**entry, "alpha_before": before, "alpha_after": self.alpha,
                             "agree_count": len(agree), "disagree_count": len(disagree)})
        return self.alpha

    def calibrate(self, prior, candidates, feature_names, operators):
        agree, disagree = [], []
        for candidate in deduplicate_by_root(candidates):
            decision = prior_agreement(prior, candidate, feature_names, operators)
            if decision is not None:
                (agree if decision else disagree).append(candidate["validation_reward"])
        self.update(agree, disagree)
        return self.history[-1]
