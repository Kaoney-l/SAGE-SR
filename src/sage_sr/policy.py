"""Autoregressive GRU policy, FiLM conditioning and elite distillation."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from .priors import CONDITION_DIM, OPERATOR_ORDER, CompiledPrior


LegalActions = Callable[[list[int]], Sequence[int]]


class SymbolicPolicy(nn.Module):
    """A symbolic policy whose inference and teacher forcing both include BOS.

    The backend provides the complete grammar mask through ``legal_actions``.
    Without that callback, the policy enforces prefix completion and the
    constant budget; additional tree constraints remain the backend's job.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 192,
        condition_dim: int = CONDITION_DIM,
        max_constants: int | None = None,
    ) -> None:
        super().__init__()
        if n_features < 1 or hidden_size < 1 or condition_dim < 1:
            raise ValueError("Feature count and policy dimensions must be positive.")
        if max_constants is not None and max_constants < 0:
            raise ValueError("max_constants must be nonnegative.")
        self.n_features, self.hidden_size, self.condition_dim = int(n_features), int(hidden_size), int(condition_dim)
        self.max_constants = max_constants
        self.names = list(OPERATOR_ORDER) + ["R"] + [f"x{i}" for i in range(n_features)]
        self.name_to_id = {name: index for index, name in enumerate(self.names)}
        self.bos_id = len(self.names)
        self.embedding = nn.Embedding(len(self.names) + 1, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, len(self.names))
        self.film = nn.Linear(condition_dim, 2 * hidden_size)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        with torch.no_grad():
            self.film.bias[:hidden_size].fill_(1.0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _condition_tensor(self, condition) -> torch.Tensor | None:
        if condition is None:
            return None
        value = torch.as_tensor(condition, dtype=self.embedding.weight.dtype, device=self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[-1] != self.condition_dim:
            raise ValueError("The semantic condition has the wrong dimension.")
        return value

    def _film(self, hidden: torch.Tensor, condition) -> torch.Tensor:
        value = self._condition_tensor(condition)
        if value is None:
            return hidden
        scale, shift = self.film(value).chunk(2, dim=-1)
        return scale.unsqueeze(1) * hidden + shift.unsqueeze(1)

    def _engine_names(self, operators: Sequence[str]) -> list[str]:
        names = list(operators) + [f"x{i}" for i in range(self.n_features)]
        if len(names) != len(set(names)):
            raise ValueError("The engine token names must be unique.")
        if any(name not in self.name_to_id for name in names):
            raise ValueError("The policy supports the paper's arithmetic and elementary-function grammar.")
        return names

    def _legal_indices(self, prefix: list[int], names: list[str], legal_actions: LegalActions | None) -> list[int]:
        if legal_actions is not None:
            indices = [int(index) for index in legal_actions(prefix)]
            if any(index < 0 or index >= len(names) for index in indices):
                raise ValueError("The backend returned an invalid action index.")
            return indices
        slots, constants = 1, 0
        for index in prefix:
            if index < 0 or index >= len(names) or slots <= 0:
                raise ValueError("Invalid expression prefix.")
            name = names[index]
            arity = 2 if name in {"+", "-", "*", "/"} else (1 if name in OPERATOR_ORDER[4:] else 0)
            slots += arity - 1
            constants += name == "R"
        if slots == 0:
            return []
        return [i for i, name in enumerate(names)
                if not (name == "R" and self.max_constants is not None and constants >= self.max_constants)]

    def _sequence_logits(self, token_ids: list[int], condition) -> torch.Tensor:
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        hidden, _ = self.gru(self.embedding(tokens))
        return self.head(self._film(hidden, condition))[0]

    def prior_callback(
        self,
        operators: Sequence[str],
        compiled: CompiledPrior,
        alpha: float = 1.0,
        use_film: bool = True,
        legal_actions: LegalActions | None = None,
    ) -> Callable[[list[int]], list[float]]:
        """Create a round-local cached action prior in backend token order."""
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("The explicit-bias trust coefficient must be finite and nonnegative.")
        self.eval()
        names = self._engine_names(operators)
        token_ids = [self.name_to_id[name] for name in names]
        engine_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        bias = torch.tensor([alpha * compiled.token_bias.get(name, 0.0) for name in names],
                            dtype=self.embedding.weight.dtype, device=self.device)
        condition = np.array(compiled.condition, dtype=np.float32, copy=True) if use_film else None
        cache: dict[tuple[int, ...], list[float]] = {}

        @torch.no_grad()
        def callback(prefix: list[int]) -> list[float]:
            key = tuple(int(index) for index in prefix)
            if key in cache:
                return list(cache[key])
            legal = self._legal_indices(list(key), names, legal_actions)
            if not legal:
                return [0.0] * len(names)
            sequence = [self.bos_id] + [token_ids[index] for index in key]
            logits = self._sequence_logits(sequence, condition)[-1].index_select(0, engine_ids) + bias
            mask = torch.ones(len(names), dtype=torch.bool, device=self.device)
            mask[legal] = False
            logits = logits.masked_fill(mask, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            cache[key] = probabilities
            return list(probabilities)

        return callback

    def log_likelihood(
        self,
        path: Sequence[int],
        operators: Sequence[str],
        condition=None,
        legal_actions: LegalActions | None = None,
    ) -> torch.Tensor:
        """Log likelihood under the conditional policy, excluding explicit bias."""
        path = [int(index) for index in path]
        if not path:
            raise ValueError("A demonstration path cannot be empty.")
        names = self._engine_names(operators)
        if any(index < 0 or index >= len(names) for index in path):
            raise ValueError("A demonstration contains an invalid token.")
        engine_ids = torch.tensor([self.name_to_id[name] for name in names], dtype=torch.long, device=self.device)
        sequence = [self.bos_id] + [self.name_to_id[names[index]] for index in path[:-1]]
        logits = self._sequence_logits(sequence, condition).index_select(1, engine_ids)
        terms = []
        for step, action in enumerate(path):
            legal = self._legal_indices(path[:step], names, legal_actions)
            if action not in legal:
                raise ValueError("A demonstration violates the configured grammar.")
            mask = torch.ones(len(names), dtype=torch.bool, device=self.device)
            mask[legal] = False
            log_probs = torch.log_softmax(logits[step].masked_fill(mask, float("-inf")), dim=-1)
            terms.append(log_probs[action])
        return torch.stack(terms).sum()

    def distill(
        self,
        demos: Sequence[tuple[Sequence[int], float]],
        condition,
        operators: Sequence[str],
        lr: float = 1e-3,
        epochs: int = 3,
        legal_actions: LegalActions | None = None,
    ) -> float:
        """Fit reward-weighted FIFO demonstrations using the current condition.

        Buffer retention is managed by the estimator. No additional age-based
        decay is applied. Only the learned policy and FiLM parameters are fit;
        the explicit token bias is not part of this imitation objective.
        """
        if not demos:
            return 0.0
        if lr <= 0 or epochs < 1:
            raise ValueError("Distillation needs a positive learning rate and epoch count.")
        rewards = np.asarray([max(float(reward), 1e-8) for _, reward in demos], dtype=np.float64)
        if not np.all(np.isfinite(rewards)):
            raise ValueError("Demonstration rewards must be finite.")
        rewards /= rewards.sum()
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        last = 0.0
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = sum(float(weight) * -self.log_likelihood(path, operators, condition, legal_actions)
                       for (path, _), weight in zip(demos, rewards))
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite elite distillation loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()
            last = float(loss.detach())
        self.eval()
        return last
