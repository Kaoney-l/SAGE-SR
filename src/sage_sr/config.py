"""Explicit configuration for the SAGE-SR search and feedback loop."""
from dataclasses import asdict, dataclass
import math


@dataclass
class SearchConfig:
    rounds: int = 4
    evaluations_per_round: int = 25000
    seed: int = 0
    operators: tuple[str, ...] = ("+", "-", "*", "/", "sin", "cos", "exp", "log", "R")
    validation_fraction: float = 0.15
    validation_seed: int = 0
    eta: float = 0.99
    max_depth: int = 6
    max_unary: int = 999
    max_constants: int = 999
    queue_size: int = 500
    exploration_c: float = 6.0
    exploration_exponent: float = 0.5
    random_exploration: float = 0.1
    prior_weight: float = 1.0
    gp_rate: float = 0.5
    mutation_rate: float = 0.1
    lm_iterations: int = 10
    training_elites: int = 8
    inherited_paths: int = 6
    representative_candidates: int = 6
    demonstration_capacity: int = 32
    hidden_size: int = 192
    distillation_lr: float = 0.001
    distillation_epochs: int = 3
    trust_cap: float = 3.0
    trust_sensitivity: float = 10.0
    conflict_threshold: float = -0.3
    conflict_patience: int = 2
    early_stop_error: float | None = 1e-6
    use_film: bool = True
    use_token_bias: bool = True
    use_semantic_mutation: bool = True
    update_semantic_prior: bool = True
    calibrate_trust: bool = True
    inherit_elites: bool = True
    distill_elites: bool = True

    def __post_init__(self):
        self.operators = tuple(self.operators)
        allowed = {"+", "-", "*", "/", "sin", "cos", "exp", "log", "R"}
        if not self.operators or len(set(self.operators)) != len(self.operators) or set(self.operators) - allowed:
            raise ValueError("Use a unique subset of the arithmetic and elementary-function grammar.")
        for name in ["rounds", "evaluations_per_round", "max_depth", "queue_size", "lm_iterations", "training_elites", "inherited_paths", "representative_candidates", "demonstration_capacity", "hidden_size", "distillation_epochs", "conflict_patience"]:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.training_elites > 8 or self.representative_candidates > self.training_elites or self.inherited_paths > self.training_elites:
            raise ValueError("At most eight training elites are exported; selection and inheritance must be subsets.")
        if not 0 < self.validation_fraction < 1 or not 0 < self.eta <= 1:
            raise ValueError("Invalid validation fraction or parsimony coefficient.")
        for name in ["gp_rate", "mutation_rate", "random_exploration"]:
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0,1].")
        for name in ["exploration_c", "exploration_exponent", "prior_weight", "distillation_lr", "trust_cap", "trust_sensitivity"]:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.max_constants < 0 or self.max_unary < 0 or self.seed < 0 or self.validation_seed < 0:
            raise ValueError("Grammar limits and seed must be nonnegative.")
        if self.early_stop_error is not None and (not math.isfinite(self.early_stop_error) or self.early_stop_error < 0):
            raise ValueError("early_stop_error must be nonnegative, or None.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
        return cls(**values)
