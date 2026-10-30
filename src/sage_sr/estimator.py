"""Round-based semantic guidance, native hybrid search and evidence updates."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import _core
from .config import SearchConfig
from .data import _validate_xy, validation_split
from .feedback import AlphaCalibrator, audit_structure
from .metrics import evaluate, reward
from .policy import SymbolicPolicy
from .priors import CompiledPrior, LLMController, compile_prior


def _raw_nmse(y, prediction):
    if not np.all(np.isfinite(prediction)):
        return float("inf")
    mse, variance = float(np.mean((prediction - y) ** 2)), float(np.var(y))
    return mse / variance if variance > 0 else (0.0 if mse == 0 else float("inf"))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class SAGESRRegressor:
    """SAGE-SR with explicit task semantics and a fixed evaluation budget.

    ``controller=None`` uses the configured LLM service. A StaticController
    must be passed explicitly for static-prior ablations or offline checks.
    Test sets never participate in fit, prior generation or early stopping.
    """
    def __init__(self, config: SearchConfig | None = None, controller=None):
        self.config = config or SearchConfig()
        self.controller = controller

    def _backend_config(self, compiled):
        cfg, c = _core.RegressorConfig(), self.config
        for name, value in {"ops": list(c.operators), "max_depth": c.max_depth, "K": c.queue_size,
                            "c": c.exploration_c, "gamma": c.exploration_exponent,
                            "gp_rate": c.gp_rate, "mutation_rate": c.mutation_rate,
                            "exploration_rate": c.random_exploration, "max_unary": c.max_unary,
                            "max_constants": c.max_constants, "lm_iterations": c.lm_iterations,
                            "max_evals": c.evaluations_per_round, "len_gamma": c.eta,
                            "puct_c": c.prior_weight, "reward_normalize": False,
                            "succ_error_tol": 0.0}.items():
            setattr(cfg, name, value)
        cfg.set_mutation_weights(list(compiled.mutation_weights) if c.use_semantic_mutation else [1.0] * 4)
        return cfg

    def fit(self, X, y, *, feature_names=None, context=None, X_val=None, y_val=None):
        c = self.config
        X, y = _validate_xy(X, y, name="fit")
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = list(feature_names or [f"x{i}" for i in range(X.shape[1])])
        if len(self.feature_names_) != X.shape[1] or len(set(self.feature_names_)) != len(self.feature_names_):
            raise ValueError("Feature names must be unique and match the input columns.")
        if (X_val is None) != (y_val is None):
            raise ValueError("Provide both X_val and y_val.")
        if X_val is None:
            Xt, yt, Xv, yv = validation_split(X, y, c.validation_fraction, c.validation_seed)
        else:
            Xt, yt = X, y
            Xv, yv = _validate_xy(X_val, y_val, name="validation")
            if Xv.shape[1] != Xt.shape[1]:
                raise ValueError("Validation input has the wrong feature count.")
        np.random.seed(c.seed)
        torch.manual_seed(c.seed)
        self.policy_ = SymbolicPolicy(X.shape[1], c.hidden_size, max_constants=c.max_constants)
        controller = self.controller if self.controller is not None else LLMController()
        task_context = {"task": context or {}, "feature_names": self.feature_names_,
                        "training_statistics": {"samples": len(yt), "target_mean": float(yt.mean()),
                                                "target_std": float(yt.std()),
                                                "feature_min": Xt.min(axis=0).tolist(),
                                                "feature_max": Xt.max(axis=0).tolist(),
                                                "feature_mean": Xt.mean(axis=0).tolist(),
                                                "feature_std": Xt.std(axis=0).tolist()}}
        self.history_, self.search_records_, self.rounds_ = [], [], []
        self.n_evaluations_, self.best_validation_reward_ = 0, -1.0
        self._best_fitted = None
        self.demonstrations_ = deque(maxlen=c.demonstration_capacity)
        inherited = []
        root_by_path = {}
        prior = calibrator = None
        for round_index in range(c.rounds):
            started = time.perf_counter()
            if round_index == 0 or c.update_semantic_prior:
                prior = controller.generate(task_context, self.history_, list(c.operators))
            compiled = compile_prior(prior, self.feature_names_, c.operators)
            if calibrator is None:
                calibrator = AlphaCalibrator(prior.self_confidence.get("operator_prior", 0.5),
                                              c.trust_sensitivity, c.trust_cap,
                                              c.conflict_threshold, c.conflict_patience)
            alpha = calibrator.alpha
            model = _core.Regressor(np.asarray(Xt.T, dtype=np.float32), np.asarray(yt, dtype=np.float32), self._backend_config(compiled))
            model.set_policy_prior(self.policy_.prior_callback(c.operators, compiled,
                                   alpha=alpha if c.use_token_bias else 0.0,
                                   use_film=c.use_film, legal_actions=model.legal_actions))
            kwargs = {"seed": c.seed * c.rounds + round_index}
            if inherited:
                kwargs.update(init_paths=[e["path"] for e in inherited],
                              init_rewards=[e["reward"] for e in inherited],
                              init_coefficients=[e["coefficients"] for e in inherited],
                              init_root_ids=[e["root_ids"] for e in inherited])
            native = model.fit(**kwargs)
            if native.n_evals > c.evaluations_per_round:
                raise RuntimeError("The backend exceeded its configured evaluation budget.")
            self.n_evaluations_ += native.n_evals
            round_records = []
            for record in native.search_records:
                path = [int(v) for v in record.path]
                roots = list(record.root_ids)
                root_by_path[tuple(path)] = roots
                row = {"round": round_index + 1, "evaluation": int(record.evaluation),
                       "kind": record.kind, "operation": record.operation, "path": path,
                       "parents": [list(p) for p in record.parents],
                       "parent_rewards": list(record.parent_rewards), "root_ids": roots,
                       "reward": float(record.reward)}
                if record.kind == "mutation" and record.parent_rewards:
                    row["delta_reward"] = float(record.reward - record.parent_rewards[0])
                round_records.append(row)
            self.search_records_.extend(round_records)
            elites, fitted_by_path = [], {}
            names = list(c.operators) + [f"x{i}" for i in range(X.shape[1])]
            for path, training_reward in zip(native.top_paths[:c.training_elites], native.top_rewards[:c.training_elites]):
                path = [int(v) for v in path]
                fitted = model.fit_expression(path)
                # The fitted native object owns the training-fitted constants.
                prediction = np.asarray(fitted.predict(np.asarray(Xv.T, dtype=np.float32)), dtype=float)
                validation_reward = reward(yv, prediction, len(path), c.eta)
                candidate = {"path": path, "expression": fitted.expression,
                             "coefficients": list(fitted.coefficients), "reward": float(training_reward),
                             "validation_reward": validation_reward,
                             "root_ids": root_by_path.get(tuple(path), ["structure:" + ",".join(map(str, path))]),
                             "token_names": [names[i] for i in path]}
                elites.append(candidate)
                fitted_by_path[tuple(path)] = fitted
            if not elites:
                raise RuntimeError("Search returned no evaluated candidate.")
            representative = max(elites[:c.representative_candidates], key=lambda e: e["validation_reward"])
            fitted = fitted_by_path[tuple(representative["path"])]
            if representative["validation_reward"] > self.best_validation_reward_:
                self.best_validation_reward_ = representative["validation_reward"]
                self._best_fitted, self._best_model = fitted, model
                self.best_candidate_ = dict(representative)
            train_error = _raw_nmse(yt, np.asarray(fitted.predict(np.asarray(Xt.T, dtype=np.float32))))
            validation_error = _raw_nmse(yv, np.asarray(fitted.predict(np.asarray(Xv.T, dtype=np.float32))))
            stop = c.early_stop_error is not None and max(train_error, validation_error) <= c.early_stop_error
            record = {"round": round_index + 1, "n_evaluations": native.n_evals,
                      "seed": kwargs["seed"], "best_training_reward": float(native.best_reward),
                      "best_validation_reward_so_far": self.best_validation_reward_,
                      "total_evaluations": self.n_evaluations_, "alpha": alpha,
                      "prior": prior.to_dict(), "mutation_weights": list(compiled.mutation_weights),
                      "representative": representative, "train_nmse": train_error,
                      "validation_nmse": validation_error, "inherited_paths": len(inherited),
                      "elapsed_seconds": time.perf_counter() - started, "early_stop": stop}
            if not stop and round_index + 1 < c.rounds:
                if c.distill_elites:
                    for elite in elites:
                        self.demonstrations_.append((elite["path"], elite["reward"]))
                    record["distillation_loss"] = self.policy_.distill(list(self.demonstrations_),
                        compiled.condition if c.use_film else None, c.operators,
                        c.distillation_lr, c.distillation_epochs, model.legal_actions)
                record["audit"] = audit_structure(elites)
                if c.calibrate_trust:
                    record["calibration"] = calibrator.calibrate(prior, elites, self.feature_names_, c.operators)
                inherited = elites[:c.inherited_paths] if c.inherit_elites else []
                self.history_.append({"round": round_index + 1, "representative": representative,
                                      "audit": record["audit"], "calibration": record.get("calibration")})
            self.rounds_.append(record)
            if stop:
                break
        self.expression_ = self.best_candidate_["expression"]
        self.controller_calls_ = getattr(controller, "calls", None)
        return self

    def predict(self, X):
        if not hasattr(self, "_best_fitted") or self._best_fitted is None:
            raise RuntimeError("Fit the regressor before prediction.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction input has the wrong feature count.")
        return np.asarray(self._best_fitted.predict(X.T), dtype=float)

    def evaluate(self, X, y):
        return evaluate(y, self.predict(X))

    def save(self, path):
        """Export configuration, fitted constants and actual search evidence."""
        if not hasattr(self, "_best_fitted") or self._best_fitted is None:
            raise RuntimeError("Fit the regressor before saving results.")
        result = {"method": "SAGE-SR", "version": "0.1.0", "config": self.config.to_dict(),
                  "feature_names": self.feature_names_, "best_candidate": self.best_candidate_,
                  "n_evaluations": self.n_evaluations_, "controller_calls": self.controller_calls_,
                  "rounds": self.rounds_, "search_records": self.search_records_}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
