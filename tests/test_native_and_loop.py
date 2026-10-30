import json
import numpy as np
import pytest
import torch
from sage_sr import _core, SearchConfig, StaticController, StructuredPrior, SAGESRRegressor
from sage_sr.metrics import reward


def test_native_uses_rmse_reward_and_fixed_fitted_constants():
    X = np.linspace(-1, 1, 80, dtype=np.float32)[None, :]
    y = (10 * X[0] + 3).astype(np.float32)
    cfg = _core.RegressorConfig()
    cfg.ops = ["+", "*", "R"]
    model = _core.Regressor(X, y, cfg)
    fitted = model.fit_expression([3])
    assert fitted.train_reward == pytest.approx(reward(y, X[0], 1, eta=float(cfg.len_gamma)), rel=2e-6)
    coefficients = model.fit_expression([0, 1, 2, 3, 2])
    before = list(coefficients.coefficients)
    coefficients.predict(X * 3)
    assert list(coefficients.coefficients) == before
    assert list(model.fit_expression([0, 1, 2, 3, 2]).coefficients) == before


def test_native_evaluation_budget_and_true_gp_records():
    X = np.linspace(-1, 1, 80, dtype=np.float32)[None, :]
    cfg = _core.RegressorConfig(); cfg.ops = ["+", "*", "R"]
    cfg.max_depth = 3; cfg.max_evals = 60; cfg.succ_error_tol = 0
    model = _core.Regressor(X, 2 * X[0] + 1, cfg)
    result = model.fit(seed=7)
    assert result.n_evals == len(result.search_records) == len(result.reward_curve) == 60
    edits = [r for r in result.search_records if r.kind in {"mutation", "crossover"}]
    assert edits and all(r.parents and r.root_ids for r in edits)


def test_full_offline_loop_exercises_conditioning_feedback_and_inheritance(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, (300, 1)); y = X[:, 0] ** 2 + X[:, 0] + .5
    prior = StructuredPrior(operator_prior={"+": .8, "-": .4, "*": .9, "R": .5},
                            variable_relevance={"x0": 1}, self_confidence={"operator_prior": .5})
    controller = StaticController(prior)
    cfg = SearchConfig(rounds=2, evaluations_per_round=100, hidden_size=16, max_depth=3,
                       operators=("+", "-", "*", "R"), distillation_epochs=1,
                       early_stop_error=None, seed=7)
    model = SAGESRRegressor(cfg, controller).fit(X, y)
    assert controller.calls == 2 and model.n_evaluations_ == 200
    assert model.rounds_[0]["distillation_loss"] > 0
    assert model.rounds_[1]["inherited_paths"] > 0
    assert model.rounds_[0]["calibration"]["alpha_before"] == 1
    assert np.isfinite(model.predict(X)).all()
    output = tmp_path / "run.json"; model.save(output)
    saved = json.loads(output.read_text())
    assert saved["config"]["eta"] == .99 and len(saved["search_records"]) == 200


def test_no_film_keeps_the_policy_but_skips_film_modulation():
    policy_prior = StructuredPrior(operator_prior={"+": .8, "R": .2})
    cfg = SearchConfig(rounds=1, evaluations_per_round=12, hidden_size=8, max_depth=2,
                       operators=("+", "R"), use_film=False, early_stop_error=None)
    X = np.linspace(0, 1, 40)[:, None]
    model = SAGESRRegressor(cfg, StaticController(policy_prior)).fit(X, X[:, 0] + .5)
    assert model.policy_ is not None and model.n_evaluations_ == 12
