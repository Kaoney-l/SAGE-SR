"""Cases that distinguish the published metric from legacy runner behavior."""

import numpy as np
import pytest

from sage_sr.metrics import RELATIVE_ERROR_EPSILON, evaluate, reward


def test_nmse_uses_retained_target_variance():
    true = np.arange(1, 21, dtype=float)
    pred = true + 0.001
    pred[-1] += 100
    result = evaluate(true, pred)
    expected_mse = np.mean((pred[:-1] - true[:-1]) ** 2)
    assert result["retained_points"] == 19
    assert result["nmse"] == pytest.approx(expected_mse / np.var(true[:-1]))
    assert not np.isclose(result["nmse"], expected_mse / np.var(true), rtol=1e-7, atol=0)
    assert result["success"] is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_one_nonfinite_prediction_fails_whole_task(bad):
    true = np.arange(1, 101, dtype=float)
    pred = true.copy()
    pred[-1] = bad
    result = evaluate(true, pred)
    assert result["success"] is False
    assert result["retained_points"] == 0
    assert np.isinf(result["nmse"])


def test_accuracy_is_one_task_indicator_not_a_point_fraction():
    true = np.ones(100)
    pred = np.ones(100)
    pred[:10] = 1.02
    result = evaluate(true, pred)
    assert result["success"] is False
    assert result["max_relative_error"] == pytest.approx(0.02)


def test_zero_target_denominator_convention_is_explicit():
    result = evaluate([0, 1, 2], [1e-6, 1, 2], trim_fraction=0)
    assert result["max_relative_error"] == pytest.approx(1e-6 / RELATIVE_ERROR_EPSILON)
    assert result["success"] is False
    exact = evaluate([0, 1, 2], [0, 1, 2])
    assert exact["success"] is True and exact["nmse"] == 0


def test_constant_target_nmse_has_no_arbitrary_variance_floor():
    assert evaluate([0, 0, 0], [0, 0, 0])["nmse"] == 0
    result = evaluate([0, 0, 0], [1, 1, 1])
    assert np.isinf(result["nmse"]) and result["success"] is False


def test_percentile_ties_are_retained():
    result = evaluate([1, 2, 3, 4], [1, 2, 3, 4])
    assert result["retained_points"] == 4


def test_reward_is_rmse_with_length_penalty_not_nrmse():
    true, pred = [100, 200], [101, 201]
    assert reward(true, pred, n_nodes=3) == pytest.approx(0.99 ** 3 / 2)
    assert reward(true, true, n_nodes=5) == pytest.approx(0.99 ** 5)
    assert reward(true, pred, n_nodes=3, eta=1) == pytest.approx(0.5)
    assert reward(true, [float("nan"), 201], n_nodes=3) == 0
    assert reward(true, pred, n_nodes=0) == 0


def test_invalid_metric_arguments_do_not_silently_change_protocol():
    with pytest.raises(ValueError, match="trim_fraction"):
        evaluate([1, 2], [1, 2], trim_fraction=1)
    with pytest.raises(ValueError, match="relative_tolerance"):
        evaluate([1, 2], [1, 2], relative_tolerance=-1)
    with pytest.raises(ValueError, match="finite target"):
        evaluate([1, float("nan")], [1, 2])
    with pytest.raises(ValueError, match="eta"):
        reward([1, 2], [1, 2], 2, eta=0)
