"""Search rewards and explicit task-level test metrics for SAGE-SR.

The test protocol trims by relative-error percentile, then normalizes MSE by
the *retained* targets' variance. Its success flag is task-level, not a
fraction of accurate sample points. Ties at the percentile are retained.

Relative errors divide by ``max(abs(y), relative_epsilon)``; the default
float64 machine epsilon is an explicit numerical guard at zero targets.
Constant targets have NMSE 0 for an exact prediction and
infinity otherwise. Any non-finite prediction fails the entire task.
"""

from __future__ import annotations

import operator

import numpy as np


RELATIVE_ERROR_EPSILON = np.finfo(np.float64).eps


def _vectors(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    if true.ndim == 2 and true.shape[1] == 1:
        true = true[:, 0]
    if true.ndim != 1 or true.size == 0 or not np.all(np.isfinite(true)):
        raise ValueError("y_true must be a nonempty finite target vector.")
    pred = np.asarray(y_pred, dtype=np.float64)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim > 1:
        raise ValueError("y_pred must contain one prediction per target.")
    try:
        pred = np.broadcast_to(pred, true.shape)
    except ValueError as exc:
        raise ValueError("y_pred must match the target shape or be a scalar.") from exc
    return true, pred


def _failed() -> dict:
    return {"nmse": float("inf"), "success": False,
            "retained_points": 0, "max_relative_error": float("inf")}


def evaluate(y_true, y_pred, trim_fraction: float = 0.05,
             relative_tolerance: float = 0.01,
             relative_epsilon: float = RELATIVE_ERROR_EPSILON) -> dict:
    """Return trimmed NMSE and a Boolean task-success flag.

    ``trim_fraction`` is in ``[0, 1)``. Points with relative error at or
    below quantile ``1 - trim_fraction`` are retained. Success requires the
    maximum retained relative error to be at most ``relative_tolerance``.
    Average ``success`` across tasks and multiply by 100 for accuracy in
    percent. Non-finite predictions are not discarded as apparent outliers.
    """
    if not np.isfinite(trim_fraction) or not 0 <= trim_fraction < 1:
        raise ValueError("trim_fraction must be in [0, 1).")
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("relative_tolerance must be finite and nonnegative.")
    if not np.isfinite(relative_epsilon) or relative_epsilon <= 0:
        raise ValueError("relative_epsilon must be positive and finite.")
    true, pred = _vectors(y_true, y_pred)
    if not np.all(np.isfinite(pred)):
        return _failed()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        errors = pred - true
        relative = np.abs(errors) / np.maximum(np.abs(true), relative_epsilon)
    if not np.all(np.isfinite(relative)):
        return _failed()
    keep = relative <= np.quantile(relative, 1.0 - trim_fraction)
    if not np.any(keep):
        return _failed()
    with np.errstate(over="ignore", invalid="ignore"):
        mse = float(np.mean(errors[keep] ** 2))
        variance = float(np.var(true[keep]))
    if variance == 0:
        nmse = 0.0 if mse == 0 else float("inf")
    else:
        nmse = mse / variance
        if not np.isfinite(nmse):
            nmse = float("inf")
    maximum = float(np.max(relative[keep]))
    return {"nmse": float(nmse), "success": bool(maximum <= relative_tolerance),
            "retained_points": int(np.count_nonzero(keep)), "max_relative_error": maximum}


def reward(y_true, y_pred, n_nodes: int, eta: float = 0.99) -> float:
    """Return ``eta**n_nodes / (1 + RMSE)``; invalid predictions receive 0.

    ``eta`` must lie in ``(0, 1]``; setting it to 1 disables the length
    penalty. A complete expression must have a positive integer node count.
    """
    if not np.isfinite(eta) or not 0 < eta <= 1:
        raise ValueError("eta must lie in (0, 1].")
    try:
        n_nodes = operator.index(n_nodes)
        if n_nodes < 1:
            return 0.0
        true, pred = _vectors(y_true, y_pred)
    except (TypeError, ValueError):
        return 0.0
    if not np.all(np.isfinite(pred)):
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    if not np.isfinite(rmse):
        return 0.0
    return float(eta ** n_nodes / (1.0 + rmse))
