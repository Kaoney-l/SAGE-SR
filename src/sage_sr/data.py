"""Portable numeric datasets and deterministic internal validation splits.

Arrays use sample-major layout: ``X.shape == (n_samples, n_features)``.
CSV files require a header and an explicitly named target column. NPZ files
are loaded without pickle support; no data is downloaded at import time.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _validate_xy(X, y, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError(f"{name}: X must have shape (samples, features).")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{name}: y must contain one target per sample.")
    if len(y) == 0:
        raise ValueError(f"{name}: the dataset is empty.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{name}: observations must be finite.")
    return np.ascontiguousarray(X), np.ascontiguousarray(y)


@dataclass
class Dataset:
    """Training observations, feature names, and optional held-out test sets."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    X_test_id: np.ndarray | None = None
    y_test_id: np.ndarray | None = None
    X_test_ood: np.ndarray | None = None
    y_test_ood: np.ndarray | None = None

    def __post_init__(self):
        self.X, self.y = _validate_xy(self.X, self.y, name="training")
        self.feature_names = list(self.feature_names)
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("Provide exactly one feature name per input column.")
        if any(not isinstance(n, str) or not n.strip() for n in self.feature_names):
            raise ValueError("Feature names must be nonempty strings.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names must be unique.")
        for suffix in ("id", "ood"):
            x_name, y_name = f"X_test_{suffix}", f"y_test_{suffix}"
            test_X, test_y = getattr(self, x_name), getattr(self, y_name)
            if (test_X is None) != (test_y is None):
                raise ValueError(f"Provide both {x_name} and {y_name}.")
            if test_X is not None:
                test_X, test_y = _validate_xy(test_X, test_y, name=f"test_{suffix}")
                if test_X.shape[1] != self.X.shape[1]:
                    raise ValueError(f"{x_name} must use the training feature columns.")
                setattr(self, x_name, test_X)
                setattr(self, y_name, test_y)


def load_csv(path: str | Path, target: str,
             features: Sequence[str] | None = None) -> Dataset:
    """Read a numeric, headered CSV; preserve requested feature-column order."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [name.strip() for name in next(reader)]
        except StopIteration as exc:
            raise ValueError("CSV file is empty.") from exc
        if any(not name for name in header) or len(set(header)) != len(header):
            raise ValueError("CSV column names must be nonempty and unique.")
        if target not in header:
            raise ValueError(f"Target column {target!r} is not in the CSV header.")
        if isinstance(features, (str, bytes)):
            raise ValueError("features must be a sequence of column names.")
        names = list(features) if features is not None else [n for n in header if n != target]
        if not names or target in names or len(set(names)) != len(names):
            raise ValueError("Select nonempty, unique feature columns excluding the target.")
        if any(name not in header for name in names):
            raise ValueError("A requested feature column is not in the CSV header.")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV row {line_number} has an incorrect number of columns.")
            try:
                rows.append([float(cell) for cell in row])
            except ValueError as exc:
                raise ValueError(f"CSV row {line_number} contains a nonnumeric value.") from exc
    if not rows:
        raise ValueError("CSV file contains no numeric observations.")
    values = np.asarray(rows, dtype=np.float64)
    return Dataset(values[:, [header.index(n) for n in names]],
                   values[:, header.index(target)], names)


def load_npz(path: str | Path) -> Dataset:
    """Load ``X_train/y_train`` or ``X/y`` from a non-pickled NPZ archive.

    Optional arrays are ``feature_names``, ``X_test_id/y_test_id`` and
    ``X_test_ood/y_test_ood``. ``X_id/y_id`` and ``X_ood/y_ood`` are accepted
    aliases; ``X_test/y_test`` is also accepted for ID data. Canonical names
    take precedence when multiple complete aliases are present.
    """
    with np.load(Path(path), allow_pickle=False) as arrays:
        if "X_train" in arrays or "y_train" in arrays:
            x_key, y_key = "X_train", "y_train"
        else:
            x_key, y_key = "X", "y"
        if x_key not in arrays or y_key not in arrays:
            raise ValueError("NPZ must contain a matching X/y or X_train/y_train pair.")
        X, y = arrays[x_key], arrays[y_key]
        n_features = 1 if X.ndim == 1 else X.shape[1] if X.ndim == 2 else 0
        if "feature_names" in arrays:
            raw_names = np.asarray(arrays["feature_names"])
            if raw_names.ndim != 1 or raw_names.dtype.kind not in "US":
                raise ValueError("feature_names must be a one-dimensional string array.")
            names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in raw_names]
        else:
            names = [f"x{i}" for i in range(n_features)]

        def optional_pair(aliases):
            selected = None
            for first, second in aliases:
                if (first in arrays) != (second in arrays):
                    raise ValueError(f"Provide both {first} and {second} in the NPZ archive.")
                if first in arrays and selected is None:
                    selected = (arrays[first], arrays[second])
            return selected or (None, None)

        X_id, y_id = optional_pair([
            ("X_test_id", "y_test_id"), ("X_id", "y_id"), ("X_test", "y_test")])
        X_ood, y_ood = optional_pair([
            ("X_test_ood", "y_test_ood"), ("X_ood", "y_ood")])
    return Dataset(X, y, names, X_id, y_id, X_ood, y_ood)


def validation_split(X, y, validation_fraction: float = 0.15,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split observations reproducibly, without modifying the input arrays.

    The validation count is ``floor(fraction * n)``, clamped to ``[1, n-1]``
    for small datasets. At least two observations are required, so neither
    the search-training set nor the internal validation set is empty.
    """
    X, y = _validate_xy(X, y, name="validation split")
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one.")
    if len(y) < 2:
        raise ValueError("At least two observations are required for a validation split.")
    n_validation = min(len(y) - 1, max(1, int(validation_fraction * len(y))))
    order = np.random.default_rng(seed).permutation(len(y))
    validation, training = order[:n_validation], order[n_validation:]
    return X[training], y[training], X[validation], y[validation]
