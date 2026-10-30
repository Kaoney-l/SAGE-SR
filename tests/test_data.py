"""Data-contract checks independent of the C++ search backend."""

import numpy as np
import pytest

from sage_sr.data import Dataset, load_csv, load_npz, validation_split


def test_csv_target_location_and_feature_order(tmp_path):
    path = tmp_path / "observations.csv"
    path.write_text("first,response,second\n1,10,3\n2,20,4\n", encoding="utf-8")
    data = load_csv(path, target="response", features=["second", "first"])
    np.testing.assert_array_equal(data.X, [[3, 1], [4, 2]])
    np.testing.assert_array_equal(data.y, [10, 20])
    assert data.feature_names == ["second", "first"]


def test_csv_rejects_ambiguous_columns_and_nonfinite_data(tmp_path):
    path = tmp_path / "observations.csv"
    path.write_text("x,x,y\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_csv(path, "y")
    path.write_text("x,y\n1,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_csv(path, "y")


def test_npz_train_and_held_out_arrays(tmp_path):
    path = tmp_path / "observations.npz"
    np.savez(path, X_train=[[1, 2], [3, 4]], y_train=[5, 6],
             feature_names=np.array(["a", "b"]),
             X_test_id=[[7, 8]], y_test_id=[9], X_ood=[[10, 11]], y_ood=[12])
    data = load_npz(path)
    assert data.feature_names == ["a", "b"]
    np.testing.assert_array_equal(data.y_test_id, [9])
    np.testing.assert_array_equal(data.X_test_ood, [[10, 11]])


def test_npz_short_keys_default_names_and_incomplete_test_pair(tmp_path):
    path = tmp_path / "observations.npz"
    np.savez(path, X=[[1], [2]], y=[3, 4])
    assert load_npz(path).feature_names == ["x0"]
    np.savez(path, X=[[1], [2]], y=[3, 4], X_test_id=[[5]])
    with pytest.raises(ValueError, match="both X_test_id and y_test_id"):
        load_npz(path)


def test_dataset_rejects_misaligned_test_features():
    with pytest.raises(ValueError, match="training feature"):
        Dataset([[1], [2]], [3, 4], ["x"], X_test_id=[[1, 2]], y_test_id=[3])


def test_validation_fraction_is_not_a_hundred_point_minimum():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float)
    first = validation_split(X, y, seed=14)
    second = validation_split(X, y, seed=14)
    assert len(first[1]) == 17 and len(first[3]) == 3
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert not set(first[1]) & set(first[3])
    assert set(first[1]) | set(first[3]) == set(y)
    np.testing.assert_array_equal(y, np.arange(20))


def test_small_validation_split_keeps_training_nonempty():
    X_train, y_train, X_val, y_val = validation_split([[1], [2]], [10, 20])
    assert X_train.shape == X_val.shape == (1, 1)
    assert len(y_train) == len(y_val) == 1
    with pytest.raises(ValueError, match="At least two"):
        validation_split([[1]], [10])


@pytest.mark.parametrize("fraction", [0, 1, -0.2, float("nan")])
def test_invalid_validation_fraction(fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        validation_split([[1], [2]], [3, 4], fraction)
