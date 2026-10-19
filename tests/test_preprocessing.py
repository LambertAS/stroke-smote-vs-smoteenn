"""Checks on encoding, split, scaling and resampling (paper profile)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stroke_imbalance.features import ENCODING_MAPS, encode
from stroke_imbalance.resampling import resample

EXPECTED_COUNTS = {"none": [3889, 199], "smote": [3889, 3889], "smoteenn": [3148, 3618]}


def test_encode_applies_notebook_codes() -> None:
    frame = pd.DataFrame(
        {
            "gender": ["Other"],
            "ever_married": ["No"],
            "work_type": ["Never_worked"],
            "smoking_status": ["Unknown"],
            "Residence_type": ["Rural"],
        }
    )
    assert encode(frame).iloc[0].tolist() == [2, 1, 4, 3, 1]


def test_encode_rejects_unknown_category() -> None:
    row = {col: [next(iter(m))] for col, m in ENCODING_MAPS.items()}
    row["gender"] = ["Unknown"]
    with pytest.raises(ValueError):
        encode(pd.DataFrame(row))


def test_split_sizes(paper_data) -> None:
    assert paper_data.X_train.shape == (4088, 10)
    assert paper_data.X_test.shape == (1022, 10)
    assert np.bincount(paper_data.y_test).tolist() == [972, 50]


def test_no_missing_values_after_imputation(paper_data) -> None:
    assert not np.isnan(paper_data.X_train).any()
    assert not np.isnan(paper_data.X_test).any()


@pytest.mark.parametrize("regime", list(EXPECTED_COUNTS))
def test_class_counts_per_regime(paper_data, regime) -> None:
    _, y_res = resample(paper_data.X_train, paper_data.y_train, regime)
    assert np.bincount(y_res).tolist() == EXPECTED_COUNTS[regime]


def test_test_set_never_resampled(paper_data) -> None:
    X_test_before, y_test_before = paper_data.X_test.copy(), paper_data.y_test.copy()
    for regime in EXPECTED_COUNTS:
        resample(paper_data.X_train, paper_data.y_train, regime)
    np.testing.assert_array_equal(paper_data.X_test, X_test_before)
    np.testing.assert_array_equal(paper_data.y_test, y_test_before)


def test_scaler_fit_on_train_only(paper_data) -> None:
    scaler = paper_data.scaler
    np.testing.assert_allclose(scaler.data_min_, paper_data.X_train_raw.min().to_numpy())
    np.testing.assert_allclose(scaler.data_max_, paper_data.X_train_raw.max().to_numpy())
    np.testing.assert_allclose(paper_data.X_train.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(paper_data.X_train.max(axis=0), 1.0, atol=1e-12)
