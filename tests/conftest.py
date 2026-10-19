"""Shared fixtures. Tests that need the dataset are skipped when it has not been downloaded."""

from __future__ import annotations

import pytest

from stroke_imbalance.config import load_config
from stroke_imbalance.data import DEFAULT_PATH, load_raw
from stroke_imbalance.features import HoldoutData, prepare_holdout

requires_data = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason="dataset missing: run `python -m stroke_imbalance.data`"
)


@pytest.fixture(scope="session")
def paper_cfg() -> dict:
    return load_config("configs/paper.yaml")


@pytest.fixture(scope="session")
def raw_df():
    if not DEFAULT_PATH.exists():
        pytest.skip("dataset missing: run `python -m stroke_imbalance.data`")
    return load_raw()


@pytest.fixture(scope="session")
def paper_data(raw_df, paper_cfg) -> HoldoutData:
    return prepare_holdout(raw_df, paper_cfg)
