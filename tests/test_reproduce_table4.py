"""Compare every Table IV cell to the original notebook output."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from stroke_imbalance.run import holdout_table

EXPECTED = pd.read_csv(Path(__file__).parent / "expected" / "table4_notebook.csv")
TOLERANCE = {"Accuracy": 1e-6, "Precision": 1e-6, "Recall": 1e-6, "F1": 1e-6, "PR_AUC": 1e-4}
XGB_SKIP_REASON = (
    "XGBoost rows are exact on Linux only (platform-dependent random sampling). See README."
)


@pytest.fixture(scope="module")
def reproduced(paper_data, paper_cfg) -> pd.DataFrame:
    return holdout_table(paper_data, paper_cfg)


def test_same_rows(reproduced) -> None:
    assert reproduced[["Model", "Resampling"]].equals(EXPECTED[["Model", "Resampling"]])


@pytest.mark.parametrize("row", range(len(EXPECTED)))
@pytest.mark.parametrize("metric", list(TOLERANCE))
def test_cell_matches_notebook(reproduced, row, metric) -> None:
    if EXPECTED.loc[row, "Model"] == "XGB" and not sys.platform.startswith("linux"):
        pytest.skip(XGB_SKIP_REASON)
    got, want = reproduced.loc[row, metric], EXPECTED.loc[row, metric]
    label = f"{EXPECTED.loc[row, 'Model']} {EXPECTED.loc[row, 'Resampling']} {metric}"
    assert got == pytest.approx(want, abs=TOLERANCE[metric]), f"{label}: got {got}, want {want}"
