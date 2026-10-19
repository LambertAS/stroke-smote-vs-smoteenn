"""Smoke tests for the corrected profile, the Table IV comparison and the EDA/headline figures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stroke_imbalance.config import load_config
from stroke_imbalance.evaluate import METRIC_COLUMNS, compare_tables, format_comparison
from stroke_imbalance.features import prepare_holdout
from stroke_imbalance.plots import (
    plot_class_balance,
    plot_bmi_glucose_by_stroke,
    plot_numeric_distributions,
    plot_pr_auc_vs_recall,
)


def test_corrected_profile_differs_only_in_imputation(paper_cfg) -> None:
    corrected = load_config("configs/corrected.yaml")
    assert corrected["preprocessing"]["bmi_imputation"] == "train_mean"
    for key in ("models", "split", "regimes", "seed"):
        assert corrected[key] == paper_cfg[key]


def test_corrected_imputation_uses_training_mean(raw_df) -> None:
    data = prepare_holdout(raw_df, load_config("configs/corrected.yaml"))
    assert not data.X_train_raw["bmi"].isna().any()
    assert not data.X_test_raw["bmi"].isna().any()
    train_ids = data.X_train_raw.index
    missing = raw_df.loc[train_ids, "bmi"].isna()
    train_mean = raw_df.loc[train_ids, "bmi"].mean()
    np.testing.assert_allclose(data.X_train_raw.loc[missing[missing].index, "bmi"], train_mean)


def test_compare_tables_shape() -> None:
    base = pd.DataFrame(
        {"Model": ["LR"], "Resampling": ["SMOTE"], **dict.fromkeys(METRIC_COLUMNS, 0.5)}
    )
    other = base.assign(Recall=0.6)
    table = compare_tables(base, other, suffixes=("paper", "corrected"))
    assert table.shape == (1, 2 + 3 * len(METRIC_COLUMNS))
    assert table.loc[0, "Recall_diff"] == 0.6 - 0.5
    assert format_comparison(table, "corrected").loc[0, "Recall"] == "0.6000 (+0.1000)"


def test_figures_written(raw_df, tmp_path) -> None:
    table4 = pd.read_csv("tests/expected/table4_notebook.csv")
    paths = [
        plot_pr_auc_vs_recall(table4, 0.049, tmp_path / "headline.png", "test"),
        plot_class_balance(raw_df, tmp_path / "fig2.png"),
        plot_numeric_distributions(raw_df, tmp_path / "fig3.png"),
        plot_bmi_glucose_by_stroke(raw_df, tmp_path / "fig4.png"),
    ]
    assert all(p.stat().st_size > 0 for p in paths)
