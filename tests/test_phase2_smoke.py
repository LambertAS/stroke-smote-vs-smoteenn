"""Smoke tests for the Phase 2 reconstructions: they run, have the right shape, write files.

These analyses have no original code, so nothing here compares against the paper's numbers.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from stroke_imbalance.calibration import calibration_table
from stroke_imbalance.cv import SCORING, cv_folds, format_mean_std, summarize_folds
from stroke_imbalance.explain import lr_coefficients, shap_importance, xgb_gain_importance
from stroke_imbalance.models import make_model
from stroke_imbalance.plots import plot_reliability, plot_shap_bar
from stroke_imbalance.resampling import resample


@pytest.fixture(scope="module")
def small_cfg(paper_cfg) -> dict:
    cfg = copy.deepcopy(paper_cfg)
    cfg["models"] = {k: cfg["models"][k] for k in ("lr", "xgb")}
    cfg["cv"].update(n_splits=2, n_repeats=1, n_jobs=1)
    return cfg


@pytest.fixture(scope="module")
def smote_models(paper_data, paper_cfg) -> dict:
    X_res, y_res = resample(paper_data.X_train, paper_data.y_train, "smote")
    return {name: make_model(name, "smote", paper_cfg).fit(X_res, y_res) for name in ("lr", "xgb")}


def test_cv_shapes(paper_data, small_cfg) -> None:
    folds = cv_folds(paper_data.X_train_raw, paper_data.y_train, small_cfg)
    assert len(folds) == 2 * 2 * 2  # models x regimes x folds
    assert folds[list(SCORING)].apply(lambda s: s.between(0, 1).all()).all()
    summary = summarize_folds(folds)
    assert summary.shape == (4, 2 + 2 * len(SCORING))
    assert format_mean_std(summary)["PR_AUC"].str.contains("±").all()


def test_explanations(paper_data, smote_models) -> None:
    names = paper_data.feature_names
    shap_df = shap_importance(smote_models["xgb"], paper_data.X_test, names)
    assert sorted(shap_df["Feature"]) == sorted(names)
    assert (shap_df["Mean_abs_SHAP"] >= 0).all()
    assert shap_df["Mean_abs_SHAP"].is_monotonic_decreasing
    gain = xgb_gain_importance(smote_models["xgb"], names)
    assert gain["Gain"].sum() == pytest.approx(1.0)
    coefs = lr_coefficients(smote_models["lr"], names)
    assert len(coefs) == len(names)
    assert set(coefs["Sign"]) <= {"+", "-"}


def test_calibration_and_figures(paper_data, smote_models, tmp_path) -> None:
    models = {"LR": smote_models["lr"], "XGB": smote_models["xgb"]}
    briers, curves = calibration_table(models, paper_data.X_test, paper_data.y_test, n_bins=10)
    assert list(briers["Model"]) == ["LR", "XGB"]
    assert briers["Brier"].between(0, 1).all()
    assert np.all((curves[["Mean_predicted", "Fraction_positive"]] >= 0).to_numpy())

    shap_df = shap_importance(smote_models["xgb"], paper_data.X_test, paper_data.feature_names)
    fig5 = plot_shap_bar(shap_df, tmp_path / "fig5.png", "test")
    fig6 = plot_reliability(curves, briers, tmp_path / "fig6.png", "test")
    assert fig5.stat().st_size > 0
    assert fig6.stat().st_size > 0
