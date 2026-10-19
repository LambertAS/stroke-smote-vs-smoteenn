"""Command-line entry point.

Usage: ``python -m stroke_imbalance.run --config configs/paper.yaml [--stages holdout,cv]``

Stages: holdout (Tables III and IV, PR-AUC vs recall chart), eda (Fig 2 to 4), cv (Table V),
explain (Fig 5, gain, LR coefficients), calibration (Fig 6). By default a profile runs the
stages listed under ``stages`` in its YAML, or all of them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stroke_imbalance.calibration import calibration_table
from stroke_imbalance.config import load_config, set_seeds
from stroke_imbalance.cv import cv_folds, format_mean_std, summarize_folds
from stroke_imbalance.data import load_raw
from stroke_imbalance.evaluate import (
    METRIC_COLUMNS,
    compare_tables,
    format_comparison,
    holdout_metrics,
)
from stroke_imbalance.explain import lr_coefficients, shap_importance, xgb_gain_importance
from stroke_imbalance.features import HoldoutData, prepare_holdout
from stroke_imbalance.models import MODEL_LABELS, make_model
from stroke_imbalance.plots import (
    plot_bmi_glucose_by_stroke,
    plot_class_balance,
    plot_numeric_distributions,
    plot_pr_auc_vs_recall,
    plot_reliability,
    plot_shap_bar,
)
from stroke_imbalance.resampling import REGIME_LABELS, resample

STAGES = ["holdout", "eda", "cv", "explain", "calibration"]
Models = dict[tuple[str, str], Any]


def class_counts(data: HoldoutData, cfg: dict) -> pd.DataFrame:
    """Table III: training-set class counts for each resampling regime."""
    rows = []
    for regime in cfg["regimes"]:
        _, y_res = resample(data.X_train, data.y_train, regime, cfg["seed"])
        neg, pos = np.bincount(y_res, minlength=2)
        rows.append(
            {"Regime": REGIME_LABELS[regime], "No stroke": neg, "Stroke": pos, "Total": neg + pos}
        )
    return pd.DataFrame(rows)


def fit_holdout_models(data: HoldoutData, cfg: dict) -> Models:
    """Fit every model on every resampled training set, keyed by (model, regime)."""
    resampled = {r: resample(data.X_train, data.y_train, r, cfg["seed"]) for r in cfg["regimes"]}
    return {
        (name, regime): make_model(name, regime, cfg).fit(X_res, y_res)
        for name in cfg["models"]
        for regime, (X_res, y_res) in resampled.items()
    }


def holdout_table(data: HoldoutData, cfg: dict, models: Models | None = None) -> pd.DataFrame:
    """Table IV: every model trained on every regime, scored on the untouched test set."""
    models = models or fit_holdout_models(data, cfg)
    rows = [
        {
            "Model": MODEL_LABELS[name],
            "Resampling": REGIME_LABELS[regime],
            **holdout_metrics(model, data.X_test, data.y_test),
        }
        for (name, regime), model in models.items()
    ]
    return pd.DataFrame(rows, columns=["Model", "Resampling", *METRIC_COLUMNS])


def to_markdown(df: pd.DataFrame, decimals: int = 3) -> str:
    """Render a DataFrame as a GitHub Markdown table, rounding floats for display only."""
    cells = df.map(lambda v: f"{v:.{decimals}f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.to_numpy()]
    return "\n".join(lines) + "\n"


def save_table(
    df: pd.DataFrame, out_dir: Path, stem: str, display: pd.DataFrame | None = None
) -> None:
    """Write a table as full-precision CSV and rounded Markdown (or a given display version)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(to_markdown(df if display is None else display), "utf-8")


def _show(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


def run_holdout(data: HoldoutData, cfg: dict, models: Models, out_dir: Path) -> None:
    """Write Tables III and IV, the PR-AUC vs recall chart and, if configured, a comparison."""
    counts = class_counts(data, cfg)
    save_table(counts, out_dir, "table3_class_counts")
    _show("Table III (training-set class counts)", counts)
    table4 = holdout_table(data, cfg, models)
    save_table(table4, out_dir, "table4")
    _show(f"Table IV (hold-out test set, n={len(data.y_test)})", table4)
    plot_pr_auc_vs_recall(
        table4,
        prevalence=float(data.y_test.mean()),
        path=out_dir / "figures" / "pr_auc_vs_recall.png",
        title="Resampling moves recall far more than PR-AUC",
    )
    if "compare_with" in cfg:
        base = pd.read_csv(cfg["compare_with"])
        comparison = compare_tables(base, table4, suffixes=("paper", cfg["name"]))
        save_table(
            comparison,
            out_dir,
            "table4_vs_paper",
            display=format_comparison(comparison, other=cfg["name"]),
        )
        _show(f"Table IV, {cfg['name']} minus paper profile", comparison)


def run_eda(raw: pd.DataFrame, out_dir: Path) -> None:
    """Write the exploratory figures (class balance and numeric distributions)."""
    fig_dir = out_dir / "figures"
    plot_class_balance(raw, fig_dir / "fig2_class_balance.png")
    plot_numeric_distributions(raw, fig_dir / "fig3_numeric_distributions.png")
    plot_bmi_glucose_by_stroke(raw, fig_dir / "fig4_bmi_glucose_by_stroke.png")
    print("\nWrote EDA figures (Fig 2 to 4)")


def run_cv(data: HoldoutData, cfg: dict, out_dir: Path) -> None:
    """Write Table V (per-fold scores and mean ± std)."""
    folds = cv_folds(data.X_train_raw, data.y_train, cfg)
    folds.to_csv(out_dir / "table5_cv_folds.csv", index=False)
    summary = summarize_folds(folds)
    save_table(summary, out_dir, "table5_cv", display=format_mean_std(summary))
    spec = cfg["cv"]
    _show(
        f"Table V ({spec['n_repeats']}x{spec['n_splits']} CV on the training set, "
        f"n={len(data.y_train)})",
        format_mean_std(summary),
    )


def run_explain(data: HoldoutData, cfg: dict, models: Models, out_dir: Path) -> None:
    """Write SHAP importance (Fig 5), XGBoost gain and LR coefficients."""
    regime = cfg["explain"]["regime"]
    label = REGIME_LABELS[regime]
    names = data.feature_names
    shap_df = shap_importance(models[("xgb", regime)], data.X_test, names)
    save_table(shap_df, out_dir, "shap_xgb_smote")
    plot_shap_bar(
        shap_df,
        out_dir / "figures" / "fig5_shap_xgb_smote.png",
        f"Feature impact, XGBoost trained on {label} (test set, n={len(data.y_test)})",
    )
    gain = xgb_gain_importance(models[("xgb", regime)], names)
    save_table(gain, out_dir, "xgb_gain_smote")
    coefs = lr_coefficients(models[("lr", regime)], names)
    save_table(coefs, out_dir, "lr_coefficients_smote")
    _show(f"Mean |SHAP|, XGB {label}", shap_df)
    _show(f"XGBoost gain importance, {label}", gain)
    _show(f"LR coefficients, {label}", coefs)


def run_calibration(data: HoldoutData, cfg: dict, models: Models, out_dir: Path) -> None:
    """Write Brier scores, reliability-curve points and Fig 6."""
    spec = cfg["calibration"]
    label = REGIME_LABELS[spec["regime"]]
    chosen = {MODEL_LABELS[name]: models[(name, spec["regime"])] for name in cfg["models"]}
    briers, curves = calibration_table(chosen, data.X_test, data.y_test, spec["n_bins"])
    save_table(briers, out_dir, "brier_smote")
    curves.to_csv(out_dir / "reliability_smote.csv", index=False)
    plot_reliability(
        curves,
        briers,
        out_dir / "figures" / "fig6_reliability_smote.png",
        f"Reliability of models trained on {label} (test set)",
    )
    _show(f"Brier scores, {label}", briers)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the requested stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml", help="YAML profile")
    parser.add_argument("--stages", help="comma-separated subset (default: the profile's stages)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    stages = args.stages.split(",") if args.stages else cfg.get("stages", STAGES)
    unknown = set(stages) - set(STAGES)
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")
    set_seeds(cfg["seed"])
    raw = load_raw(cfg["data"]["path"])
    data = prepare_holdout(raw, cfg)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Profile '{cfg['name']}', stages: {', '.join(stages)}")

    needs_models = {"holdout", "explain", "calibration"} & set(stages)
    models = fit_holdout_models(data, cfg) if needs_models else {}
    if "holdout" in stages:
        run_holdout(data, cfg, models, out_dir)
    if "eda" in stages:
        run_eda(raw, out_dir)
    if "cv" in stages:
        run_cv(data, cfg, out_dir)
    if "explain" in stages:
        run_explain(data, cfg, models, out_dir)
    if "calibration" in stages:
        run_calibration(data, cfg, models, out_dir)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
