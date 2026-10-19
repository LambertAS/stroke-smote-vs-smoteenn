"""Command-line entry point.

Usage: ``python -m stroke_imbalance.run --config configs/paper.yaml``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stroke_imbalance.config import load_config, set_seeds
from stroke_imbalance.data import load_raw
from stroke_imbalance.evaluate import METRIC_COLUMNS, holdout_metrics
from stroke_imbalance.features import HoldoutData, prepare_holdout
from stroke_imbalance.models import MODEL_LABELS, make_model
from stroke_imbalance.resampling import REGIME_LABELS, resample


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


def holdout_table(data: HoldoutData, cfg: dict) -> pd.DataFrame:
    """Table IV: every model trained on every regime, scored on the untouched test set."""
    resampled = {r: resample(data.X_train, data.y_train, r, cfg["seed"]) for r in cfg["regimes"]}
    rows = []
    for name in cfg["models"]:
        for regime, (X_res, y_res) in resampled.items():
            model = make_model(name, regime, cfg).fit(X_res, y_res)
            metrics = holdout_metrics(model, data.X_test, data.y_test)
            rows.append(
                {"Model": MODEL_LABELS[name], "Resampling": REGIME_LABELS[regime], **metrics}
            )
    return pd.DataFrame(rows, columns=["Model", "Resampling", *METRIC_COLUMNS])


def to_markdown(df: pd.DataFrame, decimals: int = 3) -> str:
    """Render a DataFrame as a GitHub Markdown table, rounding floats for display only."""
    cells = df.map(lambda v: f"{v:.{decimals}f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in cells.to_numpy()]
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    """Write a table as full-precision CSV and rounded Markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(to_markdown(df), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the hold-out experiment and write the result tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml", help="YAML profile")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seeds(cfg["seed"])
    data = prepare_holdout(load_raw(cfg["data"]["path"]), cfg)
    out_dir = Path(cfg["output_dir"])

    counts = class_counts(data, cfg)
    save_table(counts, out_dir, "table3_class_counts")
    print(f"Table III (training-set class counts), profile '{cfg['name']}':")
    print(counts.to_string(index=False))

    table4 = holdout_table(data, cfg)
    save_table(table4, out_dir, "table4")
    print(f"\nTable IV (hold-out test set, n={len(data.y_test)}), profile '{cfg['name']}':")
    print(table4.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nWrote tables to {out_dir}")


if __name__ == "__main__":
    main()
