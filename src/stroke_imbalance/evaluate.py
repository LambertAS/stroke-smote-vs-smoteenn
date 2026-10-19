"""Hold-out metrics, computed exactly as in the original notebook."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1", "PR_AUC"]


def holdout_metrics(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Accuracy, precision, recall and F1 for the positive class, plus PR-AUC.

    Labels come from ``model.predict``, as in the notebook. For LR and XGBoost this equals a
    0.5 cut on the predicted probability. For SVC it is the sign of the decision function,
    which can differ slightly from a 0.5 cut on the Platt-scaled probability.
    PR-AUC is ``average_precision_score`` on ``predict_proba[:, 1]``.
    """
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "PR_AUC": average_precision_score(y_test, y_score),
    }


def compare_tables(
    base: pd.DataFrame, other: pd.DataFrame, suffixes: tuple[str, str]
) -> pd.DataFrame:
    """Side-by-side metrics of two Table IV runs, with ``other - base`` differences."""
    keys = ["Model", "Resampling"]
    merged = base.merge(other, on=keys, suffixes=tuple(f"_{s}" for s in suffixes), validate="1:1")
    out = merged[keys].copy()
    for metric in METRIC_COLUMNS:
        a, b = merged[f"{metric}_{suffixes[0]}"], merged[f"{metric}_{suffixes[1]}"]
        out[f"{metric}_{suffixes[0]}"] = a
        out[f"{metric}_{suffixes[1]}"] = b
        out[f"{metric}_diff"] = b - a
    return out


def format_comparison(table: pd.DataFrame, other: str, decimals: int = 4) -> pd.DataFrame:
    """Display version: each cell is the other run's value with the signed difference."""
    out = table[["Model", "Resampling"]].copy()
    for metric in METRIC_COLUMNS:
        value, diff = table[f"{metric}_{other}"], table[f"{metric}_diff"]
        out[metric] = [
            f"{v:.{decimals}f} ({d:+.{decimals}f})" for v, d in zip(value, diff, strict=True)
        ]
    return out
