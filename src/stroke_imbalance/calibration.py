"""Brier scores and reliability curves on the hold-out test set."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibration_table(
    models: dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, n_bins: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Brier score per model, and reliability-curve points (uniform bins, empty bins dropped)."""
    briers, curves = [], []
    for label, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        briers.append({"Model": label, "Brier": brier_score_loss(y_test, prob)})
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=n_bins, strategy="uniform")
        curves.append(
            pd.DataFrame(
                {"Model": label, "Mean_predicted": mean_pred, "Fraction_positive": frac_pos}
            )
        )
    return pd.DataFrame(briers), pd.concat(curves, ignore_index=True)
