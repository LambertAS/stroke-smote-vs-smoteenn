"""Model explanations: SHAP for XGBoost, XGBoost gain importance, LR coefficients."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


def shap_importance(model: Any, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature (log-odds units), largest first."""
    values = shap.TreeExplainer(model).shap_values(X)
    mean_abs = np.abs(values).mean(axis=0)
    out = pd.DataFrame({"Feature": feature_names, "Mean_abs_SHAP": mean_abs})
    return out.sort_values("Mean_abs_SHAP", ascending=False, ignore_index=True)


def xgb_gain_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """XGBoost ``feature_importances_`` (total gain, normalized to sum to 1), largest first."""
    out = pd.DataFrame({"Feature": feature_names, "Gain": model.feature_importances_})
    return out.sort_values("Gain", ascending=False, ignore_index=True)


def lr_coefficients(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Logistic regression coefficients on the min-max scaled features, largest first."""
    coef = model.coef_.ravel()
    out = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": coef, "Sign": np.where(coef > 0, "+", "-")}
    )
    return out.sort_values("Coefficient", ascending=False, ignore_index=True)
