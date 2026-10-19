"""Model factory driven by the YAML profile."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODEL_LABELS = {"lr": "LR", "svm": "SVM", "xgb": "XGB"}
_CLASSES = {"lr": LogisticRegression, "svm": SVC, "xgb": XGBClassifier}


def model_params(name: str, regime: str, cfg: dict) -> dict[str, Any]:
    """Base parameters for a model, with any per-regime overrides applied."""
    spec = cfg["models"][name]
    params = dict(spec.get("params", {}))
    params.update(spec.get("overrides", {}).get(regime, {}))
    return params


def make_model(name: str, regime: str, cfg: dict) -> Any:
    """Build an unfitted estimator for a model name (lr, svm, xgb) and regime."""
    return _CLASSES[name](**model_params(name, regime, cfg))
