"""Resampling regimes: none, SMOTE, SMOTE-ENN."""

from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

REGIME_LABELS = {"none": "Original", "smote": "SMOTE", "smoteenn": "SMOTE-ENN"}


def make_sampler(regime: str, random_state: int = 42) -> SMOTE | SMOTEENN | None:
    """Return the imblearn sampler for a regime, or None for no resampling."""
    if regime == "none":
        return None
    if regime == "smote":
        return SMOTE(random_state=random_state)
    if regime == "smoteenn":
        return SMOTEENN(random_state=random_state)
    raise ValueError(f"Unknown regime: {regime}")


def resample(
    X: np.ndarray, y: np.ndarray, regime: str, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a training set. Never call this on test data."""
    sampler = make_sampler(regime, random_state)
    if sampler is None:
        return X, y
    return sampler.fit_resample(X, y)
