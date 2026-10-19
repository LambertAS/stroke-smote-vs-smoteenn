"""Table V: repeated stratified cross-validation with resampling inside each fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler

from stroke_imbalance.models import MODEL_LABELS, make_model
from stroke_imbalance.resampling import REGIME_LABELS, make_sampler

# PR-AUC on predict_proba, as on the hold-out set (the built-in "average_precision" scorer
# would use SVC's decision_function instead).
SCORING = {
    "PR_AUC": make_scorer(average_precision_score, response_method="predict_proba"),
    "Recall": "recall",
    "F1": "f1",
}


def make_pipeline(name: str, regime: str, cfg: dict) -> Pipeline:
    """Scaler, sampler and model, so scaling and resampling only ever see the training fold."""
    return Pipeline(
        [
            ("scale", MinMaxScaler()),
            ("resample", make_sampler(regime, cfg["seed"])),
            ("model", make_model(name, regime, cfg)),
        ]
    )


def cv_folds(X: pd.DataFrame, y: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Per-fold scores for every model and regime in ``cfg["cv"]``."""
    spec = cfg["cv"]
    splitter = RepeatedStratifiedKFold(
        n_splits=spec["n_splits"], n_repeats=spec["n_repeats"], random_state=spec["random_state"]
    )
    frames = []
    for name in cfg["models"]:
        for regime in spec["regimes"]:
            scores = cross_validate(
                make_pipeline(name, regime, cfg),
                X,
                y,
                cv=splitter,
                scoring=SCORING,
                n_jobs=spec.get("n_jobs", 1),
            )
            fold = pd.DataFrame({m: scores[f"test_{m}"] for m in SCORING})
            fold.insert(0, "Fold", np.arange(len(fold)))
            fold.insert(0, "Resampling", REGIME_LABELS[regime])
            fold.insert(0, "Model", MODEL_LABELS[name])
            frames.append(fold)
    return pd.concat(frames, ignore_index=True)


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation (ddof=0, as numpy) of each metric per model and regime."""
    grouped = folds.groupby(["Model", "Resampling"], sort=False)[list(SCORING)]
    out = grouped.agg(["mean", lambda s: s.std(ddof=0)])
    out.columns = [
        f"{metric}_{'mean' if stat == 'mean' else 'std'}" for metric, stat in out.columns
    ]
    return out.reset_index()


def format_mean_std(summary: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Display version of Table V with "mean ± std" cells."""
    out = summary[["Model", "Resampling"]].copy()
    for metric in SCORING:
        mean, std = summary[f"{metric}_mean"], summary[f"{metric}_std"]
        out[metric] = [
            f"{m:.{decimals}f} ± {s:.{decimals}f}" for m, s in zip(mean, std, strict=True)
        ]
    return out
