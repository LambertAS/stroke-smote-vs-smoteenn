"""All figures, in one consistent style."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order (validated palette: the first three are distinct in
# every pairing, also under colour-vision deficiency).
MODEL_COLORS = {"LR": "#2a78d6", "SVM": "#eb6834", "XGB": "#1baf7a"}
SINGLE = MODEL_COLORS["LR"]

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_MUTED,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "font.size": 10,
    "legend.frameon": False,
}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap_bar(shap_df: pd.DataFrame, path: Path, title: str) -> Path:
    """Fig 5: horizontal bars of mean |SHAP| per feature, largest at the top, values labelled."""
    df = shap_df.iloc[::-1]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(df["Feature"], df["Mean_abs_SHAP"], color=SINGLE, height=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        for y, v in enumerate(df["Mean_abs_SHAP"]):
            ax.text(v, y, f" {v:.2f}", va="center", color=TEXT_MUTED, fontsize=9)
        ax.set_xlabel("Mean |SHAP value| (log-odds)")
        ax.set_title(title)
        ax.set_xlim(0, df["Mean_abs_SHAP"].max() * 1.15)
        return _save(fig, path)


def plot_reliability(curves: pd.DataFrame, briers: pd.DataFrame, path: Path, title: str) -> Path:
    """Fig 6: reliability curves per model against the diagonal, Brier score in the legend."""
    brier = dict(zip(briers["Model"], briers["Brier"], strict=True))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=TEXT_MUTED, label="Perfect calibration")
        for model, pts in curves.groupby("Model", sort=False):
            ax.plot(
                pts["Mean_predicted"],
                pts["Fraction_positive"],
                marker="o",
                ms=5,
                lw=2,
                color=MODEL_COLORS.get(model, SINGLE),
                markeredgecolor=SURFACE,
                label=f"{model} (Brier {brier[model]:.3f})",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction of strokes")
        ax.set_title(title)
        ax.legend(loc="upper left", labelcolor=TEXT)
        return _save(fig, path)
