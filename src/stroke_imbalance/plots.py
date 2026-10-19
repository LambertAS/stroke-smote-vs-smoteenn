"""All figures, in one consistent style."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order (validated palette: the first three are distinct in
# every pairing, also under colour-vision deficiency).
MODEL_COLORS = {"LR": "#2a78d6", "SVM": "#eb6834", "XGB": "#1baf7a"}
SINGLE = MODEL_COLORS["LR"]
# Stroke status uses the first two slots; regimes are told apart by marker shape.
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
CLASS_LABELS = {0: "No stroke", 1: "Stroke"}
REGIME_MARKERS = {"Original": "o", "SMOTE": "s", "SMOTE-ENN": "^"}
NUMERIC = {"age": "Age (years)", "bmi": "BMI", "avg_glucose_level": "Average glucose (mg/dL)"}

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


def plot_pr_auc_vs_recall(table4: pd.DataFrame, prevalence: float, path: Path, title: str) -> Path:
    """Headline chart: recall at the 0.5 threshold vs PR-AUC for every model and regime."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.axhline(prevalence, ls="--", lw=1, color=TEXT_MUTED)
        ax.text(
            1.0,
            prevalence,
            f"chance level ({prevalence:.3f})",
            color=TEXT_MUTED,
            fontsize=8.5,
            ha="right",
            va="bottom",
        )
        for _, row in table4.iterrows():
            ax.scatter(
                row["Recall"],
                row["PR_AUC"],
                s=70,
                marker=REGIME_MARKERS[row["Resampling"]],
                color=MODEL_COLORS[row["Model"]],
                edgecolor=SURFACE,
                linewidth=1.5,
                zorder=3,
            )
        handles = [
            Line2D([], [], ls="", marker="o", ms=8, color=c, label=m)
            for m, c in MODEL_COLORS.items()
        ] + [
            Line2D([], [], ls="", marker=mk, ms=8, color=TEXT_MUTED, label=r)
            for r, mk in REGIME_MARKERS.items()
        ]
        ax.legend(handles=handles, ncols=2, loc="upper left", labelcolor=TEXT, fontsize=9)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(0, max(0.35, table4["PR_AUC"].max() * 1.2))
        ax.set_xlabel("Recall on the test set (threshold 0.5)")
        ax.set_ylabel("PR-AUC (threshold-free)")
        ax.set_title(title)
        return _save(fig, path)


def _panels(n: int, figsize: tuple[float, float]) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(1, n, figsize=figsize)
    return fig, list(axes)


def plot_class_balance(raw: pd.DataFrame, path: Path) -> Path:
    """Fig 2: number of patients with and without stroke."""
    counts = raw["stroke"].value_counts().sort_index()
    total = counts.sum()
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 2.4))
        labels = [CLASS_LABELS[k] for k in counts.index]
        ax.barh(
            labels, counts.to_numpy(), color=[CLASS_COLORS[k] for k in counts.index], height=0.55
        )
        for y, n in enumerate(counts.to_numpy()):
            ax.text(n, y, f" {n:,} ({n / total:.2%})", va="center", color=TEXT_MUTED, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, counts.max() * 1.3)
        ax.set_xlabel("Patients")
        ax.set_title(f"Class balance ({total:,} patients)")
        return _save(fig, path)


def plot_numeric_distributions(raw: pd.DataFrame, path: Path) -> Path:
    """Fig 3: histograms of age, BMI and average glucose (missing BMI values left out)."""
    with plt.rc_context(STYLE):
        fig, axes = _panels(len(NUMERIC), (12, 3.4))
        for ax, (col, label) in zip(axes, NUMERIC.items(), strict=True):
            sns.histplot(raw[col].dropna(), bins=30, color=SINGLE, edgecolor=SURFACE, ax=ax)
            ax.set_xlabel(label)
            ax.set_ylabel("Patients" if ax is axes[0] else "")
            ax.grid(axis="x", visible=False)
        fig.suptitle(
            "Distributions of the numeric features",
            x=0.01,
            ha="left",
            fontweight="bold",
            color=TEXT,
        )
        fig.tight_layout()
        return _save(fig, path)


def plot_bmi_glucose_by_stroke(raw: pd.DataFrame, path: Path) -> Path:
    """Fig 4: density of BMI and average glucose by stroke status (each class normalized)."""
    columns = {col: NUMERIC[col] for col in ("bmi", "avg_glucose_level")}
    with plt.rc_context(STYLE):
        fig, axes = _panels(len(columns), (9, 3.4))
        for ax, (col, label) in zip(axes, columns.items(), strict=True):
            for cls in (0, 1):
                values = raw.loc[raw["stroke"] == cls, col].dropna()
                sns.kdeplot(
                    values,
                    color=CLASS_COLORS[cls],
                    lw=2,
                    cut=0,
                    fill=True,
                    alpha=0.15,
                    label=CLASS_LABELS[cls],
                    ax=ax,
                )
            ax.set_xlabel(label)
            ax.set_ylabel("Density" if ax is axes[0] else "")
        axes[0].legend(loc="upper right", labelcolor=TEXT)
        fig.suptitle(
            "BMI and glucose by stroke status", x=0.01, ha="left", fontweight="bold", color=TEXT
        )
        fig.tight_layout()
        return _save(fig, path)


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
