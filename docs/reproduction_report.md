# Reproduction report

This file compares the paper, the original Colab notebook (`archive/original_colab_notebook.ipynb`) and this repo, and logs every difference found. It grows phase by phase.

## Implementation details that differ from the paper's description

### SVM labels come from `predict()`, not a 0.5 probability cut

The paper's Table II lists a decision threshold of 0.5 for all models, SVM included. In the notebook, class labels come from `model.predict()` for every model:

- For logistic regression and XGBoost, `predict()` is the same as cutting `predict_proba()` at 0.5.
- For `SVC(probability=True)`, `predict()` uses the sign of the decision function. The probabilities come from a separate Platt scaling fit (with internal cross-validation), so a test patient can have a predicted probability on one side of 0.5 and a label from the other side.

This repo follows the notebook (`src/stroke_imbalance/evaluate.py`), so the SVM accuracy, precision, recall and F1 in Table IV are decision-function labels, not a 0.5 cut on `predict_proba`. PR-AUC uses `predict_proba` for all models, as in the notebook.

### XGBoost `n_jobs=-1`

The notebook builds every `XGBClassifier` with `n_jobs=-1` (all available cores). Table II in the paper does not list this setting. The paper profile (`configs/paper.yaml`) includes it to match the notebook.

This matters for reproducibility: with `tree_method="hist"` (the XGBoost 3.x default), results on the same machine depend on the number of threads (see below).

## Table IV (paper profile)

Status after Phase 1: LR and SVM rows match the notebook to full precision (tolerance 1e-6 for accuracy, precision, recall and F1; 1e-4 for PR-AUC). Table III class counts match exactly.

### Open: XGBoost rows on Windows

On Windows 11 (16 logical CPUs, the pinned library versions), all three XGBoost rows differ from the notebook:

| Regime | Metric | Notebook | Windows, `n_jobs=-1` |
|---|---|---|---|
| Original | PR-AUC | 0.207306 | 0.216631 |
| SMOTE | Recall | 0.50 | 0.44 |
| SMOTE | PR-AUC | 0.198759 | 0.221470 |
| SMOTE-ENN | Recall | 0.66 | 0.58 |
| SMOTE-ENN | PR-AUC | 0.215127 | 0.206255 |

On the same machine, `n_jobs` of 1, 2, 3, 8 and -1 each give a different but repeatable result, and none matches the notebook. The inputs are identical (LR and SVM match on the same arrays), so the difference is inside XGBoost training. The notebook ran on Linux (Colab). A Linux CI run is being used to check whether the difference is due to the platform.
