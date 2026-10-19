# Reproduction report

This file compares the paper, the original Colab notebook (`archive/original_colab_notebook.ipynb`) and this repo, and logs every difference found. It grows phase by phase.

## Implementation details that differ from the paper's description

### SVM labels come from `predict()`, not a 0.5 probability cut

The paper's Table II lists a decision threshold of 0.5 for all models, SVM included. In the notebook, class labels come from `model.predict()` for every model:

- For logistic regression and XGBoost, `predict()` is the same as cutting `predict_proba()` at 0.5.
- For `SVC(probability=True)`, `predict()` uses the sign of the decision function. The probabilities come from a separate Platt scaling fit (with internal cross-validation), so a test patient can have a predicted probability on one side of 0.5 and a label from the other side.

This repo follows the notebook (`src/stroke_imbalance/evaluate.py`), so the SVM accuracy, precision, recall and F1 in Table IV are decision-function labels, not a 0.5 cut on `predict_proba`. PR-AUC uses `predict_proba` for all models, as in the notebook.

### XGBoost thread count: `n_jobs=-1` in the notebook, `n_jobs: 2` here

The notebook builds every `XGBClassifier` with `n_jobs=-1` (all available cores). Table II in the paper does not list this setting. The Colab VM had 2 CPUs, so the original run used 2 threads.

With `tree_method="hist"` (the XGBoost 3.x default), results depend on the number of threads. The paper profile (`configs/paper.yaml`) therefore pins `n_jobs: 2`, which reproduces the original run on machines with any number of cores.

## Table IV (paper profile)

**Resolved.** On Linux, all 45 cells of Table IV match the notebook exactly (maximum absolute difference 0.0 in every metric), and Table III class counts match. Reference run: GitHub Actions run [39197509940](https://github.com/LambertAS/stroke-smote-vs-smoteenn/actions/runs/39197509940) (ubuntu-latest, glibc 2.39, Python 3.12.15, xgboost 3.1.2 built with GCC 10.3.1 and OpenMP, `os.cpu_count()` = 2). The committed `results/paper/` tables come from the Linux CI artifact.

Linux is the reference platform. LR and SVM rows match on every platform tested (Linux and Windows). The XGBoost rows are exact on Linux only: XGBoost's row and column sampling draws from C++ standard-library random distributions, whose output differs between the Linux (GCC) and Windows (MSVC) builds for the same seed. On non-Linux platforms `tests/test_reproduce_table4.py` skips the XGBoost cells.

### XGBoost rows on Windows (for reference)

On Windows 11 (16 logical CPUs, the pinned library versions, `n_jobs=-1`), the XGBoost rows differ slightly from the notebook:

| Regime | Metric | Notebook | Windows, `n_jobs=-1` |
|---|---|---|---|
| Original | PR-AUC | 0.207306 | 0.216631 |
| SMOTE | Recall | 0.50 | 0.44 |
| SMOTE | PR-AUC | 0.198759 | 0.221470 |
| SMOTE-ENN | Recall | 0.66 | 0.58 |
| SMOTE-ENN | PR-AUC | 0.215127 | 0.206255 |

On the same machine, `n_jobs` of 1, 2, 3, 8 and -1 each give a different but repeatable result, and none matches the notebook (with `n_jobs=2`, XGB SMOTE recall is 0.38). The inputs are identical (LR and SVM match on the same arrays), so the difference is inside XGBoost training and comes from the platform.
