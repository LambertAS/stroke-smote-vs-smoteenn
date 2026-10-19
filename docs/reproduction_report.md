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

## Phase 2: paper vs reconstruction

The notebook contains no code for Table V, Fig 5, the importance analyses or Fig 6. The code here is a reconstruction from the paper's description, with fixed settings that were not tuned to match the paper. All values below come from the Linux CI run [39198828091](https://github.com/LambertAS/stroke-smote-vs-smoteenn/actions/runs/39198828091) (`results/paper/`).

Reconstruction choices the paper does not specify:

- **Table V:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)` on the 4,088 training rows (paper-profile preprocessing, BMI imputed with the full-data mean). Each fold runs one `imblearn` Pipeline (MinMaxScaler, then the sampler, then the model), so scaling and resampling see only the training fold. PR-AUC uses `predict_proba`; recall and F1 use `predict()`. The ± values are standard deviations over the 25 folds with ddof=0.
- **Fig 5, gain, coefficients, Fig 6:** the SMOTE-trained hold-out models from Table IV, evaluated on the 1,022-row test set. Gain is XGBoost's `feature_importances_` (total gain, normalized to sum to 1). Coefficients are on min-max scaled features. Reliability curves use 10 uniform bins.

### Table V (mean ± std over 25 folds)

| Model | Regime | PR-AUC paper | PR-AUC repo | Recall paper | Recall repo | F1 paper | F1 repo |
|---|---|---|---|---|---|---|---|
| LR | SMOTE | 0.197 ± 0.031 | 0.188 ± 0.034 | 0.769 ± 0.049 | 0.762 ± 0.055 | 0.227 ± 0.013 | 0.224 ± 0.018 |
| LR | SMOTE-ENN | 0.197 ± 0.030 | 0.188 ± 0.035 | 0.827 ± 0.054 | 0.841 ± 0.044 | 0.214 ± 0.011 | 0.216 ± 0.014 |
| SVM | SMOTE | 0.146 ± 0.031 | 0.132 ± 0.020 | 0.655 ± 0.053 | 0.612 ± 0.066 | 0.201 ± 0.012 | 0.189 ± 0.020 |
| SVM | SMOTE-ENN | 0.149 ± 0.031 | 0.132 ± 0.020 | 0.715 ± 0.049 | 0.704 ± 0.058 | 0.201 ± 0.009 | 0.194 ± 0.017 |
| XGB | SMOTE | 0.144 ± 0.020 | 0.154 ± 0.032 | 0.301 ± 0.052 | 0.272 ± 0.064 | 0.202 ± 0.031 | 0.187 ± 0.037 |
| XGB | SMOTE-ENN | 0.154 ± 0.021 | 0.155 ± 0.035 | 0.528 ± 0.072 | 0.521 ± 0.078 | 0.229 ± 0.026 | 0.221 ± 0.030 |

Every reconstructed mean is within one paper standard deviation of the paper's value, except SVM PR-AUC (0.014 and 0.017 lower, about half a standard deviation more than that) and SVM SMOTE recall (0.043 lower). The qualitative pattern is the same as in the paper: LR has the highest PR-AUC; SMOTE-ENN raises recall over SMOTE for every model; PR-AUC barely changes between the two regimes. The paper does not say where the scaler was fit inside CV or which standard deviation was reported, and either can explain gaps of this size.

### Other Phase 2 numbers

| Quantity | Paper | Reconstruction | Note |
|---|---|---|---|
| SHAP ranking (XGB, SMOTE) | age, work type, BMI, smoking status, avg glucose | age, work_type, bmi, smoking_status, avg_glucose_level | Top five identical, same order |
| Mean \|SHAP\| for age | about 2.27 | 2.198 | 3% lower |
| Age vs next feature | about 5x | 4.69x (2.198 / 0.469) | |
| XGB gain for age (SMOTE) | 0.29, first | 0.251, first | Lower, same rank. A Windows run gave 0.297; Linux is the reference platform, so 0.251 stands |
| LR coefficient signs (SMOTE) | age, avg_glucose_level, hypertension, heart_disease positive | all four positive (7.254, 1.039, 0.485, 0.293) | Match |
| Brier, XGB (SMOTE) | 0.077 | 0.080 | |
| Brier, LR (SMOTE) | 0.164 | 0.164 | Match at 3 decimals |
| Brier, SVM (SMOTE) | 0.145 | 0.145 | Match at 3 decimals |

The paper's figures were made with code that is not in the notebook, so where a value differs (XGB gain for age, the SHAP magnitude, XGB Brier) the cause cannot be traced. The XGB gain gap is in line with the platform sensitivity of XGBoost described below. All three models are poorly calibrated after SMOTE (Fig 6): predicted probabilities sit far above the observed stroke rate. The highest bins hold only a few test patients, so the right-hand end of each curve is noisy.

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
