# Data

The study uses the **Stroke Prediction Dataset** by fedesoriano on Kaggle:
<https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset>

The dataset is **not included in this repository**. It is downloaded at runtime into `data/raw/`, which is gitignored. Its terms of use are set by the dataset's Kaggle page; check them before reusing the data.

## Getting the file

```bash
make data                                   # Mac/Linux
```

```powershell
.venv\Scripts\python -m stroke_imbalance.data   # Windows (PowerShell)
```

The script tries `kagglehub` first (no Kaggle account was needed when this was written), then the Kaggle CLI if you have credentials. If both fail, it prints manual steps: download the dataset from the link above, unzip it and place `healthcare-dataset-stroke-data.csv` in `data/raw/`.

## Integrity checks

Every load checks that the file:

- has SHA-256 `644d473b05d2797006bd94865e4f8bb057f0c721617911613c82c8fcfc707420` (dataset version 1, recorded 2026-10-19),
- has 5,110 rows,
- has 249 stroke cases (4.87%).

If Kaggle publishes a new version of the file, the checksum check fails on purpose, so that results are never silently computed on different data.

## Contents

5,110 patients, 10 features plus the `stroke` label (the `id` column is dropped): gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi (201 missing values), smoking_status. The dataset's clinical source is not documented.
