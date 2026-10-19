# Stroke prediction on imbalanced data: SMOTE vs SMOTE-ENN

Reproduction code for "Learning from Imbalanced Stroke Data: A Comparative Study of SMOTE and SMOTE-ENN on Structured Health Records", IEEE ICISS 2026, [doi:10.1109/ICISS71466.2026.11715456](https://doi.org/10.1109/ICISS71466.2026.11715456).

Work in progress. The full README follows once all analyses are in place.

**Platform note.** Linux is the reference platform: there, every Table IV cell matches the original Colab notebook exactly. The XGBoost rows depend on platform-specific random sampling, so on Windows or macOS they differ slightly and their tests are skipped. See [docs/reproduction_report.md](docs/reproduction_report.md).

## Quickstart

Requires Python 3.12.

Mac/Linux:

```bash
make setup      # create .venv and install pinned packages
make data       # download the Kaggle dataset into data/raw/
make reproduce  # Table III and Table IV, written to results/paper/
make test
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv            # or: uv venv --python 3.12 --seed .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m stroke_imbalance.data
.venv\Scripts\python -m stroke_imbalance.run --config configs/paper.yaml
.venv\Scripts\python -m pytest
```
