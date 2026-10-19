# Mac/Linux shortcuts. On Windows, run the equivalent python commands shown in README.md.
PYTHON ?= python3.12
VENV := .venv
PY := $(VENV)/bin/python

.PHONY: setup data reproduce test lint all

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

data:
	$(PY) -m stroke_imbalance.data

reproduce:
	$(PY) -m stroke_imbalance.run --config configs/paper.yaml
	$(PY) -m stroke_imbalance.run --config configs/corrected.yaml

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

all: data reproduce test
