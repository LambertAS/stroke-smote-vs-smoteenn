"""Load a YAML run profile and set global seeds."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``override`` on ``base``; lists and scalars are replaced."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if "base" in cfg:
        cfg = _merge(_read_yaml(cfg.pop("base")), cfg)
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML profile (applying its ``base`` profile, if any) and resolve its paths."""
    cfg = _read_yaml(path)
    cfg["data"]["path"] = str(PROJECT_ROOT / cfg["data"]["path"])
    if "compare_with" in cfg:
        cfg["compare_with"] = str(PROJECT_ROOT / cfg["compare_with"])
    cfg["output_dir"] = str(PROJECT_ROOT / cfg["output_dir"])
    return cfg


def set_seeds(seed: int) -> None:
    """Seed Python and NumPy. Estimators get ``random_state`` explicitly from the config."""
    random.seed(seed)
    np.random.seed(seed)
