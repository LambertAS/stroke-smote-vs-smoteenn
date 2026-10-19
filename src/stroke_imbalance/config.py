"""Load a YAML run profile and set global seeds."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML profile from ``configs/`` and resolve its paths against the project root."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"]["path"] = str(PROJECT_ROOT / cfg["data"]["path"])
    cfg["output_dir"] = str(PROJECT_ROOT / cfg["output_dir"])
    return cfg


def set_seeds(seed: int) -> None:
    """Seed Python and NumPy. Estimators get ``random_state`` explicitly from the config."""
    random.seed(seed)
    np.random.seed(seed)
