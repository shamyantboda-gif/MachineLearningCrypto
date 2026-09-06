"""Config loading, merging and hashing.

A run is identified by the SHA256 of its merged config. That makes the
question "which settings produced this number" answerable later without
relying on anyone's memory.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(base_path: str | Path, *overrides: str | Path) -> dict:
    """Load ``base_path`` and layer each override on top of it."""
    config = load_yaml(base_path)
    for override in overrides:
        if override is None:
            continue
        config = _deep_merge(config, load_yaml(override))
    return config


def config_hash(config: dict, length: int = 10) -> str:
    """Stable short hash of a config dict."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def ensure_dirs() -> None:
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, RESULTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Seed every RNG we might touch. Constraint C7."""
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def get(config: dict, dotted: str, default: Any = None) -> Any:
    """Read ``config`` with a dotted path, e.g. ``get(cfg, "splits.embargo_days")``."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
