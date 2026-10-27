"""Config loading: YAML files merged onto configs/default.yaml.

Experiment and baseline configs are deltas on top of the repo-wide default
config (README section 9). Loading is a simple deep-merge so every field an
override omits falls back to the default.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a config, merged on top of configs/default.yaml.

    Parameters
    ----------
    path:
        Path to an override YAML file. If None, only the default config is
        returned. If given, its keys are deep-merged onto the default config
        (the override takes precedence).
    """
    default = load_yaml(DEFAULT_CONFIG_PATH)
    if path is None:
        return default
    override = load_yaml(path)
    return _deep_merge(default, override)
