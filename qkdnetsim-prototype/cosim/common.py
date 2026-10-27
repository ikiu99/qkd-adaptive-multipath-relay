"""E12 configuration shared by the co-simulation scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

from qkdmp.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def e12_experiment() -> dict:
    with open(ROOT / "configs" / "experiments" / "E12.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def e12_config() -> dict:
    """Default config with E12's policy and pool overrides (as experiments/e12_robustness.py)."""
    exp = e12_experiment()
    base = load_config()
    return dict(base, policy=dict(base["policy"], **exp["policy_overrides"]),
                pool=dict(base["pool"], **exp["pool_overrides"]))
