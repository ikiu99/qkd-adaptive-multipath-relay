"""Shared Level B helpers for E4/E5/E8/E11: run one configuration over many
seeds in parallel and reduce each run to the standard metric set."""

from __future__ import annotations

from typing import Callable

import pandas as pd
from joblib import Parallel, delayed

from analysis.metrics import (
    blocking_probability,
    key_efficiency,
    key_exposure_rate,
    mean_amplification,
    secure_goodput,
    service_ratio,
)
from analysis.stats import aggregate_over_seeds

METRICS = ["blocking", "failed", "service_ratio", "amplification", "exposure", "secure_goodput", "key_efficiency"]


def method_config(method: str, config: dict) -> tuple[str, dict]:
    """"Adaptive@0.5" -> ("Adaptive", config with policy.eps_target=0.5): the
    same policy at another operator-chosen security target. eps_target is
    Adaptive's operating-point knob (README 4.1); the baselines have none,
    so comparing them against more than one Adaptive operating point is the
    fair reading of a secure-goodput comparison, not a tuning advantage."""
    cfg = dict(config)
    if "@" not in method:
        return method, cfg
    name, eps = method.split("@", 1)
    cfg["policy"] = dict(config["policy"], eps_target=float(eps))
    return name, cfg


def run_metrics(df: pd.DataFrame) -> dict:
    failed = df["failed"].astype(bool) if "failed" in df and len(df) else pd.Series(False, index=df.index)
    completed = df[~df["blocked"].astype(bool) & ~failed] if len(df) else df
    nan = float("nan")
    return {
        "blocking": blocking_probability(df),
        "failed": float(failed.mean()) if len(df) else nan,
        "service_ratio": service_ratio(df),
        "amplification": mean_amplification(completed) if len(completed) else nan,
        "exposure": key_exposure_rate(completed) if len(completed) else nan,
        "secure_goodput": secure_goodput(df),
        "key_efficiency": key_efficiency(df),
    }


def over_seeds(run_one: Callable[[int], dict], n_seeds: int, n_jobs: int = -1, seed_offset: int = 0) -> dict:
    """`run_one(seed) -> run_metrics(...)` for seeds seed_offset ..
    seed_offset + n_seeds - 1; returns mean/ci columns for every metric plus
    n_seeds. A non-zero offset gives held-out seeds for evaluating an
    operating point chosen on seeds 0.. (E12 tuning/test protocol)."""
    per_seed = Parallel(n_jobs=n_jobs)(
        delayed(run_one)(seed) for seed in range(seed_offset, seed_offset + n_seeds)
    )
    row: dict = {}
    for m in METRICS:
        stats = aggregate_over_seeds([r[m] for r in per_seed])
        row[f"{m}_mean"] = stats["mean"]
        row[f"{m}_ci_low"] = stats["ci_low"]
        row[f"{m}_ci_high"] = stats["ci_high"]
    row["n_seeds"] = n_seeds
    return row
