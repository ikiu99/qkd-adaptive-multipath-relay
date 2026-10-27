"""Confidence intervals and paired comparisons across seeds (README Section 8).

`aggregate_over_seeds` takes a list of per-seed scalar values (e.g. one
key-exposure-rate per seed) and returns mean + 95% CI, via Student-t by
default (paper's stated >= 30 seeds makes normal approximation reasonable)
or bootstrap for skewed/small-sample metrics.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as scipy_stats


def student_t_ci(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean
    sem = scipy_stats.sem(arr)
    half_width = sem * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - half_width, mean + half_width


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_boot: int = 10_000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) via percentile bootstrap."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if rng is None:
        rng = np.random.default_rng(0)
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean
    boot_means = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = 1 - confidence
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi)


def aggregate_over_seeds(
    values: list[float], method: str = "t", confidence: float = 0.95
) -> dict[str, float]:
    if method == "t":
        mean, lo, hi = student_t_ci(values, confidence)
    elif method == "bootstrap":
        mean, lo, hi = bootstrap_ci(values, confidence)
    else:
        raise ValueError(f"unknown CI method: {method!r}")
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n_seeds": len(values)}


def paired_difference_ci(
    values_a: list[float], values_b: list[float], method: str = "t", confidence: float = 0.95
) -> dict[str, float]:
    """CI on the paired difference a - b (common random numbers, README 8)."""
    if len(values_a) != len(values_b):
        raise ValueError("paired comparison requires equal-length sequences")
    diffs = [a - b for a, b in zip(values_a, values_b)]
    return aggregate_over_seeds(diffs, method=method, confidence=confidence)
