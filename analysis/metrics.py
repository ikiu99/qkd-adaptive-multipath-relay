"""Metrics computed from a Level A / Level B results DataFrame (README Section 8).

Every function takes the DataFrame produced by `qkdmp.sim.graph_mc.run_level_a`
(or the Level B equivalent) and returns a single scalar. `served` rows are
non-blocked requests whose shares were delivered in sufficient number;
`blocked` rows have no further columns filled in.
"""

from __future__ import annotations

import pandas as pd


def _col(df: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    """`df[name]`, or an all-`default` Series if no row ever populated that
    column (e.g. every request in the batch was blocked/failed -- under
    extreme load or a tiny pool, `served`/`exposed`/etc. may never appear
    in any record at all, since qkdmp.sim.{graph_mc,des} only include them
    on served-request rows)."""
    if name in df.columns:
        return df[name].fillna(default)
    return pd.Series(default, index=df.index)


def service_ratio(df: pd.DataFrame) -> float:
    """Served / offered requests."""
    if len(df) == 0:
        return float("nan")
    return float((~df["blocked"] & _col(df, "served")).sum() / len(df))


def blocking_probability(df: pd.DataFrame) -> float:
    """Blocked / offered requests."""
    if len(df) == 0:
        return float("nan")
    return float(df["blocked"].sum() / len(df))


def _served(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["blocked"] & _col(df, "served")]


def key_exposure_rate(df: pd.DataFrame) -> float:
    """Exposed delivered keys / delivered keys (ground truth, README 3.4)."""
    served = _served(df)
    if len(served) == 0:
        return float("nan")
    return float(served["exposed"].sum() / len(served))


def mean_m_star(df: pd.DataFrame) -> float:
    served = _served(df)
    if len(served) == 0:
        return float("nan")
    return float(served["m_star"].mean())


def silent_failure_rate(df: pd.DataFrame) -> float:
    served = _served(df)
    if len(served) == 0:
        return float("nan")
    return float(served["silent_failure"].sum() / len(served))


def mean_amplification(df: pd.DataFrame) -> float:
    served = _served(df)
    if len(served) == 0:
        return float("nan")
    return float(served["amplification"].mean())


def success_under_attack(df: pd.DataFrame) -> float:
    """Service ratio restricted to requests that were actually exposed to a
    compromised relay (README 8: "Success under attack")."""
    exposed_candidates = df[~df["blocked"] & _col(df, "exposed")]
    if len(exposed_candidates) == 0:
        return float("nan")
    return float(_col(exposed_candidates, "served").sum() / len(exposed_candidates))


def _secure(df: pd.DataFrame) -> pd.Series:
    """Delivered, correct (no silent failure), and not exposed to the attacker."""
    def flag(name: str) -> pd.Series:
        return _col(df, name).astype(bool)

    return ~flag("blocked") & flag("served") & ~flag("exposed") & ~flag("silent_failure")


def secure_goodput(df: pd.DataFrame) -> float:
    """Headline metric (fair-comparison follow-up): securely delivered keys /
    offered requests, at a fixed offered load. Refusing requests lowers it
    directly; wasting pool bits (failed reservations, needless extra paths)
    lowers it indirectly, by leaving less key material for later requests;
    exposure and silent failures don't count as delivered."""
    if len(df) == 0:
        return float("nan")
    return float(_secure(df).sum() / len(df))


def key_efficiency(df: pd.DataFrame) -> float:
    """Secondary metric: key material that ended up in securely delivered
    keys, relative to all pool bits consumed (including bits wasted by
    failed requests). Each secure key counts its single-shortest-path cost
    `ref_bits`, so an unexposed single-path delivery with no waste scores 1."""
    # plus key spent outside any request, e.g. proactive keys (df.attrs["overhead_bits"])
    total = _col(df, "bits_consumed", 0.0).astype(float).sum() + df.attrs.get("overhead_bits", 0.0)
    if total <= 0:
        return float("nan")
    return float(_col(df, "ref_bits", 0.0).astype(float)[_secure(df)].sum() / total)


def summary_row(df: pd.DataFrame, method: str) -> dict:
    return {
        "method": method,
        "n_requests": len(df),
        "service_ratio": service_ratio(df),
        "blocking_probability": blocking_probability(df),
        "key_exposure_rate": key_exposure_rate(df),
        "mean_m_star": mean_m_star(df),
        "silent_failure_rate": silent_failure_rate(df),
        "mean_amplification": mean_amplification(df),
    }
