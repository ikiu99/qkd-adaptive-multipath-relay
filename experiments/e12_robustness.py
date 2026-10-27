"""Robustness of the E12 headline claim (reviewer-question round).

Three checks on the same grid, the same held-out seeds (1000-1029) and the
same operating point (eps_target=0.5, chosen on tuning seeds in E12):

1. Paired statistics: every method sees the same attacker set and detector
   scores for a given seed (both are drawn first from the seed's RNG), so
   per-seed differences can be tested with a paired test instead of
   comparing confidence intervals; Holm correction over all comparisons.
2. Detector miscalibration: the detector's probabilities are distorted
   (`qkdmp.suspicion.miscalibrate`) without the controller knowing, and
   eps_target is *not* re-tuned.
3. Pool size: the E12 grid at several pool capacities, scarce to plentiful.
"""

from __future__ import annotations

import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from experiments.level_b_common import METRICS, method_config, run_metrics
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.topology import build_topology

OURS = "Adaptive@0.5"


def _one(graph, policy_name: str, cfg: dict, n_requests: int, seed: int) -> dict:
    return {"seed": seed, **run_metrics(run_level_b(graph, policy_name, cfg, n_requests=n_requests, seed=seed))}


def run_per_seed(config: dict, topologies: list[str], rho: float, rho_a_values: list[float], placements: list[str],
                 methods: list[str], seeds: list[int], n_requests: int, label: dict | None = None) -> pd.DataFrame:
    """One row per (condition, method, seed) with the standard metrics."""
    rows = []
    for topology in topologies:
        graph = build_topology(topology, config)
        arrival_rate = arrival_rate_for_load(graph, config["key_size_bits"], rho)
        for placement in placements:
            for rho_a in rho_a_values:
                for method in methods:
                    policy_name, cfg = method_config(method, config)
                    cfg["des"] = dict(config["des"], arrival_rate_per_s=arrival_rate, traffic_mode="uniform")
                    cfg["attacker"] = dict(config["attacker"], placement=placement, rho_A=rho_a)
                    results = Parallel(n_jobs=-1)(
                        delayed(_one)(graph, policy_name, cfg, n_requests, s) for s in seeds
                    )
                    for r in results:
                        rows.append({**(label or {}), "topology": topology, "placement": placement,
                                     "rho_A": rho_a, "method": method, **r})
                    print(f"{label or ''} {topology}/{placement}/{rho_a}/{method} done", flush=True)
    return pd.DataFrame(rows)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def paired_comparisons(df: pd.DataFrame, ours: str = OURS, group_cols: tuple = ()) -> pd.DataFrame:
    """For every condition and baseline: mean per-seed difference (ours -
    baseline) in secure goodput and exposure, paired t-test and Wilcoxon
    signed-rank p-values (two-sided), Holm-adjusted over the whole table,
    and the verdict per metric."""
    keys = list(group_cols) + ["topology", "placement", "rho_A"]
    baselines = [m for m in df["method"].unique() if not m.startswith("Adaptive")]
    rows = []
    for cond, g in df.groupby(keys):
        cond = cond if isinstance(cond, tuple) else (cond,)
        a = g[g["method"] == ours].set_index("seed")
        for b_name in baselines:
            b = g[g["method"] == b_name].set_index("seed")
            common = a.index.intersection(b.index)
            row = dict(zip(keys, cond), baseline=b_name, n=len(common))
            for metric in ("secure_goodput", "exposure"):
                d = (a.loc[common, metric] - b.loc[common, metric]).dropna()
                row[f"{metric}_diff"] = float(d.mean())
                if len(d) < 2 or (d == 0).all():
                    row[f"{metric}_p_t"] = 1.0
                    row[f"{metric}_p_wilcoxon"] = 1.0
                    continue
                row[f"{metric}_p_t"] = float(stats.ttest_1samp(d, 0.0).pvalue)
                row[f"{metric}_p_wilcoxon"] = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
            rows.append(row)
    out = pd.DataFrame(rows)
    for metric in ("secure_goodput", "exposure"):
        for test in ("t", "wilcoxon"):
            out[f"{metric}_p_{test}_holm"] = holm(out[f"{metric}_p_{test}"].tolist())
    out["goodput_better"] = (out["secure_goodput_diff"] > 0) & (out["secure_goodput_p_t_holm"] < 0.05)
    out["goodput_worse"] = (out["secure_goodput_diff"] < 0) & (out["secure_goodput_p_t_holm"] < 0.05)
    out["exposure_lower"] = (out["exposure_diff"] < 0) & (out["exposure_p_t_holm"] < 0.05)
    out["exposure_higher"] = (out["exposure_diff"] > 0) & (out["exposure_p_t_holm"] < 0.05)
    out["dominates_on_means"] = (out["secure_goodput_diff"] >= 0) & (out["exposure_diff"] <= 0)
    return out


def summarize(pc: pd.DataFrame, group_cols: tuple = ()) -> pd.DataFrame:
    cols = ["dominates_on_means", "goodput_better", "goodput_worse", "exposure_lower", "exposure_higher"]
    if group_cols:
        s = pc.groupby(list(group_cols))[cols].sum()
        s["n_comparisons"] = pc.groupby(list(group_cols)).size()
        return s.reset_index()
    return pd.DataFrame([{**{c: int(pc[c].sum()) for c in cols}, "n_comparisons": len(pc)}])


def means_table(df: pd.DataFrame, group_cols: tuple = ()) -> pd.DataFrame:
    keys = list(group_cols) + ["topology", "placement", "rho_A", "method"]
    return df.groupby(keys)[METRICS].mean().reset_index()
