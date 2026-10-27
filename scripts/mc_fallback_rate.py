"""How often does the exposure computation fall back to Monte Carlo?

`security.exposure_prob` is exact for node-disjoint paths and for
overlapping paths with <= 20 relays in total; otherwise it estimates P_exp
by Monte Carlo. This counts, for the proposed policy on the E12 grid (all
four topologies, held-out seeds), how many (path subset, t) evaluations and
how many chosen decisions used the Monte Carlo branch. Nothing in the
policy is changed; the two security functions are only wrapped by counters.

Usage: PYTHONPATH=. python scripts/mc_fallback_rate.py [n_seeds]
Output: results/E12/mc_fallback_rate.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed

from qkdmp import security
from qkdmp.config import load_config
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.topology import build_topology

TOPOLOGIES = ["nsfnet", "abilene", "geant", "germany50"]


def _one(topo: str, placement: str, rho_a: float, seed: int, base: dict, n_requests: int) -> dict:
    counts = {"exact_subsets": 0, "mc_subsets": 0, "mc_evaluations": 0}
    tail = security.exposure_tail_exact
    mc = security.exposure_prob_monte_carlo

    def tail_counted(paths, p_v):
        out = tail(paths, p_v)
        counts["exact_subsets" if out is not None else "mc_subsets"] += 1
        return out

    def mc_counted(*args, **kwargs):
        counts["mc_evaluations"] += 1
        return mc(*args, **kwargs)

    security.exposure_tail_exact = tail_counted
    security.exposure_prob_monte_carlo = mc_counted
    try:
        graph = build_topology(topo, base)
        rate = arrival_rate_for_load(graph, base["key_size_bits"], base["_rho"])
        cfg = dict(base, des=dict(base["des"], arrival_rate_per_s=rate, traffic_mode="uniform"),
                   attacker=dict(base["attacker"], placement=placement, rho_A=rho_a))
        df = run_level_b(graph, "Adaptive", cfg, n_requests=n_requests, seed=seed)
    finally:
        security.exposure_tail_exact = tail
        security.exposure_prob_monte_carlo = mc
    return dict(topology=topo, placement=placement, rho_A=rho_a, seed=seed, decisions=len(df), **counts)


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    with open("configs/experiments/E12.yaml", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    base = load_config()
    base = dict(base, policy=dict(base["policy"], **dict(exp["policy_overrides"], eps_target=0.5)),
                pool=dict(base["pool"], **exp["pool_overrides"]), _rho=exp["rho"])
    seeds = list(range(exp["test"]["seed_offset"], exp["test"]["seed_offset"] + n_seeds))
    jobs = [(t, p, r, s) for t in TOPOLOGIES for p in exp["placements"] for r in exp["rho_A_values"] for s in seeds]
    rows = Parallel(n_jobs=-1)(delayed(_one)(t, p, r, s, base, exp["n_requests_per_seed"]) for t, p, r, s in jobs)
    df = pd.DataFrame(rows)
    out = Path("results") / "E12" / "mc_fallback_rate.csv"
    df.to_csv(out, index=False)
    g = df.groupby("topology")[["exact_subsets", "mc_subsets", "mc_evaluations", "decisions"]].sum()
    g["mc_share_of_subsets"] = g["mc_subsets"] / (g["exact_subsets"] + g["mc_subsets"])
    print(g.to_string())


if __name__ == "__main__":
    main()
