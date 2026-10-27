"""Reviewer items 4 and 5 (paper revision, branch risk-aware-baselines).

Item 5 -- one operating point everywhere: the routing-aware attacker (E3,
Level A), the drop/tamper study (E6) and the ablation (E8) used
eps_target=0.15; here the proposed method is rerun with the main
experiment's controller settings (eps_target=0.5, a_target=0.8,
fallback=block). Baselines do not depend on eps_target, so their stored rows
are reused.

Item 4 -- steady state: an E12 run covers only 0.1-0.5 s of simulated time,
less than the pool capacity of 0.5 s of generation, so pools never leave
their initial fill for long. Here every run gets a warm-up of two pool-fill
times and is then measured for four pool-fill times, (a) at the E12 pool
size on all four topologies and (b) with 3 s pools on NSFNET and Abilene.

Usage: PYTHONPATH=. python scripts/review_items.py e3|e6|e8|steady|steady-large
Outputs: results/E3/*_eps05.csv, results/E6/*_eps05.csv, results/E8/e8_ablation_eps05.csv,
         results/E12/robustness/{per_seed,means,paired}_steady[_large].csv
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import yaml

from experiments import e3, e6, e8
from experiments import e12_robustness as rb
from qkdmp.config import load_config
from qkdmp.sim.des import arrival_rate_for_load
from qkdmp.topology import build_topology

TOPOLOGIES = ["nsfnet", "abilene", "geant", "germany50"]
SEEDS = {"nsfnet": 30, "abilene": 30, "geant": 30, "germany50": 15}  # as in the original runs
MAIN_POLICY = {"eps_target": 0.5, "a_target": 0.8, "fallback": "block"}
ROB = Path("results") / "E12" / "robustness"
BASELINES = ["B0", "B0+PA", "B1+PA", "B2+PA", "B3+PA"]


def _yaml(name: str) -> dict:
    with open(f"configs/experiments/{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _suffix(topo: str) -> str:
    return "" if topo == "nsfnet" else f"_{topo}"


def item_e3() -> None:
    exp = _yaml("E3")
    base = load_config()
    cfg = dict(base, policy=dict(base["policy"], **MAIN_POLICY))
    for topo in TOPOLOGIES:
        df = e3.run_e3(cfg, rho_a_values=exp["rho_A_values"], placements=["adaptive"], methods=["Adaptive"],
                       n_seeds=SEEDS[topo], n_requests=exp["n_requests_per_seed"], topology=topo)
        df.to_csv(Path("results") / "E3" / f"e3_security_vs_attacker{_suffix(topo)}_eps05.csv", index=False)
        print(topo, df[["rho_A", "exposure_mean", "service_ratio_mean"]].to_string(index=False), flush=True)


def item_e6() -> None:
    exp = _yaml("E6")
    base = load_config()
    cfg = dict(base, policy=dict(base["policy"], **dict(exp["policy_overrides"], **MAIN_POLICY)))
    for topo in TOPOLOGIES:
        df = e6.run_e6(cfg, drop_tamper_values=[tuple(x) for x in exp["drop_tamper_values"]], methods=["Adaptive"],
                       rho=exp["rho"], n_seeds=SEEDS[topo], n_requests=exp["n_requests_per_seed"], topology=topo,
                       rho_a=exp["rho_A"])
        df.to_csv(Path("results") / "E6" / f"e6_drop_tamper{_suffix(topo)}_eps05.csv", index=False)
        print(topo, df[["p_drop", "silent_failure_rate_mean"]].to_string(index=False), flush=True)


def item_e8() -> None:
    exp = _yaml("E8")
    base = load_config()
    cfg = dict(base, policy=dict(base["policy"], **dict(exp["policy_overrides"], **MAIN_POLICY)),
               pool=dict(base["pool"], **exp["pool_overrides"]))
    df = e8.run_e8(cfg, rho=exp["rho"], n_seeds=exp["n_seeds"], n_requests=exp["n_requests_per_seed"])
    df.to_csv(Path("results") / "E8" / "e8_ablation_eps05.csv", index=False)
    print(df[["variant", "secure_goodput_mean", "exposure_mean", "failed_mean", "blocking_mean"]].to_string(index=False))


def _steady(topologies: list[str], capacity_s: float, n_seeds: int, tag: str) -> None:
    exp = _yaml("E12")
    base = load_config()
    cfg = dict(base, policy=dict(base["policy"], **exp["policy_overrides"]),
               pool=dict(base["pool"], **dict(exp["pool_overrides"], capacity_seconds=capacity_s)))
    t = exp["test"]
    seeds = list(range(t["seed_offset"], t["seed_offset"] + n_seeds))
    frames = []
    for topo in topologies:
        lam = arrival_rate_for_load(build_topology(topo, cfg), cfg["key_size_bits"], exp["rho"])
        warm = math.ceil(lam * 2 * capacity_s)
        meas = math.ceil(lam * 4 * capacity_s)
        c = dict(cfg, des=dict(cfg["des"], warmup_requests=warm))
        print(f"{topo}: lambda={lam:.0f}/s warm-up {warm} requests, measured {meas}", flush=True)
        frames.append(rb.run_per_seed(c, topologies=[topo], rho=exp["rho"], rho_a_values=exp["rho_A_values"],
                                      placements=exp["placements"], methods=BASELINES + [rb.OURS], seeds=seeds,
                                      n_requests=meas))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(ROB / f"per_seed_{tag}.csv", index=False)
    rb.means_table(df).to_csv(ROB / f"means_{tag}.csv", index=False)
    families = [[x for x in ["nsfnet", "abilene"] if x in topologies]] + [[x] for x in ["geant", "germany50"]
                                                                          if x in topologies]
    pc = pd.concat([rb.paired_comparisons(df[df.topology.isin(f)]) for f in families if f], ignore_index=True)
    pc.to_csv(ROB / f"paired_{tag}.csv", index=False)
    print(pc.groupby("topology")[["dominates_on_means", "goodput_better", "goodput_worse", "exposure_lower",
                                  "exposure_higher"]].sum().to_string())
    print(rb.means_table(df).groupby(["topology", "method"])[["secure_goodput", "exposure"]].mean().to_string())


if __name__ == "__main__":
    {"e3": item_e3, "e6": item_e6, "e8": item_e8,
     "steady": lambda: _steady(TOPOLOGIES, 0.5, 30, "steady"),
     "steady-large": lambda: _steady(["nsfnet", "abilene"], 3.0, 10, "steady_large")}[sys.argv[1]]()
