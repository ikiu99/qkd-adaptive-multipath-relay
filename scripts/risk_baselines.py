"""Reviewer check: are the gains due to the detector alone?

B0-B2 do not read the detector, so a reviewer can ask whether a baseline
that simply routes around suspicious relays would do as well. B0R/B1R/B2R
are B0/B1/B2 with paths chosen by least relay risk, sum -ln(1 - p_v), and
with the same pool-aware routing (+PA) as the other baselines. They are run
on the E12 grid (four topologies, held-out seeds 1000-1029) and paired with
the stored per-seed rows of the proposed method (Adaptive@0.5); every
method of a seed sees the same attacker set and detector output.

Usage: PYTHONPATH=. python scripts/risk_baselines.py check   (reproducibility of the stored rows)
       PYTHONPATH=. python scripts/risk_baselines.py run
Outputs: results/E12/robustness/{per_seed,paired,means}_riskaware.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from experiments import e12_robustness as rb
from qkdmp.config import load_config

OUT = Path("results") / "E12" / "robustness"
NEW = ["B0R+PA", "B1R+PA", "B2R+PA"]
TOPOLOGIES = ["nsfnet", "abilene", "geant", "germany50"]
STORED = {"nsfnet": "per_seed_calibrated.csv", "abilene": "per_seed_calibrated.csv",
          "geant": "per_seed_geant.csv", "germany50": "per_seed_germany50.csv"}
FAMILIES = [["nsfnet", "abilene"], ["geant"], ["germany50"]]  # same Holm families as the main analysis


def _setup():
    with open("configs/experiments/E12.yaml", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    base = load_config()
    cfg = dict(base, policy=dict(base["policy"], **exp["policy_overrides"]),
               pool=dict(base["pool"], **exp["pool_overrides"]))
    t = exp["test"]
    grid = dict(rho=exp["rho"], rho_a_values=exp["rho_A_values"], placements=exp["placements"],
                seeds=list(range(t["seed_offset"], t["seed_offset"] + t["n_seeds"])),
                n_requests=exp["n_requests_per_seed"])
    return cfg, grid


def stored(methods: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(OUT / f) for f in dict.fromkeys(STORED.values())]
    df = pd.concat(frames, ignore_index=True)
    return df[df["method"].isin(methods) & df["topology"].isin(TOPOLOGIES)]


def check() -> None:
    cfg, grid = _setup()
    grid = dict(grid, seeds=grid["seeds"][:3], placements=["betweenness"], rho_a_values=[0.2])
    metrics = ["secure_goodput", "exposure", "blocking", "failed"]
    for topo in ["nsfnet", "geant"]:
        new = rb.run_per_seed(cfg, topologies=[topo], methods=["B0+PA", rb.OURS], **grid)
        old = stored(["B0+PA", rb.OURS])
        old = old[(old.topology == topo) & (old.placement == "betweenness") & (old.rho_A == 0.2)
                  & old.seed.isin(grid["seeds"])]
        m = new.merge(old, on=["topology", "placement", "rho_A", "method", "seed"], suffixes=("_new", "_old"))
        diff = max((m[f"{k}_new"] - m[f"{k}_old"]).abs().max() for k in metrics)
        print(f"{topo}: {len(m)} rows compared, max abs difference {diff:.3g}")


def run() -> None:
    cfg, grid = _setup()
    new = rb.run_per_seed(cfg, topologies=TOPOLOGIES, methods=NEW, **grid)
    df = pd.concat([stored([rb.OURS]), new], ignore_index=True)
    df.to_csv(OUT / "per_seed_riskaware.csv", index=False)
    rb.means_table(df).to_csv(OUT / "means_riskaware.csv", index=False)
    pcs = [rb.paired_comparisons(df[df.topology.isin(fam)]) for fam in FAMILIES]
    pc = pd.concat(pcs, ignore_index=True)
    pc.to_csv(OUT / "paired_riskaware.csv", index=False)
    print(pc.groupby(["topology", "baseline"])[["dominates_on_means", "goodput_better", "goodput_worse",
                                                "exposure_lower", "exposure_higher"]].sum().to_string())
    print(rb.means_table(df).groupby(["topology", "method"])[["secure_goodput", "exposure"]].mean().to_string())


if __name__ == "__main__":
    {"check": check, "run": run}[sys.argv[1]]()
