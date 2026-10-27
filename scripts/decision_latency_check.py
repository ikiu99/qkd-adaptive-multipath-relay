"""Does the E12 claim survive the controller's real decision time?

The DES charges every decision `des.control_latency_s` (1 ms). Here the
proposed method is charged its measured decision time instead -- the 95th
percentile per topology from scripts/decision_time.py, and a 200 ms stress
level (about the slowest decision measured on Germany50) -- while every
baseline keeps 1 ms. The bias is deliberately against the proposed method.

Usage: PYTHONPATH=. uv run python scripts/decision_latency_check.py
Outputs: results/decision_time/latency_*.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from experiments import e12_robustness as rb
from qkdmp.config import load_config

OUT = Path("results") / "decision_time"
BASELINES = ["B0", "B0+PA", "B1+PA", "B2+PA", "B3+PA"]
TOPOLOGIES = ["nsfnet", "abilene", "geant", "germany50"]
STRESS_S = 0.2


def main() -> None:
    with open("configs/experiments/E12.yaml", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    base = load_config()
    base = dict(base, policy=dict(base["policy"], **exp["policy_overrides"]),
                pool=dict(base["pool"], **exp["pool_overrides"]))
    p95 = pd.read_csv(OUT / "summary_after.csv").set_index("topology")["p95"] / 1e3
    seeds = list(range(exp["test"]["seed_offset"], exp["test"]["seed_offset"] + exp["test"]["n_seeds"]))
    frames = []
    for topo in TOPOLOGIES:
        common = dict(topologies=[topo], rho=exp["rho"], rho_a_values=exp["rho_A_values"],
                      placements=exp["placements"], seeds=seeds, n_requests=exp["n_requests_per_seed"])
        base_df = rb.run_per_seed(base, methods=BASELINES, **common)
        for level, latency in [("p95", float(p95[topo])), ("stress", STRESS_S)]:
            cfg = dict(base, des=dict(base["des"], control_latency_s=latency))
            ours = rb.run_per_seed(cfg, methods=[rb.OURS], **common)
            frames.append(pd.concat([base_df, ours], ignore_index=True).assign(latency=level,
                                                                                ours_latency_s=latency))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "latency_per_seed.csv", index=False)
    pc = rb.paired_comparisons(df, group_cols=("latency",))
    pc.to_csv(OUT / "latency_paired.csv", index=False)
    summary = rb.summarize(pc, ("latency", "topology"))
    summary.to_csv(OUT / "latency_summary.csv", index=False)
    print(summary.to_string(index=False))
    means = rb.means_table(df, ("latency",))
    ours = means[means["method"] == rb.OURS].groupby(["latency", "topology"])["secure_goodput"].mean()
    best = means[means["method"] != rb.OURS].groupby(["latency", "topology", "placement", "rho_A"])[
        "secure_goodput"].max().groupby(level=[0, 1]).mean()
    print(((ours / best - 1) * 100).round(1).rename("lead_%").to_string())


if __name__ == "__main__":
    main()
