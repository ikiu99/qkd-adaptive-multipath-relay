"""Wall-clock time of one controller decision (policy.decide) per topology.

Runs the E12 setup (Adaptive@0.5, rho = 0.6, rho_A = 0.2, random and
betweenness placement) on held-out seeds and times every call to the
policy's `decide`, one process per run so the timings do not compete for
CPU. Reports median, 95th percentile and maximum per topology, with the
number of candidate paths the decision searched.

Usage: PYTHONPATH=. uv run python scripts/decision_time.py [--seeds N] [topology ...]
       PYTHONPATH=. uv run python scripts/decision_time.py --plot
Outputs: results/decision_time/ (the committed before/after files were made
by running this on the old and the new `qkdmp.security` implementation and
renaming the outputs), figures/decision_time/decision_time.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from experiments.level_b_common import method_config
from qkdmp.config import load_config
from qkdmp.policies.base import BLOCK
from qkdmp.sim import des
from qkdmp.sim.graph_mc import get_policy
from qkdmp.topology import build_topology

OUT = Path("results") / "decision_time"
TOPOLOGIES = ["abilene", "nsfnet", "geant", "germany50"]


def e12_config() -> dict:
    with open("configs/experiments/E12.yaml", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    base = load_config()
    return dict(base, policy=dict(base["policy"], **exp["policy_overrides"]),
                pool=dict(base["pool"], **exp["pool_overrides"])), exp


def time_run(topology: str, placement: str, seed: int, method: str = "Adaptive@0.5") -> list[dict]:
    cfg, exp = e12_config()
    graph = build_topology(topology, cfg)
    policy_name, cfg = method_config(method, cfg)
    cfg = dict(cfg, des=dict(cfg["des"], arrival_rate_per_s=des.arrival_rate_for_load(graph, cfg["key_size_bits"],
                                                                                        exp["rho"]),
                             traffic_mode="uniform"),
               attacker=dict(cfg["attacker"], placement=placement, rho_A=0.2))
    policy = get_policy(policy_name, cfg, seed=seed)
    timings: list[dict] = []
    decide = policy.decide

    def timed(request, g, p_v, c):
        t0 = time.perf_counter()
        decision = decide(request, g, p_v, c)
        timings.append({"topology": topology, "placement": placement, "seed": seed,
                        "decide_ms": (time.perf_counter() - t0) * 1e3,
                        "blocked": decision is BLOCK})
        return decision

    policy.decide = timed
    des.run_level_b(graph, policy_name, cfg, n_requests=exp["n_requests_per_seed"], seed=seed,
                    policy_override=policy)
    return timings


def plot() -> None:
    from analysis import figstyle as fs

    fs.apply()
    plt = fs.plt
    fig, ax = plt.subplots(figsize=(9.5, 5))
    runs = [("before", "per_decision_before.csv", fs.NEUTRAL, -0.19),
            ("after", "per_decision_after.csv", fs.METHOD_COLOR["Adaptive@0.5"], 0.19)]
    for label, fname, color, dx in runs:
        df = pd.read_csv(OUT / fname)
        topos = [t for t in TOPOLOGIES if t in set(df["topology"])]
        data = [df[df["topology"] == t]["decide_ms"] for t in topos]
        pos = [TOPOLOGIES.index(t) + dx for t in topos]
        ax.boxplot(data, positions=pos, widths=0.3, whis=(5, 95), showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=fs.tint(color, 0.85), edgecolor="none"),
                   medianprops=dict(color=fs.SURFACE, lw=2.2, solid_capstyle="round"),
                   whiskerprops=dict(color=fs.tint(color, 0.7), lw=1.6, solid_capstyle="round"),
                   capprops=dict(color="none"))
        ax.scatter(pos, [d.max() for d in data], marker="o", s=34, color=color, edgecolor=fs.SURFACE, lw=1.4,
                   zorder=4)
        if label == "after":
            for p_, d in zip(pos, data):
                ax.annotate(f"{d.median():.0f} ms" if d.median() >= 10 else f"{d.median():.1f} ms",
                            (p_ + 0.17, d.median()), xytext=(3, 0), textcoords="offset points", va="center",
                            fontsize=8.5, color=fs.INK, fontweight="semibold")
    ax.axhline(1000, color=fs.MUTED, lw=1.2, ls=(0, (1, 2.5)))
    ax.annotate("1 s target", (1, 1000), xycoords=("axes fraction", "data"), xytext=(-2, 5),
                textcoords="offset points", ha="right", fontsize=8.5, color=fs.MUTED)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 2e5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v / 1000:.0f} s" if v >= 1000 else (f"{v:.0f} ms" if v >= 1 else f"{v:.1f} ms")))
    ax.set_xticks(range(len(TOPOLOGIES)))
    ax.set_xticklabels(["Abilene · 11 nodes", "NSFNET · 14", "GÉANT · 22", "Germany50 · 50"])
    ax.set_title("Time per controller decision  ↓")
    fs.subtitle(ax, "box: middle 50 % of decisions · whiskers: 5-95 % · white bar: median · dot: slowest · log scale")
    ax.yaxis.set_minor_locator(plt.NullLocator())
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=fs.tint(fs.NEUTRAL, 0.85), label="before: all 2^m relay states enumerated"),
                       Patch(color=fs.tint(fs.METHOD_COLOR["Adaptive@0.5"], 0.85),
                             label="after: only shared relays enumerated (same results)")], loc="upper left")
    fs.save(fig, Path("figures") / "decision_time" / "decision_time.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("topologies", nargs="*", default=TOPOLOGIES)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--plot", action="store_true", help="only draw the before/after figure")
    args = ap.parse_args()
    if args.plot:
        plot()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(t, p, s) for t in args.topologies for p in ("random", "betweenness")
            for s in range(1000, 1000 + args.seeds)]
    results = Parallel(n_jobs=args.jobs)(delayed(time_run)(t, p, s) for t, p, s in jobs)
    df = pd.DataFrame([r for rs in results for r in rs])
    df.to_csv(OUT / "per_decision.csv", index=False)
    summary = df.groupby("topology")["decide_ms"].agg(
        n="count", median="median", p95=lambda x: float(np.percentile(x, 95)), max="max").reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
