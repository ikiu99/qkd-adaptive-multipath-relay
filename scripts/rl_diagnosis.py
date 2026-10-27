"""RL plateau diagnosis (Phase 7 follow-up): ablate the RLEnvDesign fixes one
at a time and evaluate every variant on the same 10 request batches, next to
Adaptive, a non-learned cheapest-option reference, and the batch-ILP
ceiling. Settings match E9 (configs/experiments/E9.yaml).

Usage: PYTHONPATH=. uv run python scripts/rl_diagnosis.py [topology ...]
       PYTHONPATH=. uv run python scripts/rl_diagnosis.py --seeds [topology ...]
The --seeds mode retrains legacy vs full over 3 training seeds (same
evaluation batches), so the redesign's effect is not a single-seed artefact.
       PYTHONPATH=. uv run python scripts/rl_diagnosis.py --plot
draws the ablation and the seed check from the saved CSVs
(figures/E9/rl_ablation.png).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

matplotlib.use("Agg")

from analysis.stats import aggregate_over_seeds
from qkdmp.config import load_config
from qkdmp.paths import path_edges
from qkdmp.policies.adaptive import AdaptiveThresholdPolicy
from qkdmp.policies.base import BLOCK, Request
from qkdmp.policies.batch_ilp import solve_batch_ilp
from qkdmp.policies.rl import LEGACY_DESIGN, CheapestOptionPolicy, RLEnvDesign, train_rl_policy
from qkdmp.sim.graph_mc import place_attacker, sample_request_pair
from qkdmp.suspicion import detector_p_v
from qkdmp.topology import build_topology

RESULTS_DIR = Path("results") / "E9"
FIGURES_DIR = Path("figures") / "E9"
N_BATCHES = 10
TIMESTEPS = 30_000

# name -> (design, episode_requests, align endpoints with evaluation)
VARIANTS: dict[str, tuple[RLEnvDesign, int | None, bool]] = {
    "RL legacy (Phase 5)": (LEGACY_DESIGN, 100, False),
    "RL legacy, ep=batch": (LEGACY_DESIGN, None, False),
    "RL +align": (RLEnvDesign(align_pool=True, sort_options=False, rich_obs=False, shaping_coef=0.0), None, True),
    "RL +align+sort": (RLEnvDesign(align_pool=True, sort_options=True, rich_obs=False, shaping_coef=0.0), None, True),
    "RL +align+sort+obs": (RLEnvDesign(align_pool=True, sort_options=True, rich_obs=True, shaping_coef=0.0), None, True),
    "RL full (+shaping)": (RLEnvDesign(), None, True),
}
SECONDARY_VARIANTS = ["RL legacy (Phase 5)", "RL full (+shaping)"]


def load_e9_config() -> tuple[dict, dict]:
    with open("configs/experiments/E9.yaml", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    base = load_config()
    cfg = dict(base)
    cfg["policy"] = dict(base["policy"], **exp["policy_overrides"])
    cfg["pool"] = dict(base["pool"], **exp["pool_overrides"])
    return cfg, exp


def setup(topology: str, cfg: dict, n_requests: int, n_batches: int, seed: int = 0):
    graph = build_topology(topology, cfg)
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    attacker_set = place_attacker(graph, cfg["attacker"], rng)
    p_v = detector_p_v(nodes, attacker_set, cfg, rng)
    batches = []
    for b in range(n_batches):
        brng = np.random.default_rng(1000 + b)
        batches.append([
            Request(*sample_request_pair(nodes, attacker_set, brng), cfg["policy"]["eps_target"], cfg["policy"]["a_target"])
            for _ in range(n_requests)
        ])
    return graph, attacker_set, p_v, batches


def evaluate(policy, name: str, graph_base, batches, p_v, cfg) -> list[dict]:
    key = cfg["key_size_bits"]
    rows = []
    for b, requests in enumerate(batches):
        graph = graph_base.copy()
        served, failed, cost = 0, 0, 0.0
        for req in requests:
            decision = policy.decide(req, graph, p_v, cfg)
            if decision is BLOCK:
                continue
            if any(graph.edges[u, v]["level_bits"] < key for p in decision.paths for u, v in path_edges(p)):
                failed += 1  # decided, but the shares couldn't actually be reserved
                continue
            for p in decision.paths:
                for u, v in path_edges(p):
                    graph.edges[u, v]["level_bits"] -= key
            served += 1
            cost += decision.cost or 0.0
        rows.append({"method": name, "batch": b, "served": served, "reservation_failures": failed, "cost": cost})
    return rows


def train_and_evaluate(
    name: str, topology: str, timesteps: int, n_batches: int, train_seed: int = 0,
) -> tuple[list[dict], list[float]]:
    import torch

    torch.set_num_threads(1)
    cfg, exp = load_e9_config()
    n_requests = exp["n_requests_per_batch"]
    graph, attacker_set, p_v, batches = setup(topology, cfg, n_requests, n_batches)
    design, episode_requests, align_endpoints = VARIANTS[name]
    policy, curve = train_rl_policy(
        graph.copy(), cfg, p_v, total_timesteps=timesteps, seed=train_seed,
        episode_requests=episode_requests or n_requests, design=design,
        endpoint_exclude=frozenset(attacker_set) if align_endpoints else frozenset(),
    )
    return evaluate(policy, name, graph, batches, p_v, cfg), curve


def run(topology: str, variant_names: list[str]) -> None:
    cfg, exp = load_e9_config()
    n_requests = exp["n_requests_per_batch"]
    graph, _, p_v, batches = setup(topology, cfg, n_requests, N_BATCHES)

    rows = evaluate(AdaptiveThresholdPolicy(fallback=cfg["policy"]["fallback"], seed=0), "Adaptive", graph, batches, p_v, cfg)
    rows += evaluate(CheapestOptionPolicy(seed=0), "Cheapest (non-learned)", graph, batches, p_v, cfg)
    for b, requests in enumerate(batches):
        res = solve_batch_ilp(requests, graph.copy(), p_v, cfg, window_seconds=3600.0, lambda_cost=1e-6,
                              rng=np.random.default_rng(0))
        rows.append({"method": "BatchILP", "batch": b, "served": sum(d is not None for d in res.decisions),
                     "reservation_failures": 0, "cost": sum(d.cost or 0.0 for d in res.decisions if d)})

    outputs = Parallel(n_jobs=len(variant_names))(
        delayed(train_and_evaluate)(name, topology, TIMESTEPS, N_BATCHES) for name in variant_names
    )
    curves = {}
    for name, (vrows, curve) in zip(variant_names, outputs):
        rows += vrows
        curves[name] = curve

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"rl_diagnosis_{topology}.csv", index=False)

    ilp = df[df["method"] == "BatchILP"].set_index("batch")["served"]
    summary = []
    for method, g in df.groupby("method", sort=False):
        served = g.set_index("batch")["served"]
        s = aggregate_over_seeds(list(served))
        gap = aggregate_over_seeds(list((ilp - served) / ilp))
        summary.append({
            "method": method, "served_mean": s["mean"], "served_ci_low": s["ci_low"], "served_ci_high": s["ci_high"],
            "gap_to_ilp_mean": gap["mean"], "gap_ci_low": gap["ci_low"], "gap_ci_high": gap["ci_high"],
            "reservation_failures_total": int(g["reservation_failures"].sum()), "n_batches": len(g),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_DIR / f"rl_diagnosis_summary_{topology}.csv", index=False)
    print(f"\n=== {topology} ({N_BATCHES} batches x {n_requests} requests) ===")
    print(summary_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, curve in curves.items():
        if curve:
            ma = pd.Series(curve).rolling(20, min_periods=1).mean()
            ax.plot(np.arange(1, len(curve) + 1) / len(curve), ma, label=name)
    ax.set_xlabel("Fraction of training (episodes)")
    ax.set_ylabel("Training service ratio (20-episode moving avg)")
    ax.set_title(f"RL diagnosis training curves ({topology})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rl_diagnosis_curves_{topology}.png", dpi=150)
    plt.close(fig)


TRAIN_SEEDS = [0, 1, 2]


def run_seeds(topology: str) -> None:
    cfg, exp = load_e9_config()
    graph, _, p_v, batches = setup(topology, cfg, exp["n_requests_per_batch"], N_BATCHES)
    ilp = [sum(d is not None for d in solve_batch_ilp(reqs, graph.copy(), p_v, cfg, window_seconds=3600.0,
                                                      lambda_cost=1e-6, rng=np.random.default_rng(0)).decisions)
           for reqs in batches]
    adaptive = evaluate(AdaptiveThresholdPolicy(fallback=cfg["policy"]["fallback"], seed=0), "Adaptive",
                        graph, batches, p_v, cfg)
    jobs = [(name, s) for name in SECONDARY_VARIANTS for s in TRAIN_SEEDS]
    outputs = Parallel(n_jobs=len(jobs))(
        delayed(train_and_evaluate)(name, topology, TIMESTEPS, N_BATCHES, s) for name, s in jobs
    )
    rows = []
    for r in adaptive:
        rows.append({**r, "train_seed": -1, "gap_to_ilp": (ilp[r["batch"]] - r["served"]) / ilp[r["batch"]]})
    for (name, s), (vrows, _) in zip(jobs, outputs):
        for r in vrows:
            rows.append({**r, "train_seed": s, "gap_to_ilp": (ilp[r["batch"]] - r["served"]) / ilp[r["batch"]]})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"rl_multiseed_{topology}.csv", index=False)
    per_seed = df.groupby(["method", "train_seed"], sort=False)["gap_to_ilp"].mean().reset_index()
    print(f"\n=== {topology}: mean gap to ILP per training seed ===")
    print(per_seed.to_string(index=False))
    summary = []
    for method, g in per_seed.groupby("method", sort=False):
        a = aggregate_over_seeds(list(g["gap_to_ilp"]))
        summary.append({"method": method, "n_train_seeds": len(g), "gap_mean": a["mean"],
                        "gap_min": g["gap_to_ilp"].min(), "gap_max": g["gap_to_ilp"].max()})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS_DIR / f"rl_multiseed_summary_{topology}.csv", index=False)
    print(summary_df.to_string(index=False))


ABLATION_LABELS = {
    "RL legacy (Phase 5)": "first env\n(Phase 5)",
    "RL legacy, ep=batch": "+ episode =\nevaluation batch",
    "RL +align": "+ fix 1:\nreservable only",
    "RL +align+sort": "+ fix 3a:\nsorted options",
    "RL +align+sort+obs": "+ fix 3b:\nrich observation",
    "RL full (+shaping)": "+ fix 2:\nreward shaping",
}


def plot() -> None:
    from analysis import figstyle as fs

    fs.apply()
    plt_ = fs.plt
    fig, axes = plt_.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    summ = pd.read_csv(RESULTS_DIR / "rl_diagnosis_summary_nsfnet.csv").set_index("method")
    names = list(ABLATION_LABELS)
    blue = fs.METHOD_COLOR["Adaptive@0.5"]
    for i, name in enumerate(names):
        row = summ.loc[name]
        color = fs.NEUTRAL if i < 2 else fs.tint(blue, 0.45 + 0.55 * (i - 2) / 3)
        fs.bars(ax, [i], [row["gap_to_ilp_mean"] * 100], 0.62, color)
        ax.plot([i, i], [max(0, row["gap_ci_low"]) * 100, row["gap_ci_high"] * 100], color=fs.INK_2, lw=1.1,
                alpha=0.7, zorder=4)
        ax.annotate(f"{row['gap_to_ilp_mean']:.1%}", (i, row["gap_ci_high"] * 100), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=fs.INK, fontweight="semibold")
    adaptive = summ.loc["Adaptive", "gap_to_ilp_mean"] * 100
    ax.axhline(adaptive, color=blue, lw=1.4, ls=(0, (4, 3)),
               label=f"proposed method (Adaptive): {adaptive:.1f}%")
    ax.legend(loc="upper right")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([ABLATION_LABELS[n] for n in names], fontsize=8.5)
    ax.yaxis.set_major_formatter(plt_.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, 58)
    ax.set_title("NSFNET · RL gap to the optimum, one fix at a time  ↓")
    fs.subtitle(ax, "gap to the batch-ILP ceiling · 10 batches x 25 requests · training seed 0 · "
                    "line: 95 % CI over batches")
    ax = axes[1]
    for k, topo in enumerate(["nsfnet", "random50"]):
        ms = pd.read_csv(RESULTS_DIR / f"rl_multiseed_{topo}.csv")
        per_seed = ms[ms["train_seed"] >= 0].groupby(["method", "train_seed"])["gap_to_ilp"].mean()
        ada = ms[ms["method"] == "Adaptive"]["gap_to_ilp"].mean() * 100
        for j, (method, color) in enumerate([("RL legacy (Phase 5)", fs.NEUTRAL), ("RL full (+shaping)", blue)]):
            vals = per_seed.loc[method].values * 100
            x = k * 1.0 + (j - 0.5) * 0.34
            ax.scatter([x] * len(vals), vals, s=60, color=color, edgecolor=fs.SURFACE, lw=1.4, zorder=3,
                       label=("first env" if j == 0 else "redesigned env") if k == 0 else None)
            text = f"{vals.min():.1f}%" if vals.min() == vals.max() else f"{vals.min():.1f}-{vals.max():.1f}%"
            ax.annotate(text, (x, vals.max()), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8.5, color=fs.INK_2)
        ax.plot([k - 0.35, k + 0.35], [ada, ada], color=blue, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NSFNET", "random50"])
    ax.set_xlim(-0.6, 1.6)
    ax.yaxis.set_major_formatter(plt_.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(-2, 50)
    ax.set_title("Same check over 3 training seeds  ↓")
    fs.subtitle(ax, "one dot per training seed · dashed: proposed method")
    ax.legend(loc="upper right")
    fig.suptitle("RL comparison point (E9): what held it back, and what the redesign changed", x=0.01, ha="left")
    fig.tight_layout()
    fs.save(fig, FIGURES_DIR / "rl_ablation.png")


def main() -> None:
    if sys.argv[1:2] == ["--plot"]:
        plot()
        return
    if sys.argv[1:2] == ["--seeds"]:
        for topology in sys.argv[2:] or ["nsfnet", "random50"]:
            run_seeds(topology)
        return
    topologies = sys.argv[1:] or ["nsfnet"]
    for topology in topologies:
        names = list(VARIANTS) if topology == "nsfnet" else SECONDARY_VARIANTS
        run(topology, names)


if __name__ == "__main__":
    main()
