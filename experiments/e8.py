"""E8 -- Ablation (README Section 9): proposed method without pool-awareness;
without suspicion; uniform rate model. Level B. Main figure (README 9
"Presentation emphasis").
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from experiments.level_b_common import over_seeds, run_metrics
from qkdmp.policies.adaptive import AdaptiveThresholdPolicy
from qkdmp.policies.admission import AdmissionControlled
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.topology import build_topology

RESULTS_DIR = Path("results") / "E8"
FIGURES_DIR = Path("figures") / "E8"

# "no_pool_awareness+AC/PA" (fair-comparison follow-up): the pool-unaware
# variant with the same admission check / depleted-link pruning the
# baselines get (`qkdmp.policies.admission`). Comparing them to "full"
# isolates what pool-*weighted* option selection adds beyond a pool check.
VARIANTS = ["full", "no_pool_awareness", "no_pool_awareness+AC", "no_pool_awareness+PA",
            "no_suspicion", "uniform_rate_model"]


def _policy(variant: str, fallback: str, seed: int):
    base = AdaptiveThresholdPolicy(fallback=fallback, seed=seed, pool_aware=not variant.startswith("no_pool_awareness"))
    if "+" in variant:
        return AdmissionControlled(base, variant.split("+", 1)[1])
    return base


def run_e8(config: dict, rho: float, n_seeds: int, n_requests: int, topology: str = "nsfnet") -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        cfg = dict(config)
        if variant == "uniform_rate_model":
            cfg["rate_model"] = dict(config["rate_model"], mode="uniform")
        graph = build_topology(topology, cfg)
        arrival_rate = arrival_rate_for_load(graph, cfg["key_size_bits"], rho)
        cfg["des"] = dict(cfg["des"], arrival_rate_per_s=arrival_rate, traffic_mode="uniform")

        def run_one(seed: int, variant=variant, cfg=cfg, graph=graph) -> dict:
            p_v_override = None
            if variant == "no_suspicion":
                p_v_override = {v: cfg["suspicion"]["p_min"] for v in graph.nodes()}
            df = run_level_b(graph, "Adaptive", cfg, n_requests=n_requests, seed=seed, p_v_override=p_v_override,
                             policy_override=_policy(variant, cfg["policy"]["fallback"], seed))
            return run_metrics(df)

        rows.append({"variant": variant, "rho": rho, **over_seeds(run_one, n_seeds)})
    return pd.DataFrame(rows)


def plot_e8(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    metrics = [("blocking_mean", "Blocking probability (policy)"), ("failed_mean", "Pool-exhaustion failure rate"),
               ("exposure_mean", "Key-exposure rate"), ("amplification_mean", "Mean amplification"),
               ("secure_goodput_mean", "Secure goodput")]
    for ax, (col, ylabel) in zip(axes, metrics):
        ax.bar(df["variant"], df[col])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"E8: ablation ({out_path.stem})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
