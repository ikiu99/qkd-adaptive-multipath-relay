"""E4 -- Cost & blocking vs load (README Section 9): rho in [low, overload] x
method, Level B, heterogeneous rates (NSFNET's default length-dependent rate
model). Main figure (README 9 "Presentation emphasis").
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from experiments.level_b_common import method_config, over_seeds, run_metrics
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.topology import build_topology

RESULTS_DIR = Path("results") / "E4"
FIGURES_DIR = Path("figures") / "E4"

METHODS = ["B0", "B1", "B2", "B3", "Adaptive"]


def run_e4(
    config: dict, rho_values: list[float], methods: list[str], n_seeds: int, n_requests: int,
    topology: str = "nsfnet",
) -> pd.DataFrame:
    graph = build_topology(topology, config)
    rows = []
    for method in methods:
        for rho in rho_values:
            arrival_rate = arrival_rate_for_load(graph, config["key_size_bits"], rho)
            policy_name, cfg = method_config(method, config)
            cfg["des"] = dict(config["des"], arrival_rate_per_s=arrival_rate, traffic_mode="uniform")

            def run_one(seed: int, policy_name=policy_name, cfg=cfg) -> dict:
                return run_metrics(run_level_b(graph, policy_name, cfg, n_requests=n_requests, seed=seed))

            rows.append({"method": method, "rho": rho, "arrival_rate_per_s": arrival_rate,
                         **over_seeds(run_one, n_seeds)})
    return pd.DataFrame(rows)


def plot_e4(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for method, group in df.groupby("method"):
        group = group.sort_values("rho")
        axes[0].errorbar(
            group["rho"], group["blocking_mean"],
            yerr=[group["blocking_mean"] - group["blocking_ci_low"], group["blocking_ci_high"] - group["blocking_mean"]],
            marker="o", capsize=3, label=method,
        )
        if "failed_mean" in group:
            axes[1].errorbar(
                group["rho"], group["failed_mean"],
                yerr=[group["failed_mean"] - group["failed_ci_low"], group["failed_ci_high"] - group["failed_mean"]],
                marker="o", capsize=3, label=method,
            )
        axes[2].plot(group["rho"], group["amplification_mean"], marker="o", label=method)
    axes[0].set_xlabel("Load factor rho"); axes[0].set_ylabel("Blocking probability (policy)")
    axes[0].set_title("E4: policy blocking vs load")
    axes[1].set_xlabel("Load factor rho"); axes[1].set_ylabel("Pool-exhaustion failure rate")
    axes[1].set_title("E4: pool-scarcity failures vs load")
    axes[2].set_xlabel("Load factor rho"); axes[2].set_ylabel("Mean key amplification factor")
    axes[2].set_title("E4: cost vs load")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "e4_blocking_and_cost_vs_load.png", dpi=150)
    plt.close(fig)


def plot_admission_families(df: pd.DataFrame, x: str, metric: str, ylabel: str, out_path: Path) -> None:
    """One panel per baseline family (raw, +AC, +PA) with Adaptive overlaid,
    so each panel answers "does a pool check alone close the gap?"."""
    families = [b for b in ("B0", "B1", "B2", "B3") if b in set(df["method"])]
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4.5), sharey=True, squeeze=False)
    for ax, base in zip(axes[0], families):
        for method, style in [(base, ":"), (f"{base}+AC", "--"), (f"{base}+PA", "-."), ("Adaptive", "-"),
                              ("Adaptive@0.5", "-")]:
            g = df[df["method"] == method].sort_values(x)
            if len(g):
                ax.plot(g[x], g[f"{metric}_mean"], style, marker="o", label=method,
                        linewidth=2.5 if method.startswith("Adaptive") else 1.5)
        ax.set_title(f"{base} family")
        ax.set_xlabel(x)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel(ylabel)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
