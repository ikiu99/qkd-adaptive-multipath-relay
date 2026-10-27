"""E7 -- Pareto front (README Section 9): sweep eps_target; exposure rate vs
amplification factor and vs blocking. Main figure (README 9 "Presentation
emphasis"). Both Level A (`run_e7`) and Level B (`run_e7_level_b`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from analysis.metrics import blocking_probability, key_exposure_rate, mean_amplification
from analysis.stats import aggregate_over_seeds
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.sim.graph_mc import run_level_a
from qkdmp.topology import build_topology


RESULTS_DIR = Path("results") / "E7"
FIGURES_DIR = Path("figures") / "E7"


def run_e7(
    config: dict, eps_target_values: list[float], a_target: float, rho_a: float,
    n_seeds: int, n_requests: int, topology: str = "nsfnet",
) -> pd.DataFrame:
    graph = build_topology(topology, config)
    rows = []
    for eps in eps_target_values:
        cfg = dict(config)
        cfg["policy"] = dict(config["policy"], eps_target=eps, a_target=a_target, fallback="block")
        cfg["attacker"] = dict(config["attacker"], placement="random", rho_A=rho_a)

        exp_vals, amp_vals, block_vals = [], [], []
        for seed in range(n_seeds):
            df = run_level_a(graph, "Adaptive", cfg, n_requests=n_requests, seed=seed)
            exp_vals.append(key_exposure_rate(df))
            amp_vals.append(mean_amplification(df))
            block_vals.append(blocking_probability(df))

        e_stats = aggregate_over_seeds(exp_vals)
        a_stats = aggregate_over_seeds(amp_vals)
        b_stats = aggregate_over_seeds(block_vals)
        rows.append(
            {
                "eps_target": eps,
                "exposure_mean": e_stats["mean"], "exposure_ci_low": e_stats["ci_low"], "exposure_ci_high": e_stats["ci_high"],
                "amplification_mean": a_stats["mean"], "blocking_mean": b_stats["mean"], "n_seeds": e_stats["n_seeds"],
            }
        )
    return pd.DataFrame(rows)


def plot_e7(df: pd.DataFrame, out_path: Path, level_label: str = "Level A") -> None:
    df = df.sort_values("eps_target")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df["amplification_mean"], df["exposure_mean"], marker="o")
    for _, row in df.iterrows():
        axes[0].annotate(f"eps={row['eps_target']:.2f}", (row["amplification_mean"], row["exposure_mean"]), fontsize=8)
    axes[0].set_xlabel("Mean key amplification factor"); axes[0].set_ylabel("Key-exposure rate")
    axes[0].set_title(f"E7 ({level_label}): Pareto front -- exposure vs cost")

    axes[1].plot(df["blocking_mean"], df["exposure_mean"], marker="o")
    for _, row in df.iterrows():
        axes[1].annotate(f"eps={row['eps_target']:.2f}", (row["blocking_mean"], row["exposure_mean"]), fontsize=8)
    axes[1].set_xlabel("Blocking probability"); axes[1].set_ylabel("Key-exposure rate")
    axes[1].set_title(f"E7 ({level_label}): Pareto front -- exposure vs blocking")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_e7_level_b(
    config: dict, eps_target_values: list[float], a_target: float, rho_a: float, rho: float,
    n_seeds: int, n_requests: int, topology: str = "nsfnet",
) -> pd.DataFrame:
    """Level B companion to `run_e7`: same eps_target sweep, but through the
    DES (dynamic pools, latency, drop/tamper) at a fixed load `rho`."""
    graph = build_topology(topology, config)
    arrival_rate = arrival_rate_for_load(graph, config["key_size_bits"], rho)
    rows = []
    for eps in eps_target_values:
        cfg = dict(config)
        cfg["policy"] = dict(config["policy"], eps_target=eps, a_target=a_target, fallback="block")
        cfg["attacker"] = dict(config["attacker"], placement="random", rho_A=rho_a)
        cfg["des"] = dict(config["des"], arrival_rate_per_s=arrival_rate, traffic_mode="uniform")

        exp_vals, amp_vals, block_vals = [], [], []
        for seed in range(n_seeds):
            df = run_level_b(graph, "Adaptive", cfg, n_requests=n_requests, seed=seed)
            served = df[~df["blocked"] & ~df.get("failed", False)]
            exp_vals.append(key_exposure_rate(served) if len(served) else float("nan"))
            amp_vals.append(mean_amplification(served) if len(served) else float("nan"))
            block_vals.append(blocking_probability(df))

        e_stats = aggregate_over_seeds(exp_vals)
        a_stats = aggregate_over_seeds(amp_vals)
        b_stats = aggregate_over_seeds(block_vals)
        rows.append(
            {
                "eps_target": eps,
                "exposure_mean": e_stats["mean"], "exposure_ci_low": e_stats["ci_low"], "exposure_ci_high": e_stats["ci_high"],
                "amplification_mean": a_stats["mean"], "blocking_mean": b_stats["mean"], "n_seeds": e_stats["n_seeds"],
            }
        )
    return pd.DataFrame(rows)
