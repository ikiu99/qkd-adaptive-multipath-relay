"""E6 -- Drop/tamper sweep (README Section 9): p_drop, p_tamper sweep;
success under attack, silent failures. Level B.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from analysis.metrics import silent_failure_rate, success_under_attack
from analysis.stats import aggregate_over_seeds
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.topology import build_topology

RESULTS_DIR = Path("results") / "E6"
FIGURES_DIR = Path("figures") / "E6"


def run_e6(
    config: dict, drop_tamper_values: list[tuple[float, float]], methods: list[str], rho: float,
    n_seeds: int, n_requests: int, topology: str = "nsfnet", rho_a: float | None = None,
) -> pd.DataFrame:
    """`rho_a`: attacker pervasiveness (README 3.1's rho_A), overriding
    config['attacker']['rho_A']. The default rho_A=0.1 makes the
    "simultaneously-exposed" event needed for success_under_attack too rare
    to sample even at 30 seeds x 200 requests for multipath methods (t > 1)
    -- see reports/PHASE_6_REPORT.md's "Corrections after review"."""
    graph = build_topology(topology, config)
    arrival_rate = arrival_rate_for_load(graph, config["key_size_bits"], rho)
    rows = []
    for method in methods:
        for p_drop, p_tamper in drop_tamper_values:
            cfg = dict(config)
            cfg["des"] = dict(config["des"], arrival_rate_per_s=arrival_rate, traffic_mode="uniform")
            attacker_kwargs = {"p_drop": p_drop, "p_tamper": p_tamper}
            if rho_a is not None:
                attacker_kwargs["rho_A"] = rho_a
            cfg["attacker"] = dict(config["attacker"], **attacker_kwargs)

            success_vals, silent_vals = [], []
            for seed in range(n_seeds):
                df = run_level_b(graph, method, cfg, n_requests=n_requests, seed=seed)
                served = df[~df["blocked"] & ~df.get("failed", False)]
                success_vals.append(success_under_attack(served))
                silent_vals.append(silent_failure_rate(served))

            s_stats = aggregate_over_seeds(success_vals)
            sf_stats = aggregate_over_seeds(silent_vals)
            rows.append(
                {
                    "method": method, "p_drop": p_drop, "p_tamper": p_tamper,
                    "success_under_attack_mean": s_stats["mean"],
                    "silent_failure_rate_mean": sf_stats["mean"], "n_seeds": s_stats["n_seeds"],
                }
            )
    return pd.DataFrame(rows)


def plot_e6(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method, group in df.groupby("method"):
        group = group.sort_values("p_drop")
        x = group["p_drop"]
        axes[0].plot(x, group["success_under_attack_mean"], marker="o", label=method)
        axes[1].plot(x, group["silent_failure_rate_mean"], marker="o", label=method)
    axes[0].set_xlabel("p_drop = p_tamper"); axes[0].set_ylabel("Success under attack")
    axes[0].set_title("E6: success under attack")
    axes[1].set_xlabel("p_drop = p_tamper"); axes[1].set_ylabel("Silent-failure rate")
    axes[1].set_title("E6: silent failures")
    axes[0].legend(); axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
