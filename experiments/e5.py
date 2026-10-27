"""E5 -- Hotspot traffic at medium/high load (README Section 9), Level B.
Main figure (README 9 "Presentation emphasis").
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

RESULTS_DIR = Path("results") / "E5"
FIGURES_DIR = Path("figures") / "E5"


def run_e5(
    config: dict, rho_values: dict[str, float], methods: list[str], hotspot_fraction: float,
    hotspot_pairs: int, n_seeds: int, n_requests: int, topology: str = "nsfnet",
) -> pd.DataFrame:
    """`rho_values`: {"medium": 0.3, "high": 0.8}-style label -> rho."""
    graph = build_topology(topology, config)
    rows = []
    for method in methods:
        for label, rho in rho_values.items():
            arrival_rate = arrival_rate_for_load(graph, config["key_size_bits"], rho)
            policy_name, cfg = method_config(method, config)
            cfg["des"] = dict(
                config["des"], arrival_rate_per_s=arrival_rate, traffic_mode="hotspot",
                hotspot_fraction=hotspot_fraction, hotspot_pairs=hotspot_pairs,
            )
            def run_one(seed: int, policy_name=policy_name, cfg=cfg) -> dict:
                return run_metrics(run_level_b(graph, policy_name, cfg, n_requests=n_requests, seed=seed))

            rows.append({"method": method, "load_label": label, "rho": rho, **over_seeds(run_one, n_seeds)})
    return pd.DataFrame(rows)


def plot_e5(df: pd.DataFrame, out_path: Path) -> None:
    labels = list(df["load_label"].unique())
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5), squeeze=False)
    axes = axes[0]
    for ax, label in zip(axes, labels):
        group = df[df["load_label"] == label]
        methods = group["method"]
        ax.bar(methods, group["blocking_mean"],
               yerr=[group["blocking_mean"] - group["blocking_ci_low"], group["blocking_ci_high"] - group["blocking_mean"]],
               capsize=4)
        ax.set_title(f"E5: hotspot blocking -- {label} load")
        ax.set_ylabel("Blocking probability")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
