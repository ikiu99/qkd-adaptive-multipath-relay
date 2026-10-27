"""E10 -- Detector quality (README Section 9): detector_quality in [0, 1],
Level A.

Extended (fair-comparison follow-up): besides exposure, records service
ratio and key amplification, because the first E10 run showed Adaptive at
zero exposure even with a useless detector (quality 0) -- a flat,
pessimistic prior makes it spread keys over more paths, which is safe
against a random attacker but costs key material that exposure alone
doesn't show.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from analysis.metrics import key_exposure_rate, mean_amplification, service_ratio
from analysis.stats import aggregate_over_seeds
from qkdmp.sim.graph_mc import run_level_a
from qkdmp.topology import build_topology

RESULTS_DIR = Path("results") / "E10"
FIGURES_DIR = Path("figures") / "E10"


def run_e10(
    config: dict,
    detector_quality_values: list[float],
    methods: list[str],
    rho_a: float,
    n_seeds: int,
    n_requests: int,
    topology: str = "nsfnet",
) -> pd.DataFrame:
    graph = build_topology(topology, config)
    rows = []
    for method in methods:
        for dq in detector_quality_values:
            cfg = dict(config)
            cfg["attacker"] = dict(config["attacker"], placement="random", rho_A=rho_a)
            cfg["suspicion"] = dict(config["suspicion"], detector_quality=dq)

            per_metric: dict[str, list[float]] = {"exposure": [], "service_ratio": [], "amplification": []}
            for seed in range(n_seeds):
                df = run_level_a(graph, method, cfg, n_requests=n_requests, seed=seed)
                per_metric["exposure"].append(key_exposure_rate(df))
                per_metric["service_ratio"].append(service_ratio(df))
                per_metric["amplification"].append(mean_amplification(df))
            row = {"method": method, "detector_quality": dq, "n_seeds": n_seeds}
            for name, values in per_metric.items():
                stats = aggregate_over_seeds(values)
                row.update({f"{name}_mean": stats["mean"], f"{name}_ci_low": stats["ci_low"],
                            f"{name}_ci_high": stats["ci_high"]})
            rows.append(row)
    return pd.DataFrame(rows)


def plot_e10(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    panels = [("exposure", "Key-exposure rate"), ("amplification", "Mean key amplification"),
              ("service_ratio", "Service ratio")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (name, ylabel) in zip(axes, panels):
        for method, group in df.groupby("method"):
            group = group.sort_values("detector_quality")
            ax.errorbar(
                group["detector_quality"], group[f"{name}_mean"],
                yerr=[group[f"{name}_mean"] - group[f"{name}_ci_low"], group[f"{name}_ci_high"] - group[f"{name}_mean"]],
                marker="o", capsize=3, label=method,
            )
        ax.set_xlabel("Suspicion detector quality")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle(f"E10: detector quality {title}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
