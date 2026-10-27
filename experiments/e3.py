"""E3 -- Security vs attacker (README Section 9): rho_A x placement strategy
x method, Level A, NSFNET. Also the source of Phase 3's decision gate
(README Section 12, P3): does the adaptive heuristic beat B1/B2/B3 on
exposure at comparable cost?
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

RESULTS_DIR = Path("results") / "E3"
FIGURES_DIR = Path("figures") / "E3"

METHODS = ["B0", "B1", "B2", "B3", "Adaptive"]
PLACEMENTS = ["random", "betweenness", "key_throughput", "adaptive"]


def run_e3(
    config: dict,
    rho_a_values: list[float],
    placements: list[str],
    methods: list[str],
    n_seeds: int,
    n_requests: int,
    topology: str = "nsfnet",
) -> pd.DataFrame:
    graph = build_topology(topology, config)
    rows = []
    for placement in placements:
        for method in methods:
            for rho_a in rho_a_values:
                cfg = dict(config)
                cfg["attacker"] = dict(
                    config["attacker"], placement=placement, rho_A=rho_a, warmup_requests=50
                )

                exposure_per_seed, amp_per_seed, service_per_seed = [], [], []
                for seed in range(n_seeds):
                    df = run_level_a(graph, method, cfg, n_requests=n_requests, seed=seed)
                    exposure_per_seed.append(key_exposure_rate(df))
                    amp_per_seed.append(mean_amplification(df))
                    service_per_seed.append(service_ratio(df))

                exp_stats = aggregate_over_seeds(exposure_per_seed)
                amp_stats = aggregate_over_seeds(amp_per_seed)
                srv_stats = aggregate_over_seeds(service_per_seed)
                rows.append(
                    {
                        "placement": placement,
                        "method": method,
                        "rho_A": rho_a,
                        "exposure_mean": exp_stats["mean"],
                        "exposure_ci_low": exp_stats["ci_low"],
                        "exposure_ci_high": exp_stats["ci_high"],
                        "amplification_mean": amp_stats["mean"],
                        "amplification_ci_low": amp_stats["ci_low"],
                        "amplification_ci_high": amp_stats["ci_high"],
                        "service_ratio_mean": srv_stats["mean"],
                        "n_seeds": exp_stats["n_seeds"],
                    }
                )
    return pd.DataFrame(rows)


def plot_e3_exposure(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for placement, group in df.groupby("placement"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for method, sub in group.groupby("method"):
            sub = sub.sort_values("rho_A")
            ax.errorbar(
                sub["rho_A"], sub["exposure_mean"],
                yerr=[sub["exposure_mean"] - sub["exposure_ci_low"], sub["exposure_ci_high"] - sub["exposure_mean"]],
                marker="o", capsize=3, label=method,
            )
        ax.set_xlabel("Attacker pervasiveness rho_A")
        ax.set_ylabel("Key-exposure rate")
        topo = out_dir.name if out_dir.name in ("abilene", "geant", "germany50") else "nsfnet"
        ax.set_title(f"E3: exposure vs rho_A -- placement={placement} ({topo})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"e3_exposure_{placement}.png", dpi=150)
        plt.close(fig)


def plot_e3_cost(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for placement, group in df.groupby("placement"):
        fig, ax = plt.subplots(figsize=(6, 5))
        for method, sub in group.groupby("method"):
            sub = sub.sort_values("rho_A")
            ax.plot(sub["rho_A"], sub["amplification_mean"], marker="o", label=method)
        ax.set_xlabel("Attacker pervasiveness rho_A")
        ax.set_ylabel("Mean key amplification factor")
        ax.set_title(f"E3: cost vs rho_A -- placement={placement} (NSFNET)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"e3_cost_{placement}.png", dpi=150)
        plt.close(fig)


def decision_gate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Adaptive vs each baseline: exposure delta at comparable cost, per
    placement, averaged over rho_A > 0 (rho_A=0 has ~0 exposure for
    everyone and is not informative for this comparison)."""
    rows = []
    sweep = df[df["rho_A"] > 0]
    for placement, group in sweep.groupby("placement"):
        adaptive = group[group["method"] == "Adaptive"].set_index("rho_A")
        for baseline in ("B0", "B1", "B2", "B3"):
            base = group[group["method"] == baseline].set_index("rho_A")
            common = adaptive.index.intersection(base.index)
            if len(common) == 0:
                continue
            exposure_delta = (adaptive.loc[common, "exposure_mean"] - base.loc[common, "exposure_mean"]).mean()
            amp_delta = (adaptive.loc[common, "amplification_mean"] - base.loc[common, "amplification_mean"]).mean()
            service_delta = (
                adaptive.loc[common, "service_ratio_mean"] - base.loc[common, "service_ratio_mean"]
            ).mean()
            rows.append(
                {
                    "placement": placement,
                    "baseline": baseline,
                    "mean_exposure_delta_adaptive_minus_baseline": exposure_delta,
                    "mean_amplification_delta_adaptive_minus_baseline": amp_delta,
                    "mean_service_ratio_delta_adaptive_minus_baseline": service_delta,
                    "adaptive_better_exposure": bool(exposure_delta < 0),
                    "adaptive_cheaper_or_equal": bool(amp_delta <= 0),
                    # A lower-exposure result achieved mostly by refusing
                    # service isn't a fair win (README Section 0 rule 4):
                    # flag it so the report can't silently present exposure
                    # deltas without this caveat.
                    "adaptive_serves_meaningfully_less": bool(service_delta < -0.05),
                }
            )
    return pd.DataFrame(rows)
