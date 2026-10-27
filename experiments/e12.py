"""E12 -- Secure goodput vs attacker strength (fair-comparison follow-up).
Level B. Pool-checked baselines vs Adaptive at two operating points, as the
share and placement of compromised relays grow.
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

RESULTS_DIR = Path("results") / "E12"
FIGURES_DIR = Path("figures") / "E12"


def run_e12(config: dict, topologies: list[str], rho: float, rho_a_values: list[float], placements: list[str],
            methods: list[str], n_seeds: int, n_requests: int, seed_offset: int = 0) -> pd.DataFrame:
    rows = []
    for topology in topologies:
        graph = build_topology(topology, config)
        arrival_rate = arrival_rate_for_load(graph, config["key_size_bits"], rho)
        for placement in placements:
            for rho_a in rho_a_values:
                for method in methods:
                    policy_name, cfg = method_config(method, config)
                    cfg["des"] = dict(config["des"], arrival_rate_per_s=arrival_rate, traffic_mode="uniform")
                    cfg["attacker"] = dict(config["attacker"], placement=placement, rho_A=rho_a)

                    def run_one(seed: int, policy_name=policy_name, cfg=cfg) -> dict:
                        return run_metrics(run_level_b(graph, policy_name, cfg, n_requests=n_requests, seed=seed))

                    row = {"topology": topology, "placement": placement, "rho_A": rho_a, "method": method,
                           **over_seeds(run_one, n_seeds, seed_offset=seed_offset)}
                    rows.append(row)
                    print(f"{topology}/{placement}/rho_A={rho_a}/{method}: goodput={row['secure_goodput_mean']:.3f} "
                          f"exposure={row['exposure_mean']:.3f}", flush=True)
    return pd.DataFrame(rows)


def plot_e12(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for topology, tgroup in df.groupby("topology"):
        placements = list(tgroup["placement"].unique())
        fig, axes = plt.subplots(2, len(placements), figsize=(6 * len(placements), 9), squeeze=False)
        for col, placement in enumerate(placements):
            pgroup = tgroup[tgroup["placement"] == placement]
            for row, (metric, ylabel) in enumerate([("secure_goodput", "Secure goodput"),
                                                    ("exposure", "Key-exposure rate")]):
                ax = axes[row][col]
                for method, g in pgroup.groupby("method", sort=False):
                    g = g.sort_values("rho_A")
                    ax.errorbar(g["rho_A"], g[f"{metric}_mean"],
                                yerr=[g[f"{metric}_mean"] - g[f"{metric}_ci_low"], g[f"{metric}_ci_high"] - g[f"{metric}_mean"]],
                                marker="o", capsize=3, label=method,
                                linewidth=2.5 if method.startswith("Adaptive") else 1.2)
                ax.set_xlabel("Attacker share rho_A")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{topology}, {placement} placement")
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"e12_goodput_vs_attacker_{topology}.png", dpi=140)
        plt.close(fig)


def choose_eps(tuning: pd.DataFrame) -> float:
    """Pre-registered selection rule (fixed before the test run): the
    Adaptive operating point with the highest secure goodput averaged over
    every tuning cell."""
    adaptive = tuning[tuning["method"].str.startswith("Adaptive@")]
    means = adaptive.groupby("method")["secure_goodput_mean"].mean()
    return float(means.idxmax().split("@", 1)[1])


def dominance_table(df: pd.DataFrame, adaptive_method: str) -> pd.DataFrame:
    """Per cell and baseline: does `adaptive_method` have secure goodput >=
    and exposure <= the baseline (point estimates), and is the goodput lead
    significant (non-overlapping 95% CIs)? Also whether *some* Adaptive
    operating point dominates (the Pareto claim)."""
    rows = []
    for (topology, placement, rho_a), g in df.groupby(["topology", "placement", "rho_A"]):
        a = g[g["method"] == adaptive_method].iloc[0]
        family = g[g["method"].str.startswith("Adaptive")]
        for _, b in g[~g["method"].str.startswith("Adaptive")].iterrows():
            some = family[(family["secure_goodput_mean"] >= b["secure_goodput_mean"])
                          & (family["exposure_mean"] <= b["exposure_mean"])]
            rows.append({
                "topology": topology, "placement": placement, "rho_A": rho_a, "baseline": b["method"],
                "adaptive_goodput": a["secure_goodput_mean"], "baseline_goodput": b["secure_goodput_mean"],
                "adaptive_exposure": a["exposure_mean"], "baseline_exposure": b["exposure_mean"],
                "dominates": bool(a["secure_goodput_mean"] >= b["secure_goodput_mean"]
                                  and a["exposure_mean"] <= b["exposure_mean"]),
                "goodput_lead_significant": bool(a["secure_goodput_ci_low"] > b["secure_goodput_ci_high"]),
                "goodput_significantly_worse": bool(a["secure_goodput_ci_high"] < b["secure_goodput_ci_low"]),
                "some_operating_point_dominates": len(some) > 0,
            })
    return pd.DataFrame(rows)
