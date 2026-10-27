"""E2 -- Replication (README Section 9 / Section 12 Phase 2).

(a) B1 (XOR n-of-n MNOP) empirical exposure rate vs the closed-form analytic
    bound for node-disjoint paths under iid random node compromise.

    Confirmed against Kiktenko, Taydunanov & Fedorov, "Routing Algorithm
    Within the Multiple Non-Overlapping Paths' Approach for QKD Networks",
    Entropy 26:1102, 2024 (full text supplied by the user,
    articles/entropy-26-01102-v2.pdf). Section 2 states: "Assuming a
    probability of compromise of a node operated as a trusted node is
    epsilon_compr, we can estimate that the probability of compromising a
    key between any two remote nodes in the M-paths' regime is
    upper-bounded by epsilon_compr^M." That is exactly `prod_i q_i` (q_i the
    per-path compromise probability) in the special case where every path
    has the same per-path compromise probability epsilon_compr; the general
    form used here, `prod_i (1 - (1 - rho_A)^h_i)` (h_i = relay-node count
    of path i), is the natural generalization to paths of unequal hop count,
    and is exactly `exposure_prob_disjoint` from Phase 1 (already
    unit-tested there against Monte Carlo). Note the M-paths scheme this
    paper defines (Eq. 1: K_ij = XOR over all M paths' shares) is exactly
    B1's XOR n-of-n construction, confirming B1's structure is faithful.

(b) B3 (penalty-based MTD, Ghourab et al. 2025) on `grid`: attack success
    rate vs attacker pervasiveness rho_A, for n in {2, 4, 6} paths.

    B3 is now implemented from the paper's actual equations (see
    configs/baselines/b3.yaml, full text supplied by the user,
    articles/BDCC-09-00076.pdf) rather than README's abbreviated summary.

    # ASSUMPTION / caveat: Ghourab et al. 2025's Figure 7 (Section 6, page
    # 18) reports *numeric* attack-success curves (e.g. "97.22% lower
    # success rate at 20% attacker pervasiveness with 2 paths") for their
    # exact experimental setup (their own topology/traffic/attacker model,
    # not specified in enough detail to reproduce exactly, and compared
    # against their own "traditional multipath" baseline, not our B1/B2).
    # This reproduces only the qualitative *shape* (monotonically
    # non-decreasing attack success as rho_A grows, and higher n giving
    # lower attack success since B3 is n-of-n) on our own topology/attacker
    # model -- not a numeric match to Figure 7.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from analysis.metrics import key_exposure_rate
from analysis.stats import aggregate_over_seeds
from qkdmp.paths import relay_nodes
from qkdmp.policies.base import Request
from qkdmp.policies.baselines import B1Policy
from qkdmp.sim.graph_mc import run_level_a
from qkdmp.topology import build_topology

RESULTS_DIR = Path("results") / "E2"
FIGURES_DIR = Path("figures") / "E2"


def analytic_disjoint_exposure_prob(paths: list[list], rho_a: float) -> float:
    prob = 1.0
    for path in paths:
        h = len(relay_nodes(path))
        prob *= 1 - (1 - rho_a) ** h
    return prob


def run_e2a(config: dict, rho_a_values: list[float], n_seeds: int, n_requests: int) -> pd.DataFrame:
    graph = build_topology("nsfnet", config)
    rows = []
    for rho_a in rho_a_values:
        cfg = dict(config)
        # mode="bernoulli": the analytic bound below assumes iid per-node
        # compromise, so the simulated attacker set must be drawn the same
        # way (see attacker.place_random docstring) for this to be a fair
        # comparison rather than an artifact of fixed-count rounding at
        # NSFNET's small N=14.
        cfg["attacker"] = dict(config["attacker"], placement="random", rho_A=rho_a, random_mode="bernoulli")

        empirical_per_seed = []
        analytic_per_seed = []
        for seed in range(n_seeds):
            # exclude_attacker_as_endpoint=False: see run_level_a's docstring
            # -- excluding a *fixed* attacker set from (s, d) sampling biases
            # the empirical exposure rate upward relative to this closed
            # form (a real selection effect, confirmed during Phase 2/3
            # corrections). This check validates the exposure formula in
            # isolation from that effect; every other experiment keeps the
            # README 3.1-compliant default (True).
            df = run_level_a(
                graph, "B1", cfg, n_requests=n_requests, seed=seed, exclude_attacker_as_endpoint=False
            )
            served = df[~df["blocked"]]
            if len(served) == 0:
                continue
            empirical_per_seed.append(key_exposure_rate(df))
            # Recompute each served request's own path set for the analytic
            # bound (path choice is deterministic given s,d; cheap to redo).
            b1 = B1Policy(**cfg["baselines"]["b1"])
            analytic_vals = []
            for _, row in served.iterrows():
                req = Request(row["s"], row["d"], 0.0, 0.0)
                decision = b1.decide(req, graph, {}, cfg)
                analytic_vals.append(analytic_disjoint_exposure_prob(decision.paths, rho_a))
            analytic_per_seed.append(float(np.mean(analytic_vals)))

        emp_stats = aggregate_over_seeds(empirical_per_seed)
        an_stats = aggregate_over_seeds(analytic_per_seed)
        rows.append(
            {
                "rho_A": rho_a,
                "empirical_mean": emp_stats["mean"],
                "empirical_ci_low": emp_stats["ci_low"],
                "empirical_ci_high": emp_stats["ci_high"],
                "analytic_mean": an_stats["mean"],
                "analytic_ci_low": an_stats["ci_low"],
                "analytic_ci_high": an_stats["ci_high"],
                "n_seeds": emp_stats["n_seeds"],
            }
        )
    return pd.DataFrame(rows)


def plot_e2a(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(
        df["rho_A"], df["empirical_mean"],
        yerr=[df["empirical_mean"] - df["empirical_ci_low"], df["empirical_ci_high"] - df["empirical_mean"]],
        marker="o", capsize=3, label="B1 simulated (Level A, ground truth)",
    )
    ax.plot(df["rho_A"], df["analytic_mean"], marker="x", linestyle="--", label="Analytic bound (this derivation)")
    ax.set_xlabel("Attacker pervasiveness rho_A")
    ax.set_ylabel("Key-exposure rate")
    ax.set_title("E2(a): B1 simulated exposure vs analytic bound (NSFNET)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_e2b(config: dict, rho_a_values: list[float], n_values: list[int], n_seeds: int, n_requests: int) -> pd.DataFrame:
    graph = build_topology("grid", config)
    rows = []
    for n in n_values:
        for rho_a in rho_a_values:
            cfg = dict(config)
            cfg["attacker"] = dict(config["attacker"], placement="random", rho_A=rho_a)
            cfg["baselines"] = dict(config["baselines"], b3=dict(config["baselines"]["b3"], n=n))

            per_seed = []
            for seed in range(n_seeds):
                df = run_level_a(graph, "B3", cfg, n_requests=n_requests, seed=seed)
                if (~df["blocked"]).sum() == 0:
                    continue
                per_seed.append(key_exposure_rate(df))
            stats = aggregate_over_seeds(per_seed)
            rows.append({"n": n, "rho_A": rho_a, **stats})
    return pd.DataFrame(rows)


def plot_e2b(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for n, group in df.groupby("n"):
        group = group.sort_values("rho_A")
        ax.errorbar(
            group["rho_A"], group["mean"],
            yerr=[group["mean"] - group["ci_low"], group["ci_high"] - group["mean"]],
            marker="o", capsize=3, label=f"n={n}",
        )
    ax.set_xlabel("Attacker pervasiveness rho_A")
    ax.set_ylabel("Attack success rate (key-exposure rate)")
    ax.set_title("E2(b): B3 on grid -- qualitative replication of Ghourab 2025 Fig. 7 shape")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def check_e2b_monotonic(df: pd.DataFrame) -> dict[int, bool]:
    """Qualitative check: attack success should be non-decreasing in rho_A
    for each n (the only claim we can verify without the source figure)."""
    results = {}
    for n, group in df.groupby("n"):
        group = group.sort_values("rho_A")
        values = group["mean"].to_numpy()
        results[int(n)] = bool(np.all(np.diff(values) >= -1e-9))
    return results
