"""Sensitivity sweep (Phase 7 follow-up, user-requested): several of this
project's parameters were fixed by guesswork rather than a known-correct
value (B3's lambda_cor/delta_w -- the Ghourab paper gives no numbers;
pool.capacity_seconds -- README gives no pool-size model at all). Rather
than trying to guess the "right" value, this sweeps a range of plausible
values and checks whether the *qualitative* conclusions that depend on them
hold throughout -- turning an open assumption into a robustness-checked
finding (or an honest report that it doesn't hold, per README rule 4).

Part A: B3 (grid, E2b) -- is attack success still monotonically
non-decreasing in rho_A for every (lambda_cor, delta_w) combination?
Part B: pool capacity (NSFNET, E4-style) -- does Adaptive still fail less
and cost less than B1/B2/B3 under load, for every capacity_seconds value?
Part C: B3 k_candidates (not given by the paper) -- does Adaptive's
advantage over B3 depend on how many candidate paths B3 scores?
Part D: DES latency constants (des.control_latency_s, des.hop_processing_s,
both ASSUMPTION values) scaled x0.1 / x1 / x10 -- same ranking check as B.
Part E: suspicion Beta mapping (suspicion.beta_perfect, ASSUMPTION pending
[REF-SUSPICION]) -- weaker / default / sharper "perfect detector" end;
is Adaptive's exposure still lowest of all multipath methods (Level A,
NSFNET, rho_A=0.2, random and adaptive placements)?

Usage: PYTHONPATH=. uv run python scripts/sensitivity_sweep.py [A B C D E]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.metrics import blocking_probability, mean_amplification
from experiments.e2 import analytic_disjoint_exposure_prob  # noqa: F401 (not used, kept for parity)
from qkdmp.config import load_config
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.sim.graph_mc import run_level_a
from qkdmp.topology import build_topology

OUT_DIR = Path("results") / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def part_a_b3_grid(n_seeds: int = 10, n_requests: int = 150) -> pd.DataFrame:
    config = load_config()
    graph = build_topology("grid", config)
    rho_a_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    lambda_cor_values = [0.5, 1.0, 2.0]
    delta_w_values = [0.5, 1.0, 2.0]

    rows = []
    for lambda_cor in lambda_cor_values:
        for delta_w in delta_w_values:
            means = []
            for rho_a in rho_a_values:
                cfg = dict(config)
                cfg["attacker"] = dict(config["attacker"], placement="random", rho_A=rho_a)
                cfg["baselines"] = dict(
                    config["baselines"],
                    b3=dict(config["baselines"]["b3"], lambda_cor=lambda_cor, delta_w=delta_w),
                )
                exposures = []
                for seed in range(n_seeds):
                    df = run_level_a(graph, "B3", cfg, n_requests=n_requests, seed=seed)
                    served = df[~df["blocked"]]
                    if len(served) == 0:
                        continue
                    exposures.append(served["exposed"].mean())
                means.append(float(np.mean(exposures)) if exposures else float("nan"))

            diffs = np.diff(means)
            monotonic = bool(np.all(diffs[~np.isnan(diffs)] >= -1e-9))
            rows.append(
                {
                    "lambda_cor": lambda_cor, "delta_w": delta_w,
                    **{f"exposure_rho_A_{r}": m for r, m in zip(rho_a_values, means)},
                    "monotonic_non_decreasing": monotonic,
                }
            )
    return pd.DataFrame(rows)


def part_b_pool_capacity(n_seeds: int = 10, n_requests: int = 60) -> pd.DataFrame:
    config = load_config()
    cfg_base = dict(config)
    cfg_base["policy"] = dict(config["policy"], eps_target=0.15, a_target=0.8, fallback="block")
    rho_values = [0.1, 0.6, 1.3]
    capacity_values = [0.1, 0.3, 0.5, 1.0, 2.0]
    methods = ["B1", "B2", "B3", "Adaptive"]

    rows = []
    for capacity_seconds in capacity_values:
        cfg = dict(cfg_base)
        cfg["pool"] = dict(config["pool"], capacity_seconds=capacity_seconds)
        graph = build_topology("nsfnet", cfg)

        method_results: dict[str, dict] = {}
        for method in methods:
            fail_vals, amp_vals = [], []
            for rho in rho_values:
                arrival_rate = arrival_rate_for_load(graph, cfg["key_size_bits"], rho)
                cfg_r = dict(cfg)
                cfg_r["des"] = dict(config["des"], arrival_rate_per_s=arrival_rate, traffic_mode="uniform")
                fails, amps = [], []
                for seed in range(n_seeds):
                    df = run_level_b(graph, method, cfg_r, n_requests=n_requests, seed=seed)
                    fails.append(df["failed"].mean() if "failed" in df and len(df) else float("nan"))
                    served = df[~df["blocked"] & ~df.get("failed", False)]
                    amps.append(mean_amplification(served) if len(served) else float("nan"))
                fail_vals.append(float(np.nanmean(fails)))
                amp_vals.append(float(np.nanmean(amps)))
            method_results[method] = {"fail_at_high_load": fail_vals[-1], "mean_amp": float(np.nanmean(amp_vals))}

        adaptive_fails_least = method_results["Adaptive"]["fail_at_high_load"] <= min(
            method_results[m]["fail_at_high_load"] for m in ("B1", "B2", "B3")
        )
        adaptive_cheapest = method_results["Adaptive"]["mean_amp"] <= min(
            method_results[m]["mean_amp"] for m in ("B1", "B2", "B3")
        )
        rows.append(
            {
                "capacity_seconds": capacity_seconds,
                **{f"{m}_fail_at_rho_1.3": method_results[m]["fail_at_high_load"] for m in methods},
                **{f"{m}_mean_amp": method_results[m]["mean_amp"] for m in methods},
                "adaptive_fails_least_of_multipath": adaptive_fails_least,
                "adaptive_cheapest_of_multipath": adaptive_cheapest,
            }
        )
    return pd.DataFrame(rows)


def _load_sweep(cfg: dict, methods: list[str], n_seeds: int, n_requests: int) -> dict:
    """E4-style Level B run on NSFNET at rho in {0.1, 0.6, 1.3}: per method,
    pool-exhaustion failure rate at the highest load and mean amplification."""
    graph = build_topology("nsfnet", cfg)
    rho_values = [0.1, 0.6, 1.3]
    out = {}
    for method in methods:
        fail_vals, amp_vals = [], []
        for rho in rho_values:
            cfg_r = dict(cfg)
            cfg_r["des"] = dict(cfg["des"], arrival_rate_per_s=arrival_rate_for_load(graph, cfg["key_size_bits"], rho),
                                traffic_mode="uniform")
            fails, amps = [], []
            for seed in range(n_seeds):
                df = run_level_b(graph, method, cfg_r, n_requests=n_requests, seed=seed)
                fails.append(df["failed"].mean() if "failed" in df and len(df) else float("nan"))
                served = df[~df["blocked"] & ~df.get("failed", False)]
                amps.append(mean_amplification(served) if len(served) else float("nan"))
            fail_vals.append(float(np.nanmean(fails)))
            amp_vals.append(float(np.nanmean(amps)))
        out[method] = {"fail_at_high_load": fail_vals[-1], "mean_amp": float(np.nanmean(amp_vals))}
    others = [m for m in methods if m != "Adaptive"]
    return {
        **{f"{m}_fail_at_rho_1.3": out[m]["fail_at_high_load"] for m in methods},
        **{f"{m}_mean_amp": out[m]["mean_amp"] for m in methods},
        "adaptive_fails_least": out["Adaptive"]["fail_at_high_load"] <= min(out[m]["fail_at_high_load"] for m in others),
        "adaptive_cheapest": out["Adaptive"]["mean_amp"] <= min(out[m]["mean_amp"] for m in others),
    }


def _e4_cfg() -> dict:
    config = load_config()
    cfg = dict(config)
    cfg["policy"] = dict(config["policy"], eps_target=0.15, a_target=0.8, fallback="block")
    cfg["pool"] = dict(config["pool"], capacity_seconds=0.5)  # configs/experiments/E4.yaml's value
    return cfg


def part_c_b3_k_candidates(n_seeds: int = 10, n_requests: int = 60) -> pd.DataFrame:
    rows = []
    for k in [4, 8, 16]:
        cfg = _e4_cfg()
        cfg["baselines"] = dict(cfg["baselines"], b3=dict(cfg["baselines"]["b3"], k_candidates=k))
        rows.append({"k_candidates": k, **_load_sweep(cfg, ["B3", "Adaptive"], n_seeds, n_requests)})
    return pd.DataFrame(rows)


def part_d_des_latency(n_seeds: int = 10, n_requests: int = 60) -> pd.DataFrame:
    rows = []
    for scale in [0.1, 1.0, 10.0]:
        cfg = _e4_cfg()
        cfg["des"] = dict(cfg["des"], control_latency_s=cfg["des"]["control_latency_s"] * scale,
                          hop_processing_s=cfg["des"]["hop_processing_s"] * scale)
        rows.append({"latency_scale": scale,
                     **_load_sweep(cfg, ["B0", "B1", "B2", "B3", "Adaptive"], n_seeds, n_requests)})
    return pd.DataFrame(rows)


def part_e_suspicion_beta(n_seeds: int = 10, n_requests: int = 150) -> pd.DataFrame:
    config = load_config()
    graph = build_topology("nsfnet", config)
    methods = ["B1", "B2", "B3", "Adaptive"]
    rows = []
    for label, beta in [("weak", [4.0, 1.5, 1.5, 4.0]), ("default", [8.0, 1.0, 1.0, 8.0]),
                        ("sharp", [16.0, 1.0, 1.0, 16.0])]:
        for placement in ["random", "adaptive"]:
            cfg = dict(config)
            cfg["suspicion"] = dict(config["suspicion"], beta_perfect=beta)
            cfg["attacker"] = dict(config["attacker"], placement=placement, rho_A=0.2)
            exposure = {}
            for method in methods:
                vals = []
                for seed in range(n_seeds):
                    df = run_level_a(graph, method, cfg, n_requests=n_requests, seed=seed)
                    served = df[~df["blocked"]]
                    if len(served):
                        vals.append(served["exposed"].mean())
                exposure[method] = float(np.mean(vals)) if vals else float("nan")
            rows.append({
                "beta_perfect": label, "placement": placement,
                **{f"{m}_exposure": exposure[m] for m in methods},
                "adaptive_lowest_exposure": exposure["Adaptive"] <= min(exposure[m] for m in ("B1", "B2", "B3")),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parts = sys.argv[1:] or ["A", "B", "C", "D", "E"]
    if "A" in parts:
        df_a = part_a_b3_grid()
        df_a.to_csv(OUT_DIR / "sensitivity_b3_grid.csv", index=False)
        print("Part A -- B3 on grid, monotonicity across (lambda_cor, delta_w):")
        print(df_a[["lambda_cor", "delta_w", "monotonic_non_decreasing"]].to_string(index=False))

    if "B" in parts:
        df_b = part_b_pool_capacity()
        df_b.to_csv(OUT_DIR / "sensitivity_pool_capacity.csv", index=False)
        print("\nPart B -- pool capacity_seconds, Adaptive ranking across values:")
        print(df_b[["capacity_seconds", "adaptive_fails_least_of_multipath", "adaptive_cheapest_of_multipath"]].to_string(index=False))

    for part, fn, name in [("C", part_c_b3_k_candidates, "sensitivity_b3_k_candidates.csv"),
                           ("D", part_d_des_latency, "sensitivity_des_latency.csv"),
                           ("E", part_e_suspicion_beta, "sensitivity_suspicion_beta.csv")]:
        if part in parts:
            df = fn()
            df.to_csv(OUT_DIR / name, index=False)
            print(f"\nPart {part}:")
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
