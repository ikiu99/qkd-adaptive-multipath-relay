"""Does the method's own exposure estimate hold in practice?

For every delivered key, the policy predicted P_exp (the probability that
the attacker sees >= t shares) from the detector's calibrated per-node
probabilities, then picked the cheapest option under eps_target. Selecting
the option with the lowest *estimated* risk can systematically pick
options whose risk is underestimated (the optimizer's curse), so the
predicted and the realised exposure rates are compared here, bin by bin
(reliability diagram), and against eps_target itself.

Grid: the E12 test grid (NSFNET + Abilene, random/betweenness placement,
rho_A 0.1-0.3, rho=0.6, held-out seeds 1000-1029) at eps_target 0.15, 0.5
and 0.9.

Usage: PYTHONPATH=. uv run python scripts/calibration_check.py
Outputs: results/calibration/, figures/calibration/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

matplotlib.use("Agg")

from qkdmp.config import load_config
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.topology import build_topology

OUT = Path("results") / "calibration"
FIG = Path("figures") / "calibration"
EPS_VALUES = [0.15, 0.5, 0.9]
BINS = [0.0, 0.001, 0.01, 0.03, 0.1, 0.2, 0.4, 1.0]


def _one(graph, cfg, seed: int) -> pd.DataFrame:
    df = run_level_b(graph, "Adaptive", cfg, n_requests=cfg["_n_requests"], seed=seed)
    done = df[~df["blocked"].astype(bool) & ~df["failed"].astype(bool)] if len(df) else df
    if "p_exp_pred" not in done:
        return pd.DataFrame(columns=["p_exp_pred", "exposed", "seed"])
    out = done[["p_exp_pred", "exposed"]].dropna().copy()
    out["seed"] = seed
    return out


def collect() -> pd.DataFrame:
    with open("configs/experiments/E12.yaml", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    base = load_config()
    base = dict(base, policy=dict(base["policy"], **exp["policy_overrides"]),
                pool=dict(base["pool"], **exp["pool_overrides"]), _n_requests=exp["n_requests_per_seed"])
    seeds = list(range(exp["test"]["seed_offset"], exp["test"]["seed_offset"] + exp["test"]["n_seeds"]))
    frames = []
    for topo in exp["topologies"]:
        graph = build_topology(topo, base)
        rate = arrival_rate_for_load(graph, base["key_size_bits"], exp["rho"])
        for placement in exp["placements"]:
            for rho_a in exp["rho_A_values"]:
                for eps in EPS_VALUES:
                    cfg = dict(base, policy=dict(base["policy"], eps_target=eps),
                               des=dict(base["des"], arrival_rate_per_s=rate, traffic_mode="uniform"),
                               attacker=dict(base["attacker"], placement=placement, rho_A=rho_a))
                    parts = Parallel(n_jobs=-1)(delayed(_one)(graph, cfg, s) for s in seeds)
                    df = pd.concat(parts, ignore_index=True)
                    df = df.assign(topology=topo, placement=placement, rho_A=rho_a, eps_target=eps)
                    frames.append(df)
                    print(f"{topo}/{placement}/{rho_a}/eps={eps}: n={len(df)} "
                          f"pred={df['p_exp_pred'].mean():.4f} obs={df['exposed'].astype(float).mean():.4f}",
                          flush=True)
    return pd.concat(frames, ignore_index=True)


def _ci(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z, p = 1.96, k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def reliability(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    df = df.assign(exposed=df["exposed"].astype(float), bin=pd.cut(df["p_exp_pred"], BINS, include_lowest=True))
    rows = []
    for key, g in df.groupby(group + ["bin"], observed=True):
        key = key if isinstance(key, tuple) else (key,)
        k, n = int(g["exposed"].sum()), len(g)
        lo, hi = _ci(k, n)
        rows.append({**dict(zip(group + ["bin"], key)), "n": n, "predicted": g["p_exp_pred"].mean(),
                     "observed": k / n, "observed_ci_low": lo, "observed_ci_high": hi})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = collect()
    df.to_csv(OUT / "per_request.csv", index=False)

    overall = reliability(df, [])
    overall.to_csv(OUT / "reliability_overall.csv", index=False)
    by_topo = reliability(df, ["topology", "placement"])
    by_topo.to_csv(OUT / "reliability_by_condition.csv", index=False)

    summary = df.assign(exposed=df["exposed"].astype(float)).groupby(["eps_target", "topology", "placement"]).agg(
        n=("exposed", "size"), predicted=("p_exp_pred", "mean"), observed=("exposed", "mean"),
        max_pred=("p_exp_pred", "max")).reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    print(overall.to_string(index=False))
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot([0, 0.5], [0, 0.5], color="black", lw=0.8, ls="--", label="perfect calibration")
    for (topo, placement), g in by_topo.groupby(["topology", "placement"]):
        ax.errorbar(g["predicted"], g["observed"],
                    yerr=[g["observed"] - g["observed_ci_low"], g["observed_ci_high"] - g["observed"]],
                    marker="o", capsize=3, label=f"{topo}, {placement}")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("Predicted exposure probability (bin mean)")
    ax.set_ylabel("Observed exposure frequency (95% CI)")
    ax.set_title("Reliability of the method's own risk estimate")
    ax.legend(fontsize=8)
    ax = axes[1]
    x = np.arange(len(EPS_VALUES))
    s = summary.groupby("eps_target")[["predicted", "observed"]].mean()
    ax.bar(x - 0.2, s["predicted"], 0.4, label="mean predicted", color="#7f8c99")
    ax.bar(x + 0.2, s["observed"], 0.4, label="mean observed", color="#c0392b")
    for i, e in enumerate(EPS_VALUES):
        ax.hlines(e, i - 0.45, i + 0.45, colors="black", linestyles="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"eps_target={e}" for e in EPS_VALUES])
    ax.set_ylabel("Exposure per delivered key (log)")
    ax.set_title("Predicted vs observed, against the target (dashed)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
