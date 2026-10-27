"""Robustness of the E12 headline claim: paired statistics, detector
miscalibration, pool size (see experiments/e12_robustness.py).

Usage: PYTHONPATH=. uv run python scripts/e12_robustness.py paired|miscal|pool|misrank|misrank-plot
       PYTHONPATH=. uv run python scripts/e12_robustness.py topologies geant germany50
Outputs: results/E12/robustness/
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from experiments import e12_robustness as rb
from qkdmp.config import load_config

OUT = Path("results") / "E12" / "robustness"
BASELINES = ["B0", "B0+PA", "B1+PA", "B2+PA", "B3+PA"]
MISCAL = [  # (odds_factor, temperature)
    (0.25, 1.0), (0.5, 1.0), (2.0, 1.0), (4.0, 1.0),  # risk under- / over-estimated
    (1.0, 0.5), (1.0, 2.0),                           # under- / over-confident
]
MISRANK = [0.0, 0.1, 0.2, 0.4]  # share of compromised nodes swapped with an honest one
MISRANK_TOPOLOGIES = ["nsfnet", "abilene", "geant", "germany50"]
POOL_SECONDS = [0.1, 1.0, 3.0]  # E12's own value, 0.5, comes from the "paired" run


def _setup() -> tuple[dict, dict]:
    with open("configs/experiments/E12.yaml", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    base = load_config()
    cfg = dict(base)
    cfg["policy"] = dict(base["policy"], **exp["policy_overrides"])
    cfg["pool"] = dict(base["pool"], **exp["pool_overrides"])
    return cfg, exp


def _grid(exp: dict) -> dict:
    t = exp["test"]
    return dict(topologies=exp["topologies"], rho=exp["rho"], rho_a_values=exp["rho_A_values"],
                placements=exp["placements"], seeds=list(range(t["seed_offset"], t["seed_offset"] + t["n_seeds"])),
                n_requests=exp["n_requests_per_seed"])


def paired() -> None:
    cfg, exp = _setup()
    df = rb.run_per_seed(cfg, methods=BASELINES + [rb.OURS, "Adaptive@0.15"], **_grid(exp))
    df.to_csv(OUT / "per_seed_calibrated.csv", index=False)
    pc = rb.paired_comparisons(df)
    pc.to_csv(OUT / "paired_calibrated.csv", index=False)
    print(rb.summarize(pc).to_string(index=False))
    pc15 = rb.paired_comparisons(df, ours="Adaptive@0.15")
    pc15.to_csv(OUT / "paired_calibrated_eps0.15.csv", index=False)
    print("eps 0.15:", rb.summarize(pc15).to_string(index=False))


def miscal() -> None:
    """Only Adaptive and B3+PA read p_v; B0/B0+PA/B1+PA/B2+PA are unaffected by
    the distortion and draw no extra random numbers, so their per-seed rows
    from the calibrated run are reused unchanged."""
    cfg, exp = _setup()
    base = pd.read_csv(OUT / "per_seed_calibrated.csv")
    unaffected = base[base["method"].isin(["B0", "B0+PA", "B1+PA", "B2+PA"])]
    frames = []
    for odds, temp in MISCAL:
        c = dict(cfg, suspicion=dict(cfg["suspicion"], miscal_odds_factor=odds, miscal_temperature=temp))
        label = {"odds_factor": odds, "temperature": temp}
        df = rb.run_per_seed(c, methods=[rb.OURS, "B3+PA"], label=label, **_grid(exp))
        frames.append(pd.concat([df, unaffected.assign(**label)], ignore_index=True))
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT / "per_seed_miscal.csv", index=False)
    pc = rb.paired_comparisons(all_df, group_cols=("odds_factor", "temperature"))
    pc.to_csv(OUT / "paired_miscal.csv", index=False)
    rb.means_table(all_df, ("odds_factor", "temperature")).to_csv(OUT / "means_miscal.csv", index=False)
    print(rb.summarize(pc, ("odds_factor", "temperature")).to_string(index=False))


def misrank() -> None:
    """Detector that ranks nodes wrongly (`qkdmp.suspicion.misrank`). The swap
    draws random numbers from the run's generator, so every method is re-run
    (all methods of a seed still see the same attacker set and detector)."""
    cfg, exp = _setup()
    grid = dict(_grid(exp), topologies=MISRANK_TOPOLOGIES)
    frames = []
    for fraction in MISRANK:
        c = dict(cfg, suspicion=dict(cfg["suspicion"], misrank_fraction=fraction))
        frames.append(rb.run_per_seed(c, methods=BASELINES + [rb.OURS], label={"misrank_fraction": fraction},
                                      **grid))
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT / "per_seed_misrank.csv", index=False)
    pc = rb.paired_comparisons(all_df, group_cols=("misrank_fraction",))
    pc.to_csv(OUT / "paired_misrank.csv", index=False)
    rb.means_table(all_df, ("misrank_fraction",)).to_csv(OUT / "means_misrank.csv", index=False)
    print(rb.summarize(pc, ("misrank_fraction", "topology")).to_string(index=False))


def misrank_plot() -> None:
    """Figure for the misranking run: lead in secure goodput over the best
    baseline per condition, and key exposure, against the swapped share."""
    from analysis import figstyle as fs

    fs.apply()
    plt = fs.plt
    df = pd.read_csv(OUT / "per_seed_misrank.csv")
    cond = ["misrank_fraction", "topology", "placement", "rho_A"]
    means = df.groupby(cond + ["method"])[["secure_goodput", "exposure"]].mean().reset_index()
    ours = means[means["method"] == rb.OURS].set_index(cond)
    others = means[means["method"] != rb.OURS]
    best = others.groupby(cond)["secure_goodput"].max()
    lead = ((ours["secure_goodput"] / best - 1) * 100).groupby(level=[0, 1]).mean().unstack(1)
    exp_ours = ours["exposure"].groupby(level=[0, 1]).mean().unstack(1)
    exp_low = others.groupby(["misrank_fraction", "topology", "method"])["exposure"].mean()         .groupby(level=[0, 1]).min().unstack(1)
    x = [f * 100 for f in lead.index]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    ax.axhline(0, color=fs.BASELINE, lw=1.2)
    for t in MISRANK_TOPOLOGIES:
        fs.line(ax, x, lead[t], fs.TOPO_COLOR[t], label=fs.TOPO_LABEL[t])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_title("Secure goodput: Ours vs best baseline  ↑")
    fs.subtitle(ax, "mean lead over 6 attacker conditions; no comparison is significantly worse at any level")
    ax.legend(loc="upper right")
    ax = axes[1]
    for t in MISRANK_TOPOLOGIES:
        fs.line(ax, x, exp_low[t], fs.tint(fs.TOPO_COLOR[t], 0.5), ls=(0, (3, 2.5)), lw=1.6, ms=0)
        fs.line(ax, x, exp_ours[t], fs.TOPO_COLOR[t], label=fs.TOPO_LABEL[t])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 0.48)
    ax.set_title("Key exposure  ↓")
    fs.subtitle(ax, "solid: Ours · dashed: the lowest-exposure baseline on the same network · colours as left")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.0f}%" for v in x])
        ax.set_xlabel("compromised nodes whose score is swapped with an honest node's")
    fig.suptitle("Detector that ranks nodes wrongly · E12 grid · 4 networks · 30 held-out seeds", x=0.01, ha="left")
    fig.tight_layout()
    fs.save(fig, Path("figures") / "e12_robustness" / "misrank.png")


def pool() -> None:
    cfg, exp = _setup()
    frames = []
    for seconds in POOL_SECONDS:
        c = dict(cfg, pool=dict(cfg["pool"], capacity_seconds=seconds))
        frames.append(rb.run_per_seed(c, methods=BASELINES + [rb.OURS], label={"capacity_seconds": seconds},
                                      **_grid(exp)))
    base = pd.read_csv(OUT / "per_seed_calibrated.csv")
    frames.append(base[base["method"].isin(BASELINES + [rb.OURS])].assign(capacity_seconds=0.5))
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT / "per_seed_pool.csv", index=False)
    pc = rb.paired_comparisons(all_df, group_cols=("capacity_seconds",))
    pc.to_csv(OUT / "paired_pool.csv", index=False)
    rb.means_table(all_df, ("capacity_seconds",)).to_csv(OUT / "means_pool.csv", index=False)
    print(rb.summarize(pc, ("capacity_seconds",)).to_string(index=False))


def topologies(names: list[str]) -> None:
    """The E12 grid on additional real topologies. eps_target stays at the
    value chosen on NSFNET/Abilene tuning seeds -- these networks played no
    part in any choice, so they are a held-out test of the claim."""
    cfg, exp = _setup()
    grid = dict(_grid(exp), topologies=names)
    tag = "_".join(names)
    df = rb.run_per_seed(cfg, methods=BASELINES + [rb.OURS, "Adaptive@0.15"], **grid)
    df.to_csv(OUT / f"per_seed_{tag}.csv", index=False)
    rb.means_table(df).to_csv(OUT / f"means_{tag}.csv", index=False)
    for ours, suffix in [(rb.OURS, ""), ("Adaptive@0.15", "_eps0.15")]:
        pc = rb.paired_comparisons(df, ours=ours)
        pc.to_csv(OUT / f"paired_{tag}{suffix}.csv", index=False)
        print(ours, rb.summarize(pc).to_string(index=False))
        print(pc.groupby("topology")[["dominates_on_means", "goodput_better", "goodput_worse", "exposure_lower",
                                     "exposure_higher"]].sum().to_string())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "topologies":
        topologies(sys.argv[2:])
    else:
        {"paired": paired, "miscal": miscal, "pool": pool, "misrank": misrank,
         "misrank-plot": misrank_plot}[sys.argv[1]]()
