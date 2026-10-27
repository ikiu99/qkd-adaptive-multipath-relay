"""Figures for the co-simulation report (reads the result steps).

Style: analysis/figstyle.py (shared with the report figures). In every
figure "Ours" is blue and each baseline keeps its colour; QKDNetSim pools are
drawn solid and the fluid pools as a lighter / dashed twin.

Usage: PYTHONPATH=. uv run python qkdnetsim-prototype/cosim/figures.py
Outputs: qkdnetsim-prototype/figures/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load  # noqa: E402
from run_cosim import OUT  # noqa: E402

from analysis import figstyle as fs  # noqa: E402

plt = fs.plt
FIG = OUT.parent / "figures"
OURS = "Adaptive@0.5"
ENGINE_LABEL = {"ns3": "QKDNetSim pools", "fluid": "fluid pools (Python)"}
FLUID_ALPHA = 0.42


def _engine_legend(ax, loc="upper right"):
    from matplotlib.patches import Patch

    handles = [Patch(color=fs.INK_2, label=ENGINE_LABEL["ns3"]),
               Patch(color=fs.tint(fs.INK_2, FLUID_ALPHA), label=ENGINE_LABEL["fluid"])]
    ax.legend(handles=handles, loc=loc)


def _ci(x: pd.Series) -> float:
    return 1.96 * x.std() / np.sqrt(len(x)) if len(x) > 1 else 0.0


def _method_bars(ax, per_seed: pd.DataFrame, metric: str, fmt: str) -> None:
    """Grouped bars: one colour per method, QKDNetSim solid, fluid light; a thin
    ink whisker for the 95 % CI over seeds; value printed on Ours only."""
    width = 0.38
    for j, method in enumerate(fs.METHOD_ORDER):
        for k, engine in enumerate(["ns3", "fluid"]):
            v = per_seed[(per_seed["method"] == method) & (per_seed["engine"] == engine)][metric]
            x = j + (k - 0.5) * (width + 0.03)
            color = fs.METHOD_COLOR[method]
            fs.bars(ax, [x], [v.mean()], width, color if engine == "ns3" else fs.tint(color, FLUID_ALPHA))
            ax.plot([x, x], [v.mean() - _ci(v), v.mean() + _ci(v)], color=fs.INK_2, lw=1.1, alpha=0.7, zorder=4,
                    solid_capstyle="round")
            if method == OURS and engine == "ns3":
                ax.annotate(fmt.format(v.mean()), (x, v.mean() + _ci(v)), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8.5, color=fs.INK,
                            fontweight="semibold")
    ax.set_xticks(range(len(fs.METHOD_ORDER)))
    ax.set_xticklabels([fs.METHOD_LABEL[m] for m in fs.METHOD_ORDER])
    ax.set_xlim(-0.6, len(fs.METHOD_ORDER) - 0.4)


def _agreement(ax, wide: pd.DataFrame, title: str) -> None:
    """QKDNetSim vs fluid for every (condition, method): Ours blue, baselines gray."""
    lo = min(wide["fluid"].min(), wide["ns3"].min()) - 0.03
    hi = max(wide["fluid"].max(), wide["ns3"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color=fs.BASELINE, lw=1.2, ls=(0, (4, 3)), zorder=1)
    base = wide[wide["method"] != OURS]
    ours = wide[wide["method"] == OURS]
    ax.scatter(base["fluid"], base["ns3"], s=34, color=fs.NEUTRAL, edgecolor=fs.SURFACE, lw=1.2, zorder=2,
               label="baselines")
    ax.scatter(ours["fluid"], ours["ns3"], s=46, color=fs.METHOD_COLOR[OURS], edgecolor=fs.SURFACE, lw=1.4,
               zorder=3, label="Ours")
    corr = np.corrcoef(wide["fluid"], wide["ns3"])[0, 1]
    ax.text(0.97, 0.05, f"r = {corr:.3f}", transform=ax.transAxes, ha="right", fontsize=9, color=fs.INK_2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, axis="both", color=fs.GRID, lw=0.8)
    ax.set_title(title)


def fig_e12(n_seeds: int) -> None:
    df = load("e12", n_seeds)
    df = df[df["topology"] == "nsfnet"]
    per_seed = df.groupby(["engine", "method", "seed"])[["secure_goodput", "exposure"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    _method_bars(axes[0], per_seed, "secure_goodput", "{:.3f}")
    axes[0].set_title("Secure goodput  ↑")
    fs.subtitle(axes[0], "mean of 6 attacker conditions, 95 % CI over seeds")
    _engine_legend(axes[0])
    _method_bars(axes[1], per_seed, "exposure", "{:.1%}")
    axes[1].set_title("Key exposure rate  ↓")
    fs.subtitle(axes[1], "share of delivered keys the attacker could read")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _engine_legend(axes[1], loc="upper left")
    wide = df.groupby(["placement", "rho_A", "method", "engine"])["secure_goodput"].mean().unstack("engine") \
        .reset_index()
    _agreement(axes[2], wide, "Same condition, two pool engines")
    fs.subtitle(axes[2], "one dot per (attacker condition, method)")
    axes[2].set_xlabel("secure goodput, fluid pools")
    axes[2].set_ylabel("secure goodput, QKDNetSim pools")
    axes[2].legend(loc="upper left")
    fig.suptitle(f"NSFNET · main claim (E12) · {n_seeds} held-out seeds", x=0.01, ha="left")
    fig.tight_layout()
    fs.save(fig, FIG / "cosim_e12.png")


def _lead_table(means: pd.DataFrame, cond: list[str]) -> pd.DataFrame:
    rows = []
    for (engine, topo), m in means.groupby(["engine", "topology"]):
        o = m[m["method"] == OURS].groupby(cond)["secure_goodput"].mean()
        b = m[m["method"] != OURS].groupby(cond)["secure_goodput"].max()
        rows.append({"engine": engine, "topology": topo, "ours": o.mean(), "best": b.mean()})
    return pd.DataFrame(rows).set_index(["engine", "topology"])


def fig_topologies(n_seeds: int) -> None:
    df = load("e12", n_seeds)
    means = df.groupby(["engine", "topology", "placement", "rho_A", "method"])["secure_goodput"].mean().reset_index()
    lead = _lead_table(means, ["placement", "rho_A"])
    topos = [t for t in fs.TOPO_ORDER if t in set(df["topology"])]
    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[2.3, 0.15, 1, 1], hspace=0.45, wspace=0.35, top=0.8, bottom=0.08,
                          left=0.05, right=0.98)
    ax = fig.add_subplot(gs[:, 0])
    width = 0.19
    for j, t in enumerate(topos):
        for i, (who, engine) in enumerate([("ours", "ns3"), ("ours", "fluid"), ("best", "ns3"), ("best", "fluid")]):
            color = fs.METHOD_COLOR[OURS] if who == "ours" else fs.NEUTRAL
            fs.bars(ax, [j + (i - 1.5) * (width + 0.02)], [lead.loc[(engine, t), who]], width,
                    color if engine == "ns3" else fs.tint(color, FLUID_ALPHA))
        gain = lead.loc[("ns3", t), "ours"] / lead.loc[("ns3", t), "best"] - 1
        top = max(lead.loc[(e, t), w] for e in ["ns3", "fluid"] for w in ["ours", "best"])
        ax.annotate(f"+{gain:.1%}", (j, top), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=9.5, color=fs.INK, fontweight="semibold")
    ax.set_xticks(range(len(topos)))
    ax.set_xticklabels([fs.TOPO_LABEL[t] for t in topos])
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("secure goodput")
    ax.set_title("Ours vs the best baseline, per network")
    fs.subtitle(ax, "mean over 6 attacker conditions; label = lead with QKDNetSim pools")
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=fs.METHOD_COLOR[OURS], label="Ours"),
                       Patch(color=fs.NEUTRAL, label="best baseline (per condition)"),
                       Patch(color=fs.INK_2, label=ENGINE_LABEL["ns3"]),
                       Patch(color=fs.tint(fs.INK_2, FLUID_ALPHA), label=ENGINE_LABEL["fluid"])],
              ncol=2, loc="upper left")
    wide = means.pivot_table(index=["topology", "placement", "rho_A", "method"], columns="engine",
                             values="secure_goodput").reset_index()
    for k, t in enumerate(topos):
        sub = fig.add_subplot(gs[k // 2, 2 + k % 2])
        _agreement(sub, wide[wide["topology"] == t], fs.TOPO_LABEL[t])
        sub.tick_params(labelsize=8)
        if k // 2 == 1:
            sub.set_xlabel("fluid pools", fontsize=8.5)
        if k % 2 == 0:
            sub.set_ylabel("QKDNetSim pools", fontsize=8.5)
    fig.suptitle(f"Main claim (E12) on four networks · QKDNetSim pools · {n_seeds} held-out seeds", x=0.01, ha="left")
    fs.save(fig, FIG / "cosim_topologies.png")


VIEW_COLOR = {"frozen": fs.SLOTS[1], "shadow": fs.SLOTS[0], "calibrated": fs.SLOTS[2]}
VIEW_LABEL = {"frozen": "frozen (last report)", "shadow": "shadow", "calibrated": "calibrated shadow"}


def fig_q2(n_seeds: int) -> None:
    df = load("q2", n_seeds)
    m = df[df["method"] == OURS].groupby(["engine", "rho", "view", "interval_s"])[
        ["failed", "secure_goodput"]].mean().reset_index()
    rhos = sorted(m["rho"].unique())
    fig, axes = plt.subplots(2, len(rhos), figsize=(12.5, 7.8), squeeze=False)
    for c, rho in enumerate(rhos):
        for r, metric in enumerate(["failed", "secure_goodput"]):
            ax = axes[r][c]
            for view, color in VIEW_COLOR.items():
                for engine in ["fluid", "ns3"]:
                    v = m[(m["engine"] == engine) & (m["rho"] == rho) & (m["view"] == view)].sort_values("interval_s")
                    solid = engine == "ns3"
                    # "shadow" drawn wide underneath: where it equals "calibrated" it shows as a blue halo
                    wide = view == "shadow" and solid
                    fs.line(ax, v["interval_s"], v[metric], color if solid else fs.tint(color, 0.55),
                            ls="-" if solid else (0, (3, 2.5)), lw=5.5 if wide else (2 if solid else 1.6),
                            ms=(9.5 if wide else 7) if solid else 0, zorder=2 if wide else 3)
            live = m[(m["engine"] == "ns3") & (m["rho"] == rho) & (m["view"] == "live")][metric]
            if len(live):
                ax.axhline(live.iloc[0], color=fs.MUTED, lw=1.2, ls=(0, (1, 2.5)), zorder=1)
                ax.annotate("live view", (0, live.iloc[0]), xycoords=("axes fraction", "data"), xytext=(2, 5),
                            textcoords="offset points", ha="left", fontsize=8, color=fs.MUTED)
            ax.set_xscale("log")
            ax.set_xticks([0.05, 0.1, 0.2, 0.5])
            ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5"])
            ax.minorticks_off()
            if metric == "failed":
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
                ax.set_title(f"Failed after admission  ↓   ·   load ρ = {rho}")
            else:
                ax.set_ylim(0.4, 0.7)
                ax.set_title(f"Secure goodput  ↑   ·   load ρ = {rho}")
                ax.set_xlabel("telemetry interval (s)")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=VIEW_COLOR["frozen"], lw=2, marker="o", mec=fs.SURFACE, label=VIEW_LABEL["frozen"]),
               Line2D([], [], color=VIEW_COLOR["shadow"], lw=5.5, label="shadow (blue halo where it equals calibrated)"),
               Line2D([], [], color=VIEW_COLOR["calibrated"], lw=2, marker="o", mec=fs.SURFACE,
                      label=VIEW_LABEL["calibrated"]),
               Line2D([], [], color=fs.INK_2, lw=2, label="QKDNetSim pools"),
               Line2D([], [], color=fs.tint(fs.INK_2, 0.55), lw=1.6, ls=(0, (3, 2.5)), label="fluid pools")]
    axes[0][0].legend(handles=handles, loc="upper left")
    fig.suptitle(f"NSFNET · stale controller, 20 % unreported local key use · Ours · {n_seeds} seeds",
                 x=0.01, ha="left")
    fig.tight_layout()
    fs.save(fig, FIG / "cosim_q2.png")


def fig_more(n_seeds: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6))
    # (a) load sweep
    ax = axes[0][0]
    m = load("e4", n_seeds).groupby(["engine", "rho", "method"])["secure_goodput"].mean().reset_index()
    for method in reversed(fs.METHOD_ORDER):
        for engine in ["fluid", "ns3"]:
            v = m[(m["method"] == method) & (m["engine"] == engine)].sort_values("rho")
            fs.line(ax, v["rho"], v["secure_goodput"], fs.METHOD_COLOR[method],
                    label=fs.METHOD_LABEL[method] if engine == "ns3" else None,
                    ls="-" if engine == "ns3" else (0, (3, 2.5)), alpha=1 if engine == "ns3" else 0.45,
                    lw=2.6 if method == OURS else 1.8, ms=7 if engine == "ns3" else 0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=3, loc="upper right")
    ax.set_xlabel("load ρ")
    ax.set_title("(a) Load sweep  ·  secure goodput ↑")
    fs.subtitle(ax, "solid: QKDNetSim pools · dashed: fluid pools")
    # (b) strategic attacker
    ax = axes[0][1]
    s = load("strategic", n_seeds)
    s = s.assign(detector_lag_s=s["detector_lag_s"].astype(str))
    lags = ["0.05", "0.2", "0.5", "never"]
    sm = s.groupby(["engine", "topology", "detector_lag_s", "method"])["secure_goodput"].mean().reset_index()
    ax.axhline(0, color=fs.BASELINE, lw=1.2)
    for topo in ["nsfnet", "abilene"]:
        for engine in ["fluid", "ns3"]:
            g = sm[(sm["engine"] == engine) & (sm["topology"] == topo)]
            o = g[g["method"] == OURS].set_index("detector_lag_s")["secure_goodput"]
            b = g[g["method"] != OURS].groupby("detector_lag_s")["secure_goodput"].max()
            y = [(o[k] / b[k] - 1) * 100 for k in lags]
            fs.line(ax, range(len(lags)), y, fs.TOPO_COLOR[topo],
                    label=fs.TOPO_LABEL[topo] if engine == "ns3" else None,
                    ls="-" if engine == "ns3" else (0, (3, 2.5)), alpha=1 if engine == "ns3" else 0.45,
                    ms=7 if engine == "ns3" else 0)
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels(["0.05 s", "0.2 s", "0.5 s", "never"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_xlabel("detector lag in following the attacker's moves")
    ax.set_title("(b) Strategic attacker  ·  Ours vs best baseline")
    fs.subtitle(ax, "secure goodput lead; below zero: Ours behind")
    ax.legend(loc="upper right")
    # (c) QoS classes: exposure per class
    ax = axes[1][0]
    q = load("q1", n_seeds)
    q = q[q["engine"] == "ns3"]
    methods = [("QoS-Adaptive", OURS, "Ours (QoS)"), ("B0+PA+QoS", "B0+PA", "B0+PA"),
               ("B1+PA+QoS", "B1+PA", "B1+PA"), ("B2+PA+QoS", "B2+PA", "B2+PA")]
    width = 0.2
    for i, (name, key, label) in enumerate(methods):
        vals = [q[q["method"] == name][f"{c}__exposure"].mean() for c in ["gold", "silver", "bronze"]]
        x = np.arange(3) + (i - 1.5) * (width + 0.02)
        fs.bars(ax, x, vals, width, fs.METHOD_COLOR[key], label=label)
        if key == OURS:
            for xi, v in zip(x, vals):
                ax.annotate(f"{v:.1%}", (xi, v), xytext=(0, 4), textcoords="offset points", ha="center",
                            fontsize=8.5, color=fs.INK, fontweight="semibold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["gold", "silver", "bronze"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("(c) QoS classes  ·  key exposure per class ↓")
    fs.subtitle(ax, "QKDNetSim pools (fluid pools within 1.5 points)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 0.66)
    # (d) key block size
    ax = axes[1][1]
    b = load("blocks", n_seeds)
    e = load("e12", n_seeds)
    e = e[(e["engine"] == "ns3") & (e["topology"] == "nsfnet")].assign(block_bits=256)
    d = pd.concat([e, b], ignore_index=True)
    dm = d.groupby(["block_bits", "method"])["secure_goodput"].mean().unstack("method")
    xs = range(len(dm.index))
    for method in reversed(fs.METHOD_ORDER):
        fs.line(ax, xs, dm[method], fs.METHOD_COLOR[method], label=fs.METHOD_LABEL[method],
                lw=2.6 if method == OURS else 1.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{int(k)} bits" for k in dm.index])
    ax.set_xlabel("post-processing key block")
    ax.set_title("(d) Key delivered in larger blocks  ·  secure goodput ↑")
    fs.subtitle(ax, "QKDNetSim only (the fluid model has no block size)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0.25, 0.67)
    fig.suptitle(f"NSFNET · further experiments with QKDNetSim pools · {n_seeds} seeds", x=0.01, ha="left")
    fig.tight_layout()
    fs.save(fig, FIG / "cosim_more.png")


ENGINE_ORDER = [("fluid", "fluid pools (Python)"), ("ns3", "QKDNetSim pools"),
                ("ns3relay", "QKDNetSim pools + share relay")]
ENGINE_SHADE = {"fluid": 0.35, "ns3": 0.65, "ns3relay": 1.0}


def fig_relay(n_seeds: int) -> None:
    """Lead of Ours over the best baseline per engine: E12 on four networks,
    and the strategic attacker against the detector lag."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    df = load("e12", n_seeds)
    means = df.groupby(["engine", "topology", "placement", "rho_A", "method"])["secure_goodput"].mean().reset_index()
    topos = [t for t in fs.TOPO_ORDER if t in set(df["topology"])]
    width = 0.26
    blue = fs.METHOD_COLOR[OURS]
    for i, (engine, label) in enumerate(ENGINE_ORDER):
        m = means[means["engine"] == engine]
        vals = []
        for t in topos:
            g = m[m["topology"] == t]
            o = g[g["method"] == OURS].set_index(["placement", "rho_A"])["secure_goodput"]
            b = g[g["method"] != OURS].groupby(["placement", "rho_A"])["secure_goodput"].max()
            vals.append((o.mean() / b.mean() - 1) * 100)
        x = np.arange(len(topos)) + (i - 1) * (width + 0.02)
        fs.bars(ax, x, vals, width, fs.tint(blue, ENGINE_SHADE[engine]), label=label)
        if engine == "ns3relay":
            for xi, v in zip(x, vals):
                ax.annotate(f"+{v:.1f}%", (xi, v), xytext=(0, 4), textcoords="offset points", ha="center",
                            fontsize=8.5, color=fs.INK, fontweight="semibold")
    ax.set_xticks(range(len(topos)))
    ax.set_xticklabels([fs.TOPO_LABEL[t] for t in topos])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"+{v:.0f}%"))
    ax.set_title("Main claim (E12): Ours vs best baseline, secure goodput  ↑")
    fs.subtitle(ax, "mean over 6 attacker conditions; label = with QKDNetSim share relay")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 34)
    ax = axes[1]
    s = load("strategic", n_seeds)
    s = s.assign(detector_lag_s=s["detector_lag_s"].astype(str))
    lags = ["0.05", "0.2", "0.5", "never"]
    sm = s.groupby(["engine", "detector_lag_s", "topology", "method"])["secure_goodput"].mean().reset_index()
    ax.axhline(0, color=fs.BASELINE, lw=1.2)
    for engine, label in ENGINE_ORDER:
        g = sm[sm["engine"] == engine]
        o = g[g["method"] == OURS].groupby("detector_lag_s")["secure_goodput"].mean()
        b = g[g["method"] != OURS].groupby(["detector_lag_s", "topology"])["secure_goodput"].max()             .groupby(level=0).mean()
        fs.line(ax, range(len(lags)), [(o[k] / b[k] - 1) * 100 for k in lags], fs.tint(blue, ENGINE_SHADE[engine]),
                label=label, lw=2.4 if engine == "ns3relay" else 1.8)
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels(["0.05 s", "0.2 s", "0.5 s", "never"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.set_xlabel("detector lag in following the attacker's moves")
    ax.set_title("Strategic attacker: Ours vs best baseline  ↑")
    fs.subtitle(ax, "mean over NSFNET and Abilene; below zero: Ours behind")
    ax.legend(loc="upper right")
    fig.suptitle(f"Three simulation engines, same seeds · {n_seeds} held-out seeds", x=0.01, ha="left")
    fig.tight_layout()
    fs.save(fig, FIG / "cosim_relay.png")


def main() -> None:
    fs.apply()
    FIG.mkdir(exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    fig_e12(n)
    fig_topologies(n)
    fig_q2(n)
    fig_more(n)
    fig_relay(n)


if __name__ == "__main__":
    main()
