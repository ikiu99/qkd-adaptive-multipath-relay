"""Regenerate every figure and every number reported in the paper from the
committed result CSVs.

    PYTHONPATH=. python scripts/paper_figures.py [output-dir]

Figures are written to <output-dir>/figures/*.pdf and the numbers to
<output-dir>/generated/numbers.tex (default output-dir: paper/).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper"
RES = ROOT / "results"
ROB = RES / "E12" / "robustness"
NS3 = ROOT / "qkdnetsim-prototype" / "results"
FIG = HERE / "figures"
GEN = HERE / "generated"
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from analysis import figstyle as fs  # noqa: E402

plt = fs.plt
WHITE = "#ffffff"


def style() -> None:
    fs.apply()
    plt.rcParams.update({
        "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
        "font.family": ["DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.pad_inches": 0.05, "axes.titlesize": 10, "axes.titlepad": 6,
    })


style()
OURS = "Adaptive@0.5"
BASES = ["B0", "B0+PA", "B1+PA", "B2+PA", "B3+PA"]
TOPOS = fs.TOPO_ORDER
TL = fs.TOPO_LABEL
COL = fs.METHOD_COLOR
LAB = {OURS: "Proposed", **{b: b for b in BASES}}

macros: dict[str, str] = {}


def mac(name: str, value) -> None:
    macros[name] = value if isinstance(value, str) else f"{value}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def f3(x: float) -> str:
    return f"{x:.3f}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}"


def ci95(v: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    m = v.mean()
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, m - h, m + h


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# E12: headline comparison on the four topologies (held-out seeds 1000-1029)
# ---------------------------------------------------------------------------
per_seed = pd.concat([
    pd.read_csv(ROB / "per_seed_calibrated.csv"),
    pd.read_csv(ROB / "per_seed_geant.csv"),
    pd.read_csv(ROB / "per_seed_germany50.csv"),
], ignore_index=True)
per_seed = per_seed[per_seed["method"].isin([OURS] + BASES)]
paired = {
    "nsfnet+abilene": pd.read_csv(ROB / "paired_calibrated.csv"),
    "geant": pd.read_csv(ROB / "paired_geant.csv"),
    "germany50": pd.read_csv(ROB / "paired_germany50.csv"),
}

# check the significance flags against the Holm-adjusted p-values
for key, df in paired.items():
    sig_better = (df["secure_goodput_diff"] > 0) & (df["secure_goodput_p_t_holm"] < 0.05)
    assert (sig_better == df["goodput_better"]).all(), key
    sig_lower = (df["exposure_diff"] < 0) & (df["exposure_p_t_holm"] < 0.05)
    assert (sig_lower == df["exposure_lower"]).all(), key

all_paired = pd.concat(paired.values(), ignore_index=True)
mac("NComp", len(all_paired))
mac("NDominate", int(all_paired["dominates_on_means"].sum()))
mac("NGoodputSig", int(all_paired["goodput_better"].sum()))
mac("NExposureSig", int(all_paired["exposure_lower"].sum()))
mac("NWorse", int((all_paired["goodput_worse"] | all_paired["exposure_higher"]).sum()))
mac("NWilcoxGood", int(((all_paired.secure_goodput_diff > 0) & (all_paired.secure_goodput_p_wilcoxon_holm < 0.05)).sum()))
mac("NWilcoxExp", int(((all_paired.exposure_diff < 0) & (all_paired.exposure_p_wilcoxon_holm < 0.05)).sum()))
for t in TOPOS:
    sub = all_paired[all_paired["topology"] == t]
    k = t.capitalize().replace("50", "Fifty")
    mac(f"NDom{k}", int(sub["dominates_on_means"].sum()))
    mac(f"NGood{k}", int(sub["goodput_better"].sum()))
    mac(f"NExp{k}", int(sub["exposure_lower"].sum()))
    mac(f"NCmp{k}", len(sub))

# cell means
cell = per_seed.groupby(["topology", "placement", "rho_A", "method"])[["secure_goodput", "exposure",
                                                                      "key_efficiency", "amplification",
                                                                      "blocking"]].mean().reset_index()

# per-topology table: mean over the 6 cells
rows = []
for t in TOPOS:
    c = cell[cell["topology"] == t]
    means = c.groupby("method")[["secure_goodput", "exposure", "key_efficiency"]].mean()
    best_base = means.loc[BASES, "secure_goodput"].idxmax()
    ours = means.loc[OURS]
    base_exp = c[c["method"].isin(BASES)]["exposure"]
    ours_exp = c[c["method"] == OURS]["exposure"]
    rows.append(dict(topology=t, ours=ours["secure_goodput"], best=best_base,
                     best_val=means.loc[best_base, "secure_goodput"],
                     lead=ours["secure_goodput"] / means.loc[best_base, "secure_goodput"] - 1,
                     ours_exp_max=ours_exp.max(), base_exp_min=base_exp.min(), base_exp_max=base_exp.max(),
                     ours_eff=ours["key_efficiency"]))
topo_tab = pd.DataFrame(rows)
topo_tab.to_csv(GEN / "e12_topology_table.csv", index=False)
with open(GEN / "e12_topology_table.tex", "w", encoding="utf-8") as fh:
    for r in topo_tab.itertuples():
        sub = all_paired[all_paired["topology"] == r.topology]
        fh.write(f"{TL[r.topology]} & {f3(r.ours)} & {f3(r.best_val)} ({r.best}) & {pct(r.lead)} & "
                 f"{pct(r.ours_exp_max)} & {pct(r.base_exp_min)}--{pct(r.base_exp_max)} & "
                 f"{int(sub['dominates_on_means'].sum())}/{len(sub)} & {int(sub['goodput_better'].sum())}/{len(sub)} & "
                 f"{int(sub['exposure_lower'].sum())}/{len(sub)} \\\\\n")
for r in topo_tab.itertuples():
    k = r.topology.capitalize().replace("50", "Fifty")
    mac(f"Lead{k}", pct(r.lead))
    mac(f"Ours{k}", f3(r.ours))
    mac(f"Best{k}", f3(r.best_val))
    mac(f"OursExpMax{k}", pct(r.ours_exp_max))

# pooled over the four topologies (24 cells)
m_all = cell.groupby("method")["secure_goodput"].mean()
mac("GoodputOursAll", f3(m_all[OURS]))
mac("GoodputBestBaseAll", f3(m_all[BASES].max()))
mac("BestBaseAll", m_all[BASES].idxmax())
mac("LeadAll", pct(m_all[OURS] / m_all[BASES].max() - 1))
mac("ExpOursMaxAll", pct(cell[cell.method == OURS]["exposure"].max()))
mac("ExpBaseMinAll", pct(cell[cell.method.isin(BASES)]["exposure"].min()))
mac("ExpBaseMaxAll", pct(cell[cell.method.isin(BASES)]["exposure"].max()))
eff = cell.groupby("method")["key_efficiency"].mean()
mac("EffOursAll", f2(eff[OURS]))
mac("EffBaseRange", f"{f2(eff[BASES].min())}--{f2(eff[BASES].max())}")
amp = cell.groupby("method")["amplification"].mean()
mac("AmpOursAll", f2(amp[OURS]))

# weakest cell for the proposed method (smallest lead over the best baseline)
leads = []
for (t, p, r), c in cell.groupby(["topology", "placement", "rho_A"]):
    c = c.set_index("method")
    leads.append((c.loc[OURS, "secure_goodput"] - c.loc[BASES, "secure_goodput"].max(), t, p, r,
                  c.loc[OURS, "secure_goodput"], c.loc[BASES, "secure_goodput"].max(),
                  c.loc[OURS, "exposure"], c.loc[c.loc[BASES, "secure_goodput"].idxmax(), "exposure"]))
weak = min(leads)
mac("WeakTopo", TL[weak[1]])
mac("WeakPlacement", weak[2])
mac("WeakRho", f"{weak[3]:.1f}")
mac("WeakOurs", f3(weak[4]))
mac("WeakBest", f3(weak[5]))
mac("WeakOursExp", pct(weak[6]))
mac("WeakBestExp", pct(weak[7]))

# eps sweep robustness on the tuning/test protocol (NSFNET+Abilene)
test = pd.read_csv(RES / "E12" / "e12_test.csv")
eps_rows = []
for m in sorted([x for x in test.method.unique() if x.startswith("Adaptive@")], key=lambda x: float(x.split("@")[1])):
    n_dom = 0
    for (t, p, r), c in test.groupby(["topology", "placement", "rho_A"]):
        c = c.set_index("method")
        for b in BASES:
            n_dom += int(c.loc[m, "secure_goodput_mean"] >= c.loc[b, "secure_goodput_mean"]
                         and c.loc[m, "exposure_mean"] <= c.loc[b, "exposure_mean"])
    eps_rows.append((float(m.split("@")[1]), n_dom))
eps_df = pd.DataFrame(eps_rows, columns=["eps", "dominates"])
eps_df.to_csv(GEN / "e12_eps_dominance.csv", index=False)
mid = eps_df[(eps_df.eps >= 0.05) & (eps_df.eps <= 0.9)]
mac("EpsDomMin", int(mid.dominates.min()))
mac("EpsDomMax", int(mid.dominates.max()))
mac("EpsDomLowest", int(eps_df[eps_df.eps == eps_df.eps.min()].dominates.iloc[0]))
mac("EpsLowest", f"{eps_df.eps.min():g}")
mac("EpsDomN", 60)

# Figure: secure goodput and exposure vs attacker strength, 4 topologies x 2 placements
fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), sharex=True)
for j, t in enumerate(TOPOS):
    for i, plc in enumerate(["random", "betweenness"]):
        ax = axes[i, j]
        for m in BASES + [OURS]:
            s = per_seed[(per_seed.topology == t) & (per_seed.placement == plc) & (per_seed.method == m)]
            g = s.groupby("rho_A")["secure_goodput"]
            xs = sorted(s["rho_A"].unique())
            st = [ci95(g.get_group(x).values) for x in xs]
            mean = [a[0] for a in st]
            lw, ms = (2.4, 6.5) if m == OURS else (1.4, 4.5)
            if m == OURS:
                fs.band(ax, xs, [a[1] for a in st], [a[2] for a in st], COL[m], alpha=0.18)
            fs.line(ax, xs, mean, COL[m], label=LAB[m], lw=lw, ms=ms, zorder=4 if m == OURS else 3)
        ax.set_xticks([0.1, 0.2, 0.3])
        ax.set_ylim(0, 0.95)
        if i == 0:
            ax.set_title(TL[t])
        if j == 0:
            ax.set_ylabel(f"Secure goodput\n({plc} attacker)")
        if i == 1:
            ax.set_xlabel(r"Fraction of compromised relays $\rho_A$")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig_e12_goodput")

# Figure: dominance scatter, all comparisons
fig, ax = plt.subplots(figsize=(6.2, 4.2))
for t in TOPOS:
    sub = all_paired[all_paired.topology == t]
    ax.scatter(-sub["exposure_diff"] * 100, sub["secure_goodput_diff"] * 100, s=34,
               color=fs.TOPO_COLOR[t], edgecolor=WHITE, lw=0.8, label=TL[t], zorder=3)
ax.axhline(0, color=fs.BASELINE, lw=1)
ax.axvline(0, color=fs.BASELINE, lw=1)
ax.set_xlabel("Exposure reduction vs. baseline (percentage points)")
ax.set_ylabel("Secure-goodput gain vs. baseline\n(percentage points)")
ax.grid(axis="x", color=fs.GRID, lw=0.8)
ax.legend(loc="upper left", ncol=2)
fig.tight_layout()
save(fig, "fig_dominance")
mac("GainMaxPP", f"{100 * all_paired['secure_goodput_diff'].max():.1f}")
mac("GainMinPP", f"{100 * all_paired['secure_goodput_diff'].min():.1f}")

# detector-aware baselines (B0R/B1R/B2R + PA)
RB = ["B0R+PA", "B1R+PA", "B2R+PA"]
rk_pc = pd.read_csv(ROB / "paired_riskaware.csv")
rk_m = pd.read_csv(ROB / "means_riskaware.csv")
mac("RkN", len(rk_pc))
mac("RkDom", int(rk_pc.dominates_on_means.sum()))
mac("RkGood", int(rk_pc.goodput_better.sum()))
mac("RkWorse", int(rk_pc.goodput_worse.sum()))
mac("RkExpLow", int(rk_pc.exposure_lower.sum()))
mac("RkExpHigh", int(rk_pc.exposure_higher.sum()))
mr = rk_pc[rk_pc.baseline.isin(["B1R+PA", "B2R+PA"])]
mac("RkMultiN", len(mr))
mac("RkMultiDom", int(mr.dominates_on_means.sum()))
mac("RkMultiWorse", int((mr.goodput_worse | mr.exposure_higher).sum()))
b0r = rk_pc[rk_pc.baseline == "B0R+PA"]
mac("RkSingleN", len(b0r))
mac("RkSingleDom", int(b0r.dominates_on_means.sum()))
mac("RkSingleWorse", int(b0r.goodput_worse.sum()))
mac("RkSingleExpLow", int(b0r.exposure_lower.sum()))
tm = rk_m.groupby(["topology", "method"])[["secure_goodput", "exposure", "blocking"]].mean()
with open(GEN / "riskaware_table.tex", "w", encoding="utf-8") as fh:
    for t in TOPOS:
        sub = rk_pc[rk_pc.topology == t]
        vals = [f3(tm.loc[(t, m), "secure_goodput"]) for m in [OURS] + RB]
        exps = [pct(tm.loc[(t, m), "exposure"]) for m in [OURS] + RB]
        fh.write(f"{TL[t]} & " + " & ".join(vals) + " & " + " & ".join(exps) +
                 f" & {int(sub.goodput_better.sum())}/{int(sub.goodput_worse.sum())}/{len(sub)} \\\\\n")
for t in TOPOS:
    k = t.capitalize().replace("50", "Fifty")
    mac(f"RkOurs{k}", f3(tm.loc[(t, OURS), "secure_goodput"]))
    mac(f"RkSingle{k}", f3(tm.loc[(t, "B0R+PA"), "secure_goodput"]))
    mac(f"RkOursExp{k}", pct(tm.loc[(t, OURS), "exposure"]))
    mac(f"RkSingleExp{k}", pct(tm.loc[(t, "B0R+PA"), "exposure"]))
gm = rk_m[(rk_m.topology == "geant")]
ob = gm[gm.method == OURS].blocking
mac("RkGeantBlockRange", f"{pct(ob.min(), 0)}--{pct(ob.max(), 0)}")
sb = gm[gm.method == "B0R+PA"].blocking
mac("RkGeantSingleBlockRange", f"{pct(sb.min(), 0)}--{pct(sb.max(), 0)}")
rkcell = gm[(gm.placement == "betweenness") & (gm.rho_A == 0.1)].set_index("method")
mac("RkCellOurs", f3(rkcell.loc[OURS, "secure_goodput"]))
mac("RkCellSingle", f3(rkcell.loc["B0R+PA", "secure_goodput"]))
mac("RkCellOursExp", pct(rkcell.loc[OURS, "exposure"]))
mac("RkCellSingleExp", pct(rkcell.loc["B0R+PA", "exposure"]))
mac("RkExpRatioGeant", f"{tm.loc[('geant', 'B0R+PA'), 'exposure'] / tm.loc[('geant', OURS), 'exposure']:.0f}")

# steady state: warm-up of two pool-fill times, then four pool-fill times measured
for tag, key in [("steady", "Steady"), ("steady_large", "SteadyLarge")]:
    f = ROB / f"paired_{tag}.csv"
    if not f.exists():
        continue
    sp = pd.read_csv(f)
    sm = pd.read_csv(ROB / f"means_{tag}.csv").groupby(["topology", "method"])[["secure_goodput", "exposure"]].mean()
    mac(f"{key}N", len(sp))
    mac(f"{key}Dom", int(sp.dominates_on_means.sum()))
    mac(f"{key}Good", int(sp.goodput_better.sum()))
    mac(f"{key}ExpLow", int(sp.exposure_lower.sum()))
    mac(f"{key}Worse", int((sp.goodput_worse | sp.exposure_higher).sum()))
    tops = sorted(set(sp.topology), key=TOPOS.index)
    ours = [sm.loc[(t, OURS), "secure_goodput"] for t in tops]
    best = [sm.loc[t].drop(OURS).secure_goodput.max() for t in tops]
    mac(f"{key}OursRange", f"{f2(min(ours))}--{f2(max(ours))}")
    mac(f"{key}BestRange", f"{f2(min(best))}--{f2(max(best))}")
    mac(f"{key}ExpOursMax", pct(max(sm.loc[(t, OURS), "exposure"] for t in tops)))

# predicted vs realised exposure of the decisions actually taken (E12 grid, NSFNET + Abilene)
cal = pd.read_csv(RES / "calibration" / "per_request.csv")
c5 = cal[cal.eps_target == 0.5]
mac("CalMedianPred", pct(c5.p_exp_pred.median()))
mac("CalShareBelowFive", pct((c5.p_exp_pred < 0.05).mean(), 0))
mac("CalShareAboveForty", pct((c5.p_exp_pred > 0.4).mean()))
cs = pd.read_csv(RES / "calibration" / "summary.csv")
cs5 = cs[cs.eps_target == 0.5]
mac("CalPredRange", f"{pct(cs5.predicted.min())}--{pct(cs5.predicted.max())}")
mac("CalObsRange", f"{pct(cs5.observed.min())}--{pct(cs5.observed.max())}")
mid = c5[(c5.p_exp_pred > 0.1) & (c5.p_exp_pred <= 0.2)]
mac("CalMidPred", pct(mid.p_exp_pred.mean(), 0))
mac("CalMidObs", pct(mid.exposed.astype(float).mean(), 0))
mac("CalMidShare", pct(len(mid) / len(c5)))

# how often the exposure computation fell back to Monte Carlo (E12 grid, all topologies)
mcf = pd.read_csv(RES / "E12" / "mc_fallback_rate.csv").groupby("topology")[["exact_subsets", "mc_subsets"]].sum()
mcf["share"] = mcf.mc_subsets / (mcf.exact_subsets + mcf.mc_subsets)
mac("McShareGermany", pct(mcf.loc["germany50", "share"]))
mac("McOtherZero", "yes" if (mcf.drop("germany50").mc_subsets == 0).all() else "no")

# ---------------------------------------------------------------------------
# E4: secure goodput vs offered load (proposed at eps=0.5 vs pool-aware baselines)
# ---------------------------------------------------------------------------
e4 = {t: pd.read_csv(RES / "E4" / ("e4_cost_blocking_vs_load.csv" if t == "nsfnet" else
                                    f"e4_cost_blocking_vs_load_{t}.csv")) for t in TOPOS}
fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
e4_rows = []
for ax, t in zip(axes, TOPOS):
    d = e4[t]
    for m in BASES + [OURS]:
        s = d[d.method == m].sort_values("rho")
        if s.empty:
            continue
        lw, ms = (2.4, 6.5) if m == OURS else (1.4, 4.5)
        if m == OURS:
            fs.band(ax, s.rho, s.secure_goodput_ci_low, s.secure_goodput_ci_high, COL[m], alpha=0.18)
        fs.line(ax, s.rho, s.secure_goodput_mean, COL[m], label=LAB[m], lw=lw, ms=ms)
    piv = d[d.method.isin(BASES + [OURS])].pivot(index="rho", columns="method", values="secure_goodput_mean")
    for rho, r in piv.iterrows():
        e4_rows.append(dict(topology=t, rho=rho, ours=r[OURS], best=r[BASES].max(), best_m=r[BASES].idxmax(),
                            n_seeds=int(d.n_seeds.iloc[0])))
    ax.set_title(TL[t])
    ax.set_xlabel(r"Normalised load $\rho$")
axes[0].set_ylabel("Secure goodput")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig_load")
e4t = pd.DataFrame(e4_rows)
e4t.to_csv(GEN / "e4_table.csv", index=False)
for t in TOPOS:
    s = e4t[e4t.topology == t]
    k = t.capitalize().replace("50", "Fifty")
    mac(f"LoadOurs{k}", f"{f2(s.ours.min())}--{f2(s.ours.max())}")
    mac(f"LoadBest{k}", f"{f2(s.best.min())}--{f2(s.best.max())}")
    mac(f"LoadWins{k}", int((s.ours >= s.best).sum()))
    mac(f"LoadPts{k}", len(s))
    mac(f"LoadSeeds{k}", int(s.n_seeds.iloc[0]))
hi = e4t[(e4t.topology == "germany50") & (e4t.rho == e4t.rho.max())].iloc[0]
mac("LoadGermanyHighOurs", f3(hi.ours))
mac("LoadGermanyHighBest", f3(hi.best))
mac("LoadRhoMax", f"{e4t.rho.max():.1f}")

# ---------------------------------------------------------------------------
# E3: routing-aware attacker (Level A), exposure vs rho_A
# ---------------------------------------------------------------------------
e3c = {"nsfnet": "e3_security_vs_attacker.csv", "germany50": "e3_security_vs_attacker_germany50.csv",
       "abilene": "e3_security_vs_attacker_abilene.csv", "geant": "e3_security_vs_attacker_geant.csv"}
E3COL = {"Adaptive": COL[OURS], "B0": COL["B0"], "B1": COL["B1+PA"], "B2": COL["B2+PA"], "B3": COL["B3+PA"]}
E3LAB = {"Adaptive": "Proposed", "B0": "B0", "B1": "B1", "B2": "B2", "B3": "B3"}
fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True)
for j, t in enumerate(TOPOS):
    d = pd.read_csv(RES / "E3" / e3c[t])
    d = d[(d.placement == "adaptive") & (d.method != "Adaptive")]
    ours = pd.read_csv(RES / "E3" / e3c[t].replace(".csv", "_eps05.csv")).assign(method="Adaptive")
    d = pd.concat([d, ours], ignore_index=True)
    d["secure"] = d.service_ratio_mean * (1 - d.exposure_mean)
    for i, (col, lab) in enumerate([("exposure_mean", "Key-exposure rate"),
                                    ("secure", "Requests delivered securely")]):
        ax = axes[i, j]
        for m in ["B0", "B1", "B2", "B3", "Adaptive"]:
            sm = d[d.method == m].sort_values("rho_A")
            lw, ms = (2.4, 6.5) if m == "Adaptive" else (1.4, 4.5)
            if m == "Adaptive" and col == "exposure_mean":
                fs.band(ax, sm.rho_A, sm.exposure_ci_low, sm.exposure_ci_high, E3COL[m], alpha=0.18)
            fs.line(ax, sm.rho_A, sm[col], E3COL[m], label=E3LAB[m], lw=lw, ms=ms)
        ax.set_ylim(-0.03, 1.03)
        if i == 0:
            ax.set_title(TL[t])
        else:
            ax.set_xlabel(r"Fraction of compromised relays $\rho_A$")
        if j == 0:
            ax.set_ylabel(lab)
    k = t.capitalize().replace("50", "Fifty")
    top = d[d.rho_A == d.rho_A.max()].set_index("method")
    mac(f"SmartOurs{k}", pct(top.loc["Adaptive", "exposure_mean"]))
    mac(f"SmartBase{k}", f"{pct(top.loc[['B0','B1','B2','B3'], 'exposure_mean'].min(), 0)}--"
                         f"{pct(top.loc[['B0','B1','B2','B3'], 'exposure_mean'].max(), 0)}")
    mac(f"SmartSecOurs{k}", f2(top.loc["Adaptive", "secure"]))
    mac(f"SmartSecBase{k}", f"{f2(top.loc[['B0','B1','B2','B3'], 'secure'].min())}--"
                            f"{f2(top.loc[['B0','B1','B2','B3'], 'secure'].max())}")
    mac(f"SmartServed{k}", pct(top.loc["Adaptive", "service_ratio_mean"], 0))
    mac(f"SmartSeeds{k}", int(d.n_seeds.iloc[0]))
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig_smart_attacker")

# ---------------------------------------------------------------------------
# E6: silent failures at the strongest drop/tamper setting
# ---------------------------------------------------------------------------
sf_rows = []
for t in TOPOS:
    name = "e6_drop_tamper.csv" if t == "nsfnet" else f"e6_drop_tamper_{t}.csv"
    d = pd.read_csv(RES / "E6" / name)
    d = pd.concat([d[d.method != "Adaptive"],
                   pd.read_csv(RES / "E6" / name.replace(".csv", "_eps05.csv"))], ignore_index=True)
    top = d[d.p_drop == d.p_drop.max()].set_index("method")["silent_failure_rate_mean"]
    k = t.capitalize().replace("50", "Fifty")
    mac(f"SilentOurs{k}", pct(top["Adaptive"]))
    mac(f"SilentBase{k}", f"{pct(top[['B0','B1','B2','B3']].min(), 0)}--{pct(top[['B0','B1','B2','B3']].max(), 0)}")
    sf_rows.append((t, top.to_dict()))
mac("SilentPMax", f"{d.p_drop.max():.1f}")
mac("SilentOursMax", pct(max(r[1]["Adaptive"] for r in sf_rows)))
mac("SilentBaseAll", f"{pct(min(min(r[1][m] for m in ['B0','B1','B2','B3']) for r in sf_rows), 0)}--"
                     f"{pct(max(max(r[1][m] for m in ['B0','B1','B2','B3']) for r in sf_rows), 0)}")

# ---------------------------------------------------------------------------
# Robustness: miscalibration, misranking, pool size (NSFNET + Abilene, 60 comparisons each)
# ---------------------------------------------------------------------------
def rob_summary(means_file: str, paired_file: str, key: list[str]) -> pd.DataFrame:
    mm = pd.read_csv(ROB / means_file)
    pp = pd.read_csv(ROB / paired_file)
    mm = mm[mm.method.isin([OURS] + BASES)]
    out = []
    for kv, g in mm.groupby(key):
        kv = kv if isinstance(kv, tuple) else (kv,)
        sel = np.ones(len(pp), bool)
        for k_, v_ in zip(key, kv):
            sel &= pp[k_] == v_
        p = pp[sel]
        mean = g.groupby("method")["secure_goodput"].mean()
        expo = g.groupby("method")["exposure"].mean()
        out.append(dict(zip(key, kv), ours=mean[OURS], best=mean[BASES].max(), exp_ours=expo[OURS],
                        exp_ours_max=g[g.method == OURS]["exposure"].max(),
                        n=len(p), dom=int(p.dominates_on_means.sum()), good=int(p.goodput_better.sum()),
                        expl=int(p.exposure_lower.sum()), worse=int((p.goodput_worse | p.exposure_higher).sum())))
    return pd.DataFrame(out)


miscal = rob_summary("means_miscal.csv", "paired_miscal.csv", ["odds_factor", "temperature"])
misrank_t = rob_summary("means_misrank.csv", "paired_misrank.csv", ["misrank_fraction"])
pool_t = rob_summary("means_pool.csv", "paired_pool.csv", ["capacity_seconds"])
miscal.to_csv(GEN / "rob_miscal.csv", index=False)
misrank_t.to_csv(GEN / "rob_misrank.csv", index=False)
pool_t.to_csv(GEN / "rob_pool.csv", index=False)

mac("MiscalGoodRange", f"{miscal.good.min()}--{miscal.good.max()}")
mac("MiscalDomMin", int(miscal.dom.min()))
mac("MiscalExpLowMin", int(miscal.expl.min()))
mac("MiscalWorseMax", int(miscal.worse.max()))
mac("MiscalN", int(miscal.n.iloc[0]))
mac("MiscalOursRange", f"{f3(miscal.ours.min())}--{f3(miscal.ours.max())}")
mac("MiscalExpOursMax", pct(miscal.exp_ours_max.max()))
mac("MiscalExpOursMeanMax", pct(miscal.exp_ours.max()))
for r in misrank_t.itertuples():
    k = {0.0: "Zero", 0.1: "Ten", 0.2: "Twenty", 0.4: "Forty"}[r.misrank_fraction]
    mac(f"Misrank{k}Good", r.good)
    mac(f"Misrank{k}Dom", r.dom)
    mac(f"Misrank{k}Exp", r.expl)
    mac(f"Misrank{k}Worse", r.worse)
    mac(f"Misrank{k}Ours", f3(r.ours))
    mac(f"Misrank{k}Best", f3(r.best))
mac("MisrankN", int(misrank_t.n.iloc[0]))
for r in pool_t.itertuples():
    k = {0.1: "A", 0.5: "B", 1.0: "C", 3.0: "D"}[r.capacity_seconds]
    mac(f"Pool{k}Ours", f3(r.ours))
    mac(f"Pool{k}Best", f3(r.best))
    mac(f"Pool{k}Good", r.good)
mac("PoolWorseMax", int(pool_t.worse.max()))
mac("PoolExpMin", int(pool_t.expl.min()))
mac("PoolN", int(pool_t.n.iloc[0]))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
# (a) miscalibration
ax = axes[0]
ml = []
for r in miscal.itertuples():
    if r.temperature == 1.0:
        ml.append((r.odds_factor, f"risk ×{r.odds_factor:g}" if r.odds_factor != 1 else "calibrated", r.ours, r.best))
    else:
        ml.append((10 + r.temperature, "under-conf." if r.temperature < 1 else "over-conf.", r.ours, r.best))
ref = cell[cell.topology.isin(["nsfnet", "abilene"])].groupby("method")["secure_goodput"].mean()
ml.append((1.0, "calibrated", ref[OURS], ref[BASES].max()))
ml.sort()
x = np.arange(len(ml))
fs.bars(ax, x - 0.2, [a[2] for a in ml], 0.38, COL[OURS], label="Proposed")
fs.bars(ax, x + 0.2, [a[3] for a in ml], 0.38, fs.NEUTRAL, label="Best baseline")
ax.set_xticks(x)
ax.set_xticklabels([a[1] for a in ml], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Mean secure goodput")
ax.set_ylim(0, 0.7)
ax.set_title("(a) Detector miscalibration")
ax.legend(loc="upper right", ncol=2)
# (b) misranking
ax = axes[1]
x = np.arange(len(misrank_t))
fs.bars(ax, x - 0.2, misrank_t.ours, 0.38, COL[OURS])
fs.bars(ax, x + 0.2, misrank_t.best, 0.38, fs.NEUTRAL)
ax.set_xticks(x)
ax.set_xticklabels([f"{v:.0%}" for v in misrank_t.misrank_fraction])
ax.set_xlabel("Compromised nodes swapped with honest ones")
ax.set_ylim(0, 0.7)
ax.set_title("(b) Detector misranking")
# (c) pool size
ax = axes[2]
x = np.arange(len(pool_t))
fs.bars(ax, x - 0.2, pool_t.ours, 0.38, COL[OURS])
fs.bars(ax, x + 0.2, pool_t.best, 0.38, fs.NEUTRAL)
ax.set_xticks(x)
ax.set_xticklabels([f"{v:g} s" for v in pool_t.capacity_seconds])
ax.set_xlabel(r"Pool capacity $C_e/R_e$")
ax.set_ylim(0, 0.8)
ax.set_title("(c) Key-pool size")
fig.tight_layout()
fs.round_tops(fig)
save(fig, "fig_robustness")

# ---------------------------------------------------------------------------
# Optimality, decision time, RL reference
# ---------------------------------------------------------------------------
rl_ns = pd.read_csv(RES / "E9" / "rl_diagnosis_summary_nsfnet.csv").set_index("method")
rl_r50 = pd.read_csv(RES / "E9" / "rl_diagnosis_summary_random50.csv").set_index("method")
mac("IlpServed", f"{rl_ns.loc['BatchILP', 'served_mean']:.1f}")
mac("OursServed", f"{rl_ns.loc['Adaptive', 'served_mean']:.1f}")
mac("OptGap", pct(rl_ns.loc["Adaptive", "gap_to_ilp_mean"]))
mac("OptGapLo", pct(rl_ns.loc["Adaptive", "gap_ci_low"]))
mac("OptGapHi", pct(rl_ns.loc["Adaptive", "gap_ci_high"]))
mac("OptGapRfifty", pct(rl_r50.loc["Adaptive", "gap_to_ilp_mean"]))
mac("OptBatches", int(rl_ns.loc["Adaptive", "n_batches"]))
e9 = pd.read_csv(RES / "E9" / "e9_optimality_scalability.csv").set_index(["topology", "method"])
for t in ["nsfnet", "random50"]:
    k = "Ns" if t == "nsfnet" else "Rfifty"
    mac(f"TimeOurs{k}", f"{1000 * e9.loc[(t, 'Adaptive'), 'mean_decision_time_s']:.1f}")
    mac(f"TimeExh{k}", f"{e9.loc[(t, 'Exhaustive'), 'mean_decision_time_s']:.1f}")
    mac(f"TimeIlp{k}", f"{e9.loc[(t, 'BatchILP'), 'mean_decision_time_s']:.1f}")
dt = pd.read_csv(RES / "decision_time" / "summary_after.csv").set_index("topology")
with open(GEN / "decision_time_table.tex", "w", encoding="utf-8") as fh:
    for t in TOPOS:
        r = dt.loc[t]
        fh.write(f"{TL[t]} & {int(r.n)} & {r['median']:.1f} & {r.p95:.1f} & {r['max']:.1f} \\\\\n")
mac("DtGermanyMedian", f"{dt.loc['germany50', 'median']:.1f}")
mac("DtGermanyPninetyfive", f"{dt.loc['germany50', 'p95']:.1f}")
mac("DtGermanyMax", f"{dt.loc['germany50', 'max']:.0f}")
ms = pd.read_csv(RES / "E9" / "rl_multiseed_summary_nsfnet.csv").set_index("method")
mac("RlGapMin", pct(ms.loc["RL full (+shaping)", "gap_min"]))
mac("RlGapMax", pct(ms.loc["RL full (+shaping)", "gap_max"]))
mac("RlLegacyGapMin", pct(ms.loc["RL legacy (Phase 5)", "gap_min"], 0))
mac("RlLegacyGapMax", pct(ms.loc["RL legacy (Phase 5)", "gap_max"], 0))

# latency-charged check (decision time added to delivery latency)
lat = pd.read_csv(RES / "decision_time" / "latency_paired.csv")
for L, g in lat.groupby("latency"):
    k = {"p95": "Pninetyfive"}.get(str(L), "Fixed")
    mac(f"Lat{k}Dom", int(g.dominates_on_means.sum()))
    mac(f"Lat{k}Worse", int((g.goodput_worse | g.exposure_higher).sum()))
    mac(f"Lat{k}N", len(g))
mac("LatLabels", ", ".join(map(str, sorted(lat.latency.unique()))))

# ---------------------------------------------------------------------------
# Ablation (E8, NSFNET)
# ---------------------------------------------------------------------------
e8 = pd.read_csv(RES / "E8" / "e8_ablation_eps05.csv").set_index("variant")
for v, k in [("full", "Full"), ("no_pool_awareness", "NoPool"), ("no_pool_awareness+PA", "NoPoolPA"),
             ("no_suspicion", "NoSusp")]:
    mac(f"Abl{k}Good", f3(e8.loc[v, "secure_goodput_mean"]))
    mac(f"Abl{k}Exp", pct(e8.loc[v, "exposure_mean"]))
    mac(f"Abl{k}Fail", pct(e8.loc[v, "failed_mean"]))

# ---------------------------------------------------------------------------
# QKDNetSim cross-check
# ---------------------------------------------------------------------------
ns = pd.read_csv(NS3 / "e12_summary.csv")
for eng, k in [("fluid", "Fluid"), ("ns3", "Pools"), ("ns3relay", "Relay")]:
    g = ns[ns.engine == eng]
    mac(f"Ns{k}Dom", int(g.dominates_on_means.sum()))
    mac(f"Ns{k}Good", int(g.goodput_better.sum()))
    mac(f"Ns{k}Exp", int(g.exposure_lower.sum()))
    mac(f"Ns{k}Worse", int((g.goodput_worse + g.exposure_higher).sum()))
    mac(f"Ns{k}N", int(g.n_comparisons.sum()))
ag = pd.read_csv(NS3 / "e12_engine_agreement.csv")
r_p = np.corrcoef(ag.secure_goodput_fluid, ag.secure_goodput_ns3)[0, 1]
r_r = np.corrcoef(ag.secure_goodput_fluid, ag.secure_goodput_ns3relay)[0, 1]
mac("NsCorrPools", f3(r_p))
mac("NsCorrRelay", f3(r_r))
mac("NsMadPools", pct((ag.secure_goodput_ns3 - ag.secure_goodput_fluid).abs().mean()))
mac("NsMadRelay", pct((ag.secure_goodput_ns3relay - ag.secure_goodput_fluid).abs().mean()))
mac("NsMaxRelay", pct((ag.secure_goodput_ns3relay - ag.secure_goodput_fluid).abs().max()))
with open(GEN / "qkdnetsim_table.tex", "w", encoding="utf-8") as fh:
    for t in TOPOS:
        cells = []
        for eng in ["fluid", "ns3", "ns3relay"]:
            g = ns[(ns.engine == eng) & (ns.topology == t)].iloc[0]
            cells.append(f"{g.dominates_on_means}/{g.n_comparisons} & {g.goodput_better} & {g.exposure_lower}")
        fh.write(f"{TL[t]} & " + " & ".join(cells) + " \\\\\n")

fig, ax = plt.subplots(figsize=(5.2, 4.2))
for m in BASES + [OURS]:
    s = ag[ag.method == m]
    ax.scatter(s.secure_goodput_fluid, s.secure_goodput_ns3relay, s=30 if m == OURS else 18,
               color=COL[m], edgecolor=WHITE, lw=0.6, label=LAB[m], zorder=4 if m == OURS else 3)
lim = [min(ag.secure_goodput_fluid.min(), ag.secure_goodput_ns3relay.min()) - 0.02,
       max(ag.secure_goodput_fluid.max(), ag.secure_goodput_ns3relay.max()) + 0.02]
ax.plot(lim, lim, color=fs.BASELINE, lw=1, zorder=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Secure goodput, SimPy model")
ax.set_ylabel("Secure goodput, QKDNetSim\n(pools + share relay)")
ax.grid(axis="x", color=fs.GRID, lw=0.8)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
save(fig, "fig_qkdnetsim")

# ---------------------------------------------------------------------------
with open(GEN / "numbers.tex", "w", encoding="utf-8") as fh:
    fh.write("% Generated by scripts/paper_figures.py -- do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(macros)} macros; figures in {FIG}")
for k, v in sorted(macros.items()):
    print(f"{k} = {v}")
