"""Tables for the co-simulation results.

For every experiment: per engine, paired tests of our method against every
baseline in every condition (paired t-test and Wilcoxon over seeds, Holm-
adjusted within the engine), the dominance counts, and the fluid-vs-QKDNetSim
agreement of every (condition, method) mean. Q1 also gets the per-class table.

Usage: PYTHONPATH=. uv run python qkdnetsim-prototype/cosim/analyze.py e12|q2|e4|e5|q1|strategic [--seeds N]
Outputs: qkdnetsim-prototype/results/<exp>_*.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cosim import OUT, STEPS  # noqa: E402

from experiments.e12_robustness import holm  # noqa: E402

KEYS = {"e12": ["topology", "placement", "rho_A"], "q2": ["topology", "rho", "view", "interval_s"],
        "e4": ["topology", "rho"], "e5": ["topology", "load_label"], "q1": ["topology", "placement", "rho"],
        "strategic": ["topology", "detector_lag_s"], "blocks": ["topology", "block_bits", "placement", "rho_A"]}
OURS = {"q1": "QoS-Adaptive"}
BASELINES = {"q1": ["B0+PA+QoS", "B1+PA+QoS", "B2+PA+QoS"]}


def load(exp: str, n_seeds: int) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in STEPS.glob(f"{exp}_*.csv")], ignore_index=True)
    if "topology" not in df:
        df["topology"] = "nsfnet"
    df["topology"] = df["topology"].fillna("nsfnet")
    return df[df["seed"] % 1000 < n_seeds]


def paired(df: pd.DataFrame, keys: list[str], ours: str, baselines: list[str]) -> pd.DataFrame:
    rows = []
    for (engine, *cond), g in df.groupby(["engine"] + keys):
        a = g[g["method"] == ours].set_index("seed")
        for b_name in baselines:
            b = g[g["method"] == b_name].set_index("seed")
            common = a.index.intersection(b.index)
            row = {"engine": engine, **dict(zip(keys, cond)), "baseline": b_name, "n": len(common)}
            for metric in ("secure_goodput", "exposure"):
                d = (a.loc[common, metric] - b.loc[common, metric]).dropna()
                row[f"{metric}_diff"] = float(d.mean()) if len(d) else float("nan")
                ok = len(d) >= 2 and not (d == 0).all()
                row[f"{metric}_p"] = float(stats.ttest_1samp(d, 0.0).pvalue) if ok else 1.0
                row[f"{metric}_p_wilcoxon"] = float(stats.wilcoxon(d, zero_method="zsplit").pvalue) if ok else 1.0
            rows.append(row)
    out = pd.DataFrame(rows)
    for metric in ("secure_goodput", "exposure"):
        out[f"{metric}_p_holm"] = 1.0
        for engine, idx in out.groupby("engine").groups.items():
            out.loc[idx, f"{metric}_p_holm"] = holm(out.loc[idx, f"{metric}_p"].tolist())
    out["goodput_better"] = (out["secure_goodput_diff"] > 0) & (out["secure_goodput_p_holm"] < 0.05)
    out["goodput_worse"] = (out["secure_goodput_diff"] < 0) & (out["secure_goodput_p_holm"] < 0.05)
    out["exposure_lower"] = (out["exposure_diff"] < 0) & (out["exposure_p_holm"] < 0.05)
    out["exposure_higher"] = (out["exposure_diff"] > 0) & (out["exposure_p_holm"] < 0.05)
    out["dominates_on_means"] = (out["secure_goodput_diff"] >= 0) & (out["exposure_diff"] <= 0)
    return out


def summarize(pc: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cols = ["dominates_on_means", "goodput_better", "goodput_worse", "exposure_lower", "exposure_higher"]
    s = pc.groupby(by)[cols].sum()
    s["n_comparisons"] = pc.groupby(by).size()
    return s.reset_index()


def agreement(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    means = df.groupby(["engine"] + keys + ["method"])[["secure_goodput", "exposure"]].mean().reset_index()
    wide = means.pivot_table(index=keys + ["method"], columns="engine", values=["secure_goodput", "exposure"])
    wide.columns = [f"{m}_{e}" for m, e in wide.columns]
    wide = wide.reset_index()
    if "secure_goodput_ns3" in wide:
        wide["goodput_diff"] = wide["secure_goodput_ns3"] - wide["secure_goodput_fluid"]
    return wide


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=list(KEYS))
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    exp = args.exp
    df = load(exp, args.seeds)
    keys = KEYS[exp]
    ours = OURS.get(exp, "Adaptive@0.5")
    baselines = BASELINES.get(exp) or sorted(m for m in df["method"].unique() if not m.startswith("Adaptive"))
    print(f"{exp}: {len(df)} runs, {df['seed'].nunique()} seeds, topologies {sorted(df['topology'].unique())}")

    pc = paired(df, keys, ours, baselines)
    pc.to_csv(OUT / f"{exp}_paired.csv", index=False)
    by = ["engine", "topology"]
    summary = summarize(pc, by)
    summary.to_csv(OUT / f"{exp}_summary.csv", index=False)
    print(summary.to_string(index=False))

    wide = agreement(df, keys)
    wide.to_csv(OUT / f"{exp}_engine_agreement.csv", index=False)
    for topo, w in wide.groupby("topology"):
        for engine in ("ns3", "ns3relay"):
            col = f"secure_goodput_{engine}"
            if col not in w or w[col].isna().all():
                continue
            d = (w[col] - w["secure_goodput_fluid"]).abs()
            corr = w[[col, "secure_goodput_fluid"]].corr().iloc[0, 1]
            print(f"{topo}: |{engine} - fluid| goodput mean {d.mean():.4f}, max {d.max():.4f}, corr {corr:.4f}")

    means = df.groupby(["engine"] + keys + ["method"])[["secure_goodput", "exposure"]].mean().reset_index()
    means.to_csv(OUT / f"{exp}_means.csv", index=False)
    for (engine, topo), m in means.groupby(["engine", "topology"]):
        cond = [k for k in keys if k != "topology"]
        o = m[m["method"] == ours].set_index(cond)["secure_goodput"]
        b = m[m["method"].isin(baselines)].groupby(cond)["secure_goodput"].max()
        print(f"{engine} {topo}: ours {o.mean():.3f} vs best baseline per condition {b.mean():.3f} "
              f"({(o.mean() / b.mean() - 1) * 100:+.1f}%)")

    if exp == "q1":
        cls = [c for c in df.columns if "__" in c]
        per_class = df.groupby(["engine"] + keys + ["method"])[cls].mean().reset_index()
        per_class.to_csv(OUT / "q1_per_class.csv", index=False)


if __name__ == "__main__":
    main()
