"""E11 -- Self-learned suspicion (fair-comparison follow-up). Level B.

Question: how much of Adaptive's security depends on having an external
suspicion detector, and how much can the relay learn by itself from the
outcomes of its own share deliveries (`qkdmp.suspicion_learning`)?

Views of p_v, same Adaptive policy, same traffic:
- oracle: ground truth (upper bound);
- no_info: flat prior, never updated (lower bound);
- learned: flat prior + online learning with probing (no external detector);
- learned_noprobe: the same without probing (shows why probing is needed);
- external@q: the external detector's score at quality q (README 3.2),
  never updated -- q=0.7 is the setting every other experiment uses;
- external+learned@q: external score as prior + online learning.

Swept over topology x attacker placement x attacker activity (the
compromised nodes' drop/tamper rate: a mostly-passive attacker leaves
little evidence to learn from). Reported both overall and per window of
requests, so the learning curve is visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

matplotlib.use("Agg")

from analysis.stats import aggregate_over_seeds
from experiments.level_b_common import METRICS, run_metrics
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.topology import build_topology

RESULTS_DIR = Path("results") / "E11"
FIGURES_DIR = Path("figures") / "E11"

BASE_VARIANTS = ["oracle", "no_info", "learned", "learned_noprobe"]


def variants_for(detector_qualities: list[float]) -> list[str]:
    return BASE_VARIANTS + [f"{v}@{q}" for q in detector_qualities for v in ("external", "external+learned")]


WINDOW_METRICS = ["exposure", "secure_goodput", "amplification", "auc"]


def _one_run(graph, cfg, variant: str, n_requests: int, n_windows: int, seed: int, learning: dict,
             no_info_prior: float) -> tuple[dict, list[dict], float]:
    flat = {v: no_info_prior for v in graph.nodes()}
    kwargs: dict = {}
    name, _, quality = variant.partition("@")
    if quality:
        cfg = dict(cfg, suspicion=dict(cfg["suspicion"], detector_quality=float(quality)))
    if name == "oracle":
        kwargs["oracle_p_v"] = True
    if name in ("no_info", "learned", "learned_noprobe"):
        kwargs["p_v_override"] = flat
    if name in ("learned", "external+learned"):
        kwargs["learning"] = learning
    if name == "learned_noprobe":
        kwargs["learning"] = dict(learning, probe_paths=0)
    df = run_level_b(graph, "Adaptive", cfg, n_requests=n_requests, seed=seed, **kwargs)

    windows = []
    for w, chunk in enumerate(np.array_split(np.arange(len(df)), n_windows)):
        sub = df.iloc[chunk]
        m = run_metrics(sub)
        auc = sub["auc"].astype(float).mean() if "auc" in sub and sub["auc"].notna().any() else float("nan")
        windows.append({"window": w, "exposure": m["exposure"], "secure_goodput": m["secure_goodput"],
                        "amplification": m["amplification"], "auc": auc})
    return run_metrics(df), windows, df.attrs["final_auc"]


def run_e11(config: dict, topologies: list[str], placements: list[str], activity_values: list[float],
            rho: float, rho_a: float, n_seeds: int, n_requests: int, n_windows: int, learning: dict,
            no_info_prior: float, detector_qualities: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows, window_rows = [], []
    for topology in topologies:
        graph = build_topology(topology, config)
        arrival_rate = arrival_rate_for_load(graph, config["key_size_bits"], rho)
        for placement in placements:
            for activity in activity_values:
                cfg = dict(config)
                cfg["des"] = dict(config["des"], arrival_rate_per_s=arrival_rate, traffic_mode="uniform",
                                  warmup_requests=0)
                cfg["attacker"] = dict(config["attacker"], placement=placement, rho_A=rho_a,
                                       p_drop=activity, p_tamper=activity)
                for variant in variants_for(detector_qualities):
                    results = Parallel(n_jobs=-1)(
                        delayed(_one_run)(graph, cfg, variant, n_requests, n_windows, seed, learning, no_info_prior)
                        for seed in range(n_seeds)
                    )
                    key = {"topology": topology, "placement": placement, "activity": activity, "variant": variant}
                    row = dict(key)
                    for m in METRICS:
                        stats = aggregate_over_seeds([r[0][m] for r in results])
                        row.update({f"{m}_mean": stats["mean"], f"{m}_ci_low": stats["ci_low"],
                                    f"{m}_ci_high": stats["ci_high"]})
                    auc = aggregate_over_seeds([r[2] for r in results])
                    row.update({"final_auc_mean": auc["mean"], "final_auc_ci_low": auc["ci_low"],
                                "final_auc_ci_high": auc["ci_high"], "n_seeds": n_seeds})
                    overall_rows.append(row)
                    for w in range(n_windows):
                        wrow = dict(key, window=w)
                        for m in WINDOW_METRICS:
                            stats = aggregate_over_seeds([r[1][w][m] for r in results])
                            wrow.update({f"{m}_mean": stats["mean"], f"{m}_ci_low": stats["ci_low"],
                                         f"{m}_ci_high": stats["ci_high"]})
                        window_rows.append(wrow)
                    print(f"{topology}/{placement}/p={activity}/{variant}: "
                          f"exposure={row['exposure_mean']:.3f} goodput={row['secure_goodput_mean']:.3f} "
                          f"amp={row['amplification_mean']:.2f} auc={row['final_auc_mean']:.2f}", flush=True)
    return pd.DataFrame(overall_rows), pd.DataFrame(window_rows)


def plot_e11_curves(windows: pd.DataFrame, out_dir: Path) -> None:
    """Per condition: learning curves (exposure, secure goodput, AUC) per variant."""
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [("exposure", "Key-exposure rate"), ("secure_goodput", "Secure goodput"),
              ("amplification", "Mean amplification"), ("auc", "Suspicion ranking AUC")]
    for (topology, placement, activity), group in windows.groupby(["topology", "placement", "activity"]):
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
        for ax, (m, ylabel) in zip(axes, panels):
            for variant in group["variant"].unique():
                g = group[group["variant"] == variant].sort_values("window")
                if g[f"{m}_mean"].notna().any():
                    ax.plot(g["window"] + 1, g[f"{m}_mean"], marker="o", label=variant,
                            linestyle="-" if "learned" in variant else "--")
            ax.set_xlabel("Request window")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
        fig.suptitle(f"E11: {topology}, {placement} attacker, activity p={activity}")
        fig.tight_layout()
        fig.savefig(out_dir / f"e11_{topology}_{placement}_p{activity}.png", dpi=130)
        plt.close(fig)
