"""Resumable experiment campaign on the held-out topologies (GEANT, Germany50).

Runs the report's supporting experiments -- E4 (load), E5 (hotspot), E3
(attacker placement), E6 (drop/tamper), E10 (detector quality), E11 (oracle
vs detector), detector miscalibration and pool size -- with exactly the same
per-seed code and configs as their NSFNET/Abilene versions, but split into
small *steps* (experiment x topology x condition x method), each saved to its
own file the moment it finishes:

- Resumable: on restart, finished steps are skipped; at most the steps in
  flight are lost. A step that raised is logged and retried on the next run.
- Observable: results/campaign/STATUS.md (overall %, elapsed, ETA, current
  step, per-experiment table) is rewritten after every batch, and
  results/campaign/progress.log keeps the history.
- At the end every experiment is aggregated into the same CSV layout (and
  figures) its NSFNET/Abilene version produces, suffixed with the topology.

Usage:
  PYTHONPATH=. uv run python scripts/campaign.py            # run / resume
  PYTHONPATH=. uv run python scripts/campaign.py --status   # print the plan status only
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from analysis.metrics import (
    key_exposure_rate,
    mean_amplification,
    service_ratio,
    silent_failure_rate,
    success_under_attack,
)
from analysis.stats import aggregate_over_seeds
from experiments.level_b_common import METRICS, method_config, run_metrics
from qkdmp.config import load_config
from qkdmp.sim.des import arrival_rate_for_load, run_level_b
from qkdmp.sim.graph_mc import run_level_a
from qkdmp.topology import build_topology

ROOT = Path("results") / "campaign"
STEPS = ROOT / "steps"
STATUS = ROOT / "STATUS.md"
LOG = ROOT / "progress.log"
DONE = ROOT / "DONE"
PLAN_TITLE = "GEANT full + Germany50 core"

BASELINES_E12 = ["B0", "B0+PA", "B1+PA", "B2+PA", "B3+PA"]
MISCAL = [(0.25, 1.0), (0.5, 1.0), (2.0, 1.0), (4.0, 1.0), (1.0, 0.5), (1.0, 2.0)]
POOL_SECONDS = [0.1, 1.0, 3.0]
E11_VARIANTS = ["oracle", "no_info", "external@0.7"]
# relative cost of one request's decision, measured on the E12 grid runs
TOPO_COST = {"geant": 1.0, "germany50": 6.0}

QOS_VARIANTS = {  # label -> (policy method, class_aware, reservation, deadlines)
    "NoQoS-Adaptive": ("Adaptive", False, False, False),
    "Adaptive+classes": ("Adaptive", True, False, False),
    "Adaptive+classes+reserve": ("Adaptive", True, True, False),
    "QoS-Adaptive": ("Adaptive", True, True, True),
    "B0+PA+QoS": ("B0+PA", False, True, True),
    "B1+PA+QoS": ("B1+PA", False, True, True),
    "B2+PA+QoS": ("B2+PA", False, True, True),
}

# Thesis-extension plan (packages 1-4), run after the held-out-topology plan.
THESIS_PLAN = [("nsfnet", "Q1", 30), ("abilene", "Q1", 30), ("nsfnet", "Q2", 30), ("abilene", "Q2", 30),
               ("nsfnet", "Q4", 30), ("abilene", "Q4", 30)]

# (topology, experiment, seeds) -- GEANT gets everything at 30 seeds, Germany50
# the three core experiments at 15 seeds (user decision: ~8 h + ~10 h).
PLAN = [
    ("geant", "E4", 30), ("geant", "E5", 30), ("geant", "E3", 30), ("geant", "E6", 30),
    ("geant", "E10", 30), ("geant", "E11", 30), ("geant", "MISCAL", 30), ("geant", "POOL", 30),
    ("germany50", "E4", 15), ("germany50", "E3", 15), ("germany50", "E6", 15),
]


def _yaml(exp_id: str) -> dict:
    with open(f"configs/experiments/{exp_id}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cfg(exp: dict) -> dict:
    base = load_config()
    cfg = dict(base)
    if "policy_overrides" in exp:
        cfg["policy"] = dict(base["policy"], **exp["policy_overrides"])
    if "pool_overrides" in exp:
        cfg["pool"] = dict(base["pool"], **exp["pool_overrides"])
    return cfg


# ---------------------------------------------------------------------------
# One seed of one step (runs in a worker process)
# ---------------------------------------------------------------------------

def run_seed(kind: str, graph, cfg: dict, method: str, n_requests: int, seed: int, extra: dict) -> dict:
    if kind == "levelb":
        return run_metrics(run_level_b(graph, method, cfg, n_requests=n_requests, seed=seed))
    if kind == "e6":  # same reduction as experiments/e6.py
        df = run_level_b(graph, method, cfg, n_requests=n_requests, seed=seed)
        served = df[~df["blocked"] & ~df.get("failed", False)]
        return {"success_under_attack": success_under_attack(served), "silent_failure_rate": silent_failure_rate(served)}
    if kind == "levela":  # same reduction as experiments/e3.py and e10.py
        df = run_level_a(graph, method, cfg, n_requests=n_requests, seed=seed)
        return {"exposure": key_exposure_rate(df), "amplification": mean_amplification(df),
                "service_ratio": service_ratio(df)}
    if kind == "qos":  # overall metrics plus the same metrics per QoS class, as "<class>__<metric>"
        df = run_level_b(graph, method, cfg, n_requests=n_requests, seed=seed)
        out = run_metrics(df)
        for c in cfg["qos"]["classes"]:
            sub = df[df["qos_class"] == c["name"]]
            out.update({f"{c['name']}__{k}": v for k, v in run_metrics(sub).items()} if len(sub) else {})
            out[f"{c['name']}__latency"] = float(sub["latency"].dropna().mean()) if len(sub) else float("nan")
        return out
    if kind == "e11":  # same per-seed run as experiments/e11.py
        from experiments.e11 import _one_run

        metrics, _windows, final_auc = _one_run(graph, cfg, method, n_requests, 1, seed,
                                                extra["learning"], extra["no_info_prior"])
        return {**metrics, "final_auc": final_auc}
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Step lists per experiment (same configs as the NSFNET/Abilene runs)
# ---------------------------------------------------------------------------

def steps_for(topo: str, exp_id: str, n_seeds: int) -> list[dict]:
    steps = []

    def add(key: dict, kind: str, cfg: dict, method: str, n_requests: int, seeds: list[int], extra=None):
        # the step's name uses the method *label* ("Adaptive@0.5"), not the policy
        # name ("Adaptive"), or two operating points would share one file
        key = {"exp": exp_id, "topology": topo, **key, "method": (extra or {}).get("label", method)}
        sid = "__".join(f"{v}" for v in key.values()).replace("/", "-")
        steps.append({"id": sid, "key": key, "kind": kind, "cfg": cfg, "method": method,
                      "n_requests": n_requests, "seeds": seeds, "extra": extra or {}})

    seeds = list(range(n_seeds))
    if exp_id in ("E4", "E5"):
        exp = _yaml(exp_id)
        base = _cfg(exp)
        graph = build_topology(topo, base)
        rhos = exp["rho_values"].items() if exp_id == "E5" else [(None, r) for r in exp["rho_values"]]
        for label, rho in rhos:
            for method in exp["methods"]:
                policy_name, cfg = method_config(method, base)
                des = dict(base["des"], arrival_rate_per_s=arrival_rate_for_load(graph, base["key_size_bits"], rho))
                if exp_id == "E5":
                    des.update(traffic_mode="hotspot", hotspot_fraction=exp["hotspot_fraction"],
                               hotspot_pairs=exp["hotspot_pairs"])
                else:
                    des.update(traffic_mode="uniform")
                cfg["des"] = des
                key = {"rho": rho} if exp_id == "E4" else {"load_label": label, "rho": rho}
                add(key, "levelb", cfg, policy_name, exp["n_requests_per_seed"], seeds, {"label": method})
    elif exp_id == "E3":
        exp = _yaml("E3")
        base = load_config()
        for placement in exp["placements"]:
            for method in exp["methods"]:
                for rho_a in exp["rho_A_values"]:
                    cfg = dict(base, attacker=dict(base["attacker"], placement=placement, rho_A=rho_a,
                                                   warmup_requests=50))
                    add({"placement": placement, "rho_A": rho_a}, "levela", cfg, method,
                        exp["n_requests_per_seed"], seeds)
    elif exp_id == "E6":
        exp = _yaml("E6")
        base = _cfg(exp)
        graph = build_topology(topo, base)
        rate = arrival_rate_for_load(graph, base["key_size_bits"], exp["rho"])
        for method in exp["methods"]:
            for p_drop, p_tamper in exp["drop_tamper_values"]:
                cfg = dict(base, des=dict(base["des"], arrival_rate_per_s=rate, traffic_mode="uniform"),
                           attacker=dict(base["attacker"], p_drop=p_drop, p_tamper=p_tamper, rho_A=exp["rho_A"]))
                add({"p_drop": p_drop, "p_tamper": p_tamper}, "e6", cfg, method, exp["n_requests_per_seed"], seeds)
    elif exp_id == "E10":
        exp = _yaml("E10")
        base = load_config()
        for method in exp["methods"]:
            for dq in exp["detector_quality_values"]:
                cfg = dict(base, attacker=dict(base["attacker"], placement="random", rho_A=exp["rho_A"]),
                           suspicion=dict(base["suspicion"], detector_quality=dq))
                add({"detector_quality": dq}, "levela", cfg, method, exp["n_requests_per_seed"], seeds)
    elif exp_id == "E11":
        exp = _yaml("E11")
        base = _cfg(exp)
        graph = build_topology(topo, base)
        rate = arrival_rate_for_load(graph, base["key_size_bits"], exp["rho"])
        for placement in exp["placements"]:
            for activity in exp["activity_values"]:
                cfg = dict(base, des=dict(base["des"], arrival_rate_per_s=rate, traffic_mode="uniform",
                                          warmup_requests=0),
                           attacker=dict(base["attacker"], placement=placement, rho_A=exp["rho_A"],
                                         p_drop=activity, p_tamper=activity))
                for variant in E11_VARIANTS:
                    add({"placement": placement, "activity": activity}, "e11", cfg, variant,
                        exp["n_requests_per_seed"], seeds,
                        {"learning": exp["learning"], "no_info_prior": exp["no_info_prior"]})
    elif exp_id == "Q1":
        exp = _yaml("Q1")
        base = _cfg(exp)
        base["attacker"] = dict(base["attacker"], rho_A=exp["rho_A"])
        graph = build_topology(topo, base)
        q_seeds = list(range(exp["seed_offset"], exp["seed_offset"] + n_seeds))
        for placement in exp["placements"]:
            for rho in exp["rho_values"]:
                for label, (method, aware, reserve, deadlines) in QOS_VARIANTS.items():
                    cfg = dict(base, attacker=dict(base["attacker"], placement=placement),
                               des=dict(base["des"], traffic_mode="uniform",
                                        arrival_rate_per_s=arrival_rate_for_load(graph, base["key_size_bits"], rho)),
                               qos={"classes": exp["classes"], "class_aware": aware, "reservation": reserve,
                                    "deadlines": deadlines, "retry_interval_s": exp["retry_interval_s"]})
                    add({"placement": placement, "rho": rho}, "qos", cfg, method, exp["n_requests_per_seed"],
                        q_seeds, {"label": label})
    elif exp_id == "Q2":
        exp = _yaml("Q2")
        base = _cfg(exp)
        base["attacker"] = dict(base["attacker"], rho_A=exp["rho_A"], placement=exp["placement"])
        base["kms"] = {"local_consumption_fraction": exp["local_consumption_fraction"]}
        graph = build_topology(topo, base)
        q_seeds = list(range(exp["seed_offset"], exp["seed_offset"] + n_seeds))
        views = [("live", 0.0)] + [(v, dt) for v in exp["views"] for dt in exp["intervals_s"]]
        for rho in exp["rho_values"]:
            for method in exp["methods"]:
                for view, dt in views:
                    policy_name, cfg = method_config(method, base)
                    cfg["des"] = dict(base["des"], traffic_mode="uniform",
                                      arrival_rate_per_s=arrival_rate_for_load(graph, base["key_size_bits"], rho))
                    if view != "live":
                        cfg["controller"] = {"telemetry_interval_s": dt, "view": view}
                    add({"rho": rho, "view": view, "interval_s": dt}, "levelb", cfg, policy_name,
                        exp["n_requests_per_seed"], q_seeds, {"label": method})
    elif exp_id == "Q4":
        q4 = _yaml("Q4")
        exp = _yaml("E12")
        base = _cfg(exp)
        graph = build_topology(topo, base)
        rate = arrival_rate_for_load(graph, base["key_size_bits"], exp["rho"])
        test_seeds = list(range(exp["test"]["seed_offset"], exp["test"]["seed_offset"] + n_seeds))
        for k, setting in enumerate(q4["settings"]):
            for placement in exp["placements"]:
                for rho_a in exp["rho_A_values"]:
                    for method in BASELINES_E12 + ["Adaptive@0.5"]:
                        policy_name, cfg = method_config(method, base)
                        cfg["des"] = dict(base["des"], arrival_rate_per_s=rate, traffic_mode="uniform")
                        cfg["attacker"] = dict(base["attacker"], placement=placement, rho_A=rho_a)
                        cfg["kms"] = dict(setting)
                        add({"setting": k, "placement": placement, "rho_A": rho_a}, "levelb", cfg, policy_name,
                            exp["n_requests_per_seed"], test_seeds, {"label": method})
    elif exp_id in ("MISCAL", "POOL"):
        exp = _yaml("E12")
        base = _cfg(exp)
        graph = build_topology(topo, base)
        rate = arrival_rate_for_load(graph, base["key_size_bits"], exp["rho"])
        test_seeds = list(range(exp["test"]["seed_offset"], exp["test"]["seed_offset"] + n_seeds))
        settings = ([{"odds_factor": o, "temperature": t} for o, t in MISCAL] if exp_id == "MISCAL"
                    else [{"capacity_seconds": c} for c in POOL_SECONDS])
        methods = ["Adaptive@0.5", "B3+PA"] if exp_id == "MISCAL" else BASELINES_E12 + ["Adaptive@0.5"]
        for setting in settings:
            for placement in exp["placements"]:
                for rho_a in exp["rho_A_values"]:
                    for method in methods:
                        policy_name, cfg = method_config(method, base)
                        cfg["des"] = dict(base["des"], arrival_rate_per_s=rate, traffic_mode="uniform")
                        cfg["attacker"] = dict(base["attacker"], placement=placement, rho_A=rho_a)
                        if exp_id == "MISCAL":
                            cfg["suspicion"] = dict(base["suspicion"], miscal_odds_factor=setting["odds_factor"],
                                                    miscal_temperature=setting["temperature"])
                        else:
                            cfg["pool"] = dict(base["pool"], capacity_seconds=setting["capacity_seconds"])
                            # capacity changes the graph's pool sizes: rebuild per setting in the worker
                        add({**setting, "placement": placement, "rho_A": rho_a}, "levelb", cfg, policy_name,
                            exp["n_requests_per_seed"], test_seeds, {"label": method})
    else:
        raise ValueError(exp_id)
    for s in steps:
        s["weight"] = len(s["seeds"]) * s["n_requests"] * TOPO_COST.get(topo, 1.0)
    return steps


# ---------------------------------------------------------------------------
# Execution, checkpointing, progress
# ---------------------------------------------------------------------------

def _step_file(step: dict) -> Path:
    return STEPS / f"{step['id']}.csv"


def _log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_status(all_steps: list[dict], started: float, done_weight_run: float, current: str,
                  failed: dict) -> None:
    total_w = sum(s["weight"] for s in all_steps)
    done = [s for s in all_steps if _step_file(s).exists()]
    done_w = sum(s["weight"] for s in done)
    elapsed = time.time() - started
    rate = done_weight_run / elapsed if elapsed > 0 and done_weight_run > 0 else None
    eta = timedelta(seconds=int((total_w - done_w) / rate)) if rate else None
    lines = [
        f"# Campaign status ({PLAN_TITLE})",
        "",
        f"Updated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        f"- **Progress:** {len(done)} / {len(all_steps)} steps, "
        f"**{100 * done_w / total_w:.1f}%** of estimated work",
        f"- **Elapsed this session:** {timedelta(seconds=int(elapsed))}",
        f"- **Estimated time left:** {eta if eta else 'measuring...'}"
        + (f" (finish ~{datetime.now() + eta:%Y-%m-%d %H:%M})" if eta else ""),
        f"- **Now running:** {current}",
        f"- **Failed steps (retried on next start):** {len(failed)}",
        "",
        "| topology | experiment | steps done | work done |",
        "|---|---|---|---|",
    ]
    groups: dict = {}
    for s in all_steps:
        g = groups.setdefault((s["key"]["topology"], s["key"]["exp"]), [0, 0, 0.0, 0.0])
        g[1] += 1
        g[3] += s["weight"]
        if _step_file(s).exists():
            g[0] += 1
            g[2] += s["weight"]
    for (topo, exp), (d, n, dw, tw) in groups.items():
        mark = " (done)" if d == n else ""
        lines.append(f"| {topo} | {exp} | {d} / {n}{mark} | {100 * dw / tw:.0f}% |")
    if failed:
        lines += ["", "## Failed steps", ""] + [f"- `{k}`: {v}" for k, v in failed.items()]
    lines += ["", "Log: `results/campaign/progress.log`. Restart with "
                  "`PYTHONPATH=. uv run python scripts/campaign.py` to resume."]
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, STATUS)


def _graph_for(step: dict, cache: dict):
    topo = step["key"]["topology"]
    cap = step["cfg"]["pool"]["capacity_seconds"]
    rate_mode = step["cfg"]["rate_model"]["mode"]
    k = (topo, cap, rate_mode)
    if k not in cache:
        cache[k] = build_topology(topo, step["cfg"])
    return cache[k]


def run_all(all_steps: list[dict]) -> None:
    STEPS.mkdir(parents=True, exist_ok=True)
    pending = [s for s in all_steps if not _step_file(s).exists()]
    _log(f"session start: {len(all_steps) - len(pending)} steps already done, {len(pending)} to run")
    started = time.time()
    done_weight_run = 0.0
    failed: dict = {}
    cache: dict = {}
    # batch a few steps together so each Parallel call has >= ~44 seed-jobs for 22 cores
    i = 0
    while i < len(pending):
        per = max(1, int(np.ceil(44 / len(pending[i]["seeds"]))))
        batch = pending[i:i + per]
        i += per
        current = ", ".join(s["id"] for s in batch)
        _write_status(all_steps, started, done_weight_run, current, failed)
        jobs = [(s, seed) for s in batch for seed in s["seeds"]]
        try:
            results = Parallel(n_jobs=-1)(
                delayed(run_seed)(s["kind"], _graph_for(s, cache), s["cfg"], s["method"], s["n_requests"], seed,
                                  s["extra"])
                for s, seed in jobs
            )
        except Exception:  # retry this batch step by step to isolate the failing one
            results = None
        for s in batch:
            try:
                if results is not None:
                    rows = [dict(r, seed=seed) for (st, seed), r in zip(jobs, results) if st is s]
                else:
                    rows = [dict(run_seed(s["kind"], _graph_for(s, cache), s["cfg"], s["method"], s["n_requests"],
                                          seed, s["extra"]), seed=seed) for seed in s["seeds"]]
                df = pd.DataFrame(rows)
                for k, v in s["key"].items():
                    df[k] = v
                if "label" in s["extra"]:
                    df["method"] = s["extra"]["label"]
                tmp = _step_file(s).with_suffix(".tmp")
                df.to_csv(tmp, index=False)
                os.replace(tmp, _step_file(s))
                done_weight_run += s["weight"]
                failed.pop(s["id"], None)
                _log(f"done  {s['id']}")
            except Exception as e:  # noqa: BLE001 -- keep the campaign going, retry next start
                failed[s["id"]] = f"{type(e).__name__}: {e}"
                _log(f"FAIL  {s['id']}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
    _write_status(all_steps, started, done_weight_run, "aggregating results", failed)
    return failed


# ---------------------------------------------------------------------------
# Aggregation into the standard per-experiment CSVs + figures
# ---------------------------------------------------------------------------

def _load(topo: str, exp_id: str, steps: list[dict]) -> pd.DataFrame:
    files = [_step_file(s) for s in steps if s["key"]["topology"] == topo and s["key"]["exp"] == exp_id]
    return pd.concat([pd.read_csv(f) for f in files if f.exists()], ignore_index=True)


def _agg(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        for m in metrics:
            st = aggregate_over_seeds(list(g[m]))
            row.update({f"{m}_mean": st["mean"], f"{m}_ci_low": st["ci_low"], f"{m}_ci_high": st["ci_high"]})
        row["n_seeds"] = g["seed"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(steps: list[dict]) -> None:
    from experiments import e3, e4, e6, e10
    from experiments import e12_robustness as rb

    for topo, exp_id, _ in PLAN:
        if not all(_step_file(s).exists() for s in steps if s["key"]["topology"] == topo and s["key"]["exp"] == exp_id):
            _log(f"skip aggregation {topo}/{exp_id}: steps missing")
            continue
        df = _load(topo, exp_id, steps)
        if exp_id == "E4":
            out = _agg(df, ["method", "rho"], METRICS)
            out.to_csv(f"results/E4/e4_cost_blocking_vs_load_{topo}.csv", index=False)
            e4.plot_e4(out[~out["method"].str.contains("+", regex=False)], Path("figures/E4") / topo)
            e4.plot_admission_families(out, "rho", "secure_goodput", "Secure goodput",
                                       Path("figures/E4") / topo / "e4_secure_goodput_admission.png")
        elif exp_id == "E5":
            out = _agg(df, ["method", "load_label", "rho"], METRICS)
            out.to_csv(f"results/E5/e5_hotspot_{topo}.csv", index=False)
            e4.plot_admission_families(out, "rho", "secure_goodput", "Secure goodput",
                                       Path("figures/E5") / topo / "e5_secure_goodput_admission.png")
        elif exp_id == "E3":
            out = _agg(df, ["placement", "method", "rho_A"], ["exposure", "amplification", "service_ratio"])
            out.to_csv(f"results/E3/e3_security_vs_attacker_{topo}.csv", index=False)
            e3.decision_gate_table(out).to_csv(f"results/E3/e3_decision_gate_{topo}.csv", index=False)
            e3.plot_e3_exposure(out, Path("figures/E3") / topo)
        elif exp_id == "E6":
            out = _agg(df, ["method", "p_drop", "p_tamper"], ["success_under_attack", "silent_failure_rate"])
            out.to_csv(f"results/E6/e6_drop_tamper_{topo}.csv", index=False)
            e6.plot_e6(out, Path("figures/E6") / f"e6_drop_tamper_{topo}.png")
        elif exp_id == "E10":
            out = _agg(df, ["method", "detector_quality"], ["exposure", "service_ratio", "amplification"])
            out.to_csv(f"results/E10/e10_detector_quality_{topo}.csv", index=False)
            e10.plot_e10(out, Path("figures/E10") / f"e10_detector_quality_{topo}.png", title=f"({topo})")
        elif exp_id == "E11":
            out = _agg(df.rename(columns={"method": "variant"}), ["placement", "activity", "variant"],
                       METRICS + ["final_auc"])
            out.insert(0, "topology", topo)
            out.to_csv(f"results/E11/e11_oracle_{topo}.csv", index=False)
        elif exp_id == "Q1":
            Path("results/Q1").mkdir(parents=True, exist_ok=True)
            keys = ["placement", "rho", "method"]
            metric_cols = [c for c in df.columns if c not in keys + ["seed", "exp", "topology"]]
            _agg(df, keys, metric_cols).to_csv(f"results/Q1/q1_qos_{topo}.csv", index=False)
            frames = []
            for cls in ("gold", "silver", "bronze"):
                sub = df[["topology", "placement", "rho", "method", "seed"]].copy()
                sub["secure_goodput"], sub["exposure"] = df[f"{cls}__secure_goodput"], df[f"{cls}__exposure"]
                frames.append(sub.assign(qos_class=cls, rho_A=0.2))
            pc = rb.paired_comparisons(pd.concat(frames, ignore_index=True), ours="QoS-Adaptive",
                                       group_cols=("qos_class", "rho"))
            pc = pc[pc["baseline"] != "QoS-Adaptive"]
            pc.to_csv(f"results/Q1/q1_paired_{topo}.csv", index=False)
            _log(f"{topo}/Q1 paired summary:\n{rb.summarize(pc, ('qos_class',)).to_string(index=False)}")
        elif exp_id == "Q2":
            Path("results/Q2").mkdir(parents=True, exist_ok=True)
            _agg(df, ["rho", "method", "view", "interval_s"], METRICS).to_csv(f"results/Q2/q2_stale_{topo}.csv",
                                                                                index=False)
        elif exp_id == "Q4":
            rdir = Path("results/Q4")
            rdir.mkdir(parents=True, exist_ok=True)
            df.to_csv(rdir / f"q4_per_seed_{topo}.csv", index=False)
            pc = rb.paired_comparisons(df, group_cols=("setting",))
            pc.to_csv(rdir / f"q4_paired_{topo}.csv", index=False)
            rb.means_table(df, ("setting",)).to_csv(rdir / f"q4_means_{topo}.csv", index=False)
            _log(f"{topo}/Q4 paired summary:\n{rb.summarize(pc, ('setting',)).to_string(index=False)}")
        elif exp_id in ("MISCAL", "POOL"):
            base = pd.read_csv(Path("results/E12/robustness") / f"per_seed_{topo}.csv")
            if exp_id == "MISCAL":
                group = ("odds_factor", "temperature")
                unaffected = base[base["method"].isin(["B0", "B0+PA", "B1+PA", "B2+PA"])]
                frames = [pd.concat([df[(df["odds_factor"] == o) & (df["temperature"] == t)],
                                     unaffected.assign(odds_factor=o, temperature=t)], ignore_index=True)
                          for o, t in MISCAL]
            else:
                group = ("capacity_seconds",)
                frames = [df, base[base["method"].isin(BASELINES_E12 + ["Adaptive@0.5"])].assign(capacity_seconds=0.5)]
            all_df = pd.concat(frames, ignore_index=True)
            name = exp_id.lower()
            rdir = Path("results/E12/robustness")
            all_df.to_csv(rdir / f"per_seed_{name}_{topo}.csv", index=False)
            pc = rb.paired_comparisons(all_df, group_cols=group)
            pc.to_csv(rdir / f"paired_{name}_{topo}.csv", index=False)
            rb.means_table(all_df, group).to_csv(rdir / f"means_{name}_{topo}.csv", index=False)
            _log(f"{topo}/{exp_id} paired summary:\n{rb.summarize(pc, group).to_string(index=False)}")
        _log(f"aggregated {topo}/{exp_id}")


def _select_plan(name: str) -> None:
    """--plan thesis: the thesis-extension experiments, with their own
    results/campaign_thesis/ folder (steps, STATUS.md, progress.log)."""
    global ROOT, STEPS, STATUS, LOG, DONE, PLAN, PLAN_TITLE
    if name == "thesis":
        ROOT = Path("results") / "campaign_thesis"
        PLAN, PLAN_TITLE = THESIS_PLAN, "thesis extensions"
    STEPS, STATUS, LOG, DONE = ROOT / "steps", ROOT / "STATUS.md", ROOT / "progress.log", ROOT / "DONE"


def main() -> None:
    if "--plan" in sys.argv:
        _select_plan(sys.argv[sys.argv.index("--plan") + 1])
    ROOT.mkdir(parents=True, exist_ok=True)
    all_steps = [s for topo, exp_id, n in PLAN for s in steps_for(topo, exp_id, n)]
    if "--status" in sys.argv:
        _write_status(all_steps, time.time(), 0.0, "(not running)", {})
        print(STATUS.read_text(encoding="utf-8"))
        return
    DONE.unlink(missing_ok=True)
    failed = run_all(all_steps)
    aggregate(all_steps)
    _write_status(all_steps, time.time(), 0.0, "finished" if not failed else "finished with failed steps", failed)
    if not failed:
        DONE.write_text(datetime.now().isoformat(), encoding="utf-8")
    _log("campaign finished" + ("" if not failed else f" with {len(failed)} failed steps (rerun to retry)"))


if __name__ == "__main__":
    main()
