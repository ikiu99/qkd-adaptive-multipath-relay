"""Experiments with QKDNetSim key pools (co-simulation).

Runs the unchanged qkdmp Level-B DES with each link's pool replaced by
QKDNetSim Q-buffers (`ns3_pools`), and -- on the same seeds -- with the
fluid pool model, so every number has a direct counterpart:

  e12        main claim (configs/experiments/E12.yaml): Adaptive@0.5 vs the
             pool-aware baselines, placements x rho_A, held-out seeds.
  q2         stale controller (configs/experiments/Q2.yaml).
  e4, e5     load sweep / hotspot traffic (E4.yaml, E5.yaml), restricted to
             the E12 method set.
  q1         QoS classes (Q1.yaml), all seven variants.
  strategic  strategic attacker (scripts/strategic_attacker_check.py setup).

Steps for e4 / e5 / q1 are built by scripts/campaign.py `steps_for`, so the
configurations and seeds are exactly those of the earlier runs; only the pool
engine changes. Experiments whose pools never run dry (E6: one-hour pools)
or that have no pools at all (Level A: E3, E10) are not meaningful here.

Resumable: every (step, engine) is saved to results/steps/<id>.csv; only
missing seeds are run. Progress: results/progress.log and results/STATUS.md.

Usage:
  PYTHONPATH=. uv run python qkdnetsim-prototype/cosim/run_cosim.py e12 --topology abilene --seeds 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, e12_config, e12_experiment  # noqa: E402
import ns3_relay  # noqa: E402
from ns3_pools import Ns3LinkPool, Ns3PoolServer  # noqa: E402

sys.path.insert(0, str(ROOT))
from experiments.level_b_common import method_config  # noqa: E402
from qkdmp.config import load_config  # noqa: E402
from qkdmp.sim import des  # noqa: E402
from qkdmp.topology import build_topology  # noqa: E402
from scripts import campaign  # noqa: E402

OUT = ROOT / "qkdnetsim-prototype" / "results"
STEPS = OUT / "steps"
ENGINES = ["ns3", "fluid"]
N_JOBS = 12  # each ns-3 run is one WSL process
E12_METHODS = ["B0", "B0+PA", "B1+PA", "B2+PA", "B3+PA", "Adaptive@0.5"]
STRATEGIC_LAGS = [0.05, 0.2, 0.5, None]
BLOCK_BITS = [2048, 8192]  # 256 B and 1 KB post-processing key blocks


def run_one(kind: str, graph, cfg: dict, method: str, n_requests: int, seed: int, extra: dict,
            engine: str) -> dict:
    """campaign.run_seed with the chosen pool engine. With QKDNetSim, local key
    consumption (phi, Q2) is real: the link generates only (1 - phi) of its
    nominal rate into the Q-buffers (the fluid model's net refill) while the
    controller still believes the nominal rate."""
    t0 = time.time()
    holder: dict = {}
    if engine in ("ns3", "ns3relay"):
        phi = (cfg.get("kms") or {}).get("local_consumption_fraction", 0.0)

        def attach(g):
            server = Ns3PoolServer(g, cfg["key_size_bits"], seed=seed, rate_scale=1.0 - phi,
                                   block_bits=(cfg.get("cosim") or {}).get("block_bits", 0))
            holder["server"] = server
            for i, (u, v) in enumerate(server.edges):
                d = g.edges[u, v]
                d["pool"] = Ns3LinkPool(server, i, d["capacity_bits"], d["rate_bps"])
            if engine == "ns3relay":
                holder["relay"] = g.graph["ns3_relay"] = ns3_relay.RelayPort(
                    server, cfg["attacker"]["p_drop"], cfg["attacker"]["p_tamper"])

        original = des.attach_pool_state
        des.attach_pool_state = attach
        if engine == "ns3relay":
            ns3_relay.install()
        try:
            out = campaign.run_seed(kind, graph, cfg, method, n_requests, seed, extra)
            if "relay" in holder:
                relay = holder["relay"]
                missing = int(holder["server"]._ask("INFO")[1])
                out.update(relay_shares=relay.shares_sent, relay_late=relay.late,
                           relay_exposure_mismatch=relay.mismatches, relay_missing_keys=missing)
        finally:
            des.attach_pool_state = original
            ns3_relay.uninstall()
            if "server" in holder:
                holder["server"].close()
    else:
        out = campaign.run_seed(kind, graph, cfg, method, n_requests, seed, extra)
    ipc = holder["server"].ipc_calls if "server" in holder else 0
    return {"seed": seed, **out, "ipc_calls": ipc, "wall_s": time.time() - t0}


def _step(label: dict, cfg: dict, method: str, n_requests: int, seeds: list[int], kind: str = "levelb",
          extra: dict | None = None) -> dict:
    return {"label": label, "cfg": cfg, "method": method, "n_requests": n_requests, "seeds": seeds,
            "kind": kind, "extra": extra or {}}


def _topo_label(topology: str) -> dict:
    # NSFNET steps predate the topology option; keep their ids
    return {} if topology == "nsfnet" else {"topology": topology}


def e12_plan(topology: str, n_seeds: int) -> list[dict]:
    exp = e12_experiment()
    cfg = e12_config()
    graph = build_topology(topology, cfg)
    rate = des.arrival_rate_for_load(graph, cfg["key_size_bits"], exp["rho"])
    seeds = list(range(1000, 1000 + n_seeds))
    plan = []
    for placement in exp["placements"]:
        for rho_a in exp["rho_A_values"]:
            for method in exp["baselines"] + ["Adaptive@0.5"]:
                policy, c = method_config(method, cfg)
                c = dict(c, des=dict(cfg["des"], arrival_rate_per_s=rate, traffic_mode="uniform"),
                         attacker=dict(cfg["attacker"], placement=placement, rho_A=rho_a))
                label = {"exp": "e12", **_topo_label(topology), "placement": placement, "rho_A": rho_a,
                         "method": method, "n_requests": exp["n_requests_per_seed"]}
                plan.append(_step(label, c, policy, exp["n_requests_per_seed"], seeds))
    return plan


def blocks_plan(topology: str, n_seeds: int) -> list[dict]:
    """E12 with key delivered in larger post-processing blocks (QKDNetSim only;
    the fluid model has no block size). block_bits 256 is the E12 run itself."""
    plan = []
    for block in BLOCK_BITS:
        for step in e12_plan(topology, n_seeds):
            label = {**step["label"], "exp": "blocks", "block_bits": block}
            plan.append({**step, "label": label, "cfg": dict(step["cfg"], cosim={"block_bits": block})})
    return plan


def q2_plan(topology: str, n_seeds: int) -> list[dict]:
    import yaml

    with open(ROOT / "configs" / "experiments" / "Q2.yaml", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    base = load_config()
    base = dict(base, policy=dict(base["policy"], **exp["policy_overrides"]),
                pool=dict(base["pool"], **exp["pool_overrides"]))
    graph = build_topology(topology, base)
    views = [("live", 0.0)] + [(v, dt) for v in exp["views"] for dt in exp["intervals_s"]]
    seeds = list(range(1000, 1000 + n_seeds))
    plan = []
    for rho in exp["rho_values"]:
        rate = des.arrival_rate_for_load(graph, base["key_size_bits"], rho)
        for view, dt in views:
            for method in exp["methods"]:
                policy, c = method_config(method, base)
                c = dict(c, des=dict(base["des"], arrival_rate_per_s=rate, traffic_mode="uniform"),
                         attacker=dict(base["attacker"], placement=exp["placement"], rho_A=exp["rho_A"]),
                         kms={"local_consumption_fraction": exp["local_consumption_fraction"]})
                if view != "live":
                    c["controller"] = {"telemetry_interval_s": dt, "view": view}
                label = {"exp": "q2", **_topo_label(topology), "rho": rho, "view": view, "interval_s": dt,
                         "method": method, "n_requests": exp["n_requests_per_seed"]}
                plan.append(_step(label, c, policy, exp["n_requests_per_seed"], seeds))
    return plan


def campaign_plan(exp_id: str, topology: str, n_seeds: int) -> list[dict]:
    """E4 / E5 / Q1 steps exactly as scripts/campaign.py builds them."""
    plan = []
    for s in campaign.steps_for(topology, exp_id, n_seeds):
        key = s["key"]
        if exp_id in ("E4", "E5") and key["method"] not in E12_METHODS:
            continue
        label = {"exp": exp_id.lower(), **{k: v for k, v in key.items() if k != "exp"},
                 "n_requests": s["n_requests"]}
        plan.append(_step(label, s["cfg"], s["method"], s["n_requests"], s["seeds"], s["kind"], s["extra"]))
    return plan


def strategic_plan(topology: str, n_seeds: int) -> list[dict]:
    """As scripts/strategic_attacker_check.py: E12 setup, rho_A 0.2, random
    placement, 400 requests, attacker moves every 0.1 s onto the method's most
    used relays; the detector catches up after a lag (or never)."""
    exp = e12_experiment()
    base = e12_config()
    graph = build_topology(topology, base)
    rate = des.arrival_rate_for_load(graph, base["key_size_bits"], exp["rho"])
    seeds = list(range(1000, 1000 + n_seeds))
    plan = []
    for lag in STRATEGIC_LAGS:
        strategic = {"enabled": True, "interval_s": 0.1, "window": 50, "detector_lag_s": lag}
        for method in E12_METHODS:
            policy, c = method_config(method, base)
            c = dict(c, des=dict(base["des"], arrival_rate_per_s=rate, traffic_mode="uniform"),
                     attacker=dict(base["attacker"], placement="random", rho_A=0.2, strategic=strategic))
            label = {"exp": "strategic", "topology": topology, "detector_lag_s": "never" if lag is None else lag,
                     "method": method, "n_requests": 400}
            plan.append(_step(label, c, policy, 400, seeds))
    return plan


def _step_id(label: dict) -> str:
    text = json.dumps(label, sort_keys=True)
    return f"{label['exp']}_{label['engine']}_{label['method'].replace('@', 'at').replace('+', 'p')}_" + \
        hashlib.sha1(text.encode()).hexdigest()[:10]


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _status(name: str, total: int, done: int, t_start: float, current: str) -> None:
    lines = ["# QKDNetSim co-simulation status", "",
             f"- run: {name}",
             f"- steps done: {done} / {total}",
             f"- elapsed this session: {(time.time() - t_start) / 60:.1f} min",
             f"- current: {current}", ""]
    (OUT / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def run_plan(name: str, plan: list[dict], topology: str, jobs: int, engines: list[str] = ENGINES) -> None:
    STEPS.mkdir(parents=True, exist_ok=True)
    graph = build_topology(topology, e12_config())
    t_start = time.time()
    total = len(plan) * len(engines)
    _log(f"start {name}: {total} steps")
    done = 0
    for step in plan:
        for engine in engines:
            label = {**step["label"], "engine": engine}
            path = STEPS / f"{_step_id(label)}.csv"
            desc = " ".join(f"{k}={v}" for k, v in label.items() if k not in ("exp", "n_requests"))
            old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["seed"])
            todo = [s for s in step["seeds"] if s not in set(old["seed"])]
            done += 1
            if not todo:
                continue
            _status(name, total, done - 1, t_start, desc)
            t0 = time.time()
            rows = Parallel(n_jobs=jobs)(
                delayed(run_one)(step["kind"], graph, step["cfg"], step["method"], step["n_requests"], s,
                                 step["extra"], engine) for s in todo
            )
            new = pd.DataFrame([{**label, **r} for r in rows])
            df = pd.concat([old, new], ignore_index=True) if len(old) else new
            df.to_csv(path, index=False)
            cur = df[df["seed"].isin(step["seeds"])]
            _log(f"[{done}/{total}] {desc}: goodput {cur['secure_goodput'].mean():.3f} "
                 f"exposure {cur['exposure'].mean():.3f} (+{len(todo)} seeds, {time.time() - t0:.0f}s)")
    _status(name, total, done, t_start, "finished")
    _log(f"finished {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e12", "q2", "e4", "e5", "q1", "strategic", "blocks"])
    ap.add_argument("--topology", default="nsfnet")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    ap.add_argument("--engines", default=None,
                    help="comma list of ns3 (QKDNetSim pools), ns3relay (pools + share relay), fluid")
    args = ap.parse_args()
    if args.exp == "e12":
        plan = e12_plan(args.topology, args.seeds)
    elif args.exp == "q2":
        plan = q2_plan(args.topology, args.seeds)
    elif args.exp == "blocks":
        plan = blocks_plan(args.topology, args.seeds)
    elif args.exp == "strategic":
        plan = strategic_plan(args.topology, args.seeds)
    else:
        plan = campaign_plan(args.exp.upper(), args.topology, args.seeds)
    engines = args.engines.split(",") if args.engines else (["ns3"] if args.exp == "blocks" else ENGINES)
    run_plan(f"{args.exp} {args.topology}", plan, args.topology, args.jobs, engines)


if __name__ == "__main__":
    main()
