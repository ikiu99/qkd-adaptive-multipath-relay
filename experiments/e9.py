"""E9 -- Optimality & scalability (README Section 9): heuristic vs exhaustive
optimum vs batch ILP vs RL, objective gap and decision time, topology size
up to random50.

# ASSUMPTION / scope reduction: README lists this experiment at "A + B"
# level; this implementation is Level A only (a Level B version would add a
# full DES run per method per topology on top of RL training, which does
# not fit this session's time budget). Level B follow-up is noted in the
# phase report.

Online methods (Adaptive, Exhaustive, RL) process the same batch of requests
*sequentially* against a shared starting pool, each irrevocably consuming
pool as it goes -- no lookahead. Batch ILP sees the whole window's pool
budget and request set at once and picks the jointly-optimal subset
(README 4.2: "an offline upper bound, not an online method"), so its
objective is the ceiling the online methods are compared against.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from qkdmp.paths import path_edges
from qkdmp.policies.adaptive import AdaptiveThresholdPolicy
from qkdmp.policies.base import BLOCK, Request
from qkdmp.policies.batch_ilp import solve_batch_ilp
from qkdmp.policies.exhaustive import ExhaustivePolicy
from qkdmp.policies.rl import LEGACY_DESIGN, RLEnvDesign, train_rl_policy
from qkdmp.sim.graph_mc import place_attacker, sample_request_pair
from qkdmp.suspicion import detector_p_v
from qkdmp.topology import build_topology

RESULTS_DIR = Path("results") / "E9"
FIGURES_DIR = Path("figures") / "E9"

LAMBDA_COST = 1e-6
WINDOW_SECONDS = 3600.0


def _run_online_method(policy, graph, requests, p_v, config) -> dict:
    key_size_bits = config["key_size_bits"]
    served, total_cost, times = 0, 0.0, []
    for req in requests:
        t0 = time.perf_counter()
        decision = policy.decide(req, graph, p_v, config)
        times.append(time.perf_counter() - t0)
        if decision is BLOCK:
            continue
        if any(graph.edges[u, v]["level_bits"] < key_size_bits for p in decision.paths for u, v in path_edges(p)):
            continue  # pool consumed by an earlier request in this sequence
        for path in decision.paths:
            for u, v in path_edges(path):
                graph.edges[u, v]["level_bits"] -= key_size_bits
        served += 1
        total_cost += decision.cost or 0.0
    return {
        "served": served,
        "cost": total_cost,
        "objective": served - LAMBDA_COST * total_cost,
        "mean_decision_time_s": float(np.mean(times)) if times else 0.0,
    }


def run_e9(
    config: dict,
    topologies: list[str],
    n_requests_per_batch: int,
    rl_timesteps: int,
    seed: int = 0,
    rl_ent_coef: float = 0.0,
    rl_design: RLEnvDesign = RLEnvDesign(),
) -> pd.DataFrame:
    rows = []
    training_curves: dict[str, list[float]] = {}
    for topo_name in topologies:
        graph_base = build_topology(topo_name, config)
        rng = np.random.default_rng(seed)
        nodes = list(graph_base.nodes())

        attacker_set = place_attacker(graph_base, config["attacker"], rng)
        p_v = detector_p_v(nodes, attacker_set, config, rng)

        requests = []
        for _ in range(n_requests_per_batch):
            s, d = sample_request_pair(nodes, attacker_set, rng)
            requests.append(Request(s, d, config["policy"]["eps_target"], config["policy"]["a_target"]))

        results: dict[str, Any] = {}

        adaptive = AdaptiveThresholdPolicy(fallback=config["policy"]["fallback"], seed=seed)
        results["Adaptive"] = _run_online_method(adaptive, graph_base.copy(), requests, p_v, config)

        exhaustive = ExhaustivePolicy(fallback=config["policy"]["fallback"], seed=seed)
        results["Exhaustive"] = _run_online_method(exhaustive, graph_base.copy(), requests, p_v, config)

        graph_ilp = graph_base.copy()
        t0 = time.perf_counter()
        ilp_result = solve_batch_ilp(requests, graph_ilp, p_v, config, window_seconds=WINDOW_SECONDS,
                                      lambda_cost=LAMBDA_COST, rng=rng)
        ilp_time = time.perf_counter() - t0
        ilp_served = sum(1 for d in ilp_result.decisions if d is not None)
        ilp_cost = sum(d.cost or 0.0 for d in ilp_result.decisions if d is not None)
        results["BatchILP"] = {
            "served": ilp_served, "cost": ilp_cost, "objective": ilp_result.objective,
            "mean_decision_time_s": ilp_time / max(1, len(requests)),
        }

        rl_policy, rl_training_rewards = train_rl_policy(
            graph_base.copy(), config, p_v, total_timesteps=rl_timesteps, seed=seed, ent_coef=rl_ent_coef,
            design=rl_design,
            # Train on episodes shaped like the evaluation batch; the legacy
            # design keeps the Phase 5 settings so it stays reproducible.
            episode_requests=100 if rl_design == LEGACY_DESIGN else n_requests_per_batch,
            endpoint_exclude=frozenset() if rl_design == LEGACY_DESIGN else frozenset(attacker_set),
        )
        results["RL"] = _run_online_method(rl_policy, graph_base.copy(), requests, p_v, config)
        training_curves[topo_name] = rl_training_rewards

        ilp_objective = results["BatchILP"]["objective"]
        for method_name, r in results.items():
            gap = (ilp_objective - r["objective"]) / ilp_objective if ilp_objective > 0 else None
            rows.append({"topology": topo_name, "method": method_name, "n_requests": n_requests_per_batch,
                         **r, "optimality_gap": gap})

    return pd.DataFrame(rows), training_curves


def plot_training_curves(training_curves: dict[str, list[float]], out_path: Path) -> None:
    """README Phase 5 acceptance: "RL training curve"."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for topo_name, rewards in training_curves.items():
        if not rewards:
            continue
        ax.plot(range(1, len(rewards) + 1), rewards, marker=".", label=topo_name)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Training service ratio (served / requests)")
    ax.set_title("E9: MaskablePPO training curve")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
