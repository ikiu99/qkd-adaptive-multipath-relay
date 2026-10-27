"""Proposed method C1: pool-aware adaptive (t, n) selection (README 4.1).

For each request, builds candidate paths, drops any whose pool is below the
reserve fraction, enumerates (t, n) options over bounded path subsets,
keeps the ones meeting the security/availability targets, and picks the
cheapest by pool-weighted cost.

Pool levels are read from the graph's own `level_bits` / `capacity_bits`
edge attributes (set once, statically, by `qkdmp.topology.apply_rate_model`
for Level A; Level B, Phase 4, will mutate them over time on the same graph
object -- this policy doesn't know or care which level is calling it,
README 6).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np

from qkdmp import security
from qkdmp.paths import candidate_paths, path_edges
from qkdmp.policies.base import BLOCK, Block, Decision, Request

COST_FLOOR_BITS = 1.0  # delta in cost = sum |K| / max(B_e, delta), avoids div-by-zero


def _pool_ok(path: list, graph: Any, reserve_fraction: float, min_bits: float = 0.0) -> bool:
    """README 4.1 step 1 (reserve_fraction guard). `min_bits`: also require
    one share's worth of key material on every link -- without it, a path on
    a low-capacity link can pass the fractional guard yet be unable to carry
    the share, which the Phase 7 RL diagnosis found Adaptive choosing 63
    times in 250 NSFNET requests (each one a post-decision failure)."""
    for u, v in path_edges(path):
        data = graph.edges[u, v]
        if data["level_bits"] < max(reserve_fraction * data["capacity_bits"], min_bits):
            return False
    return True


def link_ok(data: dict, reserve_fraction: float, min_bits: float = 0.0) -> bool:
    """Per-link version of `_pool_ok`."""
    return data["level_bits"] >= max(reserve_fraction * data["capacity_bits"], min_bits)


def pool_candidate_paths(graph: Any, s, d, n_max: int, reserve_fraction: float, min_bits: float = 0.0) -> list[list]:
    """Candidate paths searched on the graph with every link failing the
    pool guard removed, so every returned path passes `_pool_ok`.

    Searching first and filtering afterwards (the original Phase 3 order)
    finds only the n_max shortest paths of the *full* graph and blocks when
    all of them cross a depleted link, even if a longer route has key to
    spare -- the fair-comparison follow-up found pool-pruned baselines
    (`qkdmp.policies.admission`, tier "PA") serving far more than Adaptive
    under overload for exactly this reason.
    """
    view = nx.subgraph_view(graph, filter_edge=lambda u, v: link_ok(graph.edges[u, v], reserve_fraction, min_bits))
    return candidate_paths(view, s, d, n_max)


def _pool_weighted_cost(paths: list[list], graph: Any, key_size_bits: int) -> float:
    cost = 0.0
    for path in paths:
        for u, v in path_edges(path):
            level = graph.edges[u, v]["level_bits"]
            cost += key_size_bits / max(level, COST_FLOOR_BITS)
    return cost


def _hop_count_cost(paths: list[list], key_size_bits: int) -> float:
    """E8 ablation ("without pool-awareness", README 9): cost that ignores
    pool state entirely, counting only bits consumed (hops * |K|)."""
    return sum(len(path_edges(path)) * key_size_bits for path in paths)


def search_options(
    candidates: list[list],
    graph: Any,
    p_v: dict,
    request: Request,
    key_size_bits: int,
    p_drop: float,
    p_tamper: float,
    rng: np.random.Generator,
    max_subsets: int | None = None,
    pool_aware: bool = True,
):
    """Enumerate (subset, t) options over `candidates` (README 4.1 step 2-3).

    Returns (feasible, all_tried): each a list of dicts with keys
    paths/t/n/p_exp/p_del/cost. `feasible` holds only options meeting
    request.eps_target/a_target; `all_tried` holds every option evaluated
    (used by the `best_effort` fallback).
    """
    feasible = []
    all_tried = []
    budget = max_subsets if max_subsets is not None else float("inf")

    for n in range(1, len(candidates) + 1):
        for subset in combinations(candidates, n):
            if budget <= 0:
                break
            budget -= 1
            paths = list(subset)
            # Every threshold t of one subset shares the same distributions:
            # compute them once (exact; the Monte Carlo case is left per-t so
            # its random draws are unchanged).
            exp_tail = security.exposure_tail_exact(paths, p_v)
            survival = [security.path_survival_prob(path, p_v, p_drop, p_tamper) for path in paths]
            del_tail = np.cumsum(security.poisson_binomial_pmf(survival)[::-1])[::-1]
            cost = (
                _pool_weighted_cost(paths, graph, key_size_bits)
                if pool_aware else _hop_count_cost(paths, key_size_bits)
            )
            for t in range(1, n + 1):
                p_exp = float(exp_tail[t]) if exp_tail is not None else security.exposure_prob(paths, p_v, t, rng=rng)
                p_del = float(del_tail[t])
                option = {"paths": list(subset), "t": t, "n": n, "p_exp": p_exp, "p_del": p_del, "cost": cost}
                all_tried.append(option)
                if p_exp <= request.eps_target and p_del >= request.a_target:
                    feasible.append(option)
        if budget <= 0:
            break

    return feasible, all_tried


def option_reservable(option: dict, graph: Any, key_size_bits: float) -> bool:
    """Every link can supply one share per path crossing it (overlapping
    Yen paths may share a link and then need a multiple of |K| there)."""
    need: dict = {}
    for path in option["paths"]:
        for u, v in path_edges(path):
            edge = frozenset((u, v))
            need[edge] = need.get(edge, 0.0) + key_size_bits
    return all(graph.edges[tuple(e)]["level_bits"] >= bits for e, bits in need.items())


class AdaptiveThresholdPolicy:
    name = "Adaptive"

    def __init__(self, fallback: str = "block", seed: int = 0, pool_aware: bool = True) -> None:
        """`pool_aware=False` is the E8 ablation (README 9: "Proposed without
        pool-awareness"): skips the reserve_fraction pool filter and uses
        plain hop-count cost instead of pool-weighted cost."""
        if fallback not in ("block", "best_effort"):
            raise ValueError(f"unknown fallback: {fallback!r}")
        self.fallback = fallback
        self.pool_aware = pool_aware
        self._rng = np.random.default_rng(seed)

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        paths_cfg = config["paths"]
        attacker_cfg = config["attacker"]
        key_size_bits = config["key_size_bits"]

        if self.pool_aware:
            candidates = pool_candidate_paths(
                graph, request.s, request.d, paths_cfg["n_max"], paths_cfg["reserve_fraction"], key_size_bits
            )
        else:
            candidates = candidate_paths(graph, request.s, request.d, paths_cfg["n_max"])
        if not candidates:
            return BLOCK

        feasible, all_tried = search_options(
            candidates, graph, p_v, request, key_size_bits,
            attacker_cfg["p_drop"], attacker_cfg["p_tamper"], self._rng,
            max_subsets=paths_cfg["max_subsets"], pool_aware=self.pool_aware,
        )
        if self.pool_aware:
            feasible = [o for o in feasible if option_reservable(o, graph, key_size_bits)]
            all_tried = [o for o in all_tried if option_reservable(o, graph, key_size_bits)]

        if feasible:
            best = min(feasible, key=lambda o: o["cost"])
        elif self.fallback == "best_effort" and all_tried:
            best = min(all_tried, key=lambda o: o["p_exp"])
        else:
            return BLOCK

        return Decision(
            paths=best["paths"], t=best["t"], n=best["n"], scheme="shamir",
            p_exp=best["p_exp"], p_del=best["p_del"], cost=best["cost"],
        )
