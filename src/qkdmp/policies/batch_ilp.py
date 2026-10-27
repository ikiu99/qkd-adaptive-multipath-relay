"""Batch ILP (pool-coupled offline optimum), README 4.2.

Not a per-request `Policy` (README 11's `decide` interface): this jointly
optimizes a whole *batch* of requests against one shared pool budget over a
time window W, which a single-request `decide` call cannot express. It is an
offline upper bound, not an online method (README 4.2).

Solved with HiGHS via `scipy.optimize.milp` (scipy's MILP backend IS HiGHS),
for consistency with `qkdmp.security.m_star_ilp`'s use of the same interface,
rather than calling the `highspy` package directly -- README names either
acceptable ("HiGHS (via highspy or PuLP)").
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, milp

from qkdmp.paths import path_edges
from qkdmp.policies.adaptive import pool_candidate_paths, search_options
from qkdmp.policies.base import Decision, Request


@dataclass
class BatchILPResult:
    decisions: list[Decision | None]  # one per request, None if not served
    objective: float
    solve_time_s: float
    n_vars: int


def solve_batch_ilp(
    requests: list[Request],
    graph: Any,
    p_v: dict,
    config: dict,
    window_seconds: float,
    lambda_cost: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> BatchILPResult:
    """README 4.2: binary x_{r,o} per (request, feasible option); constraints
    `sum_o x_{r,o} <= 1` per request and `sum_{r,o} consumption_e * x_{r,o}
    <= B_e(start of W) + R_e * W` per link; objective maximize served minus
    lambda * total cost.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    paths_cfg = config["paths"]
    attacker_cfg = config["attacker"]
    key_size_bits = config["key_size_bits"]

    per_request_options: list[list[dict]] = []
    for req in requests:
        # Pool-pruned candidate search like Adaptive's (without the one-share
        # floor: the ILP's budget includes refill over the window), but over a
        # larger candidate set: Adaptive re-searches on the *depleted* graph at
        # every decision and so reaches routes outside any fixed start-of-batch
        # candidate set -- with only n_max=4 the "ceiling" served fewer
        # requests than Adaptive (calibration round). `ilp_n_max` /
        # `ilp_max_subsets` restore it as a ceiling in practice; it is exact
        # only relative to its own candidate set.
        candidates = pool_candidate_paths(
            graph, req.s, req.d, paths_cfg.get("ilp_n_max", paths_cfg["n_max"]), paths_cfg["reserve_fraction"]
        )
        if not candidates:
            per_request_options.append([])
            continue
        feasible, _all_tried = search_options(
            candidates, graph, p_v, req, key_size_bits,
            attacker_cfg["p_drop"], attacker_cfg["p_tamper"], rng,
            max_subsets=paths_cfg.get("ilp_max_subsets", paths_cfg["max_subsets"]),
        )
        per_request_options.append(feasible)

    var_request_idx: list[int] = []
    var_option: list[dict] = []
    for r, options in enumerate(per_request_options):
        for opt in options:
            var_request_idx.append(r)
            var_option.append(opt)
    n_vars = len(var_option)

    if n_vars == 0:
        return BatchILPResult([None] * len(requests), 0.0, 0.0, 0)

    edges = list(graph.edges())
    edge_index = {frozenset(e): i for i, e in enumerate(edges)}
    consumption = np.zeros((len(edges), n_vars))
    for j, opt in enumerate(var_option):
        for path in opt["paths"]:
            for u, v in path_edges(path):
                consumption[edge_index[frozenset((u, v))], j] += key_size_bits

    budgets = np.array(
        [
            min(graph.edges[u, v]["capacity_bits"], graph.edges[u, v]["level_bits"] + graph.edges[u, v]["rate_bps"] * window_seconds)
            for u, v in edges
        ]
    )
    pool_constraint = LinearConstraint(consumption, -np.inf, budgets)

    request_rows = []
    for r, options in enumerate(per_request_options):
        if not options:
            continue
        row = np.array([1.0 if ridx == r else 0.0 for ridx in var_request_idx])
        request_rows.append(row)
    constraints = [pool_constraint]
    if request_rows:
        constraints.append(LinearConstraint(np.array(request_rows), -np.inf, 1.0))

    c = np.array([-1.0 + lambda_cost * opt["cost"] for opt in var_option])
    integrality = np.ones(n_vars)
    bounds = (np.zeros(n_vars), np.ones(n_vars))

    start = time.perf_counter()
    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    solve_time = time.perf_counter() - start

    decisions: list[Decision | None] = [None] * len(requests)
    objective = 0.0
    if result.success:
        for j, val in enumerate(result.x):
            if val > 0.5:
                r = var_request_idx[j]
                opt = var_option[j]
                decisions[r] = Decision(
                    paths=opt["paths"], t=opt["t"], n=opt["n"], scheme="shamir",
                    p_exp=opt["p_exp"], p_del=opt["p_del"], cost=opt["cost"],
                )
        objective = -result.fun

    return BatchILPResult(decisions, objective, solve_time, n_vars)
