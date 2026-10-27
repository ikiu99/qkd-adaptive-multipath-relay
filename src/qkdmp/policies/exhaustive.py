"""Per-request exact optimum (README 4.2): exhaustive enumeration of all path
subsets and (t, n) over a larger candidate pool than the adaptive heuristic
uses, with no subset-count cap. Used to measure the heuristic's optimality
gap (E9, Phase 5) -- small topologies only, since the candidate pool and
subset enumeration are both larger/uncapped here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qkdmp.policies.adaptive import option_reservable, pool_candidate_paths, search_options
from qkdmp.policies.base import BLOCK, Block, Decision, Request

# ASSUMPTION: README doesn't say how much larger the exhaustive candidate
# pool should be than the heuristic's N_max; +4 gives a materially larger
# search space (more (t, n, subset) combinations to potentially beat the
# heuristic's choice) while staying tractable on the topologies in scope.
EXTRA_CANDIDATES = 4


class ExhaustivePolicy:
    name = "Exhaustive"

    def __init__(self, fallback: str = "block", seed: int = 0, extra_candidates: int = EXTRA_CANDIDATES) -> None:
        if fallback not in ("block", "best_effort"):
            raise ValueError(f"unknown fallback: {fallback!r}")
        self.fallback = fallback
        self.extra_candidates = extra_candidates
        self._rng = np.random.default_rng(seed)

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        paths_cfg = config["paths"]
        attacker_cfg = config["attacker"]
        key_size_bits = config["key_size_bits"]

        n_max = paths_cfg["n_max"] + self.extra_candidates
        candidates = pool_candidate_paths(
            graph, request.s, request.d, n_max, paths_cfg["reserve_fraction"], key_size_bits
        )
        if not candidates:
            return BLOCK

        feasible, all_tried = search_options(
            candidates, graph, p_v, request, key_size_bits,
            attacker_cfg["p_drop"], attacker_cfg["p_tamper"], self._rng,
            max_subsets=None,  # no cap: true exhaustive search
        )
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
