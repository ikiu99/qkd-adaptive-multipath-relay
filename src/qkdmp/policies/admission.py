"""Pool-checked baseline variants (fair-comparison follow-up to Phase 7).

A reviewer's obvious objection to Adaptive's low pool-exhaustion failure
rate is that any baseline could get the same by simply checking the pool
before sending. This wraps a baseline in one of two tiers of that check,
so the comparison isolates what the check alone buys vs what Adaptive's
pool-weighted (t, n) selection adds on top:

- "AC" (admission control): the baseline chooses its paths exactly as
  before; the request is refused up front if any chosen link is below the
  reserve guard or cannot supply the shares crossing it -- the same test
  Adaptive applies (`_pool_ok` + `option_reservable`).
- "PA" (pool-aware routing): the baseline chooses its paths on the graph
  with every link that fails that per-link test removed, so it routes
  *around* depleted links instead of only refusing; the final option is
  still checked like "AC" (overlapping paths can need more than one share
  on a link).

# ASSUMPTION: "PA" is implemented as depleted-link pruning, the usual way
# pool state enters QKD routing, rather than a per-baseline headroom
# re-ranking -- it applies uniformly to B0-B3 without changing any of
# their own path-selection rules.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from qkdmp.policies.adaptive import _pool_ok, link_ok, option_reservable
from qkdmp.policies.base import BLOCK, Block, Decision, Request

TIERS = ("AC", "PA")


class AdmissionControlled:
    def __init__(self, base: Any, tier: str) -> None:
        if tier not in TIERS:
            raise ValueError(f"unknown admission tier: {tier!r}")
        self.base = base
        self.tier = tier
        self.name = f"{base.name}+{tier}"

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        key_size_bits = config["key_size_bits"]
        reserve_fraction = config["paths"]["reserve_fraction"]

        view = graph
        if self.tier == "PA":
            view = nx.subgraph_view(
                graph, filter_edge=lambda u, v: link_ok(graph.edges[u, v], reserve_fraction, key_size_bits)
            )
            if not nx.has_path(view, request.s, request.d):
                return BLOCK

        decision = self.base.decide(request, view, p_v, config)
        if decision is BLOCK:
            return BLOCK
        if not all(_pool_ok(p, graph, reserve_fraction, key_size_bits) for p in decision.paths):
            return BLOCK
        if not option_reservable({"paths": decision.paths}, graph, key_size_bits):
            return BLOCK
        return decision
