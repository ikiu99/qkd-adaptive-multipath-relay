"""Attacker model (README 3.1): placement strategies and per-share drop/tamper behaviour.

Placement functions return a `set` of compromised nodes, sized either by an
explicit `count` or by `rho_A * |V \\ exclude|` (rounded). `exclude` is meant
for a request's own (s, d) (README 3.1: "never the source or destination of
a request"); for a network-wide ground-truth attacker set, leave it empty.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _target_count(candidates: list, rho_A: float | None, count: int | None) -> int:
    if count is not None:
        return min(count, len(candidates))
    if rho_A is None:
        raise ValueError("one of rho_A or count must be given")
    return min(len(candidates), round(rho_A * len(candidates)))


def place_random(
    graph: nx.Graph,
    rng: np.random.Generator,
    rho_A: float | None = None,
    count: int | None = None,
    exclude: frozenset = frozenset(),
    mode: str = "fixed_count",
) -> set:
    """README 3.1 strategy 1: "each relay node is compromised independently,
    or a fraction rho_A of nodes is chosen uniformly" -- both readings are
    supported via `mode`:
      - "fixed_count" (default): exactly round(rho_A * |candidates|) nodes,
        chosen uniformly without replacement. Lower variance; what E3/E10
        sweeps use so rho_A maps to an exact, reproducible attacker count.
      - "bernoulli": each candidate is independently included with
        probability rho_A. This is the model `exposure_prob_disjoint` /
        E2(a)'s analytic bound assume (iid per-node compromise), so use this
        mode when comparing simulation output to that closed form.
    `count` (an explicit target size) always implies "fixed_count" behaviour.
    """
    candidates = sorted((v for v in graph.nodes() if v not in exclude), key=str)
    if mode == "bernoulli" and count is None:
        if rho_A is None:
            raise ValueError("rho_A is required for mode='bernoulli'")
        draws = rng.random(len(candidates)) < rho_A
        return {v for v, drawn in zip(candidates, draws) if drawn}
    if mode not in ("fixed_count", "bernoulli"):
        raise ValueError(f"unknown mode: {mode!r}")

    k = _target_count(candidates, rho_A, count)
    if k == 0:
        return set()
    chosen = rng.choice(np.array(candidates, dtype=object), size=k, replace=False)
    return set(chosen.tolist())


def place_betweenness(
    graph: nx.Graph,
    rho_A: float | None = None,
    count: int | None = None,
    exclude: frozenset = frozenset(),
) -> set:
    """Top-rho_A nodes by betweenness centrality (README 3.1 strategy 2)."""
    centrality = nx.betweenness_centrality(graph, weight="length_km_norm")
    candidates = sorted((v for v in graph.nodes() if v not in exclude), key=str)
    k = _target_count(candidates, rho_A, count)
    ranked = sorted(candidates, key=lambda v: (-centrality[v], str(v)))
    return set(ranked[:k])


def place_key_throughput(
    graph: nx.Graph,
    rho_A: float | None = None,
    count: int | None = None,
    exclude: frozenset = frozenset(),
) -> set:
    """Top-rho_A nodes by expected relayed key volume (README 3.1 strategy 3).

    # ASSUMPTION: [REF-KEYRATE] (the user's prior work establishing key rate,
    # not node degree, as the risk proxy) is not available in this session.
    # As a stand-in, "expected relayed key volume" is approximated as
    # betweenness centrality (a load proxy) weighted by the node's mean
    # incident link rate `rate_bps` (requires the graph to have gone through
    # qkdmp.topology.apply_rate_model first). Revisit once the reference is
    # supplied.
    """
    centrality = nx.betweenness_centrality(graph, weight="length_km_norm")
    scores = {}
    for v in graph.nodes():
        incident_rates = [graph.edges[v, u]["rate_bps"] for u in graph.neighbors(v)]
        mean_rate = float(np.mean(incident_rates)) if incident_rates else 0.0
        scores[v] = centrality[v] * mean_rate

    candidates = sorted((v for v in graph.nodes() if v not in exclude), key=str)
    k = _target_count(candidates, rho_A, count)
    ranked = sorted(candidates, key=lambda v: (-scores[v], str(v)))
    return set(ranked[:k])


def place_adaptive(
    graph: nx.Graph,
    path_frequency: dict,
    rho_A: float | None = None,
    count: int | None = None,
    exclude: frozenset = frozenset(),
) -> set:
    """Top-rho_A nodes by how often they appeared in chosen path sets during a
    warm-up period (README 3.1 strategy 4). `path_frequency` is produced by
    the Level A/B simulators (counts of relay-node appearances so far);
    nodes absent from it are treated as frequency 0.
    """
    candidates = sorted((v for v in graph.nodes() if v not in exclude), key=str)
    k = _target_count(candidates, rho_A, count)
    ranked = sorted(candidates, key=lambda v: (-path_frequency.get(v, 0), str(v)))
    return set(ranked[:k])


PLACEMENT_STRATEGIES = {
    "random": place_random,
    "betweenness": place_betweenness,
    "key_throughput": place_key_throughput,
    "adaptive": place_adaptive,
}


# ---------------------------------------------------------------------------
# Per-share drop/tamper behaviour (README 3.1)
# ---------------------------------------------------------------------------

def share_outcome(rng: np.random.Generator, is_compromised: bool, p_drop: float, p_tamper: float) -> str:
    """Outcome of one share transiting one node: "delivered", "dropped" or
    "tampered". Honest nodes always deliver. A compromised node drops with
    probability p_drop; otherwise it tampers with probability p_tamper
    (drop takes precedence -- a dropped share is never also "tampered").
    """
    if not is_compromised:
        return "delivered"
    if rng.random() < p_drop:
        return "dropped"
    if rng.random() < p_tamper:
        return "tampered"
    return "delivered"


def path_outcome(
    rng: np.random.Generator, path_relay_nodes: list, compromised: set, p_drop: float, p_tamper: float
) -> str:
    """Worst outcome of a share traversing every relay node on its path:
    "dropped" if any node drops it, else "tampered" if any node tampers with
    it, else "delivered"."""
    tampered = False
    for v in path_relay_nodes:
        outcome = share_outcome(rng, v in compromised, p_drop, p_tamper)
        if outcome == "dropped":
            return "dropped"
        if outcome == "tampered":
            tampered = True
    return "tampered" if tampered else "delivered"
