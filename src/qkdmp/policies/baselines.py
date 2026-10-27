"""Baselines B0-B3 (README Section 5). All get the same inputs (graph, p_v,
config); none of them look at live pool levels ("pool-unaware").
"""

from __future__ import annotations

import math
from itertools import islice
from typing import Any

import networkx as nx

from qkdmp.paths import node_disjoint_shortest_paths, path_edges, relay_nodes
from qkdmp.policies.base import BLOCK, Block, Decision, Request


class B0Policy:
    """Single-path relay (standard trusted-relay OTP), README Section 5.

    `variant`: "hopcount" (default, unweighted shortest path) or "key_rate"
    (shortest path weighted by 1/R_e, favoring higher-rate links).
    """

    name = "B0"

    def __init__(self, variant: str = "hopcount") -> None:
        if variant not in ("hopcount", "key_rate"):
            raise ValueError(f"unknown B0 variant: {variant!r}")
        self.variant = variant

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        weight = None if self.variant == "hopcount" else (lambda u, v, d: 1.0 / d["rate_bps"])
        try:
            path = nx.shortest_path(graph, request.s, request.d, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return BLOCK
        return Decision(paths=[path], t=1, n=1, scheme="single")


class B1Policy:
    """XOR n-of-n MNOP (README Section 5): fixed n node-disjoint shortest
    paths (default n=2, or the max available <= N_max if fewer than n exist),
    XOR split, pool-unaware."""

    name = "B1"

    def __init__(self, n: int = 2, n_max: int | None = None) -> None:
        self.n = n
        self.n_max = n_max if n_max is not None else n

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        paths = node_disjoint_shortest_paths(graph, request.s, request.d, n_max=self.n_max)
        if not paths:
            return BLOCK
        actual_n = len(paths)
        return Decision(paths=paths, t=actual_n, n=actual_n, scheme="xor")


class B2Policy:
    """Fixed (t, n) Shamir (README Section 5): default (2, 3), disjoint
    shortest paths, pool- and suspicion-unaware."""

    name = "B2"

    def __init__(self, t: int = 2, n: int = 3) -> None:
        if not (1 <= t <= n):
            raise ValueError(f"require 1 <= t <= n, got t={t}, n={n}")
        self.t = t
        self.n = n

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        paths = node_disjoint_shortest_paths(graph, request.s, request.d, n_max=self.n)
        if not paths:
            return BLOCK
        actual_n = len(paths)
        actual_t = min(self.t, actual_n)
        return Decision(paths=paths, t=actual_t, n=actual_n, scheme="shamir")


class B3Policy:
    """Penalty-based multipath moving-target-defense (README Section 5),
    reimplementation of Ghourab, Azab & Gracanin, "A quantum key distribution
    routing scheme for a zero-trust QKD network system: A moving target
    defense approach", BDCC 9(4):76, 2025 (full text supplied by the user;
    equation numbers below refer to that PDF).

    Equations implemented as published:
      - Eq. (4): P(S_i) = prod_j max(0, p_j - delta_j(t)), where p_j is the
        paper's *security* probability of node j (P(node j not compromised)
        -- the complement of this codebase's p_v compromise probability:
        `security_j = 1 - p_v[j]`), and delta_j(t) = W_j / t^k (Eq. before
        Eq. 4's statement, 0 <= delta_j(t) < p_j).
      - Eq. (1): Cor_{S(P)} = sum_L(freq_L - 1) / sum_{P_i} card(links(P_i)),
        applied pairwise (S(P) = {P_i, P_j}): for two paths this reduces to
        (# shared links) / (total link count of both paths).
      - Eq. (2): TH_Corr = lambda_Cor * mean(Cor) over all candidate pairs.
      - Eq. (8): W_j = W_j^(0) + Delta_W if the node is within Manhattan
        distance d_max of a compromise detected by `record_detection`
        (README 5.3: a 2D-mesh-specific mechanism -- only meaningful on
        topologies with (x, y) node coordinates, e.g. `grid`; on other
        topologies the penalty applies to the exact detected node only).
      - Section 3 intro ("Eve would need to intercept all key segments"):
        implemented as t = n (all n selected paths required to reconstruct),
        i.e. an n-of-n scheme like B1, not a general (t, n) threshold -- the
        paper is not explicit about a threshold variant.

    # ASSUMPTION: k (the decay exponent in delta_j(t) = W_j / t^k) is not
    # given a value in the paper; config default 1.0.
    # ASSUMPTION: "t" in delta_j(t) is treated as a per-policy-instance call
    # counter (`self._t`, starts at 1 and increments every `decide` call),
    # standing in for the paper's wall-clock/round time axis, since Level A
    # has no time dimension (README 6) to draw a real "t" from.
    # ASSUMPTION: candidate generation still uses length-weighted Yen KSP
    # (README 4.2.1 mentions "dynamic link weighting" and an "ExDijkstra"
    # variant of Dijkstra using the multiplicative security metric as its
    # non-additive cost, Eq. 3, but does not give enough detail to
    # reconstruct ExDijkstra exactly) -- length-weighted KSP generates a
    # diverse candidate *pool*, which is then ranked/filtered by the
    # security (Eq. 4) and correlation (Eq. 1-2) criteria the paper does
    # specify precisely, so the physical path-finding step is the only part
    # still using our own proxy.
    # ASSUMPTION: candidates are admitted into the selected set greedily,
    # most-secure-by-Eq.4 first, keeping only those whose correlation with
    # every already-selected path is <= TH_Corr (Algorithm 1's pseudocode
    # is a generic "attempt to add if correlation < threshold and it
    # improves overall security" loop without a fully specified tie-break
    # order; this is one reasonable resolution of it, not a literal
    # transcription of a numbered procedure).
    """

    name = "B3"

    def __init__(
        self,
        n: int = 3,
        k_candidates: int = 8,
        lambda_cor: float = 1.0,
        delta_w: float = 1.0,
        d_max: float = 0.0,
        temporal_k: float = 1.0,
    ) -> None:
        self.n = n
        self.k_candidates = k_candidates
        self.lambda_cor = lambda_cor
        self.delta_w = delta_w
        self.d_max = d_max
        self.temporal_k = temporal_k
        self._penalty: dict = {}
        self._t = 1

    def record_detection(self, nodes: list, graph: Any | None = None) -> None:
        """Eq. (8): penalize the detected node(s) and, on a topology with
        (x, y) coordinates, any node within Manhattan distance d_max of one."""
        targets = set(nodes)
        if graph is not None and self.d_max > 0:
            coords = {v: (graph.nodes[v]["x"], graph.nodes[v]["y"]) for v in graph.nodes()
                      if "x" in graph.nodes[v] and "y" in graph.nodes[v]}
            for detected in nodes:
                if detected not in coords:
                    continue
                dx, dy = coords[detected]
                for v, (vx, vy) in coords.items():
                    if abs(vx - dx) + abs(vy - dy) <= self.d_max:
                        targets.add(v)
        for v in targets:
            self._penalty[v] = self._penalty.get(v, 0.0) + self.delta_w

    def _delta(self, v) -> float:
        w = self._penalty.get(v, 0.0)
        if w == 0.0:
            return 0.0
        return w / (self._t**self.temporal_k)

    def _path_security(self, path: list, p_v: dict) -> float:
        """Eq. (4): P(S_i) = prod_j max(0, security_j - delta_j(t))."""
        security = 1.0
        for v in relay_nodes(path):
            security_j = 1.0 - p_v.get(v, 0.0)
            security *= max(0.0, security_j - self._delta(v))
        return security

    @staticmethod
    def _correlation(edges_i: set, edges_j: set) -> float:
        """Eq. (1) applied to a pair of paths: shared links / total link count."""
        total = len(edges_i) + len(edges_j)
        if total == 0:
            return 0.0
        shared = len(edges_i & edges_j)
        return shared / total

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        try:
            generator = nx.shortest_simple_paths(graph, request.s, request.d, weight="length_km_norm")
            candidates = list(islice(generator, self.k_candidates))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return BLOCK
        if not candidates:
            return BLOCK
        self._t += 1

        edge_sets = [{frozenset(e) for e in path_edges(p)} for p in candidates]
        pairwise = [
            self._correlation(edge_sets[i], edge_sets[j])
            for i in range(len(candidates))
            for j in range(i + 1, len(candidates))
        ]
        mean_corr = sum(pairwise) / len(pairwise) if pairwise else 0.0
        th_corr = self.lambda_cor * mean_corr

        # Eq. (4): most secure (highest P(S_i)) first.
        ranked = sorted(range(len(candidates)), key=lambda i: -self._path_security(candidates[i], p_v))

        selected: list[int] = []
        for i in ranked:
            if len(selected) >= self.n:
                break
            if all(self._correlation(edge_sets[i], edge_sets[j]) <= th_corr for j in selected):
                selected.append(i)
        if len(selected) < self.n:
            # correlation filter left too few candidates: fill with the next
            # most-secure remaining ones regardless of correlation.
            for i in ranked:
                if len(selected) >= self.n:
                    break
                if i not in selected:
                    selected.append(i)

        if not selected:
            return BLOCK
        paths = [candidates[i] for i in selected]
        n = len(paths)
        return Decision(paths=paths, t=n, n=n, scheme="shamir")


def risk_weight(p_v: dict, s, d, hop_epsilon: float = 1e-6):
    """Edge weight whose path sum is the relay risk sum_{v relay} -ln(1 - p_v)
    (plus a constant for the endpoints and a tiny per-hop term that breaks
    ties towards shorter paths). Minimising it maximises the probability
    that no relay on the path is compromised, prod (1 - p_v)."""
    def node_cost(v) -> float:
        if v in (s, d):
            return 0.0
        return -math.log(max(1e-12, 1.0 - p_v.get(v, 0.0)))

    return lambda u, v, data: 0.5 * (node_cost(u) + node_cost(v)) + hop_epsilon


class B0RPolicy(B0Policy):
    """Reviewer-requested baseline: single path that minimises relay risk
    (the detector-aware version of B0). Same detector input as Adaptive."""

    name = "B0R"

    def __init__(self) -> None:
        super().__init__("hopcount")

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        try:
            path = nx.shortest_path(graph, request.s, request.d, weight=risk_weight(p_v, request.s, request.d))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return BLOCK
        return Decision(paths=[path], t=1, n=1, scheme="single")


class B1RPolicy(B1Policy):
    """B1 (XOR over node-disjoint paths) with paths chosen by least relay risk."""

    name = "B1R"

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        paths = node_disjoint_shortest_paths(graph, request.s, request.d, n_max=self.n_max,
                                             weight=risk_weight(p_v, request.s, request.d))
        if not paths:
            return BLOCK
        return Decision(paths=paths, t=len(paths), n=len(paths), scheme="xor")


class B2RPolicy(B2Policy):
    """B2 (fixed (2,3) Shamir over node-disjoint paths) with paths chosen by least relay risk."""

    name = "B2R"

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        paths = node_disjoint_shortest_paths(graph, request.s, request.d, n_max=self.n,
                                             weight=risk_weight(p_v, request.s, request.d))
        if not paths:
            return BLOCK
        return Decision(paths=paths, t=min(self.t, len(paths)), n=len(paths), scheme="shamir")


def build_baseline(name: str, config: dict) -> B0Policy | B1Policy | B2Policy | B3Policy:
    baselines_cfg = config.get("baselines", {})
    if name == "B0":
        return B0Policy(**baselines_cfg.get("b0", {}))
    if name == "B1":
        return B1Policy(**baselines_cfg.get("b1", {}))
    if name == "B2":
        return B2Policy(**baselines_cfg.get("b2", {}))
    if name == "B3":
        return B3Policy(**baselines_cfg.get("b3", {}))
    if name == "B0R":
        return B0RPolicy()
    if name == "B1R":
        return B1RPolicy(**baselines_cfg.get("b1", {}))
    if name == "B2R":
        return B2RPolicy(**baselines_cfg.get("b2", {}))
    raise ValueError(f"unknown baseline: {name!r}")
