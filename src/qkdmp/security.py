"""Security quantities computed by the controller (README 3.3): P_exp, P_del, m*.

All functions take `p_v: dict[node, float]` — per-node compromise
probability, as produced by `qkdmp.suspicion.suspicion_to_prob` — and a list
of candidate paths (as node lists including the s, d endpoints; only
`qkdmp.paths.relay_nodes(path)` is exposed to compromise).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from qkdmp.paths import relay_nodes

# ILP is used for m* above this many relay nodes; below it, brute force.
_M_STAR_BRUTE_FORCE_LIMIT = 15
# Exact node-level enumeration for P_exp is used up to this many relay nodes;
# above it, Monte Carlo (README 3.3).
_P_EXP_EXACT_NODE_LIMIT = 20
_MC_MIN_SAMPLES = 10_000


# ---------------------------------------------------------------------------
# Poisson-binomial distribution (independent, not-necessarily-identical Bernoullis)
# ---------------------------------------------------------------------------

def poisson_binomial_pmf(probs: list[float]) -> np.ndarray:
    """P(exactly k of the independent Bernoulli(probs[i]) succeed), k=0..len(probs).

    Standard O(n^2) DP: pmf after including trial i is the convolution of the
    pmf so far with [1-p_i, p_i].
    """
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.concatenate([pmf, [0.0]]) * (1 - p) + np.concatenate([[0.0], pmf]) * p
    return pmf


def poisson_binomial_at_least(probs: list[float], k: int) -> float:
    """P(sum of independent Bernoulli(probs) >= k)."""
    if k <= 0:
        return 1.0
    if k > len(probs):
        return 0.0
    pmf = poisson_binomial_pmf(probs)
    return float(pmf[k:].sum())


# ---------------------------------------------------------------------------
# Exposure probability P_exp (README 3.3)
# ---------------------------------------------------------------------------

def path_exposure_prob(path: list, p_v: dict) -> float:
    """q_i = 1 - prod_{v in relay(P_i)} (1 - p_v): prob. path i is exposed
    (>=1 of its relay nodes is compromised)."""
    q = 1.0
    for v in relay_nodes(path):
        q *= 1 - p_v.get(v, 0.0)
    return 1 - q


def exposure_prob_disjoint(paths: list[list], p_v: dict, t: int) -> float:
    """P_exp for node-disjoint paths: exact Poisson-binomial over per-path q_i."""
    qs = [path_exposure_prob(path, p_v) for path in paths]
    return poisson_binomial_at_least(qs, t)


def _union_relay_nodes(paths: list[list]) -> list:
    nodes: set = set()
    for path in paths:
        nodes.update(relay_nodes(path))
    return sorted(nodes, key=str)


def exposed_count_pmf(paths: list[list], p_v: dict) -> np.ndarray:
    """Exact distribution of the number of exposed paths, pmf[k] = P(k exposed).

    Same quantity `exposure_prob_exact` enumerates, computed faster: a relay
    node on a single path only matters through that path, so each path's
    private nodes fold into one Bernoulli r_i = 1 - prod(1 - p_v). Only nodes
    shared by >= 2 paths are enumerated (2^s states); given their state, the
    paths they did not expose are independent -> Poisson-binomial over r_i.
    """
    node_paths = [set(relay_nodes(p)) for p in paths]
    count: dict = {}
    for nodes in node_paths:
        for v in nodes:
            count[v] = count.get(v, 0) + 1
    shared = sorted((v for v, c in count.items() if c > 1), key=str)
    private_r = []
    for nodes in node_paths:
        q = 1.0
        for v in nodes:
            if count[v] == 1:
                q *= 1 - p_v.get(v, 0.0)
        private_r.append(1 - q)

    n = len(paths)
    pmf = np.zeros(n + 1)
    shared_probs = [p_v.get(v, 0.0) for v in shared]
    # paths each shared node lies on, as bitmasks over paths
    hits = [sum(1 << i for i, nodes in enumerate(node_paths) if v in nodes) for v in shared]
    for mask in range(1 << len(shared)):
        prob = 1.0
        exposed = 0
        for j, p in enumerate(shared_probs):
            if mask & (1 << j):
                prob *= p
                exposed |= hits[j]
            else:
                prob *= 1 - p
        if prob == 0.0:
            continue
        k0 = bin(exposed).count("1")
        rest = poisson_binomial_pmf([private_r[i] for i in range(n) if not exposed & (1 << i)])
        pmf[k0:k0 + len(rest)] += prob * rest
    return pmf


def exposure_prob_exact(paths: list[list], p_v: dict, t: int) -> float:
    """Exact P_exp for overlapping paths (README 3.3: "exact enumeration ...
    if it has <= 20 nodes"), via `exposed_count_pmf`."""
    if t <= 0:
        return 1.0
    return float(exposed_count_pmf(paths, p_v)[t:].sum())


def exposure_prob_exact_enumeration(paths: list[list], p_v: dict, t: int) -> float:
    """Reference implementation of `exposure_prob_exact`: enumerate all 2^m
    compromise subsets over the union of relay nodes. Kept for tests."""
    nodes = _union_relay_nodes(paths)
    m = len(nodes)
    node_paths = [set(relay_nodes(p)) for p in paths]
    probs = np.array([p_v.get(v, 0.0) for v in nodes])

    total = 0.0
    for mask in range(1 << m):
        compromised = {nodes[i] for i in range(m) if mask & (1 << i)}
        prob = 1.0
        for i in range(m):
            prob *= probs[i] if (mask & (1 << i)) else (1 - probs[i])
        if prob == 0.0:
            continue
        exposed = sum(1 for path_nodes in node_paths if path_nodes & compromised)
        if exposed >= t:
            total += prob
    return total


def exposure_prob_monte_carlo(
    paths: list[list], p_v: dict, t: int, rng: np.random.Generator, n_samples: int = _MC_MIN_SAMPLES
) -> tuple[float, float]:
    """Monte Carlo P_exp for overlapping paths. Returns (estimate, standard_error)."""
    nodes = _union_relay_nodes(paths)
    probs = np.array([p_v.get(v, 0.0) for v in nodes])
    node_paths = [set(relay_nodes(p)) for p in paths]

    draws = rng.random((n_samples, len(nodes))) < probs
    exposed_counts = np.zeros(n_samples, dtype=int)
    for path_nodes in node_paths:
        if not path_nodes:
            continue
        idx = [i for i, v in enumerate(nodes) if v in path_nodes]
        exposed_counts += draws[:, idx].any(axis=1)

    hits = exposed_counts >= t
    phat = hits.mean()
    se = float(np.sqrt(phat * (1 - phat) / n_samples))
    return float(phat), se


def exposure_prob(
    paths: list[list],
    p_v: dict,
    t: int,
    rng: np.random.Generator | None = None,
    n_samples: int = _MC_MIN_SAMPLES,
) -> float:
    """P_exp, dispatching on whether paths are node-disjoint and on problem size
    (README 3.3): disjoint -> exact Poisson-binomial; overlapping and small ->
    exact enumeration; overlapping and large -> Monte Carlo."""
    from qkdmp.paths import paths_are_node_disjoint

    if paths_are_node_disjoint(paths):
        return exposure_prob_disjoint(paths, p_v, t)

    m = len(_union_relay_nodes(paths))
    if m <= _P_EXP_EXACT_NODE_LIMIT:
        return exposure_prob_exact(paths, p_v, t)

    if rng is None:
        raise ValueError("rng is required for Monte Carlo P_exp (union of relay nodes > "
                          f"{_P_EXP_EXACT_NODE_LIMIT})")
    estimate, _se = exposure_prob_monte_carlo(paths, p_v, t, rng, n_samples=n_samples)
    return estimate


def exposure_tail_exact(paths: list[list], p_v: dict) -> np.ndarray | None:
    """tail[t] = P_exp(paths, t) for every t = 0..n at once, whenever
    `exposure_prob` would compute it exactly (disjoint paths, or <= 20 relay
    nodes); None when it would use Monte Carlo. Lets a caller evaluate all
    thresholds of one path subset from a single distribution without
    changing any result or any random draw."""
    from qkdmp.paths import paths_are_node_disjoint

    if paths_are_node_disjoint(paths):
        pmf = poisson_binomial_pmf([path_exposure_prob(path, p_v) for path in paths])
    elif len(_union_relay_nodes(paths)) <= _P_EXP_EXACT_NODE_LIMIT:
        pmf = exposed_count_pmf(paths, p_v)
    else:
        return None
    return np.cumsum(pmf[::-1])[::-1]


# ---------------------------------------------------------------------------
# Delivery probability P_del (README 3.3, 4.3)
# ---------------------------------------------------------------------------

def path_survival_prob(path: list, p_v: dict, p_drop: float, p_tamper: float) -> float:
    """Prob. a share on this path arrives both undropped and untampered,
    marginalizing over whether each relay node is compromised: an honest node
    never drops/tampers; a compromised node does so with probability p_drop /
    p_tamper (independent draws)."""
    survive = 1.0
    for v in relay_nodes(path):
        p = p_v.get(v, 0.0)
        survive *= (1 - p) + p * (1 - p_drop) * (1 - p_tamper)
    return survive


def delivery_prob(paths: list[list], p_v: dict, t: int, p_drop: float, p_tamper: float) -> float:
    """P_del = P(>= t shares arrive untampered)."""
    survival_probs = [path_survival_prob(path, p_v, p_drop, p_tamper) for path in paths]
    return poisson_binomial_at_least(survival_probs, t)


# ---------------------------------------------------------------------------
# Minimum compromise set size m* (README 3.3)
# ---------------------------------------------------------------------------

def _covers_at_least_t(compromised: set, node_paths: list[set], t: int) -> bool:
    exposed = sum(1 for path_nodes in node_paths if path_nodes & compromised)
    return exposed >= t


def m_star_brute_force(paths: list[list], t: int) -> int:
    """Smallest |compromise set| such that >= t paths have a compromised relay
    node, found by brute-force search over increasing subset sizes."""
    nodes = _union_relay_nodes(paths)
    node_paths = [set(relay_nodes(p)) for p in paths]
    if t <= 0:
        return 0
    for k in range(0, len(nodes) + 1):
        for combo in combinations(nodes, k):
            if _covers_at_least_t(set(combo), node_paths, t):
                return k
    return len(nodes)


def m_star_ilp(paths: list[list], t: int) -> int:
    """Same as m_star_brute_force but via ILP (partial vertex cover), for
    larger node sets (README 3.3: "brute force for small cases, ILP otherwise").
    """
    from scipy.optimize import LinearConstraint, milp

    nodes = _union_relay_nodes(paths)
    node_paths = [set(relay_nodes(p)) for p in paths]
    m = len(nodes)
    n_paths = len(paths)
    if t <= 0:
        return 0
    if t > n_paths or sum(1 for path_nodes in node_paths if path_nodes) < t:
        # Unreachable target -- too few paths, or too few paths with any relay
        # at all (a direct link has none to compromise; first hit on Germany50,
        # where the ILP then reported "infeasible"). Same convention as
        # m_star_brute_force: compromising everything is the closest we can report.
        return m

    # Variables: x_0..x_{m-1} (node selected), z_0..z_{n_paths-1} (path exposed).
    n_vars = m + n_paths
    c = np.concatenate([np.ones(m), np.zeros(n_paths)])

    # For each path i: z_i - sum_{v in path_i} x_v <= 0
    rows = []
    for i, path_nodes in enumerate(node_paths):
        row = np.zeros(n_vars)
        row[m + i] = 1.0
        for v in path_nodes:
            row[nodes.index(v)] = -1.0
        rows.append(row)
    path_constraint = LinearConstraint(np.array(rows), -np.inf, 0.0)

    # sum(z) >= t
    coverage_row = np.zeros(n_vars)
    coverage_row[m:] = 1.0
    coverage_constraint = LinearConstraint(coverage_row, t, np.inf)

    integrality = np.ones(n_vars)
    bounds_lb = np.zeros(n_vars)
    bounds_ub = np.ones(n_vars)

    result = milp(
        c,
        constraints=[path_constraint, coverage_constraint],
        integrality=integrality,
        bounds=(bounds_lb, bounds_ub),
    )
    if not result.success:
        raise RuntimeError(f"m* ILP failed to solve: {result.message}")
    return int(round(result.x[:m].sum()))


def m_star(paths: list[list], t: int) -> int:
    nodes = _union_relay_nodes(paths)
    if len(nodes) <= _M_STAR_BRUTE_FORCE_LIMIT:
        return m_star_brute_force(paths, t)
    return m_star_ilp(paths, t)
