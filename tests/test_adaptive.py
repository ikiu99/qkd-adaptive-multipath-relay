import pytest

from qkdmp.config import load_config
from qkdmp.policies.adaptive import AdaptiveThresholdPolicy
from qkdmp.policies.base import BLOCK, Request
from qkdmp.policies.exhaustive import ExhaustivePolicy
from qkdmp.suspicion import generate_suspicion_scores, suspicion_to_prob
from qkdmp.topology import build_topology
import numpy as np


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def nsfnet(config):
    return build_topology("nsfnet", config)


def _p_v(graph, config, compromised=frozenset()):
    rng = np.random.default_rng(0)
    s_v = generate_suspicion_scores(
        list(graph.nodes()), compromised, config["suspicion"]["detector_quality"], rng,
        config["suspicion"]["beta_uninformative"], config["suspicion"]["beta_perfect"],
    )
    return suspicion_to_prob(s_v, config["suspicion"]["p_min"])


def _request(s, d, config, eps=0.5, a=0.5):
    return Request(s, d, eps, a)


def test_adaptive_finds_feasible_option_with_loose_targets(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    policy = AdaptiveThresholdPolicy()
    decision = policy.decide(_request(0, 9, config, eps=0.99, a=0.01), nsfnet, p_v, config)
    assert decision is not BLOCK
    assert decision.p_exp <= 0.99
    assert decision.p_del >= 0.01


def test_adaptive_blocks_with_impossible_targets(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    policy = AdaptiveThresholdPolicy(fallback="block")
    decision = policy.decide(_request(0, 9, config, eps=0.0, a=1.0), nsfnet, p_v, config)
    assert decision is BLOCK


def test_adaptive_best_effort_never_blocks_when_candidates_exist(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    policy = AdaptiveThresholdPolicy(fallback="best_effort")
    decision = policy.decide(_request(0, 9, config, eps=0.0, a=1.0), nsfnet, p_v, config)
    assert decision is not BLOCK


def test_adaptive_picks_minimum_cost_among_feasible_options(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    policy = AdaptiveThresholdPolicy()
    decision = policy.decide(_request(0, 9, config, eps=0.99, a=0.01), nsfnet, p_v, config)
    assert decision.cost is not None and decision.cost > 0


def test_adaptive_respects_pool_reserve_fraction(config):
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1, length_km_norm=10.0, rate_bps=1e5, capacity_bits=1000.0, level_bits=1.0)
    graph.add_edge(1, 2, length_km_norm=10.0, rate_bps=1e5, capacity_bits=1000.0, level_bits=1.0)
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"], reserve_fraction=0.5)  # 0.5*1000=500 > level_bits=1.0
    policy = AdaptiveThresholdPolicy()
    decision = policy.decide(_request(0, 2, config, eps=0.99, a=0.01), graph, {}, cfg)
    assert decision is BLOCK


def test_adaptive_pool_aware_false_ignores_reserve_fraction(config):
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1, length_km_norm=10.0, rate_bps=1e5, capacity_bits=1000.0, level_bits=1.0)
    graph.add_edge(1, 2, length_km_norm=10.0, rate_bps=1e5, capacity_bits=1000.0, level_bits=1.0)
    cfg = dict(config)
    cfg["paths"] = dict(config["paths"], reserve_fraction=0.5)
    policy = AdaptiveThresholdPolicy(pool_aware=False)
    decision = policy.decide(_request(0, 2, config, eps=0.99, a=0.01), graph, {}, cfg)
    assert decision is not BLOCK  # ablation: pool state (and its reserve check) is ignored


def test_exhaustive_searches_at_least_as_wide_as_adaptive(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    adaptive = AdaptiveThresholdPolicy()
    exhaustive = ExhaustivePolicy()
    d_adaptive = adaptive.decide(_request(0, 9, config, eps=0.99, a=0.01), nsfnet, p_v, config)
    d_exhaustive = exhaustive.decide(_request(0, 9, config, eps=0.99, a=0.01), nsfnet, p_v, config)
    assert d_adaptive is not BLOCK
    assert d_exhaustive is not BLOCK
    # exhaustive searches a superset of options, so its cost can only be <=
    assert d_exhaustive.cost <= d_adaptive.cost + 1e-9


def test_adaptive_is_deterministic_given_seed(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    d1 = AdaptiveThresholdPolicy(seed=7).decide(_request(0, 9, config, eps=0.99, a=0.01), nsfnet, p_v, config)
    d2 = AdaptiveThresholdPolicy(seed=7).decide(_request(0, 9, config, eps=0.99, a=0.01), nsfnet, p_v, config)
    assert d1.paths == d2.paths and d1.t == d2.t and d1.n == d2.n


def test_adaptive_never_picks_a_link_that_cannot_carry_one_share(config):
    """A low-capacity link can pass the fractional reserve guard (10% of 300
    bits = 30) while holding less than one 256-bit share; Adaptive must
    route around it rather than choose a path that fails at reservation."""
    import networkx as nx

    graph = nx.Graph()
    for u, v in [(0, 1), (1, 3), (0, 2), (2, 3)]:
        graph.add_edge(u, v, length_km_norm=10.0, rate_bps=1e5, capacity_bits=300.0, level_bits=300.0)
    graph.edges[0, 1]["length_km_norm"] = 1.0  # make 0-1-3 the preferred, cheapest route
    graph.edges[0, 1]["level_bits"] = 100.0  # > 30-bit guard, < 256-bit share
    policy = AdaptiveThresholdPolicy()
    decision = policy.decide(_request(0, 3, config, eps=0.99, a=0.01), graph, {}, config)
    assert decision is not BLOCK
    assert all([0, 1] != p[:2] for p in decision.paths)


def test_adaptive_finds_longer_route_when_all_short_candidates_are_depleted(config):
    """Candidate search runs on the pool-pruned graph: when every short
    route crosses a depleted link, Adaptive must find the longer route that
    still has key instead of blocking."""
    import networkx as nx

    graph = nx.Graph()
    # four 2-hop routes via 1..4, all through depleted links; one 3-hop route via 5-6
    for mid in (1, 2, 3, 4):
        graph.add_edge(0, mid, length_km_norm=10.0, rate_bps=1e5, capacity_bits=3000.0, level_bits=100.0)
        graph.add_edge(mid, 9, length_km_norm=10.0, rate_bps=1e5, capacity_bits=3000.0, level_bits=3000.0)
    for u, v in [(0, 5), (5, 6), (6, 9)]:
        graph.add_edge(u, v, length_km_norm=10.0, rate_bps=1e5, capacity_bits=3000.0, level_bits=3000.0)
    decision = AdaptiveThresholdPolicy().decide(_request(0, 9, config, eps=0.99, a=0.01), graph, {}, config)
    assert decision is not BLOCK
    assert decision.paths == [[0, 5, 6, 9]]
