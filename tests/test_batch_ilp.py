import numpy as np
import pytest

from qkdmp.config import load_config
from qkdmp.policies.base import Request
from qkdmp.policies.batch_ilp import solve_batch_ilp
from qkdmp.suspicion import generate_suspicion_scores, suspicion_to_prob
from qkdmp.topology import build_topology


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def nsfnet(config):
    return build_topology("nsfnet", config)


def _p_v(graph, config):
    rng = np.random.default_rng(0)
    s_v = generate_suspicion_scores(
        list(graph.nodes()), set(), config["suspicion"]["detector_quality"], rng,
        config["suspicion"]["beta_uninformative"], config["suspicion"]["beta_perfect"],
    )
    return suspicion_to_prob(s_v, config["suspicion"]["p_min"])


def test_batch_ilp_serves_requests_with_loose_targets(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    requests = [Request(0, 9, eps_target=0.99, a_target=0.01), Request(1, 12, eps_target=0.99, a_target=0.01)]
    result = solve_batch_ilp(requests, nsfnet, p_v, config, window_seconds=3600.0)
    assert result.n_vars > 0
    assert any(d is not None for d in result.decisions)
    assert result.solve_time_s >= 0.0


def test_batch_ilp_respects_pool_budget(config):
    import networkx as nx

    graph = nx.Graph()
    graph.add_edge(0, 1, length_km_norm=10.0, rate_bps=0.0, capacity_bits=100.0, level_bits=100.0)
    graph.add_edge(1, 2, length_km_norm=10.0, rate_bps=0.0, capacity_bits=100.0, level_bits=100.0)
    # key_size_bits=256 (default) per hop far exceeds the 100-bit pool, so
    # no request can be served no matter how loose the targets are.
    requests = [Request(0, 2, eps_target=0.99, a_target=0.01)]
    result = solve_batch_ilp(requests, graph, {}, config, window_seconds=0.0)
    assert all(d is None for d in result.decisions)


def test_batch_ilp_no_feasible_options_returns_all_none(config):
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)  # disconnected -- no candidate paths at all
    requests = [Request(0, 1, eps_target=0.99, a_target=0.01)]
    result = solve_batch_ilp(requests, graph, {}, config, window_seconds=3600.0)
    assert result.decisions == [None]
    assert result.n_vars == 0


def test_batch_ilp_objective_counts_served_minus_cost(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    requests = [Request(0, 9, eps_target=0.99, a_target=0.01)]
    result = solve_batch_ilp(requests, nsfnet, p_v, config, window_seconds=3600.0, lambda_cost=1e-9)
    served = sum(1 for d in result.decisions if d is not None)
    assert result.objective == pytest.approx(served, abs=1e-3)
