import math

import networkx as nx
import pytest

from qkdmp.config import load_config
from qkdmp.topology import (
    apply_length_normalization,
    apply_rate_model,
    available_topologies,
    build_topology,
    generate_grid,
    generate_random50,
    load_topology,
    rate_from_length,
)


@pytest.fixture
def config():
    return load_config()


def test_available_topologies_lists_every_topology_in_use():
    names = available_topologies()
    assert {"nsfnet", "abilene", "geant", "germany50", "random50", "grid"} <= set(names)


@pytest.mark.parametrize("name,expected_nodes,expected_edges", [
    ("nsfnet", 14, 21),
    ("abilene", 11, 14),
    ("geant", 22, 36),
    ("germany50", 50, 88),
])
def test_static_topologies_match_spec_size(name, expected_nodes, expected_edges):
    graph = load_topology(name)
    assert graph.number_of_nodes() == expected_nodes
    assert graph.number_of_edges() == expected_edges
    assert nx.is_connected(graph)
    for _, _, data in graph.edges(data=True):
        assert data["length_km"] > 0


def test_random50_is_2_edge_connected_with_50_nodes():
    graph = generate_random50(seed=7)
    assert graph.number_of_nodes() == 50
    assert nx.is_connected(graph)
    assert nx.edge_connectivity(graph) >= 2


def test_grid_is_6x6_mesh():
    graph = generate_grid(rows=6, cols=6, unit_km=80.0)
    assert graph.number_of_nodes() == 36
    assert graph.number_of_edges() == 60  # 2 * rows * (cols - 1) for a square mesh
    for _, _, data in graph.edges(data=True):
        assert data["length_km"] == 80.0


def test_length_normalization_preserves_order_and_bounds():
    graph = nx.Graph()
    graph.add_edge(0, 1, length_km=100.0)
    graph.add_edge(1, 2, length_km=500.0)
    graph.add_edge(2, 3, length_km=300.0)

    apply_length_normalization(graph, l_min_km=10.0, l_max_km=100.0)

    lengths = {(u, v): d["length_km_norm"] for u, v, d in graph.edges(data=True)}
    assert lengths[(0, 1)] == pytest.approx(10.0)
    assert lengths[(1, 2)] == pytest.approx(100.0)
    assert 10.0 < lengths[(2, 3)] < 100.0
    # order preserved
    assert lengths[(0, 1)] < lengths[(2, 3)] < lengths[(1, 2)]


def test_length_normalization_constant_lengths_is_identity():
    graph = nx.Graph()
    graph.add_edge(0, 1, length_km=50.0)
    graph.add_edge(1, 2, length_km=50.0)

    apply_length_normalization(graph, l_min_km=10.0, l_max_km=100.0)

    for _, _, d in graph.edges(data=True):
        assert d["length_km_norm"] == 50.0


def test_rate_from_length_matches_formula():
    r0, alpha, il = 1e5, 0.2, 0.0
    length = 50.0
    expected = r0 * 10 ** (-(alpha * length + il) / 10)
    assert rate_from_length(length, r0, alpha, il) == pytest.approx(expected)


def test_rate_decreases_with_length():
    r0, alpha, il = 1e5, 0.2, 0.0
    short = rate_from_length(10.0, r0, alpha, il)
    long = rate_from_length(90.0, r0, alpha, il)
    assert long < short


def test_apply_rate_model_uniform_mode_ignores_length(config):
    graph = nx.Graph()
    graph.add_edge(0, 1, length_km=10.0, length_km_norm=10.0)
    graph.add_edge(1, 2, length_km=90.0, length_km_norm=90.0)

    cfg = dict(config)
    cfg["rate_model"] = dict(config["rate_model"], mode="uniform", uniform_rate_bps=5e4)
    apply_rate_model(graph, cfg)

    rates = {d["rate_bps"] for _, _, d in graph.edges(data=True)}
    assert rates == {5e4}


def test_apply_rate_model_sets_capacity_from_pool_config(config):
    graph = nx.Graph()
    graph.add_edge(0, 1, length_km=10.0, length_km_norm=10.0)
    apply_rate_model(graph, config)

    d = graph.edges[0, 1]
    expected_capacity = config["pool"]["capacity_seconds"] * d["rate_bps"]
    assert d["capacity_bits"] == pytest.approx(expected_capacity)
    assert d["level_bits"] == pytest.approx(expected_capacity * config["pool"]["initial_fill_fraction"])


def test_build_topology_end_to_end(config):
    for name in ("nsfnet", "abilene", "geant", "germany50", "random50", "grid"):
        graph = build_topology(name, config)
        for _, _, d in graph.edges(data=True):
            assert "length_km_norm" in d
            assert config["length_normalization"]["L_min_km"] - 1e-6 <= d["length_km_norm"] <= config["length_normalization"]["L_max_km"] + 1e-6
            assert d["rate_bps"] > 0
            assert d["capacity_bits"] > 0
