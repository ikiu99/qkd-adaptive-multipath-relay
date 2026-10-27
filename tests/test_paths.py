import networkx as nx
import pytest

from qkdmp.config import load_config
from qkdmp.paths import (
    candidate_paths,
    node_disjoint_shortest_paths,
    node_overlap,
    paths_are_node_disjoint,
    relay_nodes,
    yen_k_shortest_paths,
)
from qkdmp.topology import build_topology


def test_yen_k_shortest_paths_disconnected_graph_returns_empty():
    # nx.shortest_simple_paths raises NetworkXNoPath lazily (on first next()
    # of the generator, not at construction), so consuming it must also be
    # inside the try/except -- a disconnected graph is the regression case.
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    assert yen_k_shortest_paths(graph, 0, 1, k=3) == []


@pytest.fixture
def nsfnet():
    return build_topology("nsfnet", load_config())


def test_relay_nodes_excludes_endpoints():
    assert relay_nodes(["s", "a", "b", "d"]) == ["a", "b"]
    assert relay_nodes(["s", "d"]) == []


def test_node_disjoint_shortest_paths_are_disjoint(nsfnet):
    paths = node_disjoint_shortest_paths(nsfnet, 0, 9, n_max=3)
    assert len(paths) >= 2
    assert paths_are_node_disjoint(paths)
    for path in paths:
        assert nx.is_simple_path(nsfnet, path)


def test_candidate_paths_fills_up_to_n_max_with_overlap_allowed(nsfnet):
    n_max = 4
    paths = candidate_paths(nsfnet, 0, 9, n_max=n_max)
    assert 1 <= len(paths) <= n_max
    # all returned paths are valid simple paths between s and d
    for path in paths:
        assert path[0] == 0 and path[-1] == 9
        assert nx.is_simple_path(nsfnet, path)
    # no duplicate paths
    assert len({tuple(p) for p in paths}) == len(paths)


def test_candidate_paths_n_max_1_is_shortest_path(nsfnet):
    paths = candidate_paths(nsfnet, 0, 9, n_max=1, weight="length_km_norm")
    expected = nx.shortest_path(nsfnet, 0, 9, weight="length_km_norm")
    assert paths == [expected]


def test_node_overlap_counts_shared_relays():
    paths = [["s", "x", "a", "d"], ["s", "x", "b", "d"], ["s", "c", "d"]]
    overlap = node_overlap(paths)
    assert overlap["x"] == 2
    assert overlap["a"] == 1
    assert overlap["c"] == 1


def test_paths_are_node_disjoint_detects_sharing():
    assert paths_are_node_disjoint([["s", "a", "d"], ["s", "b", "d"]])
    assert not paths_are_node_disjoint([["s", "x", "a", "d"], ["s", "x", "b", "d"]])


def test_nsfnet_has_some_pairs_with_only_two_disjoint_paths(nsfnet):
    # README 2.4: "Known limitation ... in NSFNET some node pairs have only 2
    # node-disjoint paths." Sanity check that our topology + algorithm can
    # reproduce a pair capped at 2, without asserting it's the *only* count.
    counts = []
    nodes = list(nsfnet.nodes())
    for i, s in enumerate(nodes):
        for d in nodes[i + 1:]:
            paths = node_disjoint_shortest_paths(nsfnet, s, d, n_max=5)
            counts.append(len(paths))
    assert min(counts) <= 2
