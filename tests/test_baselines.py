import networkx as nx
import pytest

from qkdmp.config import load_config
from qkdmp.paths import relay_nodes
from qkdmp.policies.base import BLOCK, Request
from qkdmp.policies.baselines import B0Policy, B1Policy, B2Policy, B3Policy, build_baseline
from qkdmp.topology import build_topology


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def nsfnet(config):
    return build_topology("nsfnet", config)


def _request(s, d, config):
    return Request(s, d, config["policy"]["eps_target"], config["policy"]["a_target"])


def test_b0_is_shortest_hop_path(nsfnet, config):
    policy = B0Policy(variant="hopcount")
    decision = policy.decide(_request(0, 9, config), nsfnet, {}, config)
    expected = nx.shortest_path(nsfnet, 0, 9)
    assert decision.paths == [expected]
    assert decision.t == 1 and decision.n == 1
    assert decision.scheme == "single"


def test_b0_key_rate_variant_can_differ_from_hopcount(nsfnet, config):
    hop = B0Policy(variant="hopcount").decide(_request(0, 9, config), nsfnet, {}, config)
    rate = B0Policy(variant="key_rate").decide(_request(0, 9, config), nsfnet, {}, config)
    # not asserting they differ (topology-dependent) -- just that both are valid paths
    for decision in (hop, rate):
        assert decision.paths[0][0] == 0 and decision.paths[0][-1] == 9
        assert nx.is_simple_path(nsfnet, decision.paths[0])


def test_b1_xor_uses_n_disjoint_paths(nsfnet, config):
    policy = B1Policy(n=2, n_max=2)
    decision = policy.decide(_request(0, 9, config), nsfnet, {}, config)
    assert decision.scheme == "xor"
    assert decision.t == decision.n
    assert decision.n <= 2
    relay_sets = [set(relay_nodes(p)) for p in decision.paths]
    for i in range(len(relay_sets)):
        for j in range(i + 1, len(relay_sets)):
            assert relay_sets[i].isdisjoint(relay_sets[j])


def test_b2_fixed_threshold_and_share_count(nsfnet, config):
    policy = B2Policy(t=2, n=3)
    decision = policy.decide(_request(0, 9, config), nsfnet, {}, config)
    assert decision.scheme == "shamir"
    assert decision.n <= 3
    assert decision.t == min(2, decision.n)


def test_b2_blocks_when_no_path_exists(config):
    graph = nx.Graph()
    graph.add_node("s")
    graph.add_node("d")  # disconnected
    policy = B2Policy(t=2, n=3)
    decision = policy.decide(_request("s", "d", config), graph, {}, config)
    assert decision is BLOCK


def test_b3_selects_up_to_n_paths(nsfnet, config):
    p_v = {v: 0.1 for v in nsfnet.nodes()}
    policy = B3Policy(n=3, k_candidates=8, lambda_cor=1.0, delta_w=1.0)
    decision = policy.decide(_request(0, 9, config), nsfnet, p_v, config)
    assert decision is not BLOCK
    assert decision.n <= 3
    assert decision.t == decision.n
    for path in decision.paths:
        assert nx.is_simple_path(nsfnet, path)


def test_b3_record_detection_lowers_path_security(nsfnet, config):
    policy = B3Policy(n=2)
    p_v = {v: 0.1 for v in nsfnet.nodes()}
    path = [0, 6, 7, 9]  # relay nodes 6, 7 (per the verified NSFNET topology)
    security_before = policy._path_security(path, p_v)
    policy.record_detection([6])
    assert policy._penalty[6] == pytest.approx(policy.delta_w)
    security_after = policy._path_security(path, p_v)
    assert security_after < security_before


def test_b3_record_detection_spreads_by_manhattan_distance_on_grid(config):
    grid = build_topology("grid", config)
    policy = B3Policy(n=2, d_max=80.0)  # grid's unit spacing (configs/default.yaml generator)
    policy.record_detection([0], graph=grid)
    neighbors = [v for v in grid.neighbors(0)]
    assert any(policy._penalty.get(v, 0.0) > 0 for v in neighbors)


def test_build_baseline_dispatches_by_name(config):
    for name, cls in (("B0", B0Policy), ("B1", B1Policy), ("B2", B2Policy), ("B3", B3Policy)):
        assert isinstance(build_baseline(name, config), cls)
    with pytest.raises(ValueError):
        build_baseline("B99", config)


def test_risk_aware_baselines_avoid_a_suspicious_relay():
    import networkx as nx

    from qkdmp.policies.base import Request
    from qkdmp.policies.baselines import B0RPolicy, B2RPolicy

    # two 2-hop routes s-a-d and s-b-d; a is suspicious, so B0R must take b
    g = nx.Graph([("s", "a"), ("a", "d"), ("s", "b"), ("b", "d"), ("s", "c"), ("c", "x"), ("x", "d")])
    p_v = {"a": 0.9, "b": 0.01, "c": 0.01, "x": 0.01}
    req = Request(s="s", d="d", eps_target=0.5, a_target=0.8)
    assert B0RPolicy().decide(req, g, p_v, {}).paths == [["s", "b", "d"]]
    dec = B2RPolicy(t=2, n=3).decide(req, g, p_v, {})
    assert dec.paths[0] == ["s", "b", "d"] and dec.paths[-1] == ["s", "a", "d"] and dec.t == 2 and dec.n == 3
