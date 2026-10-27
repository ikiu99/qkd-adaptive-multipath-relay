import numpy as np
import pytest

from qkdmp.config import load_config
from qkdmp.policies.base import Decision
from qkdmp.sim.des import run_level_b
from qkdmp.sim.graph_mc import simulate_delivery
from qkdmp.suspicion_learning import SuspicionLearner, ranking_auc
from qkdmp.topology import build_topology


def _learner(prior=0.5):
    return SuspicionLearner({v: prior for v in "abcxy"}, q_drop=0.2, q_tamper=0.2, eps=0.01, p_min=0.01)


def test_dropped_share_raises_suspicion_of_its_relays_only():
    lr = _learner()
    lr.observe([["s", "a", "b", "d"]], ["dropped"])
    assert lr.p_v["a"] > 0.5 and lr.p_v["b"] > 0.5
    assert lr.p_v["c"] == 0.5


def test_clean_share_lowers_suspicion():
    lr = _learner()
    lr.observe([["s", "a", "d"]], ["clean"])
    assert lr.p_v["a"] < 0.5


def test_unverified_arrival_is_weaker_evidence_than_verified_clean():
    verified, unverified = _learner(), _learner()
    verified.observe([["s", "a", "d"]], ["clean"])
    unverified.observe([["s", "a", "d"]], ["arrived"])
    assert verified.p_v["a"] < unverified.p_v["a"] < 0.5


def test_blame_is_shared_on_long_paths_and_resolved_by_other_paths():
    """a is compromised (drops at the modelled rate q_drop=0.2), x is honest
    but shares a's path; x's own clean path clears it, a takes the blame."""
    lr = _learner()
    for i in range(30):
        lr.observe([["s", "a", "x", "d"], ["s", "x", "d"]], ["dropped" if i % 5 == 0 else "clean", "clean"])
    assert lr.p_v["a"] > 0.5 > lr.p_v["x"]


def test_direct_link_and_ambiguous_observations_are_ignored():
    lr = _learner()
    lr.observe([["s", "d"], ["s", "a", "d"]], ["dropped", None])
    assert all(p == 0.5 for p in lr.p_v.values())


def test_ranking_auc():
    assert ranking_auc({"a": 0.9, "b": 0.1, "c": 0.2}, {"a"}) == 1.0
    assert ranking_auc({"a": 0.5, "b": 0.5}, {"a"}) == 0.5


def test_delivery_observations_match_ground_truth_when_verifiable():
    rng = np.random.default_rng(0)
    paths = [[0, 1, 9], [0, 2, 9], [0, 3, 9], [0, 4, 9]]
    decision = Decision(paths=paths, t=2, n=4, scheme="shamir")
    seen = set()
    for _ in range(300):
        out = simulate_delivery(rng, decision, {1}, p_drop=0.3, p_tamper=0.5, key_size_bits=256)
        obs = out["observations"]
        assert all(o in ("clean", None) for o in obs[1:])  # honest relays never look bad
        seen.add(obs[0])
    assert {"dropped", "tampered", "clean"} <= seen


def test_learning_run_improves_ranking_over_a_flat_prior():
    config = load_config()
    config = dict(config, attacker=dict(config["attacker"], rho_A=0.2, p_drop=0.3, p_tamper=0.3),
                  policy=dict(config["policy"], fallback="best_effort"),
                  des=dict(config["des"], warmup_requests=0))
    graph = build_topology("nsfnet", config)
    flat = {v: 0.5 for v in graph.nodes()}
    df = run_level_b(graph, "Adaptive", config, n_requests=150, seed=1, p_v_override=flat,
                     learning={"q_drop": 0.3, "q_tamper": 0.3, "eps": 0.01})
    assert df["auc"].iloc[0] == pytest.approx(0.5)  # flat prior: every node tied
    assert df.attrs["final_auc"] > 0.8


def test_clean_history_cannot_clear_a_node_below_the_passive_floor():
    """A passive eavesdropper looks exactly like an honest node, so no amount
    of clean deliveries may push suspicion below prior * passive_fraction
    (in odds form)."""
    lr = SuspicionLearner({"a": 0.3}, q_drop=0.2, q_tamper=0.2, eps=0.01, p_min=0.01, passive_fraction=0.5)
    for _ in range(500):
        lr.observe([["s", "a", "d"]], ["clean"])
    floor = 0.3 * 0.5 / (0.7 + 0.3 * 0.5)
    assert lr.p_v["a"] == pytest.approx(floor, rel=1e-3)
