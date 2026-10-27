import numpy as np
import pytest

from qkdmp.attacker import (
    path_outcome,
    place_adaptive,
    place_betweenness,
    place_key_throughput,
    place_random,
    share_outcome,
)
from qkdmp.config import load_config
from qkdmp.topology import build_topology


@pytest.fixture
def nsfnet():
    return build_topology("nsfnet", load_config())


def test_place_random_respects_count_and_exclude(nsfnet):
    rng = np.random.default_rng(0)
    exclude = frozenset({0, 1})
    chosen = place_random(nsfnet, rng, count=3, exclude=exclude)
    assert len(chosen) == 3
    assert chosen.isdisjoint(exclude)


def test_place_random_respects_rho_a(nsfnet):
    rng = np.random.default_rng(0)
    chosen = place_random(nsfnet, rng, rho_A=0.5)
    assert len(chosen) == round(0.5 * nsfnet.number_of_nodes())


def test_place_random_bernoulli_mode_is_iid_per_node(nsfnet):
    rng = np.random.default_rng(0)
    n = nsfnet.number_of_nodes()
    rho_a = 0.3
    counts = [len(place_random(nsfnet, rng, rho_A=rho_a, mode="bernoulli")) for _ in range(2000)]
    mean_count = np.mean(counts)
    # E[count] = rho_A * n for iid Bernoulli draws; loose tolerance since this
    # is a statistical check, not an exact one.
    assert mean_count == pytest.approx(rho_a * n, abs=0.5)
    # unlike fixed_count, sizes should vary across draws
    assert len(set(counts)) > 1


def test_place_betweenness_picks_high_centrality_nodes(nsfnet):
    chosen = place_betweenness(nsfnet, count=2)
    import networkx as nx

    centrality = nx.betweenness_centrality(nsfnet, weight="length_km_norm")
    top2 = {n for n, _ in sorted(centrality.items(), key=lambda kv: -kv[1])[:2]}
    assert chosen == top2


def test_place_key_throughput_runs_and_respects_count(nsfnet):
    chosen = place_key_throughput(nsfnet, count=3)
    assert len(chosen) == 3
    assert chosen <= set(nsfnet.nodes())


def test_place_adaptive_picks_most_frequent_nodes(nsfnet):
    freq = {0: 10, 1: 5, 2: 1}
    chosen = place_adaptive(nsfnet, path_frequency=freq, count=2)
    assert chosen == {0, 1}


def test_share_outcome_honest_node_always_delivers():
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert share_outcome(rng, is_compromised=False, p_drop=1.0, p_tamper=1.0) == "delivered"


def test_share_outcome_compromised_drop_probability_one():
    rng = np.random.default_rng(0)
    assert share_outcome(rng, is_compromised=True, p_drop=1.0, p_tamper=0.0) == "dropped"


def test_share_outcome_compromised_tamper_probability_one():
    rng = np.random.default_rng(0)
    assert share_outcome(rng, is_compromised=True, p_drop=0.0, p_tamper=1.0) == "tampered"


def test_path_outcome_any_drop_dominates():
    rng = np.random.default_rng(0)
    outcome = path_outcome(rng, ["a", "b"], compromised={"a", "b"}, p_drop=1.0, p_tamper=1.0)
    assert outcome == "dropped"


def test_path_outcome_no_compromise_is_delivered():
    rng = np.random.default_rng(0)
    outcome = path_outcome(rng, ["a", "b"], compromised=set(), p_drop=1.0, p_tamper=1.0)
    assert outcome == "delivered"
