import numpy as np
import pytest

from qkdmp.policies.base import Decision
from qkdmp.sim.graph_mc import simulate_delivery


def test_shamir_n_greater_than_t_detects_tampering_and_still_serves():
    # 1 of 5 relays compromised, tamper always -- with (t=2, n=5) there are
    # enough spare shares (README 4.3: "if >= t+1 shares arrive, reconstruct
    # from several t-subsets") for the majority of t-subsets to agree on the
    # true secret despite one tampered share.
    decision = Decision(
        paths=[["s", "a", "d"], ["s", "b", "d"], ["s", "c", "d"], ["s", "e", "d"], ["s", "f", "d"]],
        t=2, n=5, scheme="shamir",
    )
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(200):
        result = simulate_delivery(rng, decision, attacker_set={"a"}, p_drop=0.0, p_tamper=1.0, key_size_bits=64)
        assert result["served"]
        assert not result["silent_failure"]
        hits += 1
    assert hits == 200


def test_shamir_n_equals_t_silently_fails_when_tampered():
    decision = Decision(paths=[["s", "a", "d"], ["s", "b", "d"]], t=2, n=2, scheme="shamir")
    rng = np.random.default_rng(0)
    result = simulate_delivery(rng, decision, attacker_set={"a"}, p_drop=0.0, p_tamper=1.0, key_size_bits=64)
    assert result["served"]
    assert result["silent_failure"]


def test_single_scheme_dropped_is_not_served():
    decision = Decision(paths=[["s", "a", "d"]], t=1, n=1, scheme="single")
    rng = np.random.default_rng(0)
    result = simulate_delivery(rng, decision, attacker_set={"a"}, p_drop=1.0, p_tamper=0.0, key_size_bits=64)
    assert not result["served"]
    assert not result["silent_failure"]


def test_single_scheme_tampered_is_silent_failure():
    decision = Decision(paths=[["s", "a", "d"]], t=1, n=1, scheme="single")
    rng = np.random.default_rng(0)
    result = simulate_delivery(rng, decision, attacker_set={"a"}, p_drop=0.0, p_tamper=1.0, key_size_bits=64)
    assert result["served"]
    assert result["silent_failure"]


def test_xor_scheme_needs_all_shares():
    decision = Decision(paths=[["s", "a", "d"], ["s", "b", "d"]], t=2, n=2, scheme="xor")
    rng = np.random.default_rng(0)
    result = simulate_delivery(rng, decision, attacker_set={"a"}, p_drop=1.0, p_tamper=0.0, key_size_bits=64)
    assert not result["served"]
