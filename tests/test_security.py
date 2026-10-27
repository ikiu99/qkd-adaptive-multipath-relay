import numpy as np
import pytest

from qkdmp.security import (
    delivery_prob,
    exposed_count_pmf,
    exposure_prob,
    exposure_prob_disjoint,
    exposure_prob_exact,
    exposure_prob_exact_enumeration,
    exposure_prob_monte_carlo,
    exposure_tail_exact,
    m_star,
    m_star_brute_force,
    m_star_ilp,
    path_exposure_prob,
    poisson_binomial_at_least,
    poisson_binomial_pmf,
)


def test_poisson_binomial_pmf_sums_to_one():
    pmf = poisson_binomial_pmf([0.1, 0.4, 0.7, 0.9])
    assert pmf.sum() == pytest.approx(1.0)
    assert len(pmf) == 5


def test_poisson_binomial_reduces_to_binomial_for_equal_probs():
    from scipy.stats import binom

    p, n = 0.3, 6
    pmf = poisson_binomial_pmf([p] * n)
    expected = binom.pmf(np.arange(n + 1), n, p)
    np.testing.assert_allclose(pmf, expected, atol=1e-10)


def test_poisson_binomial_at_least_bounds():
    probs = [0.2, 0.5, 0.8]
    assert poisson_binomial_at_least(probs, 0) == 1.0
    assert poisson_binomial_at_least(probs, 4) == 0.0


def test_path_exposure_prob_formula():
    path = ["s", "a", "b", "d"]  # relay nodes: a, b
    p_v = {"a": 0.1, "b": 0.2}
    q = path_exposure_prob(path, p_v)
    assert q == pytest.approx(1 - (1 - 0.1) * (1 - 0.2))


def test_exposure_prob_disjoint_n_equals_t_equals_1_is_path_exposure():
    path = ["s", "a", "b", "d"]
    p_v = {"a": 0.1, "b": 0.2}
    assert exposure_prob_disjoint([path], p_v, t=1) == pytest.approx(path_exposure_prob(path, p_v))


def test_exact_p_exp_matches_disjoint_formula_when_paths_are_disjoint():
    paths = [["s", "a", "d"], ["s", "b", "d"], ["s", "c", "d"]]
    p_v = {"a": 0.1, "b": 0.3, "c": 0.5}
    for t in (1, 2, 3):
        exact = exposure_prob_exact(paths, p_v, t)
        disjoint = exposure_prob_disjoint(paths, p_v, t)
        assert exact == pytest.approx(disjoint, abs=1e-9)


def test_exact_p_exp_accounts_for_shared_node():
    # two paths share relay node "x": compromising x alone exposes both shares
    paths = [["s", "x", "a", "d"], ["s", "x", "b", "d"]]
    p_v = {"x": 0.5, "a": 0.1, "b": 0.1}
    t = 2
    exact = exposure_prob_exact(paths, p_v, t)
    # naive disjoint formula (wrongly treating paths as independent) must differ
    disjoint_wrong = exposure_prob_disjoint(paths, p_v, t)
    assert exact != pytest.approx(disjoint_wrong)
    # ground truth: exposed>=2 iff x compromised (exposes both) OR (a AND b) compromised
    p_x, p_a, p_b = 0.5, 0.1, 0.1
    expected = p_x + (1 - p_x) * p_a * p_b
    assert exact == pytest.approx(expected)


def test_exact_p_exp_matches_monte_carlo_within_3_se():
    rng = np.random.default_rng(42)
    # a moderately overlapping topology: 5 paths sharing a few relay nodes
    paths = [
        ["s", "x", "a", "d"],
        ["s", "x", "b", "d"],
        ["s", "y", "b", "d"],
        ["s", "y", "c", "d"],
        ["s", "z", "d"],
    ]
    p_v = {"x": 0.3, "y": 0.2, "z": 0.6, "a": 0.1, "b": 0.15, "c": 0.25}

    for t in (1, 2, 3, 4):
        exact = exposure_prob_exact(paths, p_v, t)
        estimate, se = exposure_prob_monte_carlo(paths, p_v, t, rng, n_samples=50_000)
        assert abs(exact - estimate) <= 3 * se, (
            f"t={t}: exact={exact}, mc={estimate}, se={se}, diff={abs(exact - estimate)}"
        )


def test_delivery_prob_no_compromise_no_drop_is_one():
    paths = [["s", "a", "d"], ["s", "b", "d"]]
    p_v = {"a": 0.0, "b": 0.0}
    assert delivery_prob(paths, p_v, t=2, p_drop=0.1, p_tamper=0.1) == pytest.approx(1.0)


def test_delivery_prob_decreases_with_drop_rate():
    paths = [["s", "a", "d"], ["s", "b", "d"], ["s", "c", "d"]]
    p_v = {"a": 0.5, "b": 0.5, "c": 0.5}
    low = delivery_prob(paths, p_v, t=2, p_drop=0.1, p_tamper=0.0)
    high = delivery_prob(paths, p_v, t=2, p_drop=0.9, p_tamper=0.0)
    assert high < low


def test_m_star_single_path_needs_one_node():
    paths = [["s", "a", "b", "d"]]
    assert m_star_brute_force(paths, t=1) == 1


def test_m_star_disjoint_paths_needs_t_nodes():
    paths = [["s", "a", "d"], ["s", "b", "d"], ["s", "c", "d"]]
    assert m_star_brute_force(paths, t=2) == 2
    assert m_star_brute_force(paths, t=3) == 3


def test_m_star_shared_node_reduces_compromise_set():
    # x sits on both paths: compromising just x exposes both (t=2 reachable with 1 node)
    paths = [["s", "x", "a", "d"], ["s", "x", "b", "d"]]
    assert m_star_brute_force(paths, t=2) == 1


def test_m_star_brute_force_matches_ilp():
    rng = np.random.default_rng(7)
    paths = [
        ["s", "x", "a", "d"],
        ["s", "x", "b", "d"],
        ["s", "y", "b", "d"],
        ["s", "y", "c", "d"],
        ["s", "z", "d"],
    ]
    for t in (1, 2, 3, 4, 5):
        assert m_star_brute_force(paths, t) == m_star_ilp(paths, t)


def test_m_star_dispatches_by_size():
    paths = [["s", "a", "d"], ["s", "b", "d"]]
    assert m_star(paths, t=1) == 1


def test_m_star_ilp_matches_brute_force_when_direct_links_make_target_unreachable():
    from qkdmp.security import m_star_brute_force, m_star_ilp

    # two direct links (no relay) + one relayed path; t=2 cannot be reached
    paths = [[0, 9], [0, 9], [0, 1, 9]]
    assert m_star_ilp(paths, 2) == m_star_brute_force(paths, 2) == 1


def _random_overlapping_case(rng):
    n_nodes = int(rng.integers(6, 16))
    paths = []
    for _ in range(int(rng.integers(1, 6))):
        relays = [int(v) for v in rng.choice(n_nodes, size=int(rng.integers(0, 5)), replace=False)]
        paths.append(["s"] + relays + ["d"])
    p_v = {v: float(rng.choice([0.0, rng.random(), 1.0], p=[0.1, 0.8, 0.1])) for v in range(n_nodes)}
    return paths, p_v


def test_exposed_count_pmf_matches_full_enumeration():
    rng = np.random.default_rng(7)
    for _ in range(150):
        paths, p_v = _random_overlapping_case(rng)
        pmf = exposed_count_pmf(paths, p_v)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
        for t in range(0, len(paths) + 2):
            assert exposure_prob_exact(paths, p_v, t) == pytest.approx(
                exposure_prob_exact_enumeration(paths, p_v, t), abs=1e-12)


def test_exposure_tail_exact_matches_exposure_prob_for_every_t():
    rng = np.random.default_rng(11)
    for _ in range(100):
        paths, p_v = _random_overlapping_case(rng)
        tail = exposure_tail_exact(paths, p_v)
        assert tail is not None  # <= 15 relay nodes: always the exact branch
        for t in range(0, len(paths) + 1):
            assert tail[t] == pytest.approx(exposure_prob(paths, p_v, t), abs=1e-12)


def test_exposure_tail_exact_defers_to_monte_carlo_above_node_limit():
    paths = [["s"] + list(range(i * 7, i * 7 + 7)) + ["d"] for i in range(3)]
    paths.append(["s", 0, 7, 14, "d"])  # overlaps the others, 21 relay nodes in total
    assert exposure_tail_exact(paths, {v: 0.1 for v in range(21)}) is None
