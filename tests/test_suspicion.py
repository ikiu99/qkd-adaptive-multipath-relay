import numpy as np
import pytest

from qkdmp.suspicion import (
    detector_p_v,
    detector_quality_to_beta,
    generate_suspicion_scores,
    misrank,
    suspicion_to_prob,
)

BETA_UNINFORMATIVE = [2.0, 2.0, 2.0, 2.0]
BETA_PERFECT = [8.0, 1.0, 1.0, 8.0]


def test_detector_quality_zero_is_uninformative():
    params = detector_quality_to_beta(0.0, BETA_UNINFORMATIVE, BETA_PERFECT)
    assert params == pytest.approx(tuple(BETA_UNINFORMATIVE))


def test_detector_quality_one_is_perfect():
    params = detector_quality_to_beta(1.0, BETA_UNINFORMATIVE, BETA_PERFECT)
    assert params == pytest.approx(tuple(BETA_PERFECT))


def test_detector_quality_out_of_range_raises():
    with pytest.raises(ValueError):
        detector_quality_to_beta(1.5, BETA_UNINFORMATIVE, BETA_PERFECT)


def test_compromised_nodes_score_higher_on_average_with_good_detector():
    rng = np.random.default_rng(0)
    nodes = list(range(200))
    compromised = set(range(0, 40))
    scores = generate_suspicion_scores(
        nodes, compromised, detector_quality=0.9, rng=rng,
        beta_uninformative=BETA_UNINFORMATIVE, beta_perfect=BETA_PERFECT,
    )
    mean_compromised = np.mean([scores[v] for v in compromised])
    mean_honest = np.mean([scores[v] for v in nodes if v not in compromised])
    assert mean_compromised > mean_honest


def test_uninformative_detector_gives_similar_distributions():
    rng = np.random.default_rng(0)
    nodes = list(range(2000))
    compromised = set(range(0, 400))
    scores = generate_suspicion_scores(
        nodes, compromised, detector_quality=0.0, rng=rng,
        beta_uninformative=BETA_UNINFORMATIVE, beta_perfect=BETA_PERFECT,
    )
    mean_compromised = np.mean([scores[v] for v in compromised])
    mean_honest = np.mean([scores[v] for v in nodes if v not in compromised])
    assert abs(mean_compromised - mean_honest) < 0.05


def test_suspicion_to_prob_floor_and_monotonic():
    s_v = {"a": 0.0, "b": 0.5, "c": 1.0}
    p_v = suspicion_to_prob(s_v, p_min=0.01)
    assert p_v["a"] == pytest.approx(0.01)
    assert p_v["c"] == pytest.approx(1.0)
    assert p_v["a"] < p_v["b"] < p_v["c"]


def test_calibrated_posterior_is_calibrated():
    """Among nodes the calibrated detector scores near x, about a fraction x
    are really compromised; with an uninformative detector every node gets
    the base rate."""
    import numpy as np

    from qkdmp.config import load_config
    from qkdmp.suspicion import calibrated_posterior

    cfg = load_config()["suspicion"]
    rng = np.random.default_rng(0)
    nodes = list(range(20000))
    compromised = set(rng.choice(nodes, size=4000, replace=False).tolist())  # base rate 0.2
    s_v = generate_suspicion_scores(nodes, compromised, 0.7, rng, cfg["beta_uninformative"], cfg["beta_perfect"])
    p_v = calibrated_posterior(s_v, 0.7, cfg["beta_uninformative"], cfg["beta_perfect"], 0.2, p_min=0.0)
    p = np.array([p_v[v] for v in nodes])
    truth = np.array([v in compromised for v in nodes])
    for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 1.0)]:
        mask = (p >= lo) & (p < hi)
        assert mask.sum() > 200
        expected = p[mask].mean()
        std_err = np.sqrt(expected * (1 - expected) / mask.sum())
        assert abs(truth[mask].mean() - expected) < 3 * std_err + 0.005

    flat = calibrated_posterior(s_v, 0.0, cfg["beta_uninformative"], cfg["beta_perfect"], 0.2, p_min=0.0)
    assert all(abs(x - 0.2) < 1e-9 for x in flat.values())


def test_miscalibrate_identity_and_direction():
    from qkdmp.suspicion import miscalibrate

    p_v = {"a": 0.05, "b": 0.5, "c": 0.9}
    assert miscalibrate(p_v, 1.0, 1.0, 0.01) == p_v
    over = miscalibrate(p_v, 4.0, 1.0, 0.01)
    under = miscalibrate(p_v, 0.25, 1.0, 0.01)
    assert all(under[v] < p_v[v] < over[v] for v in p_v)
    confident = miscalibrate(p_v, 1.0, 2.0, 0.01)
    assert confident["a"] < p_v["a"] and confident["c"] > p_v["c"] and abs(confident["b"] - 0.5) < 1e-9


def test_misrank_keeps_values_and_swaps_only_across_groups():
    rng = np.random.default_rng(3)
    p_v = {v: float(v) / 20 for v in range(20)}
    compromised = {15, 16, 17, 18, 19}
    out = misrank(p_v, compromised, 1.0, rng)
    assert sorted(out.values()) == sorted(p_v.values())
    for v in compromised:  # fraction 1: every compromised node took an honest node's value
        assert out[v] < 0.75
    assert misrank(p_v, compromised, 0.0, rng) == p_v


def test_misrank_zero_draws_no_random_numbers():
    config = {"suspicion": {"detector_quality": 0.7, "p_min": 0.01, "beta_uninformative": [2, 2, 2, 2],
                            "beta_perfect": [8, 1, 1, 8], "misrank_fraction": 0.0},
              "attacker": {"rho_A": 0.2}}
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    detector_p_v(list(range(10)), {1, 2}, config, a)
    config["suspicion"].pop("misrank_fraction")
    detector_p_v(list(range(10)), {1, 2}, config, b)
    assert a.random() == b.random()
