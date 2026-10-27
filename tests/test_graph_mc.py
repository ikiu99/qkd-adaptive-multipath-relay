import pytest

from analysis.metrics import (
    blocking_probability,
    key_exposure_rate,
    mean_amplification,
    service_ratio,
    summary_row,
)
from qkdmp.config import load_config
from qkdmp.sim.graph_mc import run_level_a
from qkdmp.topology import build_topology


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def nsfnet(config):
    return build_topology("nsfnet", config)


@pytest.mark.parametrize("method", ["B0", "B1", "B2", "B3"])
def test_run_level_a_produces_valid_records(nsfnet, config, method):
    df = run_level_a(nsfnet, method, config, n_requests=200, seed=1)
    assert len(df) == 200
    assert set(df["method"].unique()) <= {method}
    served = df[~df["blocked"]]
    if len(served) > 0:
        assert served["t"].le(served["n"]).all()
        assert served["bits_consumed"].gt(0).all()
        assert served["amplification"].gt(0).all()


def test_run_level_a_is_deterministic_given_seed(nsfnet, config):
    df1 = run_level_a(nsfnet, "B2", config, n_requests=50, seed=42)
    df2 = run_level_a(nsfnet, "B2", config, n_requests=50, seed=42)
    pd_assert_frame_equal(df1, df2)


def pd_assert_frame_equal(a, b):
    import pandas.testing as pdt

    pdt.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_metrics_on_level_a_output(nsfnet, config):
    df = run_level_a(nsfnet, "B0", config, n_requests=300, seed=3)
    assert 0.0 <= service_ratio(df) <= 1.0
    assert 0.0 <= blocking_probability(df) <= 1.0
    exposure = key_exposure_rate(df)
    assert exposure != exposure or 0.0 <= exposure <= 1.0  # nan or in [0,1]
    row = summary_row(df, "B0")
    assert row["method"] == "B0"
    assert row["n_requests"] == 300


def test_b0_mean_amplification_is_one(nsfnet, config):
    # B0 IS the reference cost, so its own amplification factor must be ~1.
    df = run_level_a(nsfnet, "B0", config, n_requests=200, seed=5)
    amp = mean_amplification(df)
    assert amp == pytest.approx(1.0, abs=1e-9)


def test_multipath_methods_consume_at_least_as_much_as_b0(nsfnet, config):
    for method in ("B1", "B2", "B3"):
        df = run_level_a(nsfnet, method, config, n_requests=300, seed=9)
        served = df[~df["blocked"]]
        if len(served) == 0:
            continue
        assert (served["amplification"] >= 1.0 - 1e-9).all()


def test_exclude_attacker_as_endpoint_default_true_never_samples_compromised_endpoint(nsfnet, config):
    cfg = dict(config)
    cfg["attacker"] = dict(config["attacker"], placement="random", rho_A=0.4, random_mode="bernoulli")
    df = run_level_a(nsfnet, "B0", cfg, n_requests=200, seed=1)
    # place_attacker is deterministic given the seed; recover it to check.
    import numpy as np

    from qkdmp.sim.graph_mc import place_attacker

    rng = np.random.default_rng(1)
    attacker_set = place_attacker(nsfnet, cfg["attacker"], rng)
    assert not set(df["s"]) & attacker_set
    assert not set(df["d"]) & attacker_set


@pytest.mark.parametrize("method", ["B2", "Adaptive"])
def test_adaptive_attacker_placement_runs_warmup_and_serves_requests(nsfnet, config, method):
    cfg = dict(config)
    cfg["attacker"] = dict(config["attacker"], placement="adaptive", rho_A=0.2, warmup_requests=30)
    df = run_level_a(nsfnet, method, cfg, n_requests=50, seed=0)
    assert len(df) == 50
    assert set(df["method"].unique()) <= {method}


def test_exclude_attacker_as_endpoint_false_allows_compromised_endpoints(nsfnet, config):
    # With a high rho_A and endpoint exclusion disabled, some sampled
    # requests should have an attacker-controlled endpoint (this is the
    # setting E2(a) uses to avoid the selection bias described in
    # run_level_a's docstring).
    cfg = dict(config)
    cfg["attacker"] = dict(config["attacker"], placement="random", rho_A=0.5, random_mode="bernoulli")
    df = run_level_a(nsfnet, "B0", cfg, n_requests=300, seed=1, exclude_attacker_as_endpoint=False)

    import numpy as np

    from qkdmp.sim.graph_mc import place_attacker

    rng = np.random.default_rng(1)
    attacker_set = place_attacker(nsfnet, cfg["attacker"], rng)
    assert (set(df["s"]) | set(df["d"])) & attacker_set
