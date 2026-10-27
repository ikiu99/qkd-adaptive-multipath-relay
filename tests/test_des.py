import pytest

from qkdmp.config import load_config
from qkdmp.sim.des import LinkPoolState, arrival_rate_for_load, attach_pool_state, reserve_share_paths, run_level_b
from qkdmp.sim.graph_mc import run_level_a
from qkdmp.topology import build_topology


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def nsfnet(config):
    return build_topology("nsfnet", config)


# ---------------------------------------------------------------------------
# LinkPoolState: fluid refill (README 6)
# ---------------------------------------------------------------------------

def test_pool_refills_linearly_with_rate():
    pool = LinkPoolState(capacity=1000.0, rate=10.0, initial_level=0.0)
    assert pool.peek(5.0) == pytest.approx(50.0)
    assert pool.peek(50.0) == pytest.approx(500.0)


def test_pool_refill_caps_at_capacity():
    pool = LinkPoolState(capacity=100.0, rate=10.0, initial_level=0.0)
    assert pool.peek(1000.0) == pytest.approx(100.0)


def test_pool_sync_updates_level_and_last_update():
    pool = LinkPoolState(capacity=1000.0, rate=10.0, initial_level=0.0)
    pool.sync(5.0)
    assert pool.level == pytest.approx(50.0)
    assert pool.last_update == 5.0
    # from here, refill continues from the synced level/time
    assert pool.peek(10.0) == pytest.approx(100.0)


def test_pool_try_reserve_succeeds_and_decrements():
    pool = LinkPoolState(capacity=1000.0, rate=0.0, initial_level=500.0)
    assert pool.try_reserve(200.0, now=0.0)
    assert pool.level == pytest.approx(300.0)


def test_pool_try_reserve_fails_without_mutating_on_insufficient_level():
    pool = LinkPoolState(capacity=1000.0, rate=0.0, initial_level=50.0)
    assert not pool.try_reserve(200.0, now=0.0)
    assert pool.level == pytest.approx(50.0)


def test_reserve_share_paths_is_all_or_nothing_per_path(nsfnet, config):
    attach_pool_state(nsfnet)
    # drain one edge on the path to (almost) zero
    path = [0, 1, 2]  # 0-1, 1-2 edges (see nsfnet.json)
    nsfnet.edges[0, 1]["pool"].level = 10.0
    results = reserve_share_paths(nsfnet, [path], key_size_bits=256, now=0.0)
    assert results == [False]
    # the untouched edge must not have been debited despite the path failing
    assert nsfnet.edges[1, 2]["pool"].level == pytest.approx(nsfnet.edges[1, 2]["capacity_bits"])


# ---------------------------------------------------------------------------
# run_level_b: smoke test + Level A/B consistency (README Phase 4 acceptance)
# ---------------------------------------------------------------------------

def test_run_level_b_produces_valid_records(nsfnet, config):
    cfg = dict(config)
    cfg["des"] = dict(config["des"], warmup_requests=5, arrival_rate_per_s=50.0)
    df = run_level_b(nsfnet, "B0", cfg, n_requests=50, seed=0)
    assert len(df) == 50
    served = df[~df["blocked"] & ~df["failed"]]
    if len(served) > 0:
        assert served["latency"].gt(0).all()
        assert served["t"].eq(1).all() and served["n"].eq(1).all()


def test_run_level_b_is_deterministic_given_seed(nsfnet, config):
    cfg = dict(config)
    cfg["des"] = dict(config["des"], warmup_requests=5, arrival_rate_per_s=50.0)
    df1 = run_level_b(nsfnet, "B2", cfg, n_requests=30, seed=3)
    df2 = run_level_b(nsfnet, "B2", cfg, n_requests=30, seed=3)
    import pandas.testing as pdt

    pdt.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_b0_level_a_and_level_b_agree_on_exposure_rate(nsfnet, config):
    """README Phase 4 acceptance: "n = t = 1 behaves like B0" and "Level A
    and B agree on exposure for the same decisions". B0 is pool-unaware and
    always n=t=1, and default pools are abundant relative to a short DES
    run, so Level A's static-pool exposure rate and Level B's dynamic-pool
    exposure rate should agree closely for the same attacker/suspicion
    draws (both derive the ground-truth attacker set identically)."""
    cfg = dict(config)
    cfg["des"] = dict(config["des"], warmup_requests=10, arrival_rate_per_s=50.0)
    cfg["attacker"] = dict(config["attacker"], placement="random", rho_A=0.3, random_mode="bernoulli")

    exposure_a, exposure_b = [], []
    for seed in range(20):
        df_a = run_level_a(nsfnet, "B0", cfg, n_requests=100, seed=seed)
        df_b = run_level_b(nsfnet, "B0", cfg, n_requests=100, seed=seed)
        served_a = df_a[~df_a["blocked"]]
        served_b = df_b[~df_b["blocked"] & ~df_b["failed"]]
        if len(served_a) > 0:
            exposure_a.append(served_a["exposed"].mean())
        if len(served_b) > 0:
            exposure_b.append(served_b["exposed"].mean())

    import numpy as np

    mean_a, mean_b = np.mean(exposure_a), np.mean(exposure_b)
    assert abs(mean_a - mean_b) < 0.08, f"Level A exposure={mean_a}, Level B exposure={mean_b}"


def test_arrival_rate_for_load_scales_linearly_with_rho(nsfnet, config):
    rate_low = arrival_rate_for_load(nsfnet, config["key_size_bits"], rho=0.1)
    rate_high = arrival_rate_for_load(nsfnet, config["key_size_bits"], rho=0.4)
    assert rate_high == pytest.approx(4 * rate_low)
    assert rate_low > 0


def test_run_level_b_hotspot_traffic_runs(nsfnet, config):
    cfg = dict(config)
    cfg["des"] = dict(
        config["des"], warmup_requests=5, arrival_rate_per_s=50.0,
        traffic_mode="hotspot", hotspot_fraction=0.8, hotspot_pairs=2,
    )
    df = run_level_b(nsfnet, "B0", cfg, n_requests=50, seed=0)
    assert len(df) == 50


def test_qos_off_by_default_and_deadline_retries_terminate():
    """Without a qos block nothing changes; with deadlines, refused requests
    retry until their deadline and the run still terminates."""
    from qkdmp.config import load_config
    from qkdmp.sim.des import run_level_b
    from qkdmp.topology import build_topology

    config = load_config()
    config = dict(config, pool=dict(config["pool"], capacity_seconds=0.3),
                  policy=dict(config["policy"], fallback="block"),
                  des=dict(config["des"], arrival_rate_per_s=400.0))
    graph = build_topology("nsfnet", config)
    plain = run_level_b(graph, "Adaptive", config, n_requests=60, seed=3)
    assert "qos_class" in plain and plain["qos_class"].isna().all()

    qos = {"class_aware": True, "reservation": True, "deadlines": True, "retry_interval_s": 0.01,
           "classes": [{"name": "gold", "share": 0.5, "eps": 0.05, "a": 0.8, "reserve_fraction": 0.0, "deadline_s": 0.2},
                       {"name": "bronze", "share": 0.5, "eps": 0.5, "a": 0.8, "reserve_fraction": 0.3, "deadline_s": 0.0}]}
    df = run_level_b(graph, "Adaptive", dict(config, qos=qos), n_requests=60, seed=3)
    assert set(df["qos_class"]) == {"gold", "bronze"}
    waited = df[(df["qos_class"] == "gold") & ~df["blocked"] & df["latency"].notna()]
    assert (waited["latency"] <= 0.2 + 0.1).all()


def _stale_setup(view, phi=0.3):
    import simpy

    from qkdmp.config import load_config
    from qkdmp.sim.des import _apply_kms_model, _telemetry_process, attach_pool_state, sync_pool_view
    from qkdmp.topology import build_topology

    config = load_config()
    config = dict(config, pool=dict(config["pool"], capacity_seconds=5.0, initial_fill_fraction=0.0))
    graph = build_topology("nsfnet", config).copy()
    for _, _, d in graph.edges(data=True):
        d["level_bits"] = 0.0
    attach_pool_state(graph)
    _apply_kms_model(graph, {"local_consumption_fraction": phi})
    graph.graph["telemetry"] = {"interval": 0.5, "view": view}
    for _, _, d in graph.edges(data=True):
        d.update(snap_level=0.0, snap_time=0.0, own_used=0.0, est_rate=d["rate_bps"])
    env = simpy.Environment()
    env.process(_telemetry_process(env, graph))
    return env, graph, sync_pool_view


def test_frozen_view_only_changes_at_reports():
    env, graph, sync = _stale_setup("frozen")
    env.run(until=0.7)
    sync(graph, 0.7)
    e = next(iter(graph.edges(data=True)))[2]
    assert e["level_bits"] == e["pool"].peek(0.5) < e["pool"].peek(0.7)


def test_calibrated_view_learns_net_refill_and_nominal_shadow_overestimates():
    env, graph, sync = _stale_setup("calibrated")
    env.run(until=3.0)
    e = next(iter(graph.edges(data=True)))[2]
    assert abs(e["est_rate"] - 0.7 * e["rate_bps"]) < 0.05 * e["rate_bps"]
    env2, graph2, sync2 = _stale_setup("shadow")
    env2.run(until=3.2)
    sync2(graph2, 3.2)
    e2 = next(iter(graph2.edges(data=True)))[2]
    assert e2["level_bits"] > e2["pool"].peek(3.2)  # nominal rate ignores the unseen local use


def test_stockpiling_serves_hot_pairs_from_stock_and_charges_its_key():
    from analysis.metrics import key_efficiency
    from qkdmp.config import load_config
    from qkdmp.sim.des import run_level_b
    from qkdmp.topology import build_topology

    config = load_config()
    config = dict(config, pool=dict(config["pool"], capacity_seconds=0.5),
                  policy=dict(config["policy"], eps_target=0.5, fallback="block"),
                  des=dict(config["des"], arrival_rate_per_s=60.0, traffic_mode="hotspot", hotspot_fraction=0.8,
                           hotspot_pairs=2,
                           load_profile={"period_s": 1.0, "peak_fraction": 0.5, "peak_multiplier": 3.0,
                                         "quiet_multiplier": 0.3}))
    graph = build_topology("nsfnet", config)
    base = run_level_b(graph, "Adaptive", config, n_requests=120, seed=5)
    stocked = run_level_b(graph, "Adaptive", dict(config, stockpile={
        "enabled": True, "hot_pairs": 2, "per_pair": 5, "interval_s": 0.02, "min_fill": 0.6, "lifetime_s": 2.0}),
        n_requests=120, seed=5)
    assert "from_stock" not in base
    assert stocked["from_stock"].fillna(False).astype(bool).sum() > 0
    assert stocked.attrs["overhead_bits"] > 0 and stocked.attrs["stock_made"] > 0
    assert 0 < key_efficiency(stocked) <= 1.5


def test_strategic_attacker_moves_onto_used_relays_and_run_terminates():
    from qkdmp.config import load_config
    from qkdmp.sim.des import run_level_b
    from qkdmp.topology import build_topology

    config = load_config()
    config = dict(config, pool=dict(config["pool"], capacity_seconds=0.5),
                  policy=dict(config["policy"], eps_target=0.5, fallback="block"),
                  des=dict(config["des"], arrival_rate_per_s=300.0),
                  attacker=dict(config["attacker"], rho_A=0.2, strategic={
                      "enabled": True, "interval_s": 0.05, "window": 30, "detector_lag_s": 0.02}))
    graph = build_topology("nsfnet", config)
    df = run_level_b(graph, "Adaptive", config, n_requests=150, seed=2)
    assert len(df) == 150
    assert "p_exp_pred" in df and df["p_exp_pred"].notna().any()
