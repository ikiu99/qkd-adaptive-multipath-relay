import numpy as np
import pytest

from qkdmp.config import load_config
from qkdmp.policies.base import BLOCK, Request
from qkdmp.paths import path_edges
from qkdmp.policies.rl import (
    LEGACY_DESIGN,
    CheapestOptionPolicy,
    QKDRoutingEnv,
    RLEnvDesign,
    _enumerate_options,
    obs_size,
    pool_potential,
    train_rl_policy,
)
from qkdmp.suspicion import generate_suspicion_scores, suspicion_to_prob
from qkdmp.topology import build_topology


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def nsfnet(config):
    return build_topology("nsfnet", config)


def _p_v(graph, config):
    rng = np.random.default_rng(0)
    s_v = generate_suspicion_scores(
        list(graph.nodes()), set(), config["suspicion"]["detector_quality"], rng,
        config["suspicion"]["beta_uninformative"], config["suspicion"]["beta_perfect"],
    )
    return suspicion_to_prob(s_v, config["suspicion"]["p_min"])


def test_enumerate_options_respects_block_fallback(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    cfg = dict(config)
    cfg["policy"] = dict(config["policy"], eps_target=0.0, a_target=1.0, fallback="block")
    rng = np.random.default_rng(0)
    request = Request(0, 9, eps_target=0.0, a_target=1.0)
    options = _enumerate_options(nsfnet, request, p_v, cfg, rng)
    assert options == []  # impossible targets + block fallback -> no options offered


def test_enumerate_options_best_effort_offers_infeasible_options(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    cfg = dict(config)
    cfg["policy"] = dict(config["policy"], eps_target=0.0, a_target=1.0, fallback="best_effort")
    rng = np.random.default_rng(0)
    request = Request(0, 9, eps_target=0.0, a_target=1.0)
    options = _enumerate_options(nsfnet, request, p_v, cfg, rng)
    assert len(options) > 0  # best_effort -> falls back to all_tried


def test_env_reset_and_step_shapes(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    env = QKDRoutingEnv(nsfnet, config, p_v, max_options=8, episode_requests=5, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    mask = env.action_masks()
    assert mask.shape == (9,)
    assert mask[8]  # block always legal

    obs, reward, terminated, truncated, info = env.step(8)  # always take "block"
    assert obs.shape == env.observation_space.shape
    assert reward == 0.0
    assert not terminated


def test_env_episode_terminates_after_episode_requests(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    env = QKDRoutingEnv(nsfnet, config, p_v, max_options=8, episode_requests=3, seed=0)
    env.reset(seed=0)
    terminated = False
    steps = 0
    for _ in range(3):
        _, _, terminated, _, _ = env.step(8)
        steps += 1
    assert terminated
    assert steps == 3


def test_train_rl_policy_smoke(nsfnet, config):
    p_v = _p_v(nsfnet, config)
    policy, rewards = train_rl_policy(
        nsfnet, config, p_v, total_timesteps=200, max_options=4, episode_requests=10, seed=0
    )
    decision = policy.decide(Request(0, 9, config["policy"]["eps_target"], config["policy"]["a_target"]), nsfnet, p_v, config)
    assert decision is BLOCK or decision.n >= 1
    assert isinstance(rewards, list)


# ---------------------------------------------------------------------------
# Phase 7 redesign (RLEnvDesign): one test per fix
# ---------------------------------------------------------------------------

def _loose(config):
    cfg = dict(config)
    cfg["policy"] = dict(config["policy"], eps_target=0.99, a_target=0.01, fallback="block")
    return cfg


def test_align_pool_masks_options_that_cannot_be_reserved(nsfnet, config):
    cfg = _loose(config)
    p_v = _p_v(nsfnet, cfg)
    request = Request(0, 9, 0.99, 0.01)
    rng = np.random.default_rng(0)
    before = _enumerate_options(nsfnet, request, p_v, cfg, rng, RLEnvDesign())
    assert before
    # Starve one link used by some option below one key's worth of bits, but
    # keep it above the reserve_fraction guard so _pool_ok still admits it --
    # exactly the case the legacy env silently rewarded.
    u, v = path_edges(before[0]["paths"][0])[0]
    nsfnet.edges[u, v]["capacity_bits"] = 1000.0
    nsfnet.edges[u, v]["level_bits"] = cfg["key_size_bits"] - 1
    legacy = _enumerate_options(nsfnet, request, p_v, cfg, np.random.default_rng(0), LEGACY_DESIGN)
    aligned = _enumerate_options(nsfnet, request, p_v, cfg, np.random.default_rng(0), RLEnvDesign())

    def uses(opt):
        return any({a, b} == {u, v} for p in opt["paths"] for a, b in path_edges(p))

    assert any(uses(o) for o in legacy)
    assert not any(uses(o) for o in aligned)


def test_sort_options_is_cheapest_first(nsfnet, config):
    cfg = _loose(config)
    options = _enumerate_options(
        nsfnet, Request(0, 9, 0.99, 0.01), _p_v(nsfnet, cfg), cfg, np.random.default_rng(0), RLEnvDesign()
    )
    costs = [o["cost"] for o in options]
    assert costs == sorted(costs)


def test_rich_observation_shape_and_range(nsfnet, config):
    cfg = _loose(config)
    env = QKDRoutingEnv(nsfnet, cfg, _p_v(nsfnet, cfg), max_options=8, episode_requests=5, design=RLEnvDesign())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (obs_size(8, RLEnvDesign()),)
    assert env.observation_space.contains(obs)


def test_legacy_design_keeps_phase5_observation_size(nsfnet, config):
    env = QKDRoutingEnv(nsfnet, config, _p_v(nsfnet, config), max_options=8, episode_requests=5)
    assert env.observation_space.shape == (8 * 3 + 2,)


def test_shaping_penalizes_spending_pool(nsfnet, config):
    cfg = _loose(config)
    design = RLEnvDesign(shaping_coef=5.0)
    env = QKDRoutingEnv(nsfnet, cfg, _p_v(nsfnet, cfg), max_options=8, episode_requests=10, design=design)
    env.reset(seed=0)
    phi = pool_potential(nsfnet)
    _, r_block, _, _, _ = env.step(8)
    assert r_block == pytest.approx(5.0 * (env.gamma - 1.0) * phi, rel=1e-6)
    assert env.action_masks()[0]
    _, r_serve, _, _, _ = env.step(0)
    assert r_serve < 1.0  # +1 for serving, minus the key material it spent


def test_aligned_env_never_overdraws_under_random_actions(nsfnet, config):
    cfg = _loose(config)
    cfg["pool"] = dict(config["pool"], capacity_seconds=0.3)
    graph = build_topology("nsfnet", cfg)
    env = QKDRoutingEnv(graph, cfg, _p_v(graph, cfg), max_options=8, episode_requests=60, design=RLEnvDesign())
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(60):
        valid = np.flatnonzero(env.action_masks())
        env.step(int(rng.choice(valid)))
    assert min(d["level_bits"] for _, _, d in graph.edges(data=True)) >= 0


def test_episode_service_ratio_reported_at_termination(nsfnet, config):
    cfg = _loose(config)
    env = QKDRoutingEnv(nsfnet, cfg, _p_v(nsfnet, cfg), max_options=8, episode_requests=3, design=RLEnvDesign())
    env.reset(seed=0)
    info = {}
    for _ in range(3):
        _, _, _, _, info = env.step(8)
    assert info["episode_service_ratio"] == 0.0


def test_cheapest_option_policy_picks_minimum_cost(nsfnet, config):
    cfg = _loose(config)
    p_v = _p_v(nsfnet, cfg)
    request = Request(0, 9, 0.99, 0.01)
    decision = CheapestOptionPolicy().decide(request, nsfnet, p_v, cfg)
    options = _enumerate_options(nsfnet, request, p_v, cfg, np.random.default_rng(0), RLEnvDesign())
    assert decision.cost == pytest.approx(min(o["cost"] for o in options))
