"""RL comparison policy (README 4.4): a Gymnasium env over pre-enumerated
per-request options, trained with sb3-contrib MaskablePPO. RL is a
comparison point, not the main method.

# ASSUMPTION: README doesn't specify a within-episode pool dynamics model
# for training (it only fully specifies this for Level B). One episode here
# is `episode_requests` sequential requests against a *shared, depleting,
# non-refilling* pool budget on one topology -- the simplest model where
# pool state still meaningfully carries across steps. Pools reset to their
# initial level at the start of every episode.

Phase 7 redesign (`RLEnvDesign`). The Phase 5 env plateaued after ~50
episodes and neither 6x more steps nor a larger entropy bonus helped. A
diagnosis (reports/PHASE_7_REPORT.md) found three design causes, each fixed
by one switch so their effects can be ablated separately:

- `align_pool`: the Phase 5 env paid +1 for choosing an option even when a
  link on it lacked the key bits (the pool silently went negative), while
  evaluation counts that same choice as a failure. Options whose shares
  can't actually be reserved are now masked out, so training reward and
  evaluation agree.
- `sort_options` + `rich_obs`: options were shown in enumeration order and
  truncated arbitrarily at `max_options`, with cost scaled by a fixed
  constant and no pool-headroom information. Options are now sorted by
  pool-weighted cost (cheapest first, so truncation keeps the best ones and
  index 0 is always the heuristic's own choice), and each carries its
  cost relative to the others, its bottleneck-link headroom, and (t, n).
- `shaping_coef`: the only per-option reward difference was
  `lambda_cost * cost` (~1e-6), so every non-block action looked equally
  good and the agent had nothing to learn once it stopped blocking.
  Potential-based shaping (Ng, Harada & Russell 1999) with potential =
  mean normalized pool headroom now gives an immediate signal for key
  material spent, without changing which policy is optimal.

`LEGACY_DESIGN` reproduces the Phase 5 env exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qkdmp.paths import candidate_paths, path_edges
from qkdmp.policies.adaptive import _pool_ok, option_reservable, pool_candidate_paths, search_options
from qkdmp.policies.base import BLOCK, Block, Decision, Request

# Legacy observation only: fixed cost scale (no principled scale exists).
COST_OBS_SCALE = 10.0
LEGACY_FEATURES = 3
RICH_FEATURES = 7


@dataclass(frozen=True)
class RLEnvDesign:
    align_pool: bool = True
    sort_options: bool = True
    rich_obs: bool = True
    shaping_coef: float = 5.0


LEGACY_DESIGN = RLEnvDesign(align_pool=False, sort_options=False, rich_obs=False, shaping_coef=0.0)


def _headroom(data: dict) -> float:
    cap = data["capacity_bits"]
    return float(min(1.0, max(0.0, data["level_bits"] / cap))) if cap > 0 else 0.0


def pool_potential(graph: Any) -> float:
    """Mean normalized headroom over all links, in [0, 1]."""
    levels = [_headroom(data) for _, _, data in graph.edges(data=True)]
    return float(np.mean(levels)) if levels else 0.0


def _bottleneck_headroom(option: dict, graph: Any) -> float:
    return min(_headroom(graph.edges[u, v]) for p in option["paths"] for u, v in path_edges(p))


def obs_size(max_options: int, design: RLEnvDesign) -> int:
    if design.rich_obs:
        return max_options * RICH_FEATURES + 3
    return max_options * LEGACY_FEATURES + 2


def _request_observation(
    options: list[dict], request: Request, max_options: int,
    graph: Any = None, design: RLEnvDesign = LEGACY_DESIGN, n_scale: int = 4,
) -> np.ndarray:
    shown = options[:max_options]
    obs = np.zeros(obs_size(max_options, design), dtype=np.float32)
    if not design.rich_obs:
        for i, opt in enumerate(shown):
            obs[i * 3 + 0] = opt["p_exp"]
            obs[i * 3 + 1] = opt["p_del"]
            obs[i * 3 + 2] = min(1.0, opt["cost"] / COST_OBS_SCALE)
        obs[-2] = request.eps_target
        obs[-1] = request.a_target
        return obs

    max_cost = max((o["cost"] for o in shown), default=0.0) or 1.0
    for i, opt in enumerate(shown):
        base = i * RICH_FEATURES
        obs[base + 0] = opt["p_exp"]
        obs[base + 1] = opt["p_del"]
        obs[base + 2] = opt["cost"] / max_cost
        obs[base + 3] = _bottleneck_headroom(opt, graph)
        obs[base + 4] = min(1.0, opt["n"] / n_scale)
        obs[base + 5] = min(1.0, opt["t"] / n_scale)
        obs[base + 6] = 1.0
    obs[-3] = request.eps_target
    obs[-2] = request.a_target
    obs[-1] = pool_potential(graph)
    return obs


def _enumerate_options(
    graph: Any, request: Request, p_v: dict, config: dict, rng: np.random.Generator,
    design: RLEnvDesign = LEGACY_DESIGN,
) -> list[dict]:
    paths_cfg = config["paths"]
    attacker_cfg = config["attacker"]
    if design.align_pool:  # same pool-pruned candidate search as Adaptive
        candidates = pool_candidate_paths(
            graph, request.s, request.d, paths_cfg["n_max"], paths_cfg["reserve_fraction"], config["key_size_bits"]
        )
    else:  # legacy (Phase 5) env, kept reproducible
        candidates = candidate_paths(graph, request.s, request.d, paths_cfg["n_max"])
        candidates = [p for p in candidates if _pool_ok(p, graph, paths_cfg["reserve_fraction"])]
    if not candidates:
        return []
    feasible, all_tried = search_options(
        candidates, graph, p_v, request, config["key_size_bits"],
        attacker_cfg["p_drop"], attacker_cfg["p_tamper"], rng,
        max_subsets=paths_cfg["max_subsets"],
    )
    # Match AdaptiveThresholdPolicy/ExhaustivePolicy's `fallback` semantics
    # (README 4.1): only offer options that violate eps_target/a_target when
    # fallback == "best_effort" (found via E9: without this, RL appeared to
    # serve more requests than the batch-ILP upper bound).
    if feasible:
        options = feasible
    elif config["policy"]["fallback"] == "best_effort":
        options = all_tried
    else:
        return []

    if design.align_pool:
        options = [o for o in options if option_reservable(o, graph, config["key_size_bits"])]
    if design.sort_options:
        options = sorted(options, key=lambda o: (o["cost"], o["n"], o["p_exp"]))
    return options


class QKDRoutingEnv(gym.Env):
    """Observation: per shown option (up to `max_options`) its features --
    [p_exp, p_del, cost] (legacy) or [p_exp, p_del, relative cost,
    bottleneck headroom, n, t, present] (rich) -- plus request/pool-level
    globals. Action: option index, or `max_options` = block.
    Reward: +1 served, -lambda_cost*cost, -10 if the chosen option's P_exp
    exceeds eps_target (README 4.4), plus potential-based shaping when
    `design.shaping_coef > 0`."""

    metadata: dict = {}

    def __init__(
        self, graph: Any, config: dict, p_v: dict,
        max_options: int = 16, episode_requests: int = 100, seed: int = 0, lambda_cost: float = 1e-6,
        design: RLEnvDesign = LEGACY_DESIGN, endpoint_exclude: frozenset = frozenset(), gamma: float = 0.99,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.config = config
        self.p_v = p_v
        self.max_options = max_options
        self.episode_requests = episode_requests
        self.lambda_cost = lambda_cost
        self.design = design
        self.gamma = gamma
        self._rng = np.random.default_rng(seed)
        self._nodes = [v for v in graph.nodes() if v not in endpoint_exclude]
        self._n_scale = config["paths"]["n_max"]

        self.action_space = spaces.Discrete(max_options + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size(max_options, design),), dtype=np.float32
        )

        self._step_idx = 0
        self._episode_served = 0
        self._current_request: Request | None = None
        self._current_options: list[dict] = []

    def _reset_pools(self) -> None:
        for _, _, data in self.graph.edges(data=True):
            data["level_bits"] = data["capacity_bits"] * self.config["pool"]["initial_fill_fraction"]

    def _draw_next(self) -> None:
        s, d = self._rng.choice(np.array(self._nodes, dtype=object), size=2, replace=False)
        request = Request(s, d, self.config["policy"]["eps_target"], self.config["policy"]["a_target"])
        self._current_request = request
        options = _enumerate_options(self.graph, request, self.p_v, self.config, self._rng, self.design)
        self._current_options = options[: self.max_options]

    def _observation(self) -> np.ndarray:
        return _request_observation(
            self._current_options, self._current_request, self.max_options,
            graph=self.graph, design=self.design, n_scale=self._n_scale,
        )

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.max_options + 1, dtype=bool)
        mask[: len(self._current_options)] = True
        mask[self.max_options] = True
        return mask

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_pools()
        self._step_idx = 0
        self._episode_served = 0
        self._draw_next()
        return self._observation(), {}

    def step(self, action: int):
        action = int(action)
        options = self._current_options
        shaping = self.design.shaping_coef > 0
        phi_before = pool_potential(self.graph) if shaping else 0.0

        if action == self.max_options or action >= len(options):
            reward = 0.0
            served = False
        else:
            opt = options[action]
            for path in opt["paths"]:
                for u, v in path_edges(path):
                    self.graph.edges[u, v]["level_bits"] -= self.config["key_size_bits"]
            reward = 1.0 - self.lambda_cost * opt["cost"]
            if opt["p_exp"] > self._current_request.eps_target:
                reward -= 10.0
            served = True

        self._step_idx += 1
        self._episode_served += int(served)
        terminated = self._step_idx >= self.episode_requests
        if shaping:
            # Phi(terminal) = 0 keeps the shaping policy-invariant in an
            # episodic task (Ng, Harada & Russell 1999).
            phi_after = 0.0 if terminated else pool_potential(self.graph)
            reward += self.design.shaping_coef * (self.gamma * phi_after - phi_before)

        info: dict[str, Any] = {"served": served}
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info["episode_service_ratio"] = self._episode_served / self.episode_requests
        else:
            self._draw_next()
            obs = self._observation()
        return obs, reward, terminated, False, info


class RLPolicy:
    """Wraps a trained MaskablePPO model as a `Policy` (README 11) for
    Level A/B evaluation. Uses the same option enumeration, ordering and
    observation as the env it was trained in."""

    name = "RL"

    def __init__(self, model: Any, max_options: int, seed: int = 0, design: RLEnvDesign = LEGACY_DESIGN) -> None:
        self.model = model
        self.max_options = max_options
        self.design = design
        self._rng = np.random.default_rng(seed)

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        options = _enumerate_options(graph, request, p_v, config, self._rng, self.design)[: self.max_options]
        if not options:
            return BLOCK

        obs = _request_observation(
            options, request, self.max_options, graph=graph, design=self.design,
            n_scale=config["paths"]["n_max"],
        )
        mask = np.zeros(self.max_options + 1, dtype=bool)
        mask[: len(options)] = True
        mask[self.max_options] = True

        action, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)
        if action >= len(options):
            return BLOCK
        opt = options[action]
        return Decision(
            paths=opt["paths"], t=opt["t"], n=opt["n"], scheme="shamir",
            p_exp=opt["p_exp"], p_del=opt["p_del"], cost=opt["cost"],
        )


class CheapestOptionPolicy:
    """Non-learned reference for the RL diagnosis: always the cheapest
    reservable feasible option -- i.e. action 0 under the sorted design.
    It is what a trained agent can represent trivially, so comparing RL
    against it separates "RL learned something" from "RL learned to pick
    index 0"."""

    name = "Cheapest"

    def __init__(self, seed: int = 0) -> None:
        self.design = RLEnvDesign(align_pool=True, sort_options=True, rich_obs=False, shaping_coef=0.0)
        self._rng = np.random.default_rng(seed)

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block:
        options = _enumerate_options(graph, request, p_v, config, self._rng, self.design)
        if not options:
            return BLOCK
        opt = options[0]
        return Decision(
            paths=opt["paths"], t=opt["t"], n=opt["n"], scheme="shamir",
            p_exp=opt["p_exp"], p_del=opt["p_del"], cost=opt["cost"],
        )


def train_rl_policy(
    graph: Any, config: dict, p_v: dict,
    total_timesteps: int = 20_000, max_options: int = 16, episode_requests: int = 100, seed: int = 0,
    ent_coef: float = 0.0, learning_rate: float = 3e-4,
    design: RLEnvDesign = LEGACY_DESIGN, endpoint_exclude: frozenset = frozenset(),
):
    """README 4.4: train with sb3-contrib MaskablePPO. Report the training
    curve (README Phase 5 acceptance) -- returns `(policy, curve)` where
    `curve` is the per-episode service ratio (served / requests). That is
    comparable across designs, unlike the raw episode reward, which includes
    the shaping term when `design.shaping_coef > 0`.

    # ASSUMPTION: README's default training budget is 2e5 steps per
    # topology; `total_timesteps` defaults lower here so a full E9 sweep
    # across several topologies finishes within this session's time budget.
    # Scale up before the numbers go into the thesis (see the phase report).

    `ent_coef` (default 0.0, matching SB3's own PPO default): the first E9
    pass (Phase 5) found the training curve plateaus after ~50 episodes and
    stays flat even with 6x more steps -- consistent with early convergence
    to a local optimum rather than an undertrained policy. `ent_coef > 0`
    adds an entropy bonus to the loss that keeps the policy more stochastic
    for longer, encouraging continued exploration past that point; set it
    positive (e.g. 0.01-0.05) to test whether more exploration escapes the
    plateau (see reports/PHASE_7_REPORT.md for the result of that test).
    """
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    class _ServiceRatioCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.service_ratios: list[float] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ratio = info.get("episode_service_ratio")
                if ratio is not None:
                    self.service_ratios.append(float(ratio))
            return True

    def mask_fn(env: gym.Env) -> np.ndarray:
        # Monitor doesn't forward arbitrary method calls to the wrapped env.
        return env.unwrapped.action_masks()

    gamma = 0.99  # MaskablePPO's default; the shaping term must use the same discount
    base_env = QKDRoutingEnv(
        graph, config, p_v, max_options=max_options, episode_requests=episode_requests, seed=seed,
        design=design, endpoint_exclude=endpoint_exclude, gamma=gamma,
    )
    env = ActionMasker(Monitor(base_env), mask_fn)
    model = MaskablePPO(
        "MlpPolicy", env, verbose=0, seed=seed, ent_coef=ent_coef, learning_rate=learning_rate, gamma=gamma,
    )
    callback = _ServiceRatioCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return RLPolicy(model, max_options=max_options, seed=seed, design=design), callback.service_ratios
