"""Level B -- discrete-event simulation with SimPy (README Section 6).

Pools are fluid: refill is computed lazily at access time (`LinkPoolState`),
not via per-bit events. Requests arrive as a Poisson (or hotspot) process;
each spawns a SimPy process that reads live pool levels, asks the policy to
decide, incurs control latency, atomically reserves pool bits per share
path, incurs per-hop transport latency, then applies the same ground-truth
drop/tamper/reconstruct pipeline Level A uses (`qkdmp.sim.graph_mc.
simulate_delivery`) -- same policies, same outcome logic, only the pool/time
model differs (README 6: "Level A and Level B must share the same policy
objects; policies must not know which level calls them").
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import simpy

from qkdmp import attacker as attacker_mod
from qkdmp import security
from qkdmp.paths import path_edges, relay_nodes
from qkdmp.policies.adaptive import pool_candidate_paths
from qkdmp.policies.base import BLOCK, Decision, Request
from qkdmp.sim.graph_mc import (
    b0_reference_cost,
    bits_consumed,
    exposed_ground_truth,
    get_policy,
    place_attacker,
    sample_request_pair,
    simulate_delivery,
)
from qkdmp.suspicion import detector_p_v
from qkdmp.suspicion_learning import P_MAX, SuspicionLearner, ranking_auc

PROPAGATION_SPEED_KM_S = 2.0e5  # README 6: propagation latency = L_e / 2e5 km/s


class LinkPoolState:
    """One link's key pool, refilled as a fluid at rate `rate_bps`, capped at
    `capacity_bits`. Level is only ever materialized lazily, at `peek`/`sync`
    time (README 6: "implement as fluid refill computed lazily at access
    time, not per-bit events")."""

    __slots__ = ("capacity", "rate", "level", "last_update", "depletion_events")

    def __init__(self, capacity: float, rate: float, initial_level: float):
        self.capacity = capacity
        self.rate = rate
        self.level = initial_level
        self.last_update = 0.0
        self.depletion_events = 0

    def peek(self, now: float) -> float:
        elapsed = now - self.last_update
        if elapsed <= 0:
            return self.level
        return min(self.capacity, self.level + self.rate * elapsed)

    def sync(self, now: float) -> None:
        self.level = self.peek(now)
        self.last_update = now

    def try_reserve(self, bits: float, now: float) -> bool:
        self.sync(now)
        if self.level < bits:
            return False
        self.level -= bits
        if self.level < bits:  # README 8: "pool depletion events" -- < 1 key remaining
            self.depletion_events += 1
        return True


def _apply_kms_model(graph: nx.Graph, kms: dict | None) -> None:
    """Thesis extension (package 4, off by default): a more realistic KMS.
    - local_consumption_fraction phi: a share of every link's generated key is
      used locally -- keys handed straight to neighbouring nodes (ETSI GS QKD
      014 style) and authentication of the QKD link itself -- so the pool
      refills at (1 - phi) * rate. The controller is not told phi.
    - key_lifetime_s L: stored key expires after L seconds, so a pool can
      never hold more than L seconds of generation: capacity is capped at
      rate * L (fluid approximation of per-bit expiry; the controller knows
      this policy, so capacity_bits is capped too).
    # ASSUMPTION: phi and L are uniform across links; values are swept.
    """
    if not kms:
        return
    phi = kms.get("local_consumption_fraction", 0.0)
    lifetime = kms.get("key_lifetime_s")
    for _, _, data in graph.edges(data=True):
        pool = data["pool"]
        pool.rate = data["rate_bps"] * (1.0 - phi)
        if lifetime:
            cap = min(pool.capacity, data["rate_bps"] * lifetime)
            pool.capacity = cap
            pool.level = min(pool.level, cap)
            data["capacity_bits"] = cap
            data["level_bits"] = min(data["level_bits"], cap)


def attach_pool_state(graph: nx.Graph) -> None:
    """Create one LinkPoolState per edge from its static capacity/rate/level_bits
    (set once by `qkdmp.topology.apply_rate_model`) and store it as edge
    attribute 'pool'. Also (re)initializes 'level_bits' as the policy-visible
    snapshot `sync_pool_view` keeps in sync with the pool's true fluid level.
    """
    for _, _, data in graph.edges(data=True):
        data["pool"] = LinkPoolState(data["capacity_bits"], data["rate_bps"], data["level_bits"])


def sync_pool_view(graph: nx.Graph, now: float) -> None:
    """Refresh every edge's policy-visible 'level_bits' -- called right before
    `policy.decide` so policies (which read `graph.edges[u, v]['level_bits']`
    exactly as they do in Level A) see the pool state.

    By default the controller sees each pool's true current level. With
    periodic telemetry (thesis extension, `config["controller"]`), it only
    knows the level reported at the last telemetry instant, and fills the gap
    per `view`:
    - "frozen": the last report, unchanged;
    - "shadow": last report + nominal refill (rate_bps) since, minus the key
      it has itself allocated since (the controller knows its own decisions);
    - "calibrated": as "shadow", but with each link's *net* refill rate
      estimated from successive reports -- which also captures consumption
      the controller never sees (local key use, authentication overhead).
    """
    tel = graph.graph.get("telemetry")
    for _, _, data in graph.edges(data=True):
        if tel is None:
            data["level_bits"] = data["pool"].peek(now)
        elif tel["view"] == "frozen":
            data["level_bits"] = data["snap_level"]
        else:
            rate = data["rate_bps"] if tel["view"] == "shadow" else data["est_rate"]
            # own allocations are subtracted *before* the capacity cap: key taken
            # since the report is refilled too (capping first under-estimates
            # a full pool that is being drawn on)
            level = data["snap_level"] + rate * (now - data["snap_time"]) - data["own_used"]
            data["level_bits"] = min(data["capacity_bits"], max(0.0, level))


def note_own_use(graph: nx.Graph, paths: list[list], key_size_bits: int) -> None:
    """Controller-side bookkeeping of key it allocated since the last report."""
    if graph.graph.get("telemetry") is None:
        return
    for path in paths:
        for u, v in path_edges(path):
            graph.edges[u, v]["own_used"] += key_size_bits


def _telemetry_process(env: simpy.Environment, graph: nx.Graph):
    """Every `interval` seconds, report every pool's true level to the
    controller; in "calibrated" view also update the net refill-rate
    estimate (EWMA over reports, skipped while the pool sits at capacity,
    where the refill is not observable)."""
    tel = graph.graph["telemetry"]
    while not _run_finished(graph):
        yield env.timeout(tel["interval"])
        for _, _, data in graph.edges(data=True):
            true = data["pool"].peek(env.now)
            cap = data["capacity_bits"]
            if tel["view"] == "calibrated" and true < 0.98 * cap and data["snap_level"] < 0.98 * cap:
                observed = (true - data["snap_level"] + data["own_used"]) / tel["interval"]
                data["est_rate"] = 0.5 * data["est_rate"] + 0.5 * max(0.0, observed)
            data["snap_level"], data["snap_time"], data["own_used"] = true, env.now, 0.0


def reserve_share_paths(graph: nx.Graph, paths: list[list], key_size_bits: int, now: float) -> list[bool]:
    """README 2.3: reservation is atomic per share path -- if any link on a
    path lacks the pool, that share (only that share) cannot be sent."""
    results = []
    for path in paths:
        edges = [graph.edges[u, v] for u, v in path_edges(path)]
        pools = [e["pool"] for e in edges]
        for p in pools:
            p.sync(now)
        if all(p.level >= key_size_bits for p in pools):
            for p in pools:
                p.level -= key_size_bits
                if p.level < key_size_bits:
                    p.depletion_events += 1
            results.append(True)
        else:
            results.append(False)
    return results


def path_latency(graph: nx.Graph, path: list, hop_processing_s: float) -> float:
    """README 6: per-hop latency = propagation (L_e / 2e5 km/s) + hop_processing."""
    total = 0.0
    for u, v in path_edges(path):
        length_km = graph.edges[u, v]["length_km_norm"]
        total += length_km / PROPAGATION_SPEED_KM_S + hop_processing_s
    return total


def poisson_interarrival_times(rng: np.random.Generator, rate_per_s: float, n: int) -> np.ndarray:
    return rng.exponential(1.0 / rate_per_s, size=n)


def arrival_rate_for_load(graph: nx.Graph, key_size_bits: int, rho: float) -> float:
    """README 2.5: normalized load factor rho = offered end-to-end key
    bits/s / (sum of link rates / mean shortest-path hop count). Inverts
    this to the Poisson arrival rate (requests/s) `des.arrival_rate_per_s`
    that produces a given rho on `graph`."""
    total_rate = sum(data["rate_bps"] for _, _, data in graph.edges(data=True))
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    hops = [lengths[u][v] for u in graph.nodes() for v in graph.nodes() if u != v and v in lengths[u]]
    mean_hops = float(np.mean(hops))
    offered_bps = rho * total_rate / mean_hops
    return offered_bps / key_size_bits


def sample_traffic_pair(
    nodes: list, exclude: set, rng: np.random.Generator, traffic_cfg: dict, hotspot_pairs: list[tuple] | None
) -> tuple:
    """README 2.5: uniform random (s, d), or a hotspot_fraction of traffic
    pinned to a small set of fixed pairs."""
    if traffic_cfg["traffic_mode"] == "hotspot" and hotspot_pairs:
        if rng.random() < traffic_cfg["hotspot_fraction"]:
            pair = hotspot_pairs[rng.integers(0, len(hotspot_pairs))]
            if pair[0] not in exclude and pair[1] not in exclude:
                return pair
    return sample_request_pair(nodes, exclude, rng)


def _send_probes(graph, config, s, d, attacker_set, rng, learner: SuspicionLearner, now: float) -> float:
    """Probe up to `learner.probe_paths` candidate paths of a refused (s, d)
    request with dummy shares (see `qkdmp.suspicion_learning`); returns the
    pool bits the probes consumed."""
    paths_cfg = config["paths"]
    key_size_bits = config["key_size_bits"]
    attacker_cfg = config["attacker"]
    candidates = pool_candidate_paths(graph, s, d, paths_cfg["n_max"], paths_cfg["reserve_fraction"], key_size_bits)
    candidates = [p for p in candidates if relay_nodes(p)][: learner.probe_paths]
    reserved = reserve_share_paths(graph, candidates, key_size_bits, now)
    sent = [p for p, ok in zip(candidates, reserved) if ok]
    note_own_use(graph, sent, key_size_bits)
    outcome_to_obs = {"dropped": "dropped", "tampered": "tampered", "delivered": "clean"}
    observations = [
        outcome_to_obs[attacker_mod.path_outcome(rng, relay_nodes(p), attacker_set,
                                                 attacker_cfg["p_drop"], attacker_cfg["p_tamper"])]
        for p in sent
    ]
    learner.observe(sent, observations)
    return float(sum((len(p) - 1) * key_size_bits for p in sent))


def _strategic_attacker_process(env, graph, config, nodes, attacker_set, p_v, rng):
    """Robustness extension (off by default): a strategic attacker that
    watches the routing of the method under test and, every `interval_s`,
    moves its compromised relays (same number) onto the relays that method
    used most over its last `window` deliveries. The attacker set is
    mutated in place, so ground-truth exposure follows it at once.
    The detector re-scores the new set after `detector_lag_s` seconds
    (None: never -- the detector keeps describing the original set).
    # ASSUMPTION: the attacker can observe which relays carry shares (e.g.
    # via traffic analysis) and can re-compromise nodes instantly.
    """
    cfg = config["attacker"]["strategic"]
    usage = graph.graph["relay_usage"]
    k = len(attacker_set)
    while not _run_finished(graph):
        yield env.timeout(cfg["interval_s"])
        counts: dict = {}
        for relays in list(usage)[-cfg["window"]:]:
            for v in relays:
                counts[v] = counts.get(v, 0) + 1
        if not counts:
            continue
        ranked = sorted(counts, key=lambda v: (-counts[v], str(v)))
        new_set = set(ranked[:k])
        if len(new_set) < k:  # too few observed relays: keep the rest of the old set
            new_set |= set(sorted(attacker_set - new_set, key=str)[: k - len(new_set)])
        attacker_set.clear()
        attacker_set.update(new_set)
        graph.graph["attacker_moves"] += 1
        if cfg.get("detector_lag_s") is not None:
            env.process(_detector_catch_up(env, cfg["detector_lag_s"], nodes, frozenset(new_set), config, p_v, rng))


def _detector_catch_up(env, lag, nodes, observed_set, config, p_v, rng):
    yield env.timeout(lag)
    p_v.update(detector_p_v(nodes, set(observed_set), config, rng))


def _run_finished(graph: nx.Graph) -> bool:
    return graph.graph.get("arrivals_done", False) and graph.graph.get("active", 0) == 0


def _tracked(graph: nx.Graph, gen):
    """Wrap a request process so background processes know when to stop."""
    graph.graph["active"] = graph.graph.get("active", 0) + 1
    yield from gen
    graph.graph["active"] -= 1


def _take_stock_key(graph: nx.Graph, pair: frozenset, now: float) -> dict | None:
    """Oldest unexpired proactive key for `pair`, or None; expired keys are
    discarded (their key material was already counted when they were made)."""
    stock = graph.graph["stock"]
    keys = stock["store"].get(pair, [])
    while keys and now - keys[0]["created"] > stock["lifetime_s"]:
        keys.pop(0)
        stock["expired"] += 1
    return keys.pop(0) if keys else None


def _stockpile_process(env, graph, policy, config, attacker_set, p_v, rng):
    """Thesis extension (package 3, off by default): proactive key
    stockpiling. Every `interval_s`, for the `hot_pairs` node pairs with the
    most requests so far, build end-to-end keys ahead of demand (up to
    `per_pair` in stock) with the same policy and security target as live
    requests, but only from surplus key: links below `min_fill` of capacity
    are off-limits (a reserve floor), so live traffic keeps priority. The
    key material spent -- including on keys that later expire -- is tallied
    in `graph.graph["stock"]["bits"]` and charged in key efficiency.
    """
    stock = graph.graph["stock"]
    cfg = config["stockpile"]
    key_size_bits = config["key_size_bits"]
    policy_cfg = config["policy"]
    decide_cfg = dict(config, paths=dict(config["paths"], reserve_fraction=cfg["min_fill"]))
    while not _run_finished(graph):
        yield env.timeout(cfg["interval_s"])
        demand = graph.graph["demand"]
        hot = sorted(demand, key=lambda k: -demand[k])[: cfg["hot_pairs"]]
        for pair in hot:
            if demand[pair] < cfg.get("min_requests", 3) or len(stock["store"].get(pair, [])) >= cfg["per_pair"]:
                continue
            s, d = tuple(pair)
            sync_pool_view(graph, env.now)
            decision = policy.decide(Request(s, d, policy_cfg["eps_target"], policy_cfg["a_target"]), graph, p_v,
                                     decide_cfg)
            if decision is BLOCK:
                continue
            reserved = reserve_share_paths(graph, decision.paths, key_size_bits, env.now)
            sent = [p for p, ok in zip(decision.paths, reserved) if ok]
            note_own_use(graph, sent, key_size_bits)
            bits = float(sum((len(p) - 1) * key_size_bits for p in sent))
            stock["bits"] += bits
            if len(sent) < decision.t:
                continue
            sent_decision = Decision(paths=sent, t=decision.t, n=len(sent), scheme=decision.scheme)
            delivery = simulate_delivery(rng, sent_decision, attacker_set, config["attacker"]["p_drop"],
                                         config["attacker"]["p_tamper"], key_size_bits)
            if not delivery["served"]:
                continue
            stock["store"].setdefault(pair, []).append({
                "created": env.now, "t": sent_decision.t, "n": sent_decision.n, "bits": bits,
                "exposed": exposed_ground_truth(sent_decision, attacker_set),
                "silent_failure": delivery["silent_failure"],
            })
            stock["made"] += 1


def _request_process(
    env: simpy.Environment,
    graph: nx.Graph,
    policy,
    config: dict,
    s,
    d,
    attacker_set: set,
    p_v: dict,
    rng: np.random.Generator,
    records: list[dict[str, Any]],
    is_warmup: bool,
    learner: SuspicionLearner | None = None,
    qos_class: dict | None = None,
):
    arrival_time = env.now
    method = policy.name
    key_size_bits = config["key_size_bits"]
    des_cfg = config["des"]
    attacker_cfg = config["attacker"]
    policy_cfg = config["policy"]
    ref_bits = b0_reference_cost(graph, s, d, key_size_bits)
    auc = ranking_auc(p_v, attacker_set) if learner is not None else None

    # QoS (thesis extension; off unless config["qos"] is set): the request's
    # class sets its own security target (only if the policy is told about
    # classes -- `qos.class_aware`), its own pool reserve floor, and a
    # deadline up to which a refused request waits for the pools to refill
    # and is retried, instead of being refused at once.
    qos_name = None
    decide_cfg = config
    eps_target, a_target = policy_cfg["eps_target"], policy_cfg["a_target"]
    deadline_s, retry_s = 0.0, 0.0
    if qos_class is not None:
        qos_cfg = config["qos"]
        qos_name = qos_class["name"]
        if qos_cfg.get("class_aware", True):
            eps_target, a_target = qos_class["eps"], qos_class["a"]
        if qos_cfg.get("reservation", True):
            decide_cfg = dict(config, paths=dict(config["paths"], reserve_fraction=qos_class["reserve_fraction"]))
        if qos_cfg.get("deadlines", True):
            deadline_s, retry_s = qos_class.get("deadline_s", 0.0), qos_cfg.get("retry_interval_s", 0.01)
    request = Request(s, d, eps_target, a_target)

    stock = graph.graph.get("stock")
    if stock is not None:
        key = _take_stock_key(graph, frozenset((s, d)), env.now)
        if key is not None:  # proactive key: delivered at once, no pool use now
            yield env.timeout(des_cfg["control_latency_s"])
            if not is_warmup:
                records.append({"s": s, "d": d, "blocked": False, "failed": False, "method": method, "auc": auc,
                                "qos_class": qos_name, "from_stock": True,
                                "t": key["t"], "n": key["n"], "exposed": key["exposed"], "served": True,
                                "silent_failure": key["silent_failure"], "bits_consumed": 0.0, "ref_bits": ref_bits,
                                "amplification": key["bits"] / ref_bits if ref_bits > 0 else float("nan"),
                                "arrival_time": arrival_time, "latency": env.now - arrival_time})
            return

    sync_pool_view(graph, env.now)
    decision = policy.decide(request, graph, p_v, decide_cfg)
    while decision is BLOCK and deadline_s > 0 and retry_s > 0 and env.now + retry_s <= arrival_time + deadline_s:
        yield env.timeout(retry_s)
        sync_pool_view(graph, env.now)
        decision = policy.decide(request, graph, p_v, decide_cfg)

    if decision is BLOCK:
        probe_bits = 0.0
        if learner is not None and learner.probe_paths > 0:
            probe_bits = _send_probes(graph, config, s, d, attacker_set, rng, learner, env.now)
        if not is_warmup:
            records.append({"s": s, "d": d, "blocked": True, "failed": False, "method": method, "auc": auc,
                             "qos_class": qos_name,
                             "bits_consumed": probe_bits, "probe_bits": probe_bits, "ref_bits": ref_bits,
                             "arrival_time": arrival_time, "latency": None})
        return

    yield env.timeout(des_cfg["control_latency_s"])

    reservations = reserve_share_paths(graph, decision.paths, key_size_bits, env.now)
    sent_paths = [p for p, ok in zip(decision.paths, reservations) if ok]
    note_own_use(graph, sent_paths, key_size_bits)
    if graph.graph.get("relay_usage") is not None:  # what a strategic attacker gets to observe
        graph.graph["relay_usage"].append([v for p in sent_paths for v in relay_nodes(p)])

    if len(sent_paths) < decision.t:
        # not enough shares could even be sent -- pool-exhaustion failure,
        # distinct from a policy-level block (README 2.3/8). Shares whose
        # paths *were* reserved have already consumed their pool bits: that
        # key material is wasted, and counted as such in `bits_consumed`.
        if not is_warmup:
            wasted = sum((len(p) - 1) * key_size_bits for p in sent_paths)
            records.append({"s": s, "d": d, "blocked": False, "failed": True, "method": method, "auc": auc,
                             "qos_class": qos_name,
                             "t": decision.t, "n": decision.n, "scheme": decision.scheme,
                             "bits_consumed": wasted, "ref_bits": ref_bits,
                             "arrival_time": arrival_time, "latency": env.now - arrival_time})
        return

    sent_decision = Decision(paths=sent_paths, t=decision.t, n=len(sent_paths), scheme=decision.scheme)
    max_latency = max(path_latency(graph, p, des_cfg["hop_processing_s"]) for p in sent_paths)
    yield env.timeout(max_latency)

    exposed = exposed_ground_truth(sent_decision, attacker_set)
    m_star_val = security.m_star(sent_decision.paths, sent_decision.t)
    delivery = simulate_delivery(
        rng, sent_decision, attacker_set, attacker_cfg["p_drop"], attacker_cfg["p_tamper"], key_size_bits
    )
    bits = bits_consumed(sent_decision, key_size_bits)
    if learner is not None:
        learner.observe(sent_decision.paths, delivery["observations"])

    if not is_warmup:
        records.append(
            {
                "s": s, "d": d, "blocked": False, "failed": False, "method": method, "auc": auc,
                "qos_class": qos_name,
                "t": sent_decision.t, "n": sent_decision.n, "scheme": sent_decision.scheme,
                "exposed": exposed, "m_star": m_star_val,
                # the policy's own predicted exposure probability for what was
                # actually sent (None if some shares failed reservation, or the
                # policy does not estimate risk) -- for calibration checks
                "p_exp_pred": decision.p_exp if len(sent_paths) == len(decision.paths) else None,
                "served": delivery["served"], "silent_failure": delivery["silent_failure"],
                "bits_consumed": bits, "ref_bits": ref_bits,
                "amplification": bits / ref_bits if ref_bits > 0 else float("nan"),
                "arrival_time": arrival_time, "latency": env.now - arrival_time,
            }
        )


def _arrival_process(env, graph, policy, config, nodes, attacker_set, p_v, rng, records, total_requests,
                     warmup_requests, learner=None):
    des_cfg = config["des"]
    hotspot_pairs = None
    if des_cfg["traffic_mode"] == "hotspot":
        hotspot_pairs = [
            tuple(sample_request_pair(nodes, set(), rng)) for _ in range(des_cfg["hotspot_pairs"])
        ]
    endpoint_exclude = attacker_set  # README 3.1: A is never a request's source/destination

    qos_classes = config.get("qos", {}).get("classes") if config.get("qos") else None
    shares = np.array([c["share"] for c in qos_classes], dtype=float) if qos_classes else None
    profile = des_cfg.get("load_profile")  # thesis extension: alternating quiet / peak periods
    for i in range(total_requests):
        rate = des_cfg["arrival_rate_per_s"]
        if profile:
            in_peak = (env.now % profile["period_s"]) < profile["peak_fraction"] * profile["period_s"]
            rate *= profile["peak_multiplier"] if in_peak else profile["quiet_multiplier"]
        wait = rng.exponential(1.0 / rate)
        yield env.timeout(wait)
        s, d = sample_traffic_pair(nodes, endpoint_exclude, rng, des_cfg, hotspot_pairs)
        if graph.graph.get("demand") is not None:
            graph.graph["demand"][frozenset((s, d))] = graph.graph["demand"].get(frozenset((s, d)), 0) + 1
        qos_class = qos_classes[int(rng.choice(len(qos_classes), p=shares / shares.sum()))] if qos_classes else None
        env.process(_tracked(graph, _request_process(env, graph, policy, config, s, d, attacker_set, p_v, rng,
                                                     records, i < warmup_requests, learner, qos_class)))
    graph.graph["arrivals_done"] = True


def run_level_b(
    topology_graph: nx.Graph,
    method: str,
    config: dict,
    n_requests: int,
    seed: int,
    p_v_override: dict | None = None,
    policy_override: Any | None = None,
    learning: dict | None = None,
    oracle_p_v: bool = False,
) -> pd.DataFrame:
    """Run Level B: `config['des']['warmup_requests']` + `n_requests` Poisson
    (or hotspot) arrivals against one drawn attacker set and one dynamic pool
    state per link, returning ground-truth outcomes for the post-warmup
    requests only (README 6: "Warm-up period excluded from statistics").

    `p_v_override` (E8 ablation, "without suspicion"): use this instead of
    generating p_v from suspicion scores -- the attacker set is still drawn
    normally (it's the ground truth), only the *policy's* view of it changes.
    `policy_override`: use this policy instance instead of looking `method`
    up (E8 ablation variants like `AdaptiveThresholdPolicy(pool_aware=False)`
    aren't reachable by name through `get_policy`).
    `oracle_p_v` (E11 upper bound): the policy sees ground truth, p_v = 1 on
    compromised nodes and p_min elsewhere.
    `learning` (E11): {"q_drop", "q_tamper", "eps", optional "passive_fraction",
    "probe_paths"}
    -- refine p_v online
    from share-delivery observations (`qkdmp.suspicion_learning`), starting
    from whatever p_v the arguments above produce as the prior. The
    returned frame then carries a per-request `auc` column (ranking quality
    of the policy's current p_v against the true attacker set).
    """
    rng = np.random.default_rng(seed)
    graph = topology_graph.copy()
    attach_pool_state(graph)
    _apply_kms_model(graph, config.get("kms"))
    ctrl = config.get("controller") or {}
    if ctrl.get("telemetry_interval_s", 0) > 0:
        graph.graph["telemetry"] = {"interval": ctrl["telemetry_interval_s"], "view": ctrl.get("view", "frozen")}
        for _, _, data in graph.edges(data=True):
            data.update(snap_level=data["pool"].level, snap_time=0.0, own_used=0.0, est_rate=data["rate_bps"])
    nodes = list(graph.nodes())

    attacker_cfg = config["attacker"]
    attacker_set = place_attacker(graph, attacker_cfg, rng)

    if oracle_p_v:
        p_min = config["suspicion"]["p_min"]
        p_v = {v: (P_MAX if v in attacker_set else p_min) for v in nodes}
    elif p_v_override is not None:
        p_v = p_v_override
    else:
        p_v = detector_p_v(nodes, attacker_set, config, rng)

    learner = None
    if learning is not None:
        learner = SuspicionLearner(
            p_v, learning["q_drop"], learning["q_tamper"], learning["eps"], config["suspicion"]["p_min"],
            passive_fraction=learning.get("passive_fraction", 0.5), probe_paths=learning.get("probe_paths", 0),
        )
        p_v = learner.p_v  # same dict object: the policy sees every update

    policy = policy_override if policy_override is not None else get_policy(method, config, seed=seed)
    warmup_requests = config["des"]["warmup_requests"]

    env = simpy.Environment()
    records: list[dict[str, Any]] = []
    if graph.graph.get("telemetry") is not None:
        env.process(_telemetry_process(env, graph))
    if (config["attacker"].get("strategic") or {}).get("enabled"):
        graph.graph["relay_usage"] = []
        graph.graph["attacker_moves"] = 0
        env.process(_strategic_attacker_process(env, graph, config, nodes, attacker_set, p_v, rng))
    if (config.get("stockpile") or {}).get("enabled"):
        lifetime = config["stockpile"].get("lifetime_s") or (config.get("kms") or {}).get("key_lifetime_s") or float("inf")
        graph.graph["stock"] = {"store": {}, "bits": 0.0, "made": 0, "expired": 0, "lifetime_s": lifetime}
        graph.graph["demand"] = {}
        env.process(_stockpile_process(env, graph, policy, config, attacker_set, p_v, rng))
    env.process(
        _arrival_process(
            env, graph, policy, config, nodes, attacker_set, p_v, rng, records,
            total_requests=warmup_requests + n_requests, warmup_requests=warmup_requests, learner=learner,
        )
    )
    env.run()

    df = pd.DataFrame.from_records(records)
    df.attrs["final_auc"] = ranking_auc(p_v, attacker_set)
    if graph.graph.get("stock") is not None:  # proactive-key material, charged in key efficiency
        st = graph.graph["stock"]
        df.attrs.update(overhead_bits=st["bits"], stock_made=st["made"], stock_expired=st["expired"])
    return df
