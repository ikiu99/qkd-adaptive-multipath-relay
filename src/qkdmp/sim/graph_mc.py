"""Level A -- graph-level Monte Carlo simulator (README Section 6).

No time dimension, no pools: draws one ground-truth attacker set per run,
serves `n_requests` random (s, d) requests against it, and records the
*ground-truth* security/availability/cost outcome of each (as opposed to the
policy's own P_exp/P_del estimates, which are a decision-time input, not an
evaluation output -- README 3.4 vs 3.3).

Policies are shared with Level B (README 6): this module only supplies the
graph/p_v/config a `Policy.decide` needs and evaluates the `Decision` it
returns; it never reaches into policy internals.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from qkdmp import attacker as attacker_mod
from qkdmp import security
from qkdmp.crypto.shamir import split_secret, xor_reconstruct, xor_split
from qkdmp.paths import relay_nodes
from qkdmp.policies.adaptive import AdaptiveThresholdPolicy
from qkdmp.policies.admission import AdmissionControlled
from qkdmp.policies.base import BLOCK, Decision, Request
from qkdmp.policies.baselines import build_baseline
from qkdmp.policies.exhaustive import ExhaustivePolicy
from qkdmp.reconstruct import reconstruct_with_detection
from qkdmp.suspicion import detector_p_v

BASELINE_NAMES = {"B0", "B1", "B2", "B3", "B0R", "B1R", "B2R"}


def get_policy(method: str, config: dict, seed: int = 0):
    """`method` may carry an admission tier suffix, e.g. "B1+AC" / "B2+PA"
    (`qkdmp.policies.admission`)."""
    if "+" in method:
        base, tier = method.split("+", 1)
        return AdmissionControlled(get_policy(base, config, seed=seed), tier)
    if method in BASELINE_NAMES:
        return build_baseline(method, config)
    if method == "Adaptive":
        return AdaptiveThresholdPolicy(fallback=config["policy"]["fallback"], seed=seed)
    if method == "Exhaustive":
        return ExhaustivePolicy(fallback=config["policy"]["fallback"], seed=seed)
    raise NotImplementedError(f"policy {method!r} is not implemented yet (RL is Phase 5)")


def place_attacker(graph: nx.Graph, attacker_cfg: dict, rng: np.random.Generator) -> set:
    placement = attacker_cfg["placement"]
    rho_A = attacker_cfg["rho_A"]
    if placement == "random":
        mode = attacker_cfg.get("random_mode", "fixed_count")
        return attacker_mod.place_random(graph, rng, rho_A=rho_A, mode=mode)
    if placement == "betweenness":
        return attacker_mod.place_betweenness(graph, rho_A=rho_A)
    if placement == "key_throughput":
        return attacker_mod.place_key_throughput(graph, rho_A=rho_A)
    raise ValueError(
        f"placement {placement!r} needs traffic history (path_frequency); "
        "use place_attacker_adaptive after a warm-up period instead"
    )


def bits_consumed(decision: Decision, key_size_bits: int) -> float:
    return sum((len(path) - 1) * key_size_bits for path in decision.paths)


def b0_reference_cost(graph: nx.Graph, s, d, key_size_bits: int) -> float:
    path = nx.shortest_path(graph, s, d)
    return (len(path) - 1) * key_size_bits


def exposed_ground_truth(decision: Decision, attacker_set: set) -> bool:
    exposed_paths = sum(1 for p in decision.paths if set(relay_nodes(p)) & attacker_set)
    return exposed_paths >= decision.t


def simulate_delivery(
    rng: np.random.Generator,
    decision: Decision,
    attacker_set: set,
    p_drop: float,
    p_tamper: float,
    key_size_bits: int,
) -> dict:
    """Ground-truth, byte-level delivery simulation (README 4.3, C2):
    generates a real random secret, actually splits it per `decision.scheme`,
    drops/tampers shares per-path via the real attacker set, and reconstructs
    (with tamper detection when the scheme's redundancy allows it) to
    determine whether the request was served, and whether an undetectable
    tampering event produced a *silent failure* (README 3.4).

    Also returns `observations`, one per path: what the destination can
    actually tell about that share, without ground truth -- "dropped" (never
    arrived), "tampered" (arrived, flagged by consistency checking),
    "clean" (arrived and verified consistent, needs > t shares received),
    "arrived" (arrived, unverifiable), or None (reconstruction ambiguous).
    Consumed only by online suspicion learning (`qkdmp.suspicion_learning`).
    """
    n_bytes = key_size_bits // 8
    secret = bytes(rng.integers(0, 256, size=n_bytes, dtype=np.uint8))
    outcomes = [
        attacker_mod.path_outcome(rng, relay_nodes(p), attacker_set, p_drop, p_tamper)
        for p in decision.paths
    ]

    def _tamper(data: bytes) -> bytes:
        return bytes(rng.integers(0, 256, size=len(data), dtype=np.uint8))

    unverified = ["dropped" if o == "dropped" else "arrived" for o in outcomes]

    if decision.scheme == "shamir":
        shares = split_secret(secret, decision.t, decision.n, rng)
        arrived = [i for i in range(len(shares)) if outcomes[i] != "dropped"]
        received = [
            (shares[i][0], _tamper(shares[i][1]) if outcomes[i] == "tampered" else shares[i][1]) for i in arrived
        ]
        result = reconstruct_with_detection(received, decision.t)
        served = not result.failed
        silent_failure = served and result.secret != secret
        observations = list(unverified)
        if result.detectable:
            for j, i in enumerate(arrived):
                if result.failed:
                    observations[i] = None
                else:
                    observations[i] = "tampered" if j in result.detected_tampered_indices else "clean"
        return {"served": served, "silent_failure": silent_failure, "observations": observations}

    if decision.scheme == "xor":
        shares = xor_split(secret, decision.n, rng)
        received = [_tamper(y) if outcomes[i] == "tampered" else y for i, y in enumerate(shares) if outcomes[i] != "dropped"]
        if len(received) < decision.n:
            return {"served": False, "silent_failure": False, "observations": unverified}
        reconstructed = xor_reconstruct(received)
        served = True
        silent_failure = reconstructed != secret
        return {"served": served, "silent_failure": silent_failure, "observations": unverified}

    if decision.scheme == "single":
        outcome = outcomes[0]
        if outcome == "dropped":
            return {"served": False, "silent_failure": False, "observations": unverified}
        served = True
        silent_failure = outcome == "tampered"
        return {"served": served, "silent_failure": silent_failure, "observations": unverified}

    raise ValueError(f"unknown scheme: {decision.scheme!r}")


def sample_request_pair(nodes: list, exclude: set, rng: np.random.Generator) -> tuple:
    candidates = [v for v in nodes if v not in exclude]
    if len(candidates) < 2:
        raise RuntimeError("not enough non-compromised nodes to sample a request pair")
    s, d = rng.choice(np.array(candidates, dtype=object), size=2, replace=False)
    return s, d


def run_level_a(
    topology_graph: nx.Graph,
    method: str,
    config: dict,
    n_requests: int,
    seed: int,
    exclude_attacker_as_endpoint: bool = True,
) -> pd.DataFrame:
    """Run `n_requests` Level A requests for `method` against one drawn
    attacker set, and return one row per (non-blocked) request plus blocked
    rows, with ground-truth ("evaluated", README 3.4) ex-post outcomes.

    `exclude_attacker_as_endpoint` (default True) implements README 3.1's "A
    ... never the source or destination of a request". Leave this True for
    every normal experiment. Set it False only for a controlled analytic
    sanity check like E2(a): excluding a *fixed* attacker set from (s, d)
    sampling makes the sampled (s, d) pair's identity depend on that same
    attacker draw, which biases the empirical exposure rate upward relative
    to a closed form that assumes (s, d) sampling independent of the
    attacker set (confirmed empirically during Phase 2 -- see
    reports/PHASE_3_REPORT.md; with exclusion, NSFNET/rho_A=0.35 gave ~0.32
    vs. an analytic 0.25, and the gap vanished once endpoints were sampled
    unconditionally). It is a real selection effect, not a bug, but it means
    "the same fixed attacker set can't be both the sampling exclusion and
    the evaluated outcome's cause without inducing this bias" -- so E2(a)
    turns exclusion off specifically to validate the exposure *formula*
    in isolation from this effect.
    """
    rng = np.random.default_rng(seed)
    graph = topology_graph
    nodes = list(graph.nodes())

    attacker_cfg = config["attacker"]
    key_size_bits = config["key_size_bits"]
    policy_cfg = config["policy"]
    p_drop = attacker_cfg["p_drop"]
    p_tamper = attacker_cfg["p_tamper"]

    policy = get_policy(method, config, seed=seed)

    if attacker_cfg["placement"] == "adaptive":
        # README 3.1 strategy 4: the attacker targets nodes that appear most
        # often in chosen path sets during a warm-up period. Run `method`
        # for `warmup_requests` with no suspicion information at all (p_v
        # all 0 -- there is no attacker set yet to derive it from: this is
        # the chicken-and-egg point the strategy itself has to start from)
        # to observe its "natural" routing pattern, then place the attacker
        # on the most-used relay nodes from that.
        warmup_requests = attacker_cfg.get("warmup_requests", 100)
        p_v_zero = {v: 0.0 for v in nodes}
        path_frequency: dict = {}
        for _ in range(warmup_requests):
            s, d = sample_request_pair(nodes, set(), rng)
            request = Request(s, d, policy_cfg["eps_target"], policy_cfg["a_target"])
            decision = policy.decide(request, graph, p_v_zero, config)
            if decision is BLOCK:
                continue
            for path in decision.paths:
                for v in relay_nodes(path):
                    path_frequency[v] = path_frequency.get(v, 0) + 1
        attacker_set = attacker_mod.place_adaptive(graph, path_frequency, rho_A=attacker_cfg["rho_A"])
    else:
        attacker_set = place_attacker(graph, attacker_cfg, rng)

    p_v = detector_p_v(nodes, attacker_set, config, rng)

    endpoint_exclude = attacker_set if exclude_attacker_as_endpoint else set()
    records: list[dict[str, Any]] = []
    for _ in range(n_requests):
        s, d = sample_request_pair(nodes, endpoint_exclude, rng)
        request = Request(s, d, policy_cfg["eps_target"], policy_cfg["a_target"])
        decision = policy.decide(request, graph, p_v, config)

        if decision is BLOCK:
            records.append({"s": s, "d": d, "blocked": True, "method": method})
            continue

        exposed = exposed_ground_truth(decision, attacker_set)
        m_star_val = security.m_star(decision.paths, decision.t)
        # README 4.3 / C2: real secret, real split, real drop/tamper per
        # path, real reconstruction (with majority-vote tamper detection
        # when n > t; silently wrong, undetectable, when n == t).
        delivery = simulate_delivery(rng, decision, attacker_set, p_drop, p_tamper, key_size_bits)
        served = delivery["served"]
        silent_failure = delivery["silent_failure"]

        bits = bits_consumed(decision, key_size_bits)
        ref_bits = b0_reference_cost(graph, s, d, key_size_bits)

        records.append(
            {
                "s": s,
                "d": d,
                "blocked": False,
                "method": method,
                "t": decision.t,
                "n": decision.n,
                "scheme": decision.scheme,
                "exposed": exposed,
                "m_star": m_star_val,
                "served": served,
                "silent_failure": silent_failure,
                "bits_consumed": bits,
                "amplification": bits / ref_bits if ref_bits > 0 else float("nan"),
            }
        )

    return pd.DataFrame.from_records(records)
