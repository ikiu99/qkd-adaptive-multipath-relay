"""Online suspicion learning from share-delivery outcomes.

The external detector (README 3.2) hands the controller one number per node
-- its suspicion score -- and nothing else. This module refines that number
online, using only what the key relay itself observes about each share it
sends: whether the share arrived, and, when more than t shares arrive,
whether the destination's consistency check flagged it as tampered
(`qkdmp.sim.graph_mc.simulate_delivery`'s `observations`). The external
score is the prior; the relay's own evidence updates it. With no external
detector at all, the prior is flat and the relay learns from scratch.

Model: each relay node v is honest (H), actively compromised (A) or
passively compromised (P), with p_v = P(A) + P(P). An active node drops a
share with probability q_drop, otherwise tampers with it with probability
q_tamper; honest and passive nodes drop with a small benign probability
eps and never tamper -- a passive node only eavesdrops, which is exactly
what exposes keys and exactly what leaves no trace. For a share path with
relays R and an observation o, each v in R is updated by P(o | state of v),
where the other relays' contribution uses their current P(A) (a mean-field
approximation of the exact joint posterior, exponential in |R|).

Why three states: with a two-state model (H vs "compromised = active"),
every clean delivery through a compromised node counts as evidence of its
honesty, and a rarely-misbehaving node gets exonerated -- in E11's
development runs this *raised* exposure above the no-information baseline
when the attacker was mostly passive. With the passive state, clean
history can only move a node's suspicion down to the floor
P(C) * passive_fraction / (P(H) + P(C) * passive_fraction) of its prior:
relay-side evidence can convict an active node but never clear one.

# ASSUMPTION: the controller's misbehavior model (q_drop, q_tamper, eps) is
# a configured estimate, not the attacker's true rates; see the
# `learning` block of configs/experiments/E11.yaml (its assumed rate
# matches neither attacker activity level E11 runs).
# ASSUMPTION: `passive_fraction`, the prior share of compromised nodes that
# are passive, is a configured estimate (the simulated attacker is a single
# activity level, not a mixture). A passive node produces no evidence at all
# -- only the external prior can flag it; E11 varies attacker activity.

Probing: a relay only learns about paths it actually uses, and a cautious
policy that refuses every request through uncertain nodes never uses them
-- with a flat prior it serves direct links only and learns nothing. So
when a request is refused, the controller may send dummy *probe* shares
along up to `probe_paths` of that request's candidate paths
(`qkdmp.sim.des`). A probe carries no key, so it cannot expose one; it
costs pool bits like a real share; and since the destination knows its
content, a tampered probe is always detected ("clean" is exact).
# ASSUMPTION: probes are indistinguishable from real shares to a relay.
# Under hop-by-hop OTP a relay sees each share's plaintext, but a Shamir
# share is uniformly random, so a random probe looks the same provided the
# controller gives it the same framing/metadata as a real share.
"""

from __future__ import annotations

from qkdmp.paths import relay_nodes

P_MAX = 1.0 - 1e-6


class SuspicionLearner:
    def __init__(
        self,
        prior_p_v: dict,
        q_drop: float,
        q_tamper: float,
        eps: float,
        p_min: float,
        passive_fraction: float = 0.5,
        probe_paths: int = 0,
    ) -> None:
        self.q_drop = q_drop
        self.q_tamper = q_tamper
        self.eps = eps
        self.p_min = p_min
        self.probe_paths = probe_paths
        # per node: (P(honest), P(active), P(passive))
        self._state = {
            v: (1 - p, p * (1 - passive_fraction), p * passive_fraction) for v, p in prior_p_v.items()
        }
        self.p_v = dict(prior_p_v)  # updated in place; policies read it on every decide

    def _drop(self, p_active: float) -> float:
        return p_active * self.q_drop + (1 - p_active) * self.eps

    def _tamper(self, p_active: float) -> float:
        return p_active * self.q_tamper

    def _likelihoods(self, obs: str, others_active: list[float]) -> tuple[float, float]:
        """(P(obs | v active), P(obs | v honest or passive)) for one share path."""
        no_drop_others = 1.0
        no_tamper_others = 1.0
        for p in others_active:
            no_drop_others *= 1 - self._drop(p)
            no_tamper_others *= 1 - self._tamper(p)

        def lik(drop_v: float, tamper_v: float) -> float:
            arrived = (1 - drop_v) * no_drop_others
            if obs == "dropped":
                return 1 - arrived
            if obs == "arrived":
                return arrived
            clean = arrived * (1 - tamper_v) * no_tamper_others
            if obs == "clean":
                return clean
            return arrived - clean  # "tampered"

        return lik(self.q_drop, self.q_tamper), lik(self.eps, 0.0)

    def observe(self, paths: list[list], observations: list[str | None]) -> None:
        for path, obs in zip(paths, observations):
            if obs is None:
                continue
            relays = relay_nodes(path)
            if not relays:
                continue  # direct link: no relay to learn about
            active = [self._state[v][1] for v in relays]
            for i, v in enumerate(relays):
                l_active, l_quiet = self._likelihoods(obs, active[:i] + active[i + 1:])
                h, a, p = self._state[v]
                h, a, p = h * l_quiet, a * l_active, p * l_quiet
                total = h + a + p
                if total <= 0:
                    continue
                self._state[v] = (h / total, a / total, p / total)
                self.p_v[v] = min(P_MAX, max(self.p_min, (a + p) / total))


def ranking_auc(p_v: dict, compromised: set) -> float:
    """Probability that a random compromised node gets a higher p_v than a
    random honest one (ties count half) -- how well the scores *rank*
    nodes, independent of calibration."""
    bad = [p_v[v] for v in p_v if v in compromised]
    good = [p_v[v] for v in p_v if v not in compromised]
    if not bad or not good:
        return float("nan")
    wins = sum(1.0 if b > g else 0.5 if b == g else 0.0 for b in bad for g in good)
    return wins / (len(bad) * len(good))
