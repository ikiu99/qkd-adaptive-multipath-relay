"""Share relay through QKDNetSim (co-simulation, engine "ns3relay").

With `ns3_pools` alone, QKDNetSim supplies the key pools and the DES still
decides, in Python, what happens to each share on its path. Here the shares
themselves travel through the ns-3 network: after the DES reserves the key on
every hop (as before, `reserve_share_paths`), each share is sent as a UDP
packet from the source KMS along its path, one-time-pad encrypted per hop with
the key reserved on that link (qkdmp-pool-server "Share relay"). A compromised
relay sees the plaintext and drops or randomises the share with the same
probabilities as `qkdmp.attacker.share_outcome`. The destination then
reconstructs from the bytes that actually arrived, with the same Shamir / XOR
code as `qkdmp.sim.graph_mc.simulate_delivery`.

Exposure is what the attacker actually handled: a share counts as exposed
when a compromised relay saw it in plaintext, and a key is exposed when at
least t of its shares were. `RelayPort.mismatches` counts requests where this
differs from the DES ground truth (`exposed_ground_truth`); by construction it
should stay 0 (only compromised relays drop, and they see the share first).

The DES itself is not modified: `install()` rebuilds `des._request_process`
from its own source with only the delivery block replaced; if that block ever
changes, `install()` fails loudly instead of silently diverging.
"""

from __future__ import annotations

import inspect
import textwrap

import numpy as np

from qkdmp.crypto.shamir import split_secret, xor_reconstruct, xor_split
from qkdmp.reconstruct import reconstruct_with_detection
from qkdmp.sim import des
from qkdmp.sim.graph_mc import exposed_ground_truth

_OLD_BLOCK = '''
    max_latency = max(path_latency(graph, p, des_cfg["hop_processing_s"]) for p in sent_paths)
    yield env.timeout(max_latency)

    exposed = exposed_ground_truth(sent_decision, attacker_set)
    m_star_val = security.m_star(sent_decision.paths, sent_decision.t)
    delivery = simulate_delivery(
        rng, sent_decision, attacker_set, attacker_cfg["p_drop"], attacker_cfg["p_tamper"], key_size_bits
    )
'''
_NEW_BLOCK = '''
    max_latency = max(path_latency(graph, p, des_cfg["hop_processing_s"]) for p in sent_paths)
    _relay = graph.graph["ns3_relay"]
    _ticket = _relay.send(sent_decision, attacker_set, rng, key_size_bits, env.now)
    yield env.timeout(max_latency + _relay.margin_s)
    delivery, exposed = _relay.receive(_ticket, attacker_set, env.now)
    m_star_val = security.m_star(sent_decision.paths, sent_decision.t)
'''

_ORIGINAL = des._request_process


def install() -> None:
    """Swap in the relay-aware `_request_process` (idempotent)."""
    src = textwrap.dedent(inspect.getsource(_ORIGINAL))
    if src.count(_OLD_BLOCK.strip("\n")) != 1:
        raise RuntimeError("qkdmp.sim.des._request_process changed: the relay patch no longer applies")
    patched = src.replace(_OLD_BLOCK.strip("\n"), _NEW_BLOCK.strip("\n"))
    namespace: dict = {}
    exec(compile(patched, "<ns3relay _request_process>", "exec"), des.__dict__, namespace)
    des._request_process = namespace["_request_process"]


def uninstall() -> None:
    des._request_process = _ORIGINAL


class RelayPort:
    """The DES side of the share relay for one run (one Ns3PoolServer)."""

    margin_s = 1e-3  # extra wait beyond the nominal path latency before shares count as lost

    def __init__(self, server, p_drop: float, p_tamper: float):
        self.server = server
        self.p_drop, self.p_tamper = p_drop, p_tamper
        self._attack_sent: frozenset | None = None
        self._next_sid = 1
        self.mismatches = 0
        self.late = 0
        self.shares_sent = 0

    def _sync_attacker(self, attacker_set: set) -> None:
        current = frozenset(attacker_set)  # a strategic attacker mutates the set in place
        if current != self._attack_sent:
            nodes = sorted(current)
            self.server._ask(f"ATTACK {self.p_drop} {self.p_tamper} {len(nodes)} " + " ".join(map(str, nodes)))
            self._attack_sent = current

    def send(self, decision, attacker_set: set, rng: np.random.Generator, key_size_bits: int, now: float) -> dict:
        self._sync_attacker(attacker_set)
        secret = bytes(rng.integers(0, 256, size=key_size_bits // 8, dtype=np.uint8))
        if decision.scheme == "shamir":
            shares = split_secret(secret, decision.t, decision.n, rng)
            payloads = [y for _x, y in shares]
        elif decision.scheme == "xor":
            shares = xor_split(secret, decision.n, rng)
            payloads = list(shares)
        elif decision.scheme == "single":
            shares = [secret]
            payloads = [secret]
        else:
            raise ValueError(f"unknown scheme: {decision.scheme!r}")
        sids = []
        for path, payload in zip(decision.paths, payloads):
            sid = self._next_sid
            self._next_sid += 1
            self.server._ask(f"SEND {now:.9f} {sid} {len(path)} " + " ".join(map(str, path)) + " " + payload.hex())
            sids.append(sid)
        self.shares_sent += len(sids)
        return {"decision": decision, "secret": secret, "shares": shares, "payloads": payloads, "sids": sids}

    def receive(self, ticket: dict, attacker_set: set, now: float) -> tuple[dict, bool]:
        decision = ticket["decision"]
        reply = self.server._ask(f"RECV {now:.9f} " + " ".join(map(str, ticket["sids"])))[1:]
        arrived: dict[int, bytes] = {}
        seen = 0
        for i, item in enumerate(reply):
            _sid, status, was_seen, hexdata = item.split(",")
            seen += int(was_seen)
            if status == "A":
                arrived[i] = bytes.fromhex(hexdata)
            elif status == "P":
                self.late += 1
        exposed = seen >= decision.t
        if exposed != exposed_ground_truth(decision, attacker_set):
            self.mismatches += 1
        return self._reconstruct(ticket, arrived), exposed

    @staticmethod
    def _reconstruct(ticket: dict, arrived: dict[int, bytes]) -> dict:
        """Same outcome logic as graph_mc.simulate_delivery, on the bytes that arrived."""
        decision, secret, payloads = ticket["decision"], ticket["secret"], ticket["payloads"]
        n = len(payloads)
        unverified = ["arrived" if i in arrived else "dropped" for i in range(n)]
        if decision.scheme == "shamir":
            xs = [x for x, _y in ticket["shares"]]
            order = sorted(arrived)
            received = [(xs[i], arrived[i]) for i in order]
            result = reconstruct_with_detection(received, decision.t)
            served = not result.failed
            observations = list(unverified)
            if result.detectable:
                for j, i in enumerate(order):
                    observations[i] = None if result.failed else (
                        "tampered" if j in result.detected_tampered_indices else "clean")
            return {"served": served, "silent_failure": served and result.secret != secret,
                    "observations": observations}
        if decision.scheme == "xor":
            if len(arrived) < n:
                return {"served": False, "silent_failure": False, "observations": unverified}
            return {"served": True, "silent_failure": xor_reconstruct([arrived[i] for i in range(n)]) != secret,
                    "observations": unverified}
        if 0 not in arrived:
            return {"served": False, "silent_failure": False, "observations": unverified}
        return {"served": True, "silent_failure": arrived[0] != secret, "observations": unverified}
