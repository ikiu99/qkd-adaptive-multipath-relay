"""Bridge from the qkdmp Level-B DES to QKDNetSim key pools.

`Ns3PoolServer` runs `qkdmp-pool-server` (ns-3.46 + QKDNetSim, inside WSL)
as a subprocess and talks to it over stdin/stdout. `Ns3LinkPool` has the
same interface as `qkdmp.sim.des.LinkPoolState` (peek / sync / level /
try_reserve / capacity / rate / depletion_events), so the DES and every
policy run unchanged: reading a level asks QKDNetSim for its Q-buffers,
and lowering it removes that many keys from both end KMSs.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import networkx as nx

WSL_DISTRO = os.environ.get("QKDMP_WSL_DISTRO", "Ubuntu-22.04")
SERVER = os.environ.get(
    "QKDMP_NS3_SERVER", "$HOME/qkdmp-ns3/ns-3-dev/build/scratch/ns3.46-qkdmp-pool-server-optimized"
)


def to_wsl_path(path: Path) -> str:
    p = Path(path).resolve()
    drive = p.drive.rstrip(":").lower()
    return f"/mnt/{drive}" + p.as_posix()[len(p.drive):]


def write_topology(graph: nx.Graph, key_size_bits: int, path: Path, rate_scale: float = 1.0) -> list[tuple]:
    """Edge list in the server's format; returns the edge order used.
    `rate_scale` scales the generation rate QKDNetSim uses (not the capacity)."""
    edges = list(graph.edges())
    lines = [f"{len(edges)} {key_size_bits}"]
    for u, v in edges:
        d = graph.edges[u, v]
        lines.append(f"{u} {v} {d['rate_bps'] * rate_scale:.3f} {d['capacity_bits']:.3f} {d['length_km_norm']:.3f}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return edges


class Ns3PoolServer:
    """One QKDNetSim process = one DES run (one seed, one method)."""

    def __init__(self, graph: nx.Graph, key_size_bits: int, seed: int, preroll_s: float = 2.0,
                 rate_scale: float = 1.0, block_bits: int = 0):
        self.key_size_bits = key_size_bits
        self._tmp = tempfile.TemporaryDirectory(prefix="qkdmp_ns3_")
        topo = Path(self._tmp.name) / "topology.txt"
        self.edges = write_topology(graph, key_size_bits, topo, rate_scale)
        self.index = {frozenset(e): i for i, e in enumerate(self.edges)}
        cmd = (f"cd \"$(mktemp -d)\" && exec {SERVER} --topo={to_wsl_path(topo)} "
               f"--seed={seed} --preroll={preroll_s}" + (f" --blockBits={block_bits}" if block_bits else ""))
        self.proc = subprocess.Popen(
            ["wsl.exe", "-d", WSL_DISTRO, "-e", "bash", "-c", cmd],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        ready = self.proc.stdout.readline().split()
        if not ready or ready[0] != "READY":
            raise RuntimeError(f"pool server failed to start: {ready} {self.proc.stderr.read()[-2000:]}")
        self._levels: list[float] | None = None
        self._levels_time = -1.0
        self.ipc_calls = 0

    def _ask(self, line: str) -> list[str]:
        self.ipc_calls += 1
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline().split()
        if not reply:
            raise RuntimeError(f"pool server died on {line!r}")
        return reply

    def levels(self, now: float) -> list[float]:
        if self._levels is None or now != self._levels_time:
            reply = self._ask(f"SYNC {now:.9f}")
            self._levels = [float(x) for x in reply[1:]]
            self._levels_time = now
        return self._levels

    def take(self, edge_idx: int, bits: float, now: float) -> float:
        taken = float(self._ask(f"TAKE {now:.9f} {edge_idx} {bits:.3f}")[1])
        if self._levels is not None and now == self._levels_time:
            self._levels[edge_idx] -= taken
        return taken

    def raw(self, now: float) -> list[tuple[float, float]]:
        """Untrimmed Q-buffer bits at both ends of every edge."""
        return [tuple(float(y) for y in x.split(":")) for x in self._ask(f"RAW {now:.9f}")[1:]]

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
        self._tmp.cleanup()


class Ns3LinkPool:
    """Drop-in for `LinkPoolState`, backed by one edge of the pool server."""

    def __init__(self, server: Ns3PoolServer, edge_idx: int, capacity: float, rate: float):
        self.server = server
        self.idx = edge_idx
        self.capacity = capacity
        self.rate = rate
        self.depletion_events = 0
        self._now = 0.0

    def peek(self, now: float) -> float:
        return self.server.levels(now)[self.idx]

    def sync(self, now: float) -> None:
        self._now = now

    @property
    def level(self) -> float:
        return self.server.levels(self._now)[self.idx]

    @level.setter
    def level(self, value: float) -> None:
        current = self.level
        if value < current:  # the DES only ever lowers a level by consuming key
            self.server.take(self.idx, current - value, self._now)

    def try_reserve(self, bits: float, now: float) -> bool:
        self.sync(now)
        if self.level < bits:
            return False
        self.level = self.level - bits
        if self.level < bits:
            self.depletion_events += 1
        return True
