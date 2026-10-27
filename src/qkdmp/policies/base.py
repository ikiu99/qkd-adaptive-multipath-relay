"""Policy interface (README 11): `decide(request, network_state, suspicion) -> Decision | Block`.

All methods (baselines, the proposed heuristic, exhaustive, batch ILP, RL)
implement `Policy`. Level A and Level B share the same policy objects and
must not leak which level is calling them (README 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Request:
    s: Any
    d: Any
    eps_target: float
    a_target: float


@dataclass
class Decision:
    """A policy's choice for one request.

    `scheme` is "single" (B0: one path, whole key, no splitting), "xor"
    (n-of-n XOR split) or "shamir" ((t, n) Shamir split). `t == n` for
    "single" and "xor" schemes.
    """

    paths: list[list]
    t: int
    n: int
    scheme: str = "shamir"
    p_exp: float | None = None
    p_del: float | None = None
    cost: float | None = None

    def __post_init__(self) -> None:
        if len(self.paths) != self.n:
            raise ValueError(f"Decision has {len(self.paths)} paths but n={self.n}")
        if not (1 <= self.t <= self.n):
            raise ValueError(f"require 1 <= t <= n, got t={self.t}, n={self.n}")


class Block:
    """Sentinel: the policy could not find a feasible option for this request."""

    _instance: "Block | None" = None

    def __new__(cls) -> "Block":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Block()"

    def __bool__(self) -> bool:
        return False


BLOCK = Block()


class Policy(Protocol):
    name: str

    def decide(self, request: Request, graph: Any, p_v: dict, config: dict) -> Decision | Block: ...
