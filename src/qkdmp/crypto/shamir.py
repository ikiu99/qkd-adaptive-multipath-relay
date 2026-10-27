"""Byte-wise Shamir (t, n) secret sharing over GF(2^8), and n-of-n XOR splitting.

Shamir shares are keyed by an integer index x in [1, 255] (0 is reserved: the
polynomial's value at x=0 is the secret itself, so x=0 must never be handed
out as a share). Reconstruction is Lagrange interpolation at x=0.
"""

from __future__ import annotations

import numpy as np

from qkdmp.crypto import gf256

Share = tuple[int, bytes]


def split_secret(secret: bytes, t: int, n: int, rng: np.random.Generator) -> list[Share]:
    """Split `secret` into n shares such that any t reconstruct it.

    Each byte of the secret gets its own random degree-(t-1) polynomial over
    GF(2^8), with the secret byte as the constant term. Shares are evaluated
    at x = 1..n.
    """
    if not (1 <= t <= n):
        raise ValueError(f"require 1 <= t <= n, got t={t}, n={n}")
    if n > 255:
        raise ValueError("GF(2^8) Shamir supports at most 255 shares")

    # coeffs[byte_idx][0] is the secret byte; coeffs[byte_idx][1:t] are random
    coeffs = np.empty((len(secret), t), dtype=np.uint8)
    coeffs[:, 0] = np.frombuffer(secret, dtype=np.uint8)
    if t > 1:
        coeffs[:, 1:] = rng.integers(0, 256, size=(len(secret), t - 1), dtype=np.uint8)

    shares: list[Share] = []
    for x in range(1, n + 1):
        y = bytearray(len(secret))
        for byte_idx in range(len(secret)):
            acc = 0
            # Horner's method in GF(2^8), evaluating highest degree coeff first.
            for power in range(t - 1, -1, -1):
                acc = gf256.add(gf256.mul(acc, x), int(coeffs[byte_idx, power]))
            y[byte_idx] = acc
        shares.append((x, bytes(y)))
    return shares


def reconstruct_secret(shares: list[Share]) -> bytes:
    """Reconstruct the secret from >= t shares via Lagrange interpolation at x=0.

    Any t (or more, if all are consistent with the same underlying
    polynomial) shares reconstruct the secret exactly; using fewer raises.
    """
    if len(shares) < 1:
        raise ValueError("need at least one share to reconstruct")
    xs = [x for x, _ in shares]
    if len(set(xs)) != len(xs):
        raise ValueError("duplicate share indices")
    if any(x == 0 for x in xs):
        raise ValueError("x=0 is not a valid share index")

    length = len(shares[0][1])
    if any(len(y) != length for _, y in shares):
        raise ValueError("all shares must have the same length")

    secret = bytearray(length)
    for byte_idx in range(length):
        acc = 0
        for i, (xi, yi) in enumerate(shares):
            yi_byte = yi[byte_idx]
            if yi_byte == 0:
                continue
            # Lagrange basis polynomial L_i(0) = prod_{j!=i} (0 - x_j) / (x_i - x_j)
            # In GF(2^8), subtraction == addition == XOR, so (0 - x_j) == x_j.
            num = 1
            den = 1
            for j, (xj, _) in enumerate(shares):
                if i == j:
                    continue
                num = gf256.mul(num, xj)
                den = gf256.mul(den, gf256.add(xi, xj))
            term = gf256.mul(yi_byte, gf256.div(num, den))
            acc = gf256.add(acc, term)
        secret[byte_idx] = acc
    return bytes(secret)


# ---------------------------------------------------------------------------
# XOR n-of-n splitting (B1 baseline, README Section 5)
# ---------------------------------------------------------------------------

def xor_split(secret: bytes, n: int, rng: np.random.Generator) -> list[bytes]:
    """Split `secret` into n shares that XOR back to it; all n are required."""
    if n < 1:
        raise ValueError("n must be >= 1")
    shares = [bytes(rng.integers(0, 256, size=len(secret), dtype=np.uint8)) for _ in range(n - 1)]
    acc = bytearray(secret)
    for share in shares:
        for i, b in enumerate(share):
            acc[i] ^= b
    shares.append(bytes(acc))
    return shares


def xor_reconstruct(shares: list[bytes]) -> bytes:
    if not shares:
        raise ValueError("need at least one share to reconstruct")
    length = len(shares[0])
    if any(len(s) != length for s in shares):
        raise ValueError("all shares must have the same length")
    acc = bytearray(length)
    for share in shares:
        for i, b in enumerate(share):
            acc[i] ^= b
    return bytes(acc)
