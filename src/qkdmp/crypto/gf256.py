"""GF(2^8) arithmetic (AES reduction polynomial x^8+x^4+x^3+x+1 = 0x11B).

Log/antilog tables give O(1) multiply/divide/pow after an O(256) setup.
Byte values 0-255 are field elements throughout.
"""

from __future__ import annotations

_POLY = 0x11B

_EXP = [0] * 512  # doubled so exp lookups never need to wrap with a modulo
_LOG = [0] * 256


def _xtime(a: int) -> int:
    """Multiply by x (0x02), reducing mod the field polynomial."""
    a <<= 1
    if a & 0x100:
        a ^= _POLY
    return a


def _build_tables() -> None:
    # 0x02 is NOT a primitive element for poly 0x11B (its multiplicative
    # order is only 51), so log/exp tables built from it would not cover all
    # 255 nonzero elements. 0x03 is primitive for this polynomial and is the
    # generator conventionally used for Rijndael's GF(2^8) log tables.
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x = _xtime(x) ^ x  # multiply by 3
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]


_build_tables()


def add(a: int, b: int) -> int:
    """GF(2^8) addition (= subtraction) is XOR."""
    return a ^ b


def mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def pow_(a: int, n: int) -> int:
    if n == 0:
        return 1
    if a == 0:
        return 0
    return _EXP[(_LOG[a] * n) % 255]


def inv(a: int) -> int:
    if a == 0:
        raise ZeroDivisionError("0 has no inverse in GF(2^8)")
    return _EXP[255 - _LOG[a]]


def div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("division by 0 in GF(2^8)")
    if a == 0:
        return 0
    return _EXP[(_LOG[a] - _LOG[b]) % 255]
