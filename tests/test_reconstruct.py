import numpy as np
import pytest

from qkdmp.crypto.shamir import split_secret
from qkdmp.reconstruct import reconstruct_with_detection


def test_not_enough_shares_fails():
    rng = np.random.default_rng(0)
    secret = b"abcdefgh"
    shares = split_secret(secret, t=3, n=5, rng=rng)
    result = reconstruct_with_detection(shares[:2], t=3)
    assert result.failed
    assert result.secret is None


def test_exactly_t_shares_undetectable_but_reconstructs_if_untampered():
    rng = np.random.default_rng(0)
    secret = b"abcdefgh"
    shares = split_secret(secret, t=3, n=5, rng=rng)
    result = reconstruct_with_detection(shares[:3], t=3)
    assert not result.detectable
    assert not result.failed
    assert result.secret == secret
    assert result.detected_tampered_indices == frozenset()


def test_exactly_t_shares_silently_wrong_if_one_tampered():
    rng = np.random.default_rng(0)
    secret = b"abcdefgh"
    shares = list(split_secret(secret, t=3, n=5, rng=rng)[:3])
    x, y = shares[0]
    tampered_y = bytes((y[0] ^ 0xFF,)) + y[1:]
    shares[0] = (x, tampered_y)

    result = reconstruct_with_detection(shares, t=3)
    assert not result.detectable  # n == t: no way to know this happened
    assert not result.failed
    assert result.secret != secret  # silently wrong


def test_n_greater_than_t_detects_single_tampered_share():
    rng = np.random.default_rng(0)
    secret = b"abcdefgh"
    shares = list(split_secret(secret, t=3, n=5, rng=rng))
    x, y = shares[0]
    tampered_y = bytes((y[0] ^ 0xFF,)) + y[1:]
    shares[0] = (x, tampered_y)

    result = reconstruct_with_detection(shares, t=3)
    assert result.detectable
    assert not result.failed
    assert result.secret == secret  # majority of t-subsets still agree on the truth
    assert 0 in result.detected_tampered_indices


def test_n_greater_than_t_no_tampering_reconstructs_cleanly():
    rng = np.random.default_rng(0)
    secret = b"abcdefgh"
    shares = split_secret(secret, t=3, n=5, rng=rng)
    result = reconstruct_with_detection(shares, t=3)
    assert result.detectable
    assert not result.failed
    assert result.secret == secret
    assert result.detected_tampered_indices == frozenset()


def test_too_many_tampered_shares_does_not_confidently_recover_truth():
    # With 4 of 5 shares tampered (independently, so their corruptions don't
    # correlate into a false consensus), a majority-vote scheme is not
    # expected to recover the true secret -- it must either discard the
    # request as failed, or (an inherent limitation of majority voting when
    # the attacker controls most shares) converge on a wrong answer. What it
    # must NOT do is silently return the *correct* secret with unflagged
    # confidence, since that would misrepresent how much protection t-of-n
    # detection actually provides once compromise exceeds its threshold.
    rng = np.random.default_rng(0)
    secret = b"abcdefgh"
    shares = list(split_secret(secret, t=3, n=5, rng=rng))
    for i in range(4):
        x, y = shares[i]
        mask = bytes(rng.integers(1, 256, size=len(y), dtype=np.uint8))
        tampered_y = bytes(b ^ m for b, m in zip(y, mask))
        shares[i] = (x, tampered_y)

    result = reconstruct_with_detection(shares, t=3)
    assert result.failed or result.secret != secret
