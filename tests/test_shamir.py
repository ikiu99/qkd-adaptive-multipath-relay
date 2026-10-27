import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qkdmp.crypto.shamir import (
    reconstruct_secret,
    split_secret,
    xor_reconstruct,
    xor_split,
)


def _secret_and_tn():
    return st.integers(min_value=1, max_value=32).flatmap(
        lambda length: st.tuples(
            st.binary(min_size=length, max_size=length),
            st.integers(min_value=1, max_value=6).flatmap(
                lambda t: st.tuples(st.just(t), st.integers(min_value=t, max_value=t + 4))
            ),
        )
    )


@given(data=_secret_and_tn(), seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_any_t_shares_reconstruct(data, seed):
    secret, (t, n) = data
    rng = np.random.default_rng(seed)
    shares = split_secret(secret, t, n, rng)

    # any t-subset (not just the first t) reconstructs the secret
    subset_rng = np.random.default_rng(seed + 1)
    subset_idx = subset_rng.choice(n, size=t, replace=False)
    subset = [shares[i] for i in subset_idx]
    assert reconstruct_secret(subset) == secret

    # using all n shares also reconstructs correctly
    assert reconstruct_secret(shares) == secret


def test_split_rejects_invalid_t_n():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        split_secret(b"x", t=0, n=3, rng=rng)
    with pytest.raises(ValueError):
        split_secret(b"x", t=4, n=3, rng=rng)


def test_t_minus_1_shares_are_uniformly_distributed():
    """Statistical no-information test: with only t-1 shares, for a family of
    secrets the observed share bytes at a fixed x should look uniform over
    GF(256), since for any target secret there is exactly one polynomial
    completing those t-1 points to it (information-theoretic security of
    Shamir sharing). We check this with a chi-square goodness-of-fit test on
    one held-out share's byte values across many independent secrets.
    """
    t, n = 3, 5
    rng = np.random.default_rng(1234)
    n_trials = 4000

    held_out_share_bytes = []
    for _ in range(n_trials):
        secret = bytes(rng.integers(0, 256, size=1, dtype=np.uint8))
        shares = split_secret(secret, t, n, rng)
        # Reveal only t-1 = 2 shares' worth of information by looking at a
        # share NOT in that subset: its value should be independent of secret.
        held_out_share_bytes.append(shares[-1][1][0])

    counts = np.bincount(held_out_share_bytes, minlength=256)
    expected = n_trials / 256
    chi2 = float(((counts - expected) ** 2 / expected).sum())

    # 255 degrees of freedom; critical value at alpha=0.001 is ~330.
    assert chi2 < 330.0, f"chi2={chi2} suggests held-out share is not uniform"


def test_xor_split_reconstruct_roundtrip():
    rng = np.random.default_rng(0)
    secret = bytes(rng.integers(0, 256, size=32, dtype=np.uint8))
    shares = xor_split(secret, n=4, rng=rng)
    assert len(shares) == 4
    assert xor_reconstruct(shares) == secret


def test_xor_reconstruct_needs_all_shares():
    rng = np.random.default_rng(0)
    secret = bytes(rng.integers(0, 256, size=8, dtype=np.uint8))
    shares = xor_split(secret, n=3, rng=rng)
    assert xor_reconstruct(shares[:2]) != secret
