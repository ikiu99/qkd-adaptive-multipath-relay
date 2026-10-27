"""Destination reconstruction and tamper detection (README 4.3, C2).

With n > t shares, reconstructing from several t-subsets and cross-checking
lets the destination detect (and discard) tampered shares. With n == t there
is no redundancy: a tampered-but-undropped share is silently accepted as
correct -- this is the "silent failure" case README 3.4 and 4.3 call out.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from qkdmp.crypto.shamir import Share, reconstruct_secret

MAX_SUBSETS_TRIED = 100


@dataclass
class ReconstructionResult:
    secret: bytes | None
    detected_tampered_indices: frozenset[int]
    failed: bool  # no consistent majority found among t-subsets (discarded)
    detectable: bool  # True iff n > t, i.e. tamper detection was possible at all


def reconstruct_with_detection(shares: list[Share], t: int) -> ReconstructionResult:
    """Reconstruct from received `shares` (already past drop simulation, so
    len(shares) may be < the original n), detecting tampering when possible.

    - len(shares) < t: not enough shares to reconstruct at all -> failed.
    - len(shares) == t: exactly threshold shares, no redundancy to check
      consistency against -> reconstructs whatever the (possibly tampered)
      shares yield, with `detectable=False` (README 4.3: undetectable with
      n == t; the caller must compare against ground truth to know whether
      this was a *silent failure*).
    - len(shares) > t: reconstructs from every t-subset (bounded by
      MAX_SUBSETS_TRIED), takes the majority result across subsets as the
      secret, and flags shares that never participate in a majority-result
      subset as tampered. If no strict majority exists, the request is
      discarded as failed (README 4.3: "Otherwise discard the request as
      failed").
    """
    if len(shares) < t:
        return ReconstructionResult(None, frozenset(), failed=True, detectable=False)

    if len(shares) == t:
        secret = reconstruct_secret(shares)
        return ReconstructionResult(secret, frozenset(), failed=False, detectable=False)

    subset_indices = list(combinations(range(len(shares)), t))
    if len(subset_indices) > MAX_SUBSETS_TRIED:
        subset_indices = subset_indices[:MAX_SUBSETS_TRIED]

    results_by_subset: list[tuple[tuple[int, ...], bytes]] = []
    for idx_tuple in subset_indices:
        subset = [shares[i] for i in idx_tuple]
        secret = reconstruct_secret(subset)
        results_by_subset.append((idx_tuple, secret))

    # Plurality vote, not a strict >50% majority of all C(n, t) subsets:
    # subsets that omit every tampered share all agree on the true secret,
    # while subsets touched by (independently, arbitrarily) tampered shares
    # generically disagree with each other too -- so the true secret is the
    # single most-repeated result as long as fewer than about half the
    # shares are tampered, well before "> 50% of all subsets" would hold.
    # Ambiguous (tied top count, or every subset gave a different secret)
    # means discard as failed, per README 4.3.
    counts = Counter(secret for _, secret in results_by_subset)
    top = counts.most_common(2)
    majority_secret, majority_count = top[0]
    runner_up_count = top[1][1] if len(top) > 1 else 0
    if majority_count < 2 or majority_count <= runner_up_count:
        return ReconstructionResult(None, frozenset(), failed=True, detectable=True)

    majority_share_indices: set[int] = set()
    for idx_tuple, secret in results_by_subset:
        if secret == majority_secret:
            majority_share_indices.update(idx_tuple)

    all_indices = set(range(len(shares)))
    tampered = frozenset(all_indices - majority_share_indices)
    return ReconstructionResult(majority_secret, tampered, failed=False, detectable=True)
