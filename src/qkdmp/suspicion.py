"""Synthetic suspicion scores (README 3.2): external input, referenced to
[REF-SUSPICION] (not yet supplied -- the Beta parameter mapping below is a
documented placeholder, see configs/default.yaml `suspicion` section).
"""

from __future__ import annotations

import numpy as np


def detector_quality_to_beta(
    detector_quality: float, beta_uninformative: list[float], beta_perfect: list[float]
) -> tuple[float, float, float, float]:
    """Linearly interpolate [a_c, b_c, a_h, b_h] between the uninformative and
    near-perfect Beta parameter sets by detector_quality in [0, 1]."""
    if not 0.0 <= detector_quality <= 1.0:
        raise ValueError(f"detector_quality must be in [0, 1], got {detector_quality}")
    lo = np.array(beta_uninformative)
    hi = np.array(beta_perfect)
    a_c, b_c, a_h, b_h = lo + detector_quality * (hi - lo)
    return float(a_c), float(b_c), float(a_h), float(b_h)


def generate_suspicion_scores(
    nodes: list,
    compromised: set,
    detector_quality: float,
    rng: np.random.Generator,
    beta_uninformative: list[float],
    beta_perfect: list[float],
) -> dict:
    """s_v ~ Beta(a_c, b_c) if v in `compromised` else Beta(a_h, b_h) (README 3.2)."""
    a_c, b_c, a_h, b_h = detector_quality_to_beta(detector_quality, beta_uninformative, beta_perfect)
    scores = {}
    for v in nodes:
        if v in compromised:
            scores[v] = float(rng.beta(a_c, b_c))
        else:
            scores[v] = float(rng.beta(a_h, b_h))
    return scores


def calibrated_posterior(
    s_v: dict, detector_quality: float, beta_uninformative: list[float], beta_perfect: list[float],
    base_rate: float, p_min: float,
) -> dict:
    """Bayes posterior P(compromised | score) under the same Beta score model
    that generated the scores: pi f_C(s) / (pi f_C(s) + (1 - pi) f_H(s)).

    This is what a *calibrated* detector outputs -- among nodes it scores
    0.2, about 20% really are compromised. The user's detector (the
    project's external input, README 3.2) outputs calibrated probabilities,
    so this is the default mapping (`suspicion.output: calibrated`); the
    original linear map, which scores honest nodes ~0.18 at quality 0.7
    whatever the true base rate, is kept as `output: linear` to reproduce
    earlier results.
    # ASSUMPTION: calibrated with respect to the true base rate, taken as
    # the attacker's rho_A -- the definition of calibration presupposes the
    # detector's designer knows how common compromise is.
    """
    from scipy.stats import beta as beta_dist

    a_c, b_c, a_h, b_h = detector_quality_to_beta(detector_quality, beta_uninformative, beta_perfect)
    out = {}
    for v, s in s_v.items():
        s = min(max(s, 1e-9), 1 - 1e-9)
        f_c = beta_dist.pdf(s, a_c, b_c)
        f_h = beta_dist.pdf(s, a_h, b_h)
        den = base_rate * f_c + (1 - base_rate) * f_h
        post = base_rate if den <= 0 else base_rate * f_c / den
        out[v] = min(1 - 1e-6, max(p_min, float(post)))
    return out


def detector_p_v(nodes: list, compromised: set, config: dict, rng: np.random.Generator) -> dict:
    """The external detector's per-node compromise probability, as every
    simulator and experiment consumes it: draw scores, then map them per
    `suspicion.output` ("calibrated", default, or legacy "linear")."""
    cfg = config["suspicion"]
    s_v = generate_suspicion_scores(
        nodes, compromised, cfg["detector_quality"], rng, cfg["beta_uninformative"], cfg["beta_perfect"],
    )
    if cfg.get("output", "calibrated") == "linear":
        return suspicion_to_prob(s_v, cfg["p_min"])
    p_v = calibrated_posterior(
        s_v, cfg["detector_quality"], cfg["beta_uninformative"], cfg["beta_perfect"],
        base_rate=config["attacker"]["rho_A"], p_min=cfg["p_min"],
    )
    p_v = miscalibrate(p_v, cfg.get("miscal_odds_factor", 1.0), cfg.get("miscal_temperature", 1.0), cfg["p_min"])
    if cfg.get("misrank_fraction", 0.0) > 0:
        p_v = misrank(p_v, compromised, cfg["misrank_fraction"], rng)
    return p_v


def misrank(p_v: dict, compromised: set, fraction: float, rng: np.random.Generator) -> dict:
    """Controlled ranking error, for robustness tests: each compromised node,
    with probability `fraction`, swaps its probability with a random honest
    node (each honest node swaps at most once). The multiset of values -- and
    so the detector's average calibration -- is unchanged, but the swapped
    compromised nodes now look as harmless as an honest node did, and those
    honest nodes as suspicious. The controller is not told.
    """
    out = dict(p_v)
    honest = sorted((v for v in p_v if v not in compromised), key=str)
    for v in sorted((v for v in p_v if v in compromised), key=str):
        if not honest or rng.random() >= fraction:
            continue
        h = honest.pop(int(rng.integers(len(honest))))
        out[v], out[h] = out[h], out[v]
    return out


def miscalibrate(p_v: dict, odds_factor: float, temperature: float, p_min: float) -> dict:
    """Controlled distortion of a calibrated detector, for robustness tests:
    logit(p') = temperature * logit(p) + ln(odds_factor).

    - odds_factor > 1: systematic over-estimation of risk (every node looks
      more suspicious than it is, e.g. a wrongly high assumed base rate);
      < 1: under-estimation.
    - temperature > 1: over-confident (probabilities pushed towards 0 and
      1); < 1: under-confident (pushed towards 0.5).
    (1, 1) is the identity. The controller is not told about the distortion.
    """
    if odds_factor == 1.0 and temperature == 1.0:
        return p_v
    out = {}
    for v, p in p_v.items():
        p = min(max(p, 1e-9), 1 - 1e-9)
        logit = temperature * np.log(p / (1 - p)) + np.log(odds_factor)
        q = 1.0 / (1.0 + np.exp(-logit))
        out[v] = min(1 - 1e-6, max(p_min, float(q)))
    return out


def suspicion_to_prob(s_v: dict, p_min: float) -> dict:
    """Legacy linear map from suspicion score to prior compromise
    probability, with floor p_min (README 3.2): p_v = p_min + (1 - p_min) * s_v.
    Not calibrated -- see `calibrated_posterior`.
    """
    return {v: p_min + (1 - p_min) * s for v, s in s_v.items()}
