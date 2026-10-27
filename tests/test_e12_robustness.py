import numpy as np
import pandas as pd

from experiments.e12_robustness import holm, paired_comparisons


def test_holm_matches_hand_computation():
    # sorted p: 0.01, 0.02, 0.04 -> x3, x2, x1 = 0.03, 0.04, 0.04 (monotone)
    assert holm([0.04, 0.01, 0.02]) == [0.04, 0.03, 0.04]


def test_paired_comparison_detects_consistent_lead():
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(30):
        base = rng.uniform(0.3, 0.7)  # shared per-seed difficulty -> pairing matters
        rows.append({"topology": "t", "placement": "p", "rho_A": 0.1, "method": "Adaptive@0.5", "seed": seed,
                     "secure_goodput": base + 0.02, "exposure": 0.0})
        rows.append({"topology": "t", "placement": "p", "rho_A": 0.1, "method": "B0", "seed": seed,
                     "secure_goodput": base, "exposure": 0.2 + rng.normal(0, 0.01)})
    pc = paired_comparisons(pd.DataFrame(rows))
    assert bool(pc["goodput_better"].iloc[0]) and bool(pc["exposure_lower"].iloc[0])
