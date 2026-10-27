"""Strategic attacker: how fast must the detector track it?

The attacker watches the routing of the method under test and every 0.1 s
moves its compromised relays onto that method's most-used relays (last 50
deliveries). The detector re-scores the new compromised set after a lag;
this sweeps the lag (0.05 / 0.2 / 0.5 s, and never) to locate where the
proposed method stops beating the pool-aware baselines. Pools refill in
0.5 s, so the lags span "much faster than" to "as slow as" the network's
own time scale.

Usage: PYTHONPATH=. uv run python scripts/strategic_attacker_check.py
Outputs: results/strategic/
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from experiments import e12_robustness as rb
from qkdmp.config import load_config

OUT = Path("results") / "strategic"
BASELINES = ["B0", "B0+PA", "B1+PA", "B2+PA", "B3+PA"]
LAGS = [0.05, 0.2, 0.5, None]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open("configs/experiments/E12.yaml", encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    base = load_config()
    base = dict(base, policy=dict(base["policy"], **exp["policy_overrides"]),
                pool=dict(base["pool"], **exp["pool_overrides"]))
    seeds = list(range(exp["test"]["seed_offset"], exp["test"]["seed_offset"] + exp["test"]["n_seeds"]))
    frames = []
    for lag in LAGS:
        strategic = {"enabled": True, "interval_s": 0.1, "window": 50, "detector_lag_s": lag}
        cfg = dict(base, attacker=dict(base["attacker"], strategic=strategic))
        frames.append(rb.run_per_seed(cfg, topologies=exp["topologies"], rho=exp["rho"], rho_a_values=[0.2],
                                      placements=["random"], methods=BASELINES + [rb.OURS], seeds=seeds,
                                      n_requests=400, label={"detector_lag_s": "never" if lag is None else lag}))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "per_seed.csv", index=False)
    pc = rb.paired_comparisons(df, group_cols=("detector_lag_s",))
    pc.to_csv(OUT / "paired.csv", index=False)
    means = rb.means_table(df, ("detector_lag_s",))
    means.to_csv(OUT / "means.csv", index=False)
    print(rb.summarize(pc, ("detector_lag_s",)).to_string(index=False))
    print(means.pivot_table(index=["detector_lag_s", "topology"], columns="method",
                            values="secure_goodput").round(3).to_string())


if __name__ == "__main__":
    main()
