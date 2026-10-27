"""Realised QKDNetSim key rate per NSFNET link vs the configured rate.

No key is consumed; the Q-buffers are read untrimmed at DES time 0 and T.
Usage: PYTHONPATH=. uv run python qkdnetsim-prototype/cosim/calibrate_rates.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import e12_config  # noqa: E402
from ns3_pools import Ns3PoolServer  # noqa: E402

from qkdmp.topology import build_topology  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def main(horizon_s: float = 10.0) -> None:
    OUT.mkdir(exist_ok=True)
    cfg = e12_config()
    graph = build_topology("nsfnet", cfg)
    t0 = time.time()
    server = Ns3PoolServer(graph, cfg["key_size_bits"], seed=0)
    start = server.raw(0.0)
    end = server.raw(horizon_s)
    server.close()
    rows = []
    for (u, v), a, b in zip(server.edges, start, end):
        rate = graph.edges[u, v]["rate_bps"]
        realised = (min(b) - min(a)) / horizon_s
        rows.append({"u": u, "v": v, "configured_bps": rate, "realised_bps": realised, "ratio": realised / rate,
                     "ends_equal": a[0] == a[1] and b[0] == b[1]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rate_calibration.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"mean ratio {df['ratio'].mean():.4f}, min {df['ratio'].min():.4f}, max {df['ratio'].max():.4f}; "
          f"wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
