# QKDNetSim co-simulation

Repeats the main experiment with QKDNetSim (ns-3.46 + QKDNetSim v2) instead of
the fluid key-pool model. The Level-B simulator (requests, attacker, detector,
policies, metrics) runs unchanged; only the key pools, and in the `ns3relay`
engine also the share transport, are simulated in ns-3.

- Every node runs a QKD module, a KMS and a QKDN controller; every edge is a QKD
  link whose post-processing generates 256-bit keys at the link rate into the
  Q-buffers of both end KMSs.
- `ns3/qkdmp-pool-server.cc` answers pool queries over stdin/stdout (advance
  time, read levels, take keys) and, for `ns3relay`, forwards each share hop by
  hop over UDP, one-time-pad encrypted with a key from that link's buffer. A
  compromised relay records the share and drops or randomises it with the
  attacker's probabilities.
- `cosim/ns3_pools.py` replaces the fluid pool of the Level-B simulator;
  `cosim/ns3_relay.py` replaces its delivery step.

Engines: `fluid` (the original model), `ns3` (QKDNetSim pools), `ns3relay`
(QKDNetSim pools and share relay). All engines use the same seeds.

## QKDNetSim fix

`ns3/qkdnetsim-queue-init-fix.patch` initialises two members of
`QKDKMSQueueLogic` that its constructor reads before ns-3 sets the
corresponding attributes; left uninitialised they can trigger an allocation of
many gigabytes (`std::bad_alloc`). No other change is made to QKDNetSim.

## Reproduce

Inside WSL (Ubuntu 22.04 with the QKDNetSim prerequisites):

```bash
bash qkdnetsim-prototype/scripts/setup_ns3.sh     # ns-3.46 + QKDNetSim in ~/qkdmp-ns3, patched and built
bash qkdnetsim-prototype/scripts/build_server.sh  # pool server
```

From the repository root (Windows):

```bash
PYTHONPATH=. uv run python qkdnetsim-prototype/cosim/calibrate_rates.py
PYTHONPATH=. uv run python qkdnetsim-prototype/cosim/run_cosim.py e12 --topology nsfnet --seeds 30 --engines fluid,ns3,ns3relay
PYTHONPATH=. uv run python qkdnetsim-prototype/cosim/analyze.py e12
```

Repeat `run_cosim.py` for `abilene`, `geant` and `germany50`. The run is
resumable. `results/e12_*.csv` hold the tables used in the paper.
