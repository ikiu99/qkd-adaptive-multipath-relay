# Key-budget-aware adaptive threshold secret sharing for multipath QKD key relay

Simulation code and results for the paper *Key-budget-aware adaptive threshold
secret sharing for multipath key relay in SDN-controlled quantum key
distribution networks*.

For every key request, an SDN controller chooses the relay paths and the Shamir
parameters (t, n) together, using calibrated per-node compromise probabilities
and the current key-pool level of every link. The predicted exposure probability
is kept below a bound, and the cheapest option is chosen.

## Contents

| Path | Contents |
|---|---|
| `src/qkdmp/` | Library: Shamir and XOR sharing over GF(2^8), exposure and delivery probabilities, path search, attacker and detector models, the proposed policy and the baselines, graph-level Monte Carlo (Level A) and SimPy discrete-event simulator (Level B) |
| `experiments/`, `configs/` | Experiment runners and one YAML configuration per experiment |
| `scripts/` | Robustness checks, detector-aware baselines, steady-state runs, decision time, calibration, topology conversion |
| `data/topologies/` | NSFNET, Abilene, GÉANT, Germany50 and the synthetic 50-node network, with sources |
| `results/` | Per-seed and aggregated results behind every figure and table of the paper |
| `qkdnetsim-prototype/` | ns-3/QKDNetSim co-simulation (key pools and hop-by-hop share relay) and its results |
| `tests/` | Test suite |

## Setup

```bash
uv sync --extra dev
uv run pytest -q
```

## Reproducing the paper

All figures and every number in the paper are rebuilt from the committed CSVs:

```bash
PYTHONPATH=. uv run python scripts/paper_figures.py
```

The results themselves are produced by:

| Paper section | Command |
|---|---|
| Main comparison, robustness (miscalibration, misranking, pool size) | `scripts/e12_robustness.py paired\|miscal\|misrank\|pool`, `scripts/e12_robustness.py topologies geant germany50` |
| Detector-aware baselines | `scripts/risk_baselines.py run` |
| Load, routing-aware attacker, silent failures (GÉANT, Germany50) | `scripts/campaign.py` |
| Same at the main operating point; ablation; steady state | `scripts/review_items.py e3\|e6\|e8\|steady\|steady-large` |
| Predicted vs realised exposure | `scripts/calibration_check.py` |
| Monte Carlo fallback rate | `scripts/mc_fallback_rate.py` |
| Optimality and RL reference | `scripts/rl_diagnosis.py`, `python -m experiments.run --config configs/experiments/E9.yaml` |
| Decision time | `scripts/decision_time.py`, `scripts/decision_latency_check.py` |
| QKDNetSim co-simulation | `qkdnetsim-prototype/README.md` |

Prefix each command with `PYTHONPATH=. uv run python`. Runs are deterministic
given the seed, and every method sees the same attacker set and detector output
for a given seed.
