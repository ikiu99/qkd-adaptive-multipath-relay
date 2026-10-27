"""Generate data/topologies/random50.json and grid.json with a fixed seed.

Run once (or whenever the generator parameters change):
    python scripts/generate_topologies.py
"""

from __future__ import annotations

from qkdmp.topology import TOPOLOGY_DIR, generate_grid, generate_random50, save_topology_json

RANDOM50_SEED = 7


def main() -> None:
    random50 = generate_random50(seed=RANDOM50_SEED)
    save_topology_json(random50, TOPOLOGY_DIR / "random50.json")
    print(f"Wrote random50: {random50.number_of_nodes()} nodes, {random50.number_of_edges()} edges")

    grid = generate_grid(rows=6, cols=6, unit_km=80.0)
    save_topology_json(grid, TOPOLOGY_DIR / "grid.json")
    print(f"Wrote grid: {grid.number_of_nodes()} nodes, {grid.number_of_edges()} edges")


if __name__ == "__main__":
    main()
