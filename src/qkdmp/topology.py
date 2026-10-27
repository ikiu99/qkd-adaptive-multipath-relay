"""Topology loading, length normalization and the key-rate model (README 2.1-2.2, 7).

A topology is a `networkx.Graph`. Every edge carries at least `length_km`
(raw, pre-normalization). `apply_length_normalization` adds `length_km_norm`.
`apply_rate_model` adds `rate_bps`, `capacity_bits` and `level_bits`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from qkdmp.config import REPO_ROOT

TOPOLOGY_DIR = REPO_ROOT / "data" / "topologies"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_topology_json(path: str | Path) -> nx.Graph:
    """Load a topology stored as the JSON edge-list format used in data/topologies/."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    graph: nx.Graph = nx.Graph(name=data.get("name", path.stem))
    for node in data["nodes"]:
        node_id = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        graph.add_node(node_id, **attrs)
    for edge in data["edges"]:
        u, v = edge["u"], edge["v"]
        attrs = {k: v for k, v in edge.items() if k not in ("u", "v")}
        if "length_km" not in attrs:
            raise ValueError(f"Edge ({u}, {v}) in {path} is missing length_km")
        graph.add_edge(u, v, **attrs)
    return graph


def load_topology_graphml(path: str | Path) -> nx.Graph:
    graph = nx.read_graphml(path)
    # GraphML round-trips node ids as strings and numeric attrs as strings;
    # normalize both so downstream code can rely on int ids / float lengths.
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    for _, _, attrs in graph.edges(data=True):
        if "length_km" in attrs:
            attrs["length_km"] = float(attrs["length_km"])
    return graph


def load_topology(name: str, topology_dir: str | Path | None = None) -> nx.Graph:
    """Load a topology by name from data/topologies/ (JSON preferred, GraphML fallback)."""
    base = Path(topology_dir) if topology_dir is not None else TOPOLOGY_DIR
    json_path = base / f"{name}.json"
    if json_path.exists():
        return load_topology_json(json_path)
    graphml_path = base / f"{name}.graphml"
    if graphml_path.exists():
        return load_topology_graphml(graphml_path)
    raise FileNotFoundError(f"No topology named '{name}' found in {base}")


def available_topologies(topology_dir: str | Path | None = None) -> list[str]:
    base = Path(topology_dir) if topology_dir is not None else TOPOLOGY_DIR
    names = {p.stem for p in base.glob("*.json")} | {p.stem for p in base.glob("*.graphml")}
    return sorted(names)


# ---------------------------------------------------------------------------
# Length normalization (README 2.2)
# ---------------------------------------------------------------------------

def apply_length_normalization(
    graph: nx.Graph, l_min_km: float, l_max_km: float, enabled: bool = True
) -> nx.Graph:
    """Rescale link lengths into [l_min_km, l_max_km], preserving their order.

    Adds a `length_km_norm` edge attribute; `length_km` (raw) is left intact.
    If `enabled` is False, or all lengths are equal, `length_km_norm` is set
    equal to `length_km` (no negative/zero-width rescale is attempted).
    """
    lengths = [d["length_km"] for _, _, d in graph.edges(data=True)]
    lo, hi = min(lengths), max(lengths)

    if not enabled or math.isclose(lo, hi):
        for _, _, d in graph.edges(data=True):
            d["length_km_norm"] = d["length_km"]
        return graph

    scale = (l_max_km - l_min_km) / (hi - lo)
    for _, _, d in graph.edges(data=True):
        d["length_km_norm"] = l_min_km + (d["length_km"] - lo) * scale
    return graph


# ---------------------------------------------------------------------------
# Key-rate model (README 2.2)
# ---------------------------------------------------------------------------

def rate_from_length(length_km: float, r0_bps: float, alpha_db_per_km: float, il_db: float) -> float:
    """R_e = R0 * 10^(-(alpha * L_e + IL) / 10)."""
    exponent = -(alpha_db_per_km * length_km + il_db) / 10.0
    return r0_bps * (10.0 ** exponent)


def apply_rate_model(graph: nx.Graph, config: dict[str, Any]) -> nx.Graph:
    """Assign `rate_bps`, `capacity_bits` and `level_bits` to every edge.

    Requires `length_km_norm` to already be set (call apply_length_normalization
    first) when mode == "length_dependent".
    """
    rm = config["rate_model"]
    pool = config["pool"]
    mode = rm["mode"]

    for _, _, d in graph.edges(data=True):
        if mode == "length_dependent":
            length = d.get("length_km_norm", d["length_km"])
            rate = rate_from_length(length, rm["R0_bps"], rm["alpha_db_per_km"], rm["IL_db"])
        elif mode == "uniform":
            rate = rm["uniform_rate_bps"]
        else:
            raise ValueError(f"Unknown rate_model.mode: {mode!r}")

        capacity = pool["capacity_seconds"] * rate
        d["rate_bps"] = rate
        d["capacity_bits"] = capacity
        d["level_bits"] = capacity * pool["initial_fill_fraction"]
    return graph


def build_topology(name: str, config: dict[str, Any], topology_dir: str | Path | None = None) -> nx.Graph:
    """Load a topology and apply length normalization + the rate model, per config."""
    graph = load_topology(name, topology_dir=topology_dir)
    ln = config["length_normalization"]
    apply_length_normalization(graph, ln["L_min_km"], ln["L_max_km"], enabled=ln["enabled"])
    apply_rate_model(graph, config)
    return graph


# ---------------------------------------------------------------------------
# Synthetic topology generators (README 7)
# ---------------------------------------------------------------------------

def generate_random50(
    seed: int = 0,
    n_nodes: int = 50,
    beta: float = 0.4,
    area_km: float = 2000.0,
    max_resamples: int = 200,
) -> nx.Graph:
    """Waxman random graph, resampled with rising density until 2-edge-connected.

    Node coordinates are drawn uniformly in [0, area_km]^2; link lengths are
    the Euclidean distance between endpoints (README 7: "lengths from coordinates").
    """
    rng = np.random.default_rng(seed)

    waxman = None
    for attempt in range(max_resamples):
        # Waxman's alpha controls density; widen it if earlier attempts failed
        # to reach 2-edge-connectivity, so this terminates for small n_nodes too.
        alpha = 0.2 + 0.05 * (attempt // 10)
        py_seed = int(rng.integers(0, 2**31 - 1))
        candidate = nx.waxman_graph(n_nodes, beta=beta, alpha=alpha, seed=py_seed)
        if nx.is_connected(candidate) and nx.edge_connectivity(candidate) >= 2:
            waxman = candidate
            break
    if waxman is None:
        raise RuntimeError(
            f"Could not generate a 2-edge-connected random50 graph in {max_resamples} resamples"
        )

    graph = nx.Graph(name="random50")
    for node, (x, y) in nx.get_node_attributes(waxman, "pos").items():
        graph.add_node(node, x=x * area_km, y=y * area_km)
    for u, v in waxman.edges():
        xu, yu = graph.nodes[u]["x"], graph.nodes[u]["y"]
        xv, yv = graph.nodes[v]["x"], graph.nodes[v]["y"]
        length_km = math.dist((xu, yu), (xv, yv))
        graph.add_edge(u, v, length_km=length_km)
    return graph


def generate_grid(rows: int = 6, cols: int = 6, unit_km: float = 80.0) -> nx.Graph:
    """2D mesh topology (README 7), unit spacing mapped to lengths in km."""
    mesh = nx.grid_2d_graph(rows, cols)
    graph = nx.Graph(name="grid")
    label_of = {}
    for idx, (r, c) in enumerate(sorted(mesh.nodes())):
        label_of[(r, c)] = idx
        graph.add_node(idx, x=c * unit_km, y=r * unit_km)
    for (r1, c1), (r2, c2) in mesh.edges():
        graph.add_edge(label_of[(r1, c1)], label_of[(r2, c2)], length_km=unit_km)
    return graph


def graph_to_json_dict(graph: nx.Graph) -> dict[str, Any]:
    nodes = [{"id": n, **graph.nodes[n]} for n in sorted(graph.nodes())]
    edges = [{"u": u, "v": v, **d} for u, v, d in graph.edges(data=True)]
    return {"name": graph.name, "nodes": nodes, "edges": edges}


def save_topology_json(graph: nx.Graph, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(graph_to_json_dict(graph), f, indent=2)
