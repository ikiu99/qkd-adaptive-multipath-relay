"""Candidate path construction and overlap analysis (README 2.4).

`candidate_paths(graph, s, d, n_max)` returns up to `n_max` paths: first the
node-disjoint ones (Suurballe/Bhandari-style: repeatedly take the shortest
remaining path, then remove its internal nodes from the graph, so each path
found is shortest given the previous removals), then, if `n_max` isn't
reached, Yen k-shortest simple paths (which may overlap) to fill the rest.
"""

from __future__ import annotations

from itertools import islice

import networkx as nx

WEIGHT_ATTR = "length_km_norm"


def node_disjoint_shortest_paths(
    graph: nx.Graph, s, d, n_max: int, weight: str = WEIGHT_ATTR
) -> list[list]:
    """Greedy Suurballe-like construction: shortest path, remove its internal
    nodes, repeat. Not guaranteed globally min-cost like true Suurballe, but
    every path returned is node-disjoint from every other and this is
    deterministic and cheap.
    """
    working = graph.copy()
    paths: list[list] = []
    for _ in range(n_max):
        try:
            path = nx.shortest_path(working, s, d, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            break
        paths.append(path)
        internal = [v for v in path if v not in (s, d)]
        working.remove_nodes_from(internal)
    return paths


def yen_k_shortest_paths(
    graph: nx.Graph, s, d, k: int, weight: str = WEIGHT_ATTR
) -> list[list]:
    """Yen's k-shortest simple paths (may overlap), via networkx."""
    try:
        # nx.shortest_simple_paths raises NetworkXNoPath lazily, on the
        # first `next()` -- constructing the generator alone can't fail,
        # so the whole consumption must be inside the try, not just this line.
        generator = nx.shortest_simple_paths(graph, s, d, weight=weight)
        return list(islice(generator, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def candidate_paths(
    graph: nx.Graph, s, d, n_max: int, weight: str = WEIGHT_ATTR
) -> list[list]:
    """Up to n_max candidate paths for (s, d): node-disjoint first, then Yen
    k-shortest simple paths to fill any remaining slots (README 2.4)."""
    disjoint = node_disjoint_shortest_paths(graph, s, d, n_max, weight=weight)
    if len(disjoint) >= n_max:
        return disjoint[:n_max]

    seen = {tuple(p) for p in disjoint}
    extra_needed = n_max - len(disjoint)
    # Ask Yen for more than we need since some of its paths will duplicate
    # the disjoint ones already found.
    yen_paths = yen_k_shortest_paths(graph, s, d, k=n_max * 3, weight=weight)
    extra: list[list] = []
    for path in yen_paths:
        if tuple(path) in seen:
            continue
        seen.add(tuple(path))
        extra.append(path)
        if len(extra) >= extra_needed:
            break
    return disjoint + extra


def relay_nodes(path: list) -> list:
    """Internal (non-endpoint) nodes of a path -- the ones that can see a share."""
    return path[1:-1]


def path_edges(path: list) -> list[tuple]:
    return list(zip(path[:-1], path[1:]))


def node_overlap(paths: list[list]) -> dict:
    """Map relay node -> number of paths (out of `paths`) it appears on.

    Used to move security computation from path level to node level
    (README 2.4: "a node on several paths sees several shares").
    """
    overlap: dict = {}
    for path in paths:
        for node in relay_nodes(path):
            overlap[node] = overlap.get(node, 0) + 1
    return overlap


def paths_are_node_disjoint(paths: list[list]) -> bool:
    seen: set = set()
    for path in paths:
        for node in relay_nodes(path):
            if node in seen:
                return False
            seen.add(node)
    return True
