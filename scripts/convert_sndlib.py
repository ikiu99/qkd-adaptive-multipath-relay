"""Convert SNDlib native-format network files (data/topologies/raw/sndlib_*.txt)
into this project's topology JSON (data/topologies/<name>.json).

Node coordinates are taken verbatim from the SNDlib file (longitude,
latitude); link length is the great-circle (haversine) distance between the
endpoints, as for Abilene -- SNDlib link records carry capacity/cost modules,
not a physical length. Parallel links between the same node pair are merged.

Usage: PYTHONPATH=. uv run python scripts/convert_sndlib.py geant germany50
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import networkx as nx

RAW = Path("data") / "topologies" / "raw"
OUT = Path("data") / "topologies"
EARTH_RADIUS_KM = 6371.0

DESCRIPTIONS = {
    "geant": "GEANT pan-European research backbone (SNDlib 'geant', 2005 snapshot; includes the "
             "transatlantic New York and the Israel PoPs).",
    "germany50": "Germany50 realistic German national reference network (SNDlib 'germany50').",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def section(text: str, name: str) -> list[str]:
    block = text.split(f"{name} (", 1)[1].split("\n)", 1)[0]
    return [line.strip() for line in block.strip().splitlines() if line.strip()]


def convert(name: str) -> dict:
    text = (RAW / f"sndlib_{name}.txt").read_text(encoding="utf-8", errors="replace")
    nodes = []
    index = {}
    for line in section(text, "NODES"):
        m = re.match(r"(\S+)\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", line)
        label, lon, lat = m.group(1), float(m.group(2)), float(m.group(3))
        index[label] = len(nodes)
        nodes.append({"id": len(nodes), "label": label, "lat": lat, "lon": lon})
    pairs = set()
    for line in section(text, "LINKS"):
        m = re.match(r"\S+\s*\(\s*(\S+)\s+(\S+)\s*\)", line)
        u, v = index[m.group(1)], index[m.group(2)]
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    edges = []
    for u, v in sorted(pairs):
        a, b = nodes[u], nodes[v]
        edges.append({"u": u, "v": v, "length_km": round(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]), 1)})

    graph = nx.Graph()
    graph.add_edges_from((e["u"], e["v"]) for e in edges)
    return {
        "name": name,
        "description": f"{DESCRIPTIONS.get(name, name)} {len(nodes)} nodes, {len(edges)} links; "
                       f"min node-connectivity {nx.node_connectivity(graph)}.",
        "source": f"SNDlib (Orlowski, Wessaely, Pioro, Tomaszewski, 'SNDlib 1.0 -- Survivable Network Design "
                  f"Library', Networks 55(3), 2010), https://sndlib.put.poznan.pl/download/sndlib-networks-native/"
                  f"{name}.txt, downloaded to data/topologies/raw/sndlib_{name}.txt. Node lon/lat verbatim from "
                  f"that file; length_km is the haversine distance between endpoints (the file has no link-length "
                  f"field); parallel links merged.",
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    for name in sys.argv[1:]:
        data = convert(name)
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(data["description"])


if __name__ == "__main__":
    main()
