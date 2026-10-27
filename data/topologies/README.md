# Topology data sources

| File | Nodes | Links | Source |
|---|---|---|---|
| `nsfnet.json` | 14 | 21 | **Verified** -- supplied by the user, see note below |
| `abilene.json` | 11 | 14 | **Verified** -- Internet Topology Zoo, see note below. Second real topology, used in place of COST239. |
| `geant.json` | 22 | 36 | **Verified** -- SNDlib `geant` (real GEANT research backbone), see note below |
| `germany50.json` | 50 | 88 | **Verified** -- SNDlib `germany50` (German national reference network), see note below |
| `cost239.json` | 11 | 26 | **Deprecated, unverified, no longer used in any experiment** -- kept only for history, see note below |
| `random50.json` | 50 | 140 | Generated (`scripts/generate_topologies.py`, `waxman_graph`, seed 7) |
| `grid.json` | 36 | 60 | Generated (`scripts/generate_topologies.py`, 6x6 mesh, 80 km unit spacing) |

## GEANT and Germany50 (SNDlib) -- held-out topologies for the headline claim

Downloaded (with the user's permission) from SNDlib, the Survivable Network
Design Library (Orlowski, Wessaely, Pioro, Tomaszewski, *Networks* 55(3),
2010): `https://sndlib.put.poznan.pl/download/sndlib-networks-native/{geant,germany50}.txt`,
kept verbatim in `raw/sndlib_*.txt`. `scripts/convert_sndlib.py` turns them
into the JSON format: node longitude/latitude verbatim from the file;
`length_km` is the haversine distance between endpoints (SNDlib link records
hold capacity/cost modules, not a length); parallel links merged. Both are
2-node-connected. GEANT (2005 snapshot) includes the transatlantic New York
and the Israel PoPs, so raw lengths span 116-6795 km; like every topology here
they are rescaled into single QKD spans by `length_normalization`. Chosen
over deployed QKD networks (Beijing-Shanghai, SECOQC, Tokyo), which are
mostly chains with little path diversity to test multipath routing on.

## NSFNET -- now verified

`nsfnet.json` was replaced with the file the user supplied
(`articles/nsfnet.json`): the classic 14-city NSFNET T1 backbone (Seattle,
Palo Alto, San Diego, Salt Lake City, Boulder, Houston, Lincoln, Champaign,
Pittsburgh, Atlanta, Ann Arbor, Ithaca, Princeton, College Park), with
lat/lon coordinates (stored as `lat`/`lon` node attributes) and real
geographic distances (`meta.geographic_km` in the source file -> `length_km`
here). This replaces the earlier memory-reconstructed version from Phase 0/2
that was explicitly flagged as unverified. The source file's own precomputed
`edges` values (already rescaled into ~8.7-52.1 km) and its `exposure`/
`qber_unit` fields were **not** carried over -- this project applies its own
`apply_length_normalization` + rate model (README 2.2) to the raw
`geographic_km`, and the `exposure`/`qber_unit` fields don't correspond to
this project's security or rate model, so including them would risk being
silently misread as this project's own output.

## Abilene -- second verified real topology (replaces COST239)

`abilene.json` is the Internet2/Abilene US research backbone (Feb 2005
snapshot), sourced from the Internet Topology Zoo GitHub archive
(`mroughan/InternetTopologyZoo`, `gml/Abilene.gml`; primary reference:
Knight, Nguyen, Falkner, Bowden, Roughan, "The Internet Topology Zoo", IEEE
JSAC 29(9), 2011, itself citing Internet2's own published network map as the
data source). 11 nodes, 14 links, min node-connectivity 2 -- comparable size
to COST239's 11 nodes, and a topology that is itself extremely widely used
as a standard reference network in the networking literature. Node lat/lon
are verbatim from the GML file; `length_km` is the great-circle (haversine)
distance between each edge's endpoints, computed from those coordinates (the
source GML has no link-length field of its own, only a link-type label like
"OC-192"). Used in E3 and E4 as a second data point alongside NSFNET (see
`reports/PHASE_7_REPORT.md`); found and verified after the user asked
whether a legitimate second real topology could be sourced to strengthen the
single-topology limitation flagged in `reports/FINDINGS.md`.

## COST239 -- deprecated, no longer used

`cost239.json` was built from **memory of a commonly reproduced edge list**
in the optical networking literature (11-node/26-link dense European mesh),
and could never be verified against a primary source. COST239 also turns
out not to be a real deployed network in the Internet Topology Zoo's sense
(it's a synthetic ITU/COST-project reference topology, confirmed absent
from that archive), so there was no real-network primary source to check it
against even in principle. Per an explicit user decision, this file is kept
for history only and is **not used in any experiment** as of Phase 7;
`abilene.json` is its replacement as the project's second topology.

All topologies are re-scaled by `apply_length_normalization` (README 2.2)
into `[L_min_km, L_max_km]` (default 10-100 km) before the rate model is
applied. NSFNET's and Abilene's raw `length_km` values are real distances;
`cost239.json`'s (unused) values would only have needed to be correct in
relative order for the rate model, had it remained in use.

## Generated topologies (random50, grid)

- `random50.json`: `networkx.waxman_graph(50, beta=0.4, alpha=...)`, resampled
  with increasing alpha until 2-edge-connected, seed 7. Node coordinates are
  uniform in a 2000x2000 km square; `length_km` is Euclidean distance between
  endpoints. Regenerate with `python scripts/generate_topologies.py`.
- `grid.json`: 6x6 mesh (`networkx.grid_2d_graph`), 80 km unit spacing.

## File format

JSON edge list: `{"name", "nodes": [{"id", ...}], "edges": [{"u", "v",
"length_km", ...}]}`. Loaded by `qkdmp.topology.load_topology`.
