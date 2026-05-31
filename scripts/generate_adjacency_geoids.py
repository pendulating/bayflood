#!/usr/bin/env python
"""
Generate GEOID sidecar files for existing adjacency edge lists.

The committed adjacency edge lists (``*_node1.txt`` / ``*_node2.txt``) store
1-indexed positions in the GeoJSON feature order they were built from. Those
positions are only meaningful if the dataset the model fits is in the *same* row
order — which is not guaranteed (see docs/CODE_REVIEW_FINDINGS.md C1). This
script writes a ``*_geoids.txt`` sidecar (one GEOID per node position) next to
each edge list so ``util.read_real_data`` can align the adjacency to any dataset
row order by GEOID.

For each target it also VERIFIES the sidecar: it checks that, under the GeoJSON
feature order, edge endpoints are genuinely within the adjacency buffer. A high
pass rate confirms the GeoJSON order matches the adjacency node order.

Usage:
    python scripts/generate_adjacency_geoids.py
"""

import sys
from pathlib import Path

import numpy as np
import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parent.parent

# (label, node1_path, [candidate build_geojsons], id_column, buffer_ft)
# Multiple candidate GeoJSONs are tried; the sidecar is written from the FIRST
# whose feature order matches the adjacency node order (verified spatially). A
# sidecar is NEVER written from a GeoJSON that fails verification, since a
# wrong-order sidecar would silently remap edges to the wrong areas.
TARGETS = [
    (
        "CT (ct_)",
        BASE_DIR / "data/adjacency/cg_500/ct_nyc_adj_list_custom_geometric_node1.txt",
        [
            BASE_DIR / "aggregation/geo/data/ct-nyc-2020.geojson",
            BASE_DIR / "data/ct-nyc-2020.geojson",
        ],
        "GEOID",
        500.0,
    ),
    (
        "CBG (cbg_)",
        BASE_DIR / "data/adjacency/cbg_cg_300/cbg_nyc_adj_list_custom_geometric_node1.txt",
        [
            BASE_DIR / "aggregation/geo/data/cbg-nyc-2020.geojson",
            BASE_DIR / "data/cbg-nyc-2020.geojson",
        ],
        "GEOID",
        300.0,
    ),
]


def sidecar_path(node1_path: Path) -> Path:
    name = node1_path.name
    assert name.endswith("_node1.txt"), name
    return node1_path.with_name(name[: -len("_node1.txt")] + "_geoids.txt")


def verify(geoms, node1, node2, buffer_ft, n_sample=600):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(node1), size=min(n_sample, len(node1)), replace=False)
    ok = sum(
        1 for k in idx
        if geoms[node1[k] - 1].distance(geoms[node2[k] - 1]) <= buffer_ft + 1.0
    )
    return ok / len(idx)


def process(label, node1_path, geojson_candidates, id_column, buffer_ft):
    node2_path = node1_path.with_name(node1_path.name.replace("_node1.txt", "_node2.txt"))
    out = sidecar_path(node1_path)
    if not node1_path.exists():
        print(f"[skip] {label}: {node1_path} not found")
        return

    node1 = [int(x) for x in node1_path.read_text().split()]
    node2 = [int(x) for x in node2_path.read_text().split()]
    max_idx = max(max(node1), max(node2))

    best = None  # (frac, geoids, geojson_path)
    for geojson_path in geojson_candidates:
        if not geojson_path.exists():
            continue
        g = gpd.read_file(str(geojson_path)).to_crs(2263)
        if id_column not in g.columns or len(g) < max_idx:
            continue
        geoids = [int(x) for x in g[id_column].astype("int64").tolist()]
        frac = verify(g.geometry.values, node1, node2, buffer_ft)
        if best is None or frac > best[0]:
            best = (frac, geoids, geojson_path)
        if frac >= 0.99:
            break

    if best is None:
        print(f"[skip] {label}: no usable candidate GeoJSON found.")
        return

    frac, geoids, geojson_path = best
    if frac >= 0.9:
        out.write_text("\n".join(str(x) for x in geoids) + "\n")
        status = "OK" if frac >= 0.99 else "WARN"
        print(
            f"[{status}] {label}: wrote {out.relative_to(BASE_DIR)} "
            f"({len(geoids)} GEOIDs) from {geojson_path.name}; "
            f"{100*frac:.1f}% of sampled edges within {int(buffer_ft)}ft."
        )
    else:
        # Do NOT persist a wrong-order sidecar. Remove any stale one so the reader
        # falls back to positional alignment + the spatial tripwire.
        if out.exists():
            out.unlink()
        print(
            f"[FAIL] {label}: no candidate GeoJSON matches the adjacency node "
            f"order (best {100*frac:.1f}% within {int(buffer_ft)}ft). No sidecar "
            f"written. The adjacency for this geometry should be regenerated with "
            f"tract_weights.py so it emits a sidecar; until then runs rely on the "
            f"read_real_data tripwire to catch misalignment."
        )


def main():
    for t in TARGETS:
        process(*t)


if __name__ == "__main__":
    main()
