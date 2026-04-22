#!/usr/bin/env python3
"""Generate Graph-Coloring COLOR instances for the QQA4CO benchmark.

The PQQA paper (Ichikawa & Arai 2024, §5.5) uses the COLOR dataset of
Trick (2002) — the canonical DIMACS-era graph-coloring benchmark. To keep
this repository self-contained (no external downloads required), we
generate the subset that is **procedurally defined**:

* ``myciel_k`` — the Mycielski construction of order ``k``; chromatic
  number is exactly ``k`` and clique number is 2. Available from
  ``networkx.mycielski_graph(k)``.
* ``queen_k_k`` — the ``k × k`` queen-attack graph; two vertices are
  adjacent iff the corresponding squares share a row, column, or
  diagonal. Known chromatic numbers for small ``k`` are tabulated in the
  literature (Chvátal; DeLaVina et al.).

All instances are saved as ``.gpickle`` + ``manifest.jsonl`` under
``<out>/coloring/{family}/``. Each manifest row carries:
  - ``num_colors``  — the recommended K for ``Coloring(num_category=K)``,
  - ``best_known``   — 0 when a proper K-coloring exists (i.e. we set K
    equal to the known chromatic number), else the minimum number of
    conflicts the problem is lower-bounded by. ``ApR`` on this benchmark
    is the complement ``1 - conflicts / |E|``.

Usage:
    python scripts/generate_coloring_instances.py --out data
"""

from __future__ import annotations

import argparse
import json
import pickle
import urllib.request
from pathlib import Path

import networkx as nx

# -------- DIMACS COLOR real-world instances --------------------------------
# The PQQA paper (Table 6) also evaluates three non-procedural COLOR
# instances that do not have a closed-form generator: ``anna`` and ``jean``
# come from Knuth's Stanford GraphBase (book-character intersection
# graphs), and ``queen8_12`` is a non-square queen graph. We fetch the
# canonical DIMACS ``.col`` files from Trick's COLOR repository. The
# chromatic numbers below are the paper's target ``num_colors`` (anna: 11,
# jean: 10, queen8_12: 12; see Schuetz et al. 2022 Table 7 and Yang et al.
# 2021 Table 3).
DIMACS_COLOR_URL = "https://mat.tepper.cmu.edu/COLOR/instances"
DIMACS_INSTANCES: dict[str, int] = {
    "anna": 11,
    "jean": 10,
    "queen8_12": 12,
}


def _parse_dimacs_col(text: str) -> nx.Graph:
    """Parse a DIMACS ``.col`` ascii edge list into an undirected graph.

    DIMACS format: comment lines start with ``c``; a ``p edge N M`` header
    declares ``N`` 1-indexed vertices and ``M`` edges; each edge is a line
    ``e u v``. We store 0-indexed nodes for consistency with the rest of
    QQA4CO.
    """
    g = nx.Graph()
    n_vertices: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        tok = line.split()
        if tok[0] == "p":
            n_vertices = int(tok[2])
            g.add_nodes_from(range(n_vertices))
        elif tok[0] == "e":
            u = int(tok[1]) - 1
            v = int(tok[2]) - 1
            if u != v:
                g.add_edge(u, v, weight=1.0)
    if n_vertices is None:
        raise ValueError("DIMACS .col file missing 'p edge ...' header")
    return g


def build_dimacs(name: str, cache_dir: Path) -> nx.Graph:
    """Download (once) and parse ``<name>.col`` from Trick's COLOR site."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fpath = cache_dir / f"{name}.col"
    if not fpath.exists():
        url = f"{DIMACS_COLOR_URL}/{name}.col"
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            fpath.write_bytes(resp.read())
    g = _parse_dimacs_col(fpath.read_text())
    g.graph["problem"] = "coloring"
    g.graph["graph_type"] = "dimacs"
    g.graph["subset"] = name
    g.graph["num_colors"] = DIMACS_INSTANCES[name]
    g.graph["source"] = f"DIMACS COLOR {name}.col (Trick 2002; Stanford GraphBase)"
    return g

# -------- Mycielski graphs -------------------------------------------------
# Chromatic number of M_k is k by construction (Mycielski 1955).
# We start from k=3 so N grows: M_3=5, M_4=11, M_5=23, M_6=47, M_7=95.
MYCIEL_ORDERS = [3, 4, 5, 6, 7]


def build_mycielski(k: int) -> nx.Graph:
    g = nx.mycielski_graph(k).copy()
    g.graph["problem"] = "coloring"
    g.graph["graph_type"] = "myciel"
    g.graph["subset"] = f"myciel{k}"
    g.graph["num_colors"] = k
    g.graph["source"] = f"networkx.mycielski_graph({k}) (Mycielski 1955)"
    return g


# -------- Queen graphs -----------------------------------------------------
# Vertices: (r, c) for r,c in [0, k). Edge iff same row / col / diagonal.
# Chromatic number — from literature — for k=5..13:
#   5:5, 6:7, 7:7, 8:9, 9:10, 10:11, 11:11, 12:12, 13:13
QUEEN_CHROMATIC: dict[int, int] = {
    5: 5,
    6: 7,
    7: 7,
    8: 9,
    9: 10,
    10: 11,
    11: 11,
    12: 12,
    13: 13,
}
QUEEN_SIZES = sorted(QUEEN_CHROMATIC.keys())


def build_queen(k: int) -> nx.Graph:
    g = nx.Graph()
    n = k * k
    g.add_nodes_from(range(n))
    for a in range(n):
        r1, c1 = divmod(a, k)
        for b in range(a + 1, n):
            r2, c2 = divmod(b, k)
            if r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2):
                g.add_edge(a, b, weight=1.0)
    g.graph["problem"] = "coloring"
    g.graph["graph_type"] = "queen"
    g.graph["subset"] = f"queen{k}_{k}"
    g.graph["num_colors"] = QUEEN_CHROMATIC[k]
    g.graph["source"] = f"queen attack graph on a {k}x{k} chessboard"
    return g


def write_family(
    out_root: Path,
    family: str,
    items: list[tuple[str, nx.Graph, int]],
) -> None:
    """items = list of (subset_name, graph, num_colors)."""
    subset_dir = out_root / "coloring" / family
    subset_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, (subset, g, K) in enumerate(items, start=1):
        fname = f"{i:04d}.gpickle"
        with open(subset_dir / fname, "wb") as fh:
            pickle.dump(g, fh)
        records.append(
            {
                "id": f"coloring-{family}-{subset}",
                "file": fname,
                "problem": "coloring",
                "graph_type": family,
                "subset": subset,
                "num_nodes": g.number_of_nodes(),
                "num_edges": g.number_of_edges(),
                # best_known (minimisation): 0 conflicts iff a proper
                # colouring with ``num_colors`` exists. For chromatic K
                # the known minimum is 0.
                "best_known": 0,
                "best_known_source": (
                    "chromatic number from Mycielski 1955"
                    if family == "myciel"
                    else (
                        "DIMACS COLOR tabulated chromatic number (Trick 2002)"
                        if family == "dimacs"
                        else "tabulated chromatic number"
                    )
                ),
                "num_colors": K,
                "source": g.graph.get("source", ""),
            }
        )
    with open(subset_dir / "manifest.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(records)} {family} instances to {subset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument(
        "--skip-dimacs",
        action="store_true",
        help="Skip fetching DIMACS COLOR real-world instances (anna, jean, queen8_12). "
        "Use this for offline rebuilds.",
    )
    parser.add_argument(
        "--dimacs-cache",
        type=Path,
        default=Path("data") / "coloring" / "_dimacs_cache",
        help="Directory to cache downloaded .col files (default: data/coloring/_dimacs_cache).",
    )
    args = parser.parse_args()

    mycs = [(f"myciel{k}", build_mycielski(k), k) for k in MYCIEL_ORDERS]
    write_family(args.out, "myciel", mycs)

    queens = [(f"queen{k}_{k}", build_queen(k), QUEEN_CHROMATIC[k]) for k in QUEEN_SIZES]
    write_family(args.out, "queen", queens)

    if not args.skip_dimacs:
        dimacs = [
            (name, build_dimacs(name, args.dimacs_cache), DIMACS_INSTANCES[name])
            for name in DIMACS_INSTANCES
        ]
        write_family(args.out, "dimacs", dimacs)


if __name__ == "__main__":
    main()
