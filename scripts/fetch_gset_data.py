"""Fetch the MaxCut G-set benchmark from Stanford and materialise it as
``data/gset/standard/``.

G-set is the canonical MaxCut benchmark — Helmberg & Rendl published the
original G1–G54 instances in 2000 and Yinyu Ye has hosted the full
G1–G67 + G70/G72/G77/G81 superset at

    https://web.stanford.edu/~yyye/yyye/Gset/G<n>

ever since. Not every slot between G1 and G81 exists upstream (G68, G69,
G71, G73–G76, G78–G80 return HTTP 404). This script silently skips those.

The output layout is the same as every other family in this repo::

    data/gset/standard/
        G1.gpickle       pickle.dump(networkx.Graph)
        G2.gpickle
        ...
        G81.gpickle
        manifest.jsonl   one JSON record per instance

Each manifest record carries at least::

    {
        "id": "G70",
        "file": "G70.gpickle",
        "nodes": 10000,
        "edges": 9999,
        "best_known": 9591,
        "best_known_source": "Benlic & Hao 2013 + Ichikawa 2024"
    }

Best-known cuts are the highest values reported in the peer-reviewed
literature we could verify (Benlic & Hao 2013, Matsuda 2018,
Martí et al. 2009, Ichikawa NeurIPS 2024). When a fresh upper bound
becomes available, update ``BEST_KNOWN`` and re-run this script.

Usage::

    python scripts/fetch_gset_data.py                    # download + convert
    python scripts/fetch_gset_data.py --skip-download    # convert only
    python scripts/fetch_gset_data.py --out data/gset_v2 # alternate dest
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import urllib.error
import urllib.request
from pathlib import Path

import networkx as nx

GSET_URL = "https://web.stanford.edu/~yyye/yyye/Gset/G{n}"

# Canonical G-set best-known cuts.
#
# Sources (values preferred in this order when they disagree):
#   1. Benlic & Hao (2013) — "Breakout Local Search for the Max-Cut
#      Problem", Engineering Applications of Artificial Intelligence 26
#      (1162-1173). Tables 1 & 2 cover G1-G67 comprehensively.
#   2. Matsuda (2018) — "Benchmark Instances for the Max-Cut Problem",
#      updated G43-G67.
#   3. Ichikawa (NeurIPS 2024) — G70 best-known (9591).
#   4. Martí, Duarte & Laguna (2009) — G1-G50 updates.
BEST_KNOWN: dict[int, int] = {
    # G1..G5: n=800 random weighted
    1: 11624,
    2: 11620,
    3: 11622,
    4: 11646,
    5: 11631,
    # G6..G10: n=800 random ±1 weighted
    6: 2178,
    7: 2006,
    8: 2005,
    9: 2054,
    10: 2000,
    # G11..G13: n=800 toroidal 3-regular
    11: 564,
    12: 556,
    13: 582,
    # G14..G17: n=800 planar ±1
    14: 3064,
    15: 3050,
    16: 3052,
    17: 3047,
    # G18..G21: n=800 toroidal ±1
    18: 992,
    19: 906,
    20: 941,
    21: 931,
    # G22..G26: n=2000 random dense
    22: 13359,
    23: 13344,
    24: 13337,
    25: 13340,
    26: 13328,
    # G27..G31: n=2000 random ±1 dense
    27: 3341,
    28: 3298,
    29: 3405,
    30: 3412,
    31: 3309,
    # G32..G34: n=2000 toroidal 3-regular
    32: 1410,
    33: 1382,
    34: 1384,
    # G35..G38: n=2000 planar ±1
    35: 7684,
    36: 7677,
    37: 7689,
    38: 7681,
    # G39..G42: n=2000 toroidal ±1
    39: 2408,
    40: 2400,
    41: 2405,
    42: 2481,
    # G43..G47: n=1000 dense
    43: 6660,
    44: 6650,
    45: 6654,
    46: 6649,
    47: 6657,
    # G48..G50: n=3000 toroidal (±1 / bipartite-ish)
    48: 6000,
    49: 6000,
    50: 5880,
    # G51..G54: n=1000 planar ±1
    51: 3848,
    52: 3851,
    53: 3850,
    54: 3852,
    # G55..G67: sparse large
    55: 10299,
    56: 4017,
    57: 3494,
    58: 19293,
    59: 6086,
    60: 14188,
    61: 5796,
    62: 4870,
    63: 27045,
    64: 8751,
    65: 5562,
    66: 6364,
    67: 6950,
    # G70, G72, G77, G81 — sparse near-bipartite large
    70: 9591,
    72: 7006,
    77: 9938,
    81: 14048,
}
BEST_KNOWN_SOURCE = "Benlic & Hao 2013; Matsuda 2018; Ichikawa NeurIPS 2024"


def _download_one(n: int, dest: Path, *, verbose: bool = True) -> bool:
    """Fetch a single Gset file. Returns ``False`` on 404."""
    url = GSET_URL.format(n=n)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        if verbose:
            print(f"[gset] G{n:<2d} cached at {dest}")
        return True
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
        if verbose:
            print(f"[gset] G{n:<2d} downloaded ({dest.stat().st_size / 1024:.0f} KB)")
        return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            if verbose:
                print(f"[gset] G{n:<2d} skip (upstream 404)")
            return False
        raise


def _parse_gset(path: Path) -> nx.Graph:
    """Parse the canonical G-set text format.

    Line 1: ``n m`` (num nodes, num edges).
    Subsequent lines: ``u v w`` (1-indexed, integer weight, usually ±1).

    We store the graph 0-indexed (matching every other loader in this repo)
    and attach ``weight=float(w)`` to each edge so downstream solvers see
    the original signed weight.
    """
    with path.open() as fh:
        header = fh.readline().split()
        n, m = int(header[0]), int(header[1])
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            u, v, w = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
            g.add_edge(u, v, weight=w)
    if g.number_of_edges() != m:
        # G-set occasionally has duplicate edges in the ±1 variants; tolerate
        # but warn instead of failing hard.
        print(
            f"[gset] {path.name}: parsed {g.number_of_edges()} edges but header says {m}; "
            f"using the parsed graph (duplicate-edge collisions are common)",
            file=sys.stderr,
        )
    return g


def materialise(raw_dir: Path, out_dir: Path) -> list[dict]:
    """Convert every Gset text file under ``raw_dir`` into a .gpickle +
    append a manifest record.

    Returns the list of manifest records actually written so callers can
    chain this with an upload step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for n in sorted(BEST_KNOWN):
        src = raw_dir / f"G{n}"
        if not src.is_file():
            print(f"[gset] G{n} missing on disk, skip")
            continue
        g = _parse_gset(src)
        dst = out_dir / f"G{n}.gpickle"
        with dst.open("wb") as fh:
            pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)
        records.append(
            {
                "id": f"G{n}",
                "file": dst.name,
                "nodes": g.number_of_nodes(),
                "edges": g.number_of_edges(),
                "best_known": BEST_KNOWN[n],
                "best_known_source": BEST_KNOWN_SOURCE,
                "source_url": GSET_URL.format(n=n),
                "problem": "maxcut",
                "graph_type": "gset",
                "subset": "standard",
            }
        )
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[gset] wrote {len(records)} records to {manifest}")
    return records


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_out = repo_root / "data" / "gset" / "standard"

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument(
        "--raw",
        type=Path,
        default=default_out,
        help=(
            "Directory holding the raw text files (G1, G2, ...). Defaults to "
            "--out, so the download+convert flow lands everything in place."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the Stanford download step (useful in CI / air-gapped).",
    )
    args = parser.parse_args()

    args.raw.mkdir(parents=True, exist_ok=True)
    got = 0
    if not args.skip_download:
        for n in sorted(BEST_KNOWN):
            if _download_one(n, args.raw / f"G{n}"):
                got += 1
        print(f"[gset] downloaded / cached {got} / {len(BEST_KNOWN)} instances")
    materialise(args.raw, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
