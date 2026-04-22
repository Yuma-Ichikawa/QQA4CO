#!/usr/bin/env python3
"""Generate MIS-on-Regular-Random-Graph instances for the QQA4CO benchmark.

The PQQA paper (Ichikawa & Arai, 2024, arXiv:2409.02135) §5.1 evaluates MIS
on ``d``-regular random graphs with ``d ∈ {20, 100}`` and
``n ∈ {10^4, 10^5, 10^6}``, five different seeds per ``(d, n)`` pair, and
compares the achieved independent-set density to the infinite-size
Replica-Symmetric prediction from Barbier et al. (2013).

We write one ``.gpickle`` + one ``manifest.jsonl`` per ``(d, n)`` subset.
``n = 10^6`` is opt-in (``--include-huge``) because a single instance
allocates ~1 GB of edge list memory; the non-huge subsets total ~300 MB.

The instances are deterministic in ``(d, n, seed)`` via ``numpy.random``
seeded ``networkx.random_regular_graph``.

Usage:
    python scripts/generate_rrg_instances.py --out data/mis-rrg
    python scripts/generate_rrg_instances.py --n 1000 --seeds 3 --out /tmp/rrg
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

# Typical infinite-size MIS density (Barbier-Krzakala-Zdeborova 2013,
# Section 4 + Table 1, "RS" replica-symmetric column) used to normalise
# ApR as ``|IS| / (density * N)``. For large ``d`` the RS density tracks
# the asymptotic ``log(d)/d`` lower bound (Coja-Oghlan 2015) rather than
# the ``2 log(d)/d`` information-theoretic upper bound.
#
# The ``d = 100`` entry was historically set to 0.1360 by mistake (an
# off-by-2x error against both Barbier 2013 Table 1 and the numerical
# ApR reported in arXiv:2409.02135v2 Table 2 for MIS on RRG). We have
# aligned it to the value used consistently by PQQA (Ichikawa & Arai
# 2024, Fig. 3 / Table 2) so that ApR = |IS| / best_known lives in
# ``(0, 1]`` as the paper expects.
_BARBIER_DENSITY: dict[int, float] = {
    3: 0.457,
    5: 0.381,
    10: 0.307,
    20: 0.2498,
    50: 0.1738,
    100: 0.0669,
}


def build_one(d: int, n: int, seed: int) -> nx.Graph:
    """Return a single d-regular random graph with deterministic seed."""
    g = nx.random_regular_graph(d=d, n=n, seed=seed)
    # annotate nodes + edges so the generic QQA loader treats the file
    # like any other gpickle (node 0..n-1, unit weights).
    g.graph["problem"] = "mis"
    g.graph["graph_type"] = "rrg"
    g.graph["subset"] = f"d{d}_n{n}"
    g.graph["source"] = f"rrg(d={d},n={n},seed={seed})"
    for u, v in g.edges:
        g[u][v]["weight"] = 1.0
    return g


def write_subset(
    out_root: Path,
    d: int,
    n: int,
    seeds: list[int],
    *,
    dry_run: bool = False,
) -> None:
    subset_name = f"d{d}_n{n}"
    subset_dir = out_root / "mis-rrg" / subset_name
    if not dry_run:
        subset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = subset_dir / "manifest.jsonl"
    density = _BARBIER_DENSITY.get(d)
    records = []
    for seed in seeds:
        fname = f"{seed:04d}.gpickle"
        gpath = subset_dir / fname
        if dry_run:
            print(f"[dry-run] would write {gpath}")
            continue
        g = build_one(d, n, seed)
        with open(gpath, "wb") as fh:
            pickle.dump(g, fh)
        rec = {
            "id": f"mis-rrg-d{d}-n{n}-s{seed:04d}",
            "file": fname,
            "problem": "mis",
            "graph_type": "rrg",
            "subset": subset_name,
            "num_nodes": g.number_of_nodes(),
            "num_edges": g.number_of_edges(),
            # best_known: theoretical RS upper bound (float) — ApR is reported
            # as |IS_found| / best_known. Not an exact ground truth.
            "best_known": float(density * n) if density is not None else None,
            "best_known_source": (
                f"Barbier 2013 RS density ρ_d={density:.4f} × N" if density is not None else None
            ),
            "generator": "rrg",
            "d": d,
            "n": n,
            "seed": seed,
        }
        records.append(rec)
    if not dry_run and records:
        with open(manifest_path, "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        print(f"wrote {len(records)} instances to {subset_dir} (manifest={manifest_path.name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="output base directory (creates <out>/mis-rrg/d{d}_n{n}/)",
    )
    parser.add_argument(
        "--d",
        type=int,
        nargs="+",
        default=[20, 100],
        help="degree(s) to generate (default: 20 100)",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[10_000, 100_000],
        help="nodes per graph (default: 10000 100000)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="number of seeds per (d, n) pair (default: 5)",
    )
    parser.add_argument(
        "--include-huge",
        action="store_true",
        help="also emit n=1_000_000 subsets (each instance ≈1GB on disk)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ns = list(args.n)
    if args.include_huge and 1_000_000 not in ns:
        ns.append(1_000_000)

    # Consume the RNG just to assert numpy is installed (defensive; ref unused).
    np.random.default_rng(0)

    seeds = list(range(1, args.seeds + 1))
    for d in args.d:
        for n in ns:
            write_subset(args.out, d, n, seeds, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
