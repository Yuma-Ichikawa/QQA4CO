#!/usr/bin/env python3
"""Generate 3D Edwards-Anderson spin-glass benchmark instances.

The 3D EA Ising model on an ``L × L × L`` cubic lattice with nearest-
neighbour couplings ``J_ij`` drawn i.i.d. from a fixed distribution and
periodic boundary conditions. Hamiltonian:

    H(s) = − Σ_{⟨i,j⟩} J_ij s_i s_j,   s ∈ {−1, +1}^N,   N = L^3.

This is the canonical hard benchmark for spin-glass heuristics (Houdayer,
Young, Pal, ...). The PQQA paper does not benchmark 3D EA directly; we
include it per user request as a physics-grade stress test for QQA.

We emit both coupling distributions used in the literature:
  * ``gaussian``   — ``J_ij ~ N(0, 1)`` (Parisi-like, continuous).
  * ``bimodal``    — ``J_ij ∈ {+1, -1}`` with equal probability
    (±J model; classic ground-state search benchmark).

Couplings are stored as ``.npz`` (sparse pair lists) + ``manifest.jsonl``,
not ``.gpickle``, because the graph is fully determined by the pair list
and we want provenance (seed, L, dist) in the manifest. The QQA loader
reconstructs ``EdwardsAnderson`` via a helper that ingests this format.

Usage:
    python scripts/generate_ea3d_instances.py --out data
    python scripts/generate_ea3d_instances.py --L 4 6 8 --seeds 10 --out data
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

_DEFAULT_L = [4, 6, 8, 10]
_DEFAULT_DISTS = ["gaussian", "bimodal"]


def _nn_edges(L: int) -> list[tuple[int, int]]:
    """Nearest-neighbour edge list on a periodic L^3 lattice (i < j)."""
    edges: list[tuple[int, int]] = []
    for x, y, z in itertools.product(range(L), repeat=3):
        idx = (x * L + y) * L + z
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nx_ = (x + dx) % L
            ny_ = (y + dy) % L
            nz_ = (z + dz) % L
            nidx = (nx_ * L + ny_) * L + nz_
            a, b = min(idx, nidx), max(idx, nidx)
            if a != b:
                edges.append((a, b))
    return edges


def _sample_couplings(edges: list[tuple[int, int]], dist: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_edges = len(edges)
    if dist == "gaussian":
        return rng.standard_normal(n_edges).astype(np.float64)
    if dist == "bimodal":
        return rng.choice([-1.0, 1.0], size=n_edges).astype(np.float64)
    raise ValueError(f"unknown dist {dist!r}")


def write_instance(out_dir: Path, L: int, dist: str, seed: int) -> dict:
    edges = _nn_edges(L)
    J = _sample_couplings(edges, dist, seed)
    N = L**3
    fname = f"{seed:04d}.npz"
    path = out_dir / fname
    arr_i = np.asarray([e[0] for e in edges], dtype=np.int32)
    arr_j = np.asarray([e[1] for e in edges], dtype=np.int32)
    np.savez_compressed(path, i=arr_i, j=arr_j, J=J, L=np.int32(L), dist=dist)
    # Ground-state energy is generally unknown for L ≥ 4 without
    # branch-and-bound / spin-glass server. For L = 2 (N=8) we can
    # compute it exactly via brute-force enumeration.
    e_ground: float | None = None
    if N <= 20:
        e_ground = float(_brute_ground(N, edges, J))
    return {
        "id": f"ea3d-{dist}-L{L}-s{seed:04d}",
        "file": fname,
        "problem": "ea3d",
        "graph_type": dist,
        "subset": f"L{L}",
        "num_spins": N,
        "num_edges": len(edges),
        "L": L,
        "dist": dist,
        "seed": seed,
        "periodic": True,
        "best_known": e_ground,
        "best_known_source": "exact brute-force" if e_ground is not None else None,
        "source": f"ea3d({dist}, L={L}, seed={seed})",
    }


def _brute_ground(N: int, edges: list[tuple[int, int]], J: np.ndarray) -> float:
    """Brute-force ground-state energy for tiny instances (N ≤ ~20)."""
    best = float("inf")
    for k in range(1 << N):
        s = np.fromiter(((k >> b) & 1 for b in range(N)), count=N, dtype=np.int8)
        s = 2 * s - 1
        e = -float(sum(J[idx] * s[i] * s[j] for idx, (i, j) in enumerate(edges)))
        if e < best:
            best = e
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--L", type=int, nargs="+", default=_DEFAULT_L)
    parser.add_argument("--dist", nargs="+", default=_DEFAULT_DISTS)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    for dist in args.dist:
        for L in args.L:
            subset_dir = args.out / "ea3d" / dist / f"L{L}"
            subset_dir.mkdir(parents=True, exist_ok=True)
            records = []
            for seed in range(1, args.seeds + 1):
                records.append(write_instance(subset_dir, L, dist, seed))
            with open(subset_dir / "manifest.jsonl", "w") as fh:
                for rec in records:
                    fh.write(json.dumps(rec) + "\n")
            print(f"wrote {len(records)} {dist} L={L} instances to {subset_dir}")


if __name__ == "__main__":
    main()
