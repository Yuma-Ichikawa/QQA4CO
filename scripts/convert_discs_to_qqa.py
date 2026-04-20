#!/usr/bin/env python3
"""Convert raw DISCS combinatorial-optimization instances to a unified format.

The DISCS NeurIPS-2023 release ships its CO benchmark as a soup of pickle
shapes (1 graph per file vs. many, ``(g, [sol])`` tuples vs. ``[(obj, g), ...]``
lists, raw DIMACS CNF, custom ``.mc`` text, ...). This script normalises
*all* of them into the same on-disk format::

    <dst>/<problem>/<graph_type>/<subset>/{0001.gpickle, 0002.gpickle, ...,
                                            manifest.jsonl}

Each ``*.gpickle`` is ``pickle.dump(networkx.Graph)``; each line in
``manifest.jsonl`` is one JSON record describing the instance and its known
best objective (when the source carries one).

The script is **idempotent**: re-running over an already-populated ``dst`` is
fine; existing instances are overwritten with identical bytes.

Usage::

    python scripts/convert_discs_to_qqa.py \\
        --src  data/discs/_raw         \\
        --dst  data/discs              \\
        --problem all                  \\
        # optional:
        --subsets satlib,ba-200,rb    # only convert these
        --limit   5                    # only first N per subset (smoke)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import pickle
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

LOG = logging.getLogger("discs2qqa")

# --------------------------------------------------------------------------- #
# Generic helpers                                                             #
# --------------------------------------------------------------------------- #


def _iter_pickle_stream(path: Path) -> Iterator:
    """Yield successive pickle records until EOF.

    DISCS' MaxCut/Optsicom files concatenate many ``pickle.dump(...)`` calls in
    a single binary file, terminated by EOF. Standard ``pickle.load`` only
    returns the first object; we loop until ``EOFError``.
    """
    with open(path, "rb") as fh:
        while True:
            try:
                yield pickle.load(fh)
            except EOFError:
                return


def _load_single_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _normalize_graph(
    g_in: nx.Graph,
    *,
    fill_weight: float | None = 1.0,
) -> nx.Graph:
    """Return a fresh ``nx.Graph`` with nodes relabelled to ``0..N-1``.

    - Drops parallel edges / self-loops (DISCS sources occasionally include
      them; QQA4CO ``qubo`` builders silently double-count otherwise).
    - Ensures every edge carries a ``weight`` attribute (defaults to
      ``fill_weight``; pass ``None`` to leave un-weighted edges as-is).
    """
    if g_in.is_directed():
        g_in = g_in.to_undirected()

    g_in = nx.Graph(g_in)
    g_in.remove_edges_from(nx.selfloop_edges(g_in))

    mapping = {n: i for i, n in enumerate(sorted(g_in.nodes()))}
    g = nx.relabel_nodes(g_in, mapping, copy=True)

    if fill_weight is not None:
        for _, _, data in g.edges(data=True):
            if "weight" not in data:
                data["weight"] = float(fill_weight)
            else:
                data["weight"] = float(data["weight"])

    return g


@dataclass
class _Subset:
    problem: str
    graph_type: str
    subset: str
    out_dir: Path
    manifest_lines: list[dict]

    @classmethod
    def open(
        cls,
        dst_root: Path,
        problem: str,
        graph_type: str,
        subset: str,
    ) -> _Subset:
        out_dir = dst_root / problem / graph_type / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        # Wipe any pre-existing manifest so the rewrite is clean.
        manifest = out_dir / "manifest.jsonl"
        if manifest.exists():
            manifest.unlink()
        return cls(problem, graph_type, subset, out_dir, [])

    def emit(
        self,
        graph: nx.Graph,
        *,
        index: int,
        best_known: float | None,
        source: str,
    ) -> None:
        gid = f"{self.problem}-{self.graph_type}-{self.subset}-{index:04d}"
        fname = f"{index:04d}.gpickle"
        path = self.out_dir / fname

        with open(path, "wb") as fh:
            pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)

        record = {
            "id": gid,
            "file": fname,
            "problem": self.problem,
            "graph_type": self.graph_type,
            "subset": self.subset,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "best_known": float(best_known) if best_known is not None else None,
            "source": source,
        }
        self.manifest_lines.append(record)

    def close(self) -> int:
        if not self.manifest_lines:
            # Nothing emitted; remove empty dir to keep the tree tidy.
            with contextlib.suppress(OSError):
                self.out_dir.rmdir()
            return 0
        manifest = self.out_dir / "manifest.jsonl"
        with open(manifest, "w") as fh:
            for rec in self.manifest_lines:
                fh.write(json.dumps(rec) + "\n")
        LOG.info(
            "  -> %s/%s: %d instances",
            self.graph_type,
            self.subset,
            len(self.manifest_lines),
        )
        return len(self.manifest_lines)


# --------------------------------------------------------------------------- #
# MaxCut                                                                      #
# --------------------------------------------------------------------------- #


def convert_maxcut_random(src_root: Path, dst_root: Path, kind: str, *, limit: int | None) -> int:
    """``maxcut-ba`` / ``maxcut-er`` → unified.

    Source layout::
        <src>/maxcut-{ba|er}/maxcut-<rand_type>/test-*  (pickle stream)
    Each pickled record is ``(g, [sol, ...])``.
    """
    src_dir = src_root / f"maxcut-{kind}"
    if not src_dir.is_dir():
        LOG.warning("skip maxcut-%s: %s not found", kind, src_dir)
        return 0

    total = 0
    for sub in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        # sub is e.g. ``maxcut-200`` or ``maxcut-1024-1100``
        subset = sub.name.removeprefix("maxcut-")
        out = _Subset.open(dst_root, "maxcut", kind, subset)
        idx = 1
        for fname in sorted(sub.iterdir()):
            if not fname.name.startswith("test-"):
                continue
            for record in _iter_pickle_stream(fname):
                g_raw, sols = record[0], record[1]
                g = _normalize_graph(g_raw, fill_weight=1.0)
                best = float(sols[0]) if sols else None
                out.emit(g, index=idx, best_known=best, source=str(fname.relative_to(src_root)))
                idx += 1
                if limit is not None and idx > limit:
                    break
            if limit is not None and idx > limit:
                break
        total += out.close()
    return total


def convert_maxcut_optsicom(src_root: Path, dst_root: Path, *, limit: int | None) -> int:
    src_file = src_root / "optsicom" / "b.pkl"
    if not src_file.is_file():
        LOG.warning("skip maxcut-optsicom: %s not found", src_file)
        return 0
    out = _Subset.open(dst_root, "maxcut", "optsicom", "b")
    idx = 1
    for record in _iter_pickle_stream(src_file):
        g_raw, sols = record[0], record[1]
        g = _normalize_graph(g_raw, fill_weight=1.0)
        best = float(sols[0]) if sols else None
        out.emit(g, index=idx, best_known=best, source=str(src_file.relative_to(src_root)))
        idx += 1
        if limit is not None and idx > limit:
            break
    return out.close()


# --------------------------------------------------------------------------- #
# MIS                                                                         #
# --------------------------------------------------------------------------- #


def convert_mis_er_test(
    src_root: Path, dst_root: Path, rand_type: str, *, limit: int | None
) -> int:
    """``er_<rand_type>_test/ER*`` — 1 file = 1 graph; node ``label`` summed."""
    src_dir = src_root / f"er_{rand_type}_test"
    if not src_dir.is_dir():
        LOG.warning("skip mis-er_%s: %s not found", rand_type, src_dir)
        return 0

    out = _Subset.open(dst_root, "mis", "er", rand_type)
    idx = 1
    for fname in sorted(src_dir.iterdir()):
        if not fname.name.startswith("ER"):
            continue
        g_raw = _load_single_pickle(fname)
        g = _normalize_graph(g_raw, fill_weight=None)  # MIS is unweighted

        # The DISCS loader sums per-node 'label' for a known-best — we keep
        # whichever is non-zero.
        best = sum(d.get("label", 0) for _, d in g.nodes(data=True)) or None
        out.emit(g, index=idx, best_known=best, source=str(fname.relative_to(src_root)))
        idx += 1
        if limit is not None and idx > limit:
            break
    return out.close()


def convert_mis_er_density(src_root: Path, dst_root: Path, *, limit: int | None) -> int:
    """``er_700_800/ER-700-800-<density>.pkl`` — 1 file = list of graphs."""
    src_dir = src_root / "er_700_800"
    if not src_dir.is_dir():
        LOG.warning("skip mis-er_density: %s not found", src_dir)
        return 0
    total = 0
    for fname in sorted(src_dir.iterdir()):
        if not fname.name.endswith(".pkl"):
            continue
        # ``ER-700-800-0.10.pkl`` → density="0.10"
        density = fname.stem.replace("ER-700-800-", "")
        out = _Subset.open(dst_root, "mis", "er_density", density)
        g_list = _load_single_pickle(fname)
        for idx, g_raw in enumerate(g_list, start=1):
            g = _normalize_graph(g_raw, fill_weight=None)
            out.emit(g, index=idx, best_known=None, source=str(fname.relative_to(src_root)))
            if limit is not None and idx >= limit:
                break
        total += out.close()
    return total


def convert_mis_satlib(src_root: Path, dst_root: Path, *, limit: int | None) -> int:
    """``satlib_test/*.cnf`` → 3-clause graph encoding."""
    try:
        from pysat.formula import CNF
    except ImportError as e:
        LOG.error("satlib needs python-sat: %s", e)
        return 0

    src_dir = src_root / "satlib_test"
    if not src_dir.is_dir():
        LOG.warning("skip mis-satlib: %s not found", src_dir)
        return 0

    out = _Subset.open(dst_root, "mis", "satlib", "uf")
    idx = 1
    for fname in sorted(src_dir.iterdir()):
        if fname.suffix != ".cnf":
            continue
        cnf = CNF(str(fname))
        g_raw = _cnf_to_graph(cnf)
        g = _normalize_graph(g_raw, fill_weight=None)
        # Best-known MIS for an n-clause SAT formula = n iff satisfiable
        # (one literal per clause picked); the DISCS pickle does not carry it,
        # but for SATLIB uf-* benchmarks we know all clauses are satisfiable,
        # so #clauses is the optimum.
        best = len(list(filter(lambda c: c, cnf.clauses)))
        out.emit(g, index=idx, best_known=best, source=str(fname.relative_to(src_root)))
        idx += 1
        if limit is not None and idx > limit:
            break
    return out.close()


def _cnf_to_graph(cnf) -> nx.Graph:
    """Replicates ``mis_loader.SatLibGraphGen.cnf2graph`` from DISCS."""
    import numpy as np

    nv = cnf.nv
    clauses = list(filter(lambda x: x, cnf.clauses))
    ind = {k: [] for k in np.concatenate([np.arange(1, nv + 1), -np.arange(1, nv + 1)])}
    edges: list[tuple[int, int]] = []
    for i, clause in enumerate(clauses):
        a, b, c = clause[0], clause[1], clause[2]
        aa, bb, cc = 3 * i, 3 * i + 1, 3 * i + 2
        ind[a].append(aa)
        ind[b].append(bb)
        ind[c].append(cc)
        edges.extend([(aa, bb), (aa, cc), (bb, cc)])
    for i in np.arange(1, nv + 1):
        for u in ind[i]:
            for v in ind[-i]:
                edges.append((u, v))
    return nx.from_edgelist(edges)


# --------------------------------------------------------------------------- #
# MaxClique                                                                   #
# --------------------------------------------------------------------------- #


def convert_maxclique_rb(src_root: Path, dst_root: Path, *, limit: int | None) -> int:
    """``RB_test/RB*`` — 1 file = list[(obj, g)]."""
    src_dir = src_root / "RB_test"
    if not src_dir.is_dir():
        LOG.warning("skip maxclique-rb: %s not found", src_dir)
        return 0
    out = _Subset.open(dst_root, "maxclique", "rb", "all")
    idx = 1
    for fname in sorted(src_dir.iterdir()):
        if not fname.name.startswith("RB"):
            continue
        g_list = _load_single_pickle(fname)
        for obj, g_raw in g_list:
            g = _normalize_graph(g_raw, fill_weight=None)
            out.emit(g, index=idx, best_known=float(obj), source=str(fname.relative_to(src_root)))
            idx += 1
            if limit is not None and idx > limit:
                break
        if limit is not None and idx > limit:
            break
    return out.close()


def convert_maxclique_twitter(src_root: Path, dst_root: Path, *, limit: int | None) -> int:
    """``twitter/twitter*`` — 1 file = (graphs[], objs[])."""
    src_dir = src_root / "twitter"
    if not src_dir.is_dir():
        LOG.warning("skip maxclique-twitter: %s not found", src_dir)
        return 0
    out = _Subset.open(dst_root, "maxclique", "twitter", "all")
    idx = 1
    for fname in sorted(src_dir.iterdir()):
        if not fname.name.startswith("twitter"):
            continue
        data = _load_single_pickle(fname)
        graphs, objs = data[0], data[1]
        for g_raw, obj in zip(graphs, objs, strict=False):
            g = _normalize_graph(g_raw, fill_weight=None)
            out.emit(g, index=idx, best_known=float(obj), source=str(fname.relative_to(src_root)))
            idx += 1
            if limit is not None and idx > limit:
                break
        if limit is not None and idx > limit:
            break
    return out.close()


# --------------------------------------------------------------------------- #
# NormCut                                                                     #
# --------------------------------------------------------------------------- #


def convert_normcut_nets(src_root: Path, dst_root: Path, *, limit: int | None) -> int:
    """``nets/<NAME>.pkl`` — 1 file = 1 (large) computation graph.

    The DISCS NeurIPS-2023 release ships a few empty / corrupted nets
    pickles (notably ``TRANSFORMER.pkl`` — 0 nodes, 0 edges). We skip
    those with a warning so downstream solvers never see a degenerate
    instance.
    """
    src_dir = src_root / "nets"
    if not src_dir.is_dir():
        LOG.warning("skip normcut-nets: %s not found", src_dir)
        return 0
    total = 0
    for idx, fname in enumerate(sorted(src_dir.iterdir()), start=1):
        if fname.suffix != ".pkl":
            continue
        name = fname.stem
        g_raw = _load_single_pickle(fname)
        g = _normalize_graph(g_raw, fill_weight=None)
        if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
            LOG.warning(
                "skip normcut-nets/%s: source pickle is empty (N=%d, E=%d)",
                name,
                g.number_of_nodes(),
                g.number_of_edges(),
            )
            continue
        out = _Subset.open(dst_root, "normcut", "nets", name)
        out.emit(g, index=1, best_known=None, source=str(fname.relative_to(src_root)))
        total += out.close()
        if limit is not None and idx >= limit:
            break
    return total


def convert_normcut_gap_rand(src_root: Path, dst_root: Path, *, limit: int | None) -> int:
    """``gap_rand/<rand_type>/*.pkl`` — each file = 1 graph."""
    src_dir = src_root / "gap_rand"
    if not src_dir.is_dir():
        LOG.warning("skip normcut-gap_rand: %s not found", src_dir)
        return 0
    total = 0
    for sub in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        out = _Subset.open(dst_root, "normcut", "gap_rand", sub.name)
        idx = 1
        for fname in sorted(sub.iterdir()):
            if fname.suffix != ".pkl":
                continue
            g_raw = _load_single_pickle(fname)
            g = _normalize_graph(g_raw, fill_weight=None)
            out.emit(g, index=idx, best_known=None, source=str(fname.relative_to(src_root)))
            idx += 1
            if limit is not None and idx > limit:
                break
        total += out.close()
    return total


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


_PROBLEMS = ("maxcut", "mis", "maxclique", "normcut")

_DISPATCH = {
    "maxcut": [
        ("ba", lambda s, d, lim: convert_maxcut_random(s, d, "ba", limit=lim)),
        ("er", lambda s, d, lim: convert_maxcut_random(s, d, "er", limit=lim)),
        ("optsicom", lambda s, d, lim: convert_maxcut_optsicom(s, d, limit=lim)),
    ],
    "mis": [
        ("er-800", lambda s, d, lim: convert_mis_er_test(s, d, "800", limit=lim)),
        ("er-10k", lambda s, d, lim: convert_mis_er_test(s, d, "10k", limit=lim)),
        ("er_density", lambda s, d, lim: convert_mis_er_density(s, d, limit=lim)),
        ("satlib", lambda s, d, lim: convert_mis_satlib(s, d, limit=lim)),
    ],
    "maxclique": [
        ("rb", lambda s, d, lim: convert_maxclique_rb(s, d, limit=lim)),
        ("twitter", lambda s, d, lim: convert_maxclique_twitter(s, d, limit=lim)),
    ],
    "normcut": [
        ("nets", lambda s, d, lim: convert_normcut_nets(s, d, limit=lim)),
        ("gap_rand", lambda s, d, lim: convert_normcut_gap_rand(s, d, limit=lim)),
    ],
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", type=Path, required=True, help="raw DISCS sco/ root")
    p.add_argument("--dst", type=Path, required=True, help="unified output root (data/discs)")
    p.add_argument(
        "--problem",
        choices=("all", *_PROBLEMS),
        default="all",
        help="restrict to a single problem",
    )
    p.add_argument(
        "--subsets",
        type=str,
        default=None,
        help="comma-separated subset selectors (e.g. 'satlib,ba,rb'); matches against the loader name",
    )
    p.add_argument("--limit", type=int, default=None, help="cap #instances per subset (smoke)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.src.is_dir():
        LOG.error("src directory does not exist: %s", args.src)
        return 2

    args.dst.mkdir(parents=True, exist_ok=True)
    selectors = set(args.subsets.split(",")) if args.subsets else None

    problems = _PROBLEMS if args.problem == "all" else (args.problem,)
    grand_total = 0
    for prob in problems:
        LOG.info("== %s ==", prob)
        for label, fn in _DISPATCH[prob]:
            if selectors and not any(s in label for s in selectors):
                continue
            grand_total += fn(args.src, args.dst, args.limit)

    LOG.info("done: %d instances written under %s", grand_total, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
