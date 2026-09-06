"""Tests for the non-DISCS benchmark loaders (``qqa.datasets.{coloring, mis_rrg, ea3d, balanced_partition, list_benchmark_families}``).

Each test fabricates a tiny on-disk tree under ``tmp_path`` and points
``QQA_DATA_DIR`` at it, so the tests run without any network / HF Hub
access. They check that the loader returns a ``DiscsBenchmark`` wrapping
the right problem class, propagates the manifest metadata, and that
``list_benchmark_families`` discovers the fabricated layout.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from qqa.datasets import (
    DiscsBenchmark,
    balanced_partition,
    coloring,
    ea3d,
    gset,
    list_benchmark_families,
    mis_rrg,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _write_gpickle_subset(
    subset_dir: Path,
    graphs: list[nx.Graph],
    records_extra: list[dict],
) -> None:
    subset_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    for i, (g, extra) in enumerate(zip(graphs, records_extra, strict=True), start=1):
        fname = f"{i:04d}.gpickle"
        with open(subset_dir / fname, "wb") as fh:
            pickle.dump(g, fh)
        rec = {
            "id": f"{subset_dir.name}-{i:04d}",
            "file": fname,
            "num_nodes": g.number_of_nodes(),
            "num_edges": g.number_of_edges(),
            "best_known": None,
            **extra,
        }
        manifest_lines.append(rec)
    with open(subset_dir / "manifest.jsonl", "w") as fh:
        for rec in manifest_lines:
            fh.write(json.dumps(rec) + "\n")


def _write_ea3d_subset(
    subset_dir: Path,
    L: int,
    instances: list[tuple[np.ndarray, np.ndarray, np.ndarray, float | None]],
) -> None:
    """Write tiny EA3D instances as ``.npz`` + manifest row.

    ``instances`` is a list of ``(i, j, J, best_known)`` tuples matching
    the on-disk format used by ``scripts/generate_ea3d_instances.py``.
    """
    subset_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx, (i, j, J, best) in enumerate(instances, start=1):
        fname = f"{idx:04d}.npz"
        np.savez(
            subset_dir / fname,
            i=i.astype(np.int64),
            j=j.astype(np.int64),
            J=J.astype(np.float64),
            L=np.asarray(L, dtype=np.int64),
        )
        lines.append(
            {
                "id": f"ea3d-{subset_dir.name}-{idx:04d}",
                "file": fname,
                "num_spins": L**3,
                "L": L,
                "best_known": best,
            }
        )
    with open(subset_dir / "manifest.jsonl", "w") as fh:
        for rec in lines:
            fh.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    root.mkdir()

    # Coloring: myciel/ (flat) and queen/ (flat)
    myciel3 = nx.mycielski_graph(3)
    _write_gpickle_subset(
        root / "coloring" / "myciel",
        [myciel3, nx.mycielski_graph(4)],
        [
            {"graph_type": "myciel", "k": 3, "num_colors": 4, "best_known": 0},
            {"graph_type": "myciel", "k": 4, "num_colors": 5, "best_known": 0},
        ],
    )
    _write_gpickle_subset(
        root / "coloring" / "queen",
        [nx.cycle_graph(4)],
        [{"graph_type": "queen", "num_colors": 3, "best_known": 0}],
    )

    # MIS on RRG: d4_n20/ (flat)
    rng = np.random.default_rng(0)
    g_rrg = nx.random_regular_graph(4, 20, seed=int(rng.integers(0, 1_000_000)))
    _write_gpickle_subset(
        root / "mis-rrg" / "d4_n20",
        [g_rrg],
        [{"degree": 4, "num_nodes": 20, "seed": 0, "best_known": 9.0}],
    )

    # EA3D: gaussian/L2 (two-level layout, tiny 2x2x2 cube)
    L = 2
    i_arr = np.array([0, 1, 2], dtype=np.int64)
    j_arr = np.array([1, 2, 3], dtype=np.int64)
    J_arr = np.array([-0.5, 0.7, -1.1], dtype=np.float64)
    _write_ea3d_subset(
        root / "ea3d" / "gaussian" / "L2",
        L,
        [(i_arr, j_arr, J_arr, -1.5), (i_arr, j_arr, J_arr * 0.5, None)],
    )

    # Balanced partition source (DISCS normcut/nets)
    _write_gpickle_subset(
        root / "discs" / "normcut" / "nets" / "MINI",
        [nx.path_graph(6)],
        [
            {
                "problem": "normcut",
                "graph_type": "nets",
                "subset": "MINI",
                "best_known": None,
            }
        ],
    )

    # G-set (flat: data/gset/standard/manifest.jsonl with two records).
    # The records mirror the fields written by scripts/fetch_gset_data.py.
    _write_gpickle_subset(
        root / "gset" / "standard",
        [nx.complete_graph(5), nx.cycle_graph(6)],
        [
            {
                "problem": "maxcut",
                "graph_type": "gset",
                "subset": "standard",
                "best_known": 6,
                "best_known_source": "fixture",
                "source_url": "fixture://G-fake-1",
            },
            {
                "problem": "maxcut",
                "graph_type": "gset",
                "subset": "standard",
                "best_known": 6,
                "best_known_source": "fixture",
                "source_url": "fixture://G-fake-2",
            },
        ],
    )

    monkeypatch.setenv("QQA_DATA_DIR", str(root))
    return root


# --------------------------------------------------------------------------- #
# Coloring                                                                    #
# --------------------------------------------------------------------------- #


def test_coloring_myciel_loads(fake_root):
    bench = coloring(graph_type="myciel")
    assert isinstance(bench, DiscsBenchmark)
    assert len(bench) == 2
    assert all(p.__class__.__name__ == "Coloring" for p in bench.problems)
    # num_colors picked up from manifest
    assert bench.problems[0].num_category == 4
    assert bench.manifest[1]["k"] == 4


def test_coloring_all_types_no_filter(fake_root):
    bench = coloring()  # no graph_type => every subset
    assert len(bench) == 3  # 2 myciel + 1 queen


def test_coloring_num_colors_override(fake_root):
    bench = coloring(graph_type="myciel", num_colors=7)
    assert all(p.num_category == 7 for p in bench.problems)


def test_coloring_limit(fake_root):
    bench = coloring(graph_type="myciel", limit=1)
    assert len(bench) == 1


def test_coloring_limit_does_not_open_records_beyond_the_limit(fake_root):
    manifest = fake_root / "coloring" / "myciel" / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    records[1]["file"] = "missing.gpickle"
    manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    bench = coloring(graph_type="myciel", limit=1)
    assert len(bench) == 1


# --------------------------------------------------------------------------- #
# MIS on RRG                                                                  #
# --------------------------------------------------------------------------- #


def test_mis_rrg_loads(fake_root):
    bench = mis_rrg(subset="d4_n20")
    assert isinstance(bench, DiscsBenchmark)
    assert len(bench) == 1
    assert bench.problems[0].__class__.__name__ == "MaximumIndependentSet"
    assert bench.best_known[0] == 9.0
    assert bench.manifest[0]["degree"] == 4


def test_mis_rrg_all_subsets(fake_root):
    bench = mis_rrg()
    assert len(bench) == 1  # only one subset in the fixture


# --------------------------------------------------------------------------- #
# EA3D                                                                        #
# --------------------------------------------------------------------------- #


def test_ea3d_gaussian_loads(fake_root):
    bench = ea3d(dist="gaussian", subset="L2")
    assert isinstance(bench, DiscsBenchmark)
    assert len(bench) == 2
    ea = bench.problems[0]
    assert ea.__class__.__name__ == "EdwardsAnderson"
    assert ea.num_spins == 2**3 == 8
    # Coupling matrix is symmetric and contains the fixture values.
    assert ea.J.shape == (8, 8)
    assert ea.J[0, 1].item() == pytest.approx(-0.5)
    assert ea.J[1, 0].item() == pytest.approx(-0.5)
    assert np.isfinite(bench.best_known[0])
    assert np.isnan(bench.best_known[1])


def test_ea3d_all_dists_no_filter(fake_root):
    bench = ea3d()  # recurse into every dist/subset
    assert len(bench) == 2


def test_ea3d_missing_dist_raises(fake_root):
    with pytest.raises(FileNotFoundError):
        ea3d(dist="nope")


# --------------------------------------------------------------------------- #
# Balanced partition (DISCS normcut/nets reuse)                               #
# --------------------------------------------------------------------------- #


def test_balanced_partition_from_normcut(fake_root):
    bench = balanced_partition(graph_type="nets", subset="MINI", num_category=3)
    assert isinstance(bench, DiscsBenchmark)
    assert len(bench) == 1
    prob = bench.problems[0]
    assert prob.__class__.__name__ == "BalancedGraphPartition"
    assert prob.num_category == 3
    assert np.isnan(bench.best_known[0])
    assert bench.manifest[0]["num_category"] == 3


def test_balanced_partition_missing_graph_type_raises(fake_root):
    with pytest.raises(FileNotFoundError, match="balanced-partition"):
        balanced_partition(graph_type="nope")


# --------------------------------------------------------------------------- #
# Family catalog                                                              #
# --------------------------------------------------------------------------- #


def test_list_benchmark_families(fake_root):
    cat = list_benchmark_families()
    # Families present:
    assert set(cat.keys()) >= {"coloring", "mis-rrg", "ea3d", "discs", "gset"}
    # coloring is flat-per-graph_type => everything sits under key ""
    assert set(cat["coloring"][""]) == {"myciel", "queen"}
    assert cat["mis-rrg"][""] == ["d4_n20"]
    # ea3d uses a two-level {dist: [L-tag, ...]} layout.
    assert cat["ea3d"]["gaussian"] == ["L2"]
    # discs structure is preserved verbatim.
    assert cat["discs"]["normcut"] == ["nets"]
    # gset ships as a flat single-subset family under data/gset/standard/.
    assert cat["gset"][""] == ["standard"]


# --------------------------------------------------------------------------- #
# G-set                                                                       #
# --------------------------------------------------------------------------- #


def test_gset_loads(fake_root):
    bench = gset()
    assert isinstance(bench, DiscsBenchmark)
    assert len(bench) == 2
    assert all(p.__class__.__name__ == "MaxCut" for p in bench.problems)
    # best_known and source metadata propagated from the manifest.
    assert bench.best_known[0] == 6.0
    assert bench.manifest[0]["best_known_source"] == "fixture"
    assert bench.manifest[0]["source_url"].startswith("fixture://")


def test_gset_limit(fake_root):
    bench = gset(limit=1)
    assert len(bench) == 1


def test_gset_bench_discs_catalog_registers(fake_root, monkeypatch):
    """The bench_discs.py catalog must pick up ``gset`` via
    ``list_benchmark_families`` so ``--suite gset`` resolves end-to-end
    without any DISCS data on disk (regression test for the rename +
    Gset integration).
    """
    from qqa.benchmarking import runner as bench_discs

    catalog = bench_discs._build_catalog()
    assert "gset" in catalog
    resolved = bench_discs._resolve_suite("gset")
    assert any(fam == "gset" for fam, _, _ in resolved)
