"""Tests for the unified DISCS loader (``qqa.datasets.discs_*``).

These tests fabricate a tiny on-disk DISCS layout (3 nodes graphs, 2-3
instances per subset) and verify that ``discs_*`` returns the right problem
classes, ``best_known`` values, and survives both single-subset and "load all
subsets" calls.
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
    discs_maxclique,
    discs_maxcut,
    discs_mis,
    discs_normcut,
    list_discs_subsets,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_subset(
    base: Path,
    problem: str,
    graph_type: str,
    subset: str,
    graphs: list[nx.Graph],
    bests: list[float | None],
) -> Path:
    sdir = base / problem / graph_type / subset
    sdir.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    for i, (g, b) in enumerate(zip(graphs, bests, strict=True), start=1):
        fname = f"{i:04d}.gpickle"
        with open(sdir / fname, "wb") as fh:
            pickle.dump(g, fh)
        manifest_lines.append(
            {
                "id": f"{problem}-{graph_type}-{subset}-{i:04d}",
                "file": fname,
                "problem": problem,
                "graph_type": graph_type,
                "subset": subset,
                "num_nodes": g.number_of_nodes(),
                "num_edges": g.number_of_edges(),
                "best_known": b,
                "source": "synthetic",
            }
        )
    with open(sdir / "manifest.jsonl", "w") as fh:
        for rec in manifest_lines:
            fh.write(json.dumps(rec) + "\n")
    return sdir


@pytest.fixture
def fake_discs_root(tmp_path, monkeypatch) -> Path:
    """Build a minimal but realistic ``data/discs/...`` tree under tmp_path."""
    discs_root = tmp_path / "discs"

    # MaxCut: 2 subsets, 2 instances each, weighted cycles
    g1 = nx.cycle_graph(4)
    g2 = nx.cycle_graph(5)
    for u, v in g1.edges():
        g1[u][v]["weight"] = 1.0
    for u, v in g2.edges():
        g2[u][v]["weight"] = 1.0
    _make_subset(discs_root, "maxcut", "ba", "200", [g1, g2], [4.0, 4.0])
    _make_subset(discs_root, "maxcut", "ba", "400", [g1], [4.0])

    # MIS: SATLIB-style (unweighted)
    _make_subset(discs_root, "mis", "satlib", "uf", [nx.path_graph(5)], [3.0])

    # MaxClique: RB
    _make_subset(discs_root, "maxclique", "rb", "all", [nx.complete_graph(4)], [4.0])

    # NormCut: nets (no best_known)
    _make_subset(discs_root, "normcut", "nets", "VGG", [nx.cycle_graph(6)], [None])

    monkeypatch.setenv("QQA_DATA_DIR", str(tmp_path))
    return discs_root


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_discs_maxcut_single_subset(fake_discs_root):
    bench = discs_maxcut(graph_type="ba", subset="200")
    assert isinstance(bench, DiscsBenchmark)
    assert len(bench) == 2
    assert all(p.__class__.__name__ == "MaxCut" for p in bench.problems)
    assert bench.best_known.tolist() == [4.0, 4.0]
    assert bench.manifest[0]["graph_type"] == "ba"


def test_discs_maxcut_all_subsets_concatenated(fake_discs_root):
    # Loading without subset returns the union of all subsets under graph_type.
    bench = discs_maxcut(graph_type="ba", subset=None)
    assert len(bench) == 3  # 2 (200) + 1 (400)


def test_discs_maxcut_limit(fake_discs_root):
    bench = discs_maxcut(graph_type="ba", subset="200", limit=1)
    assert len(bench) == 1


def test_discs_mis(fake_discs_root):
    bench = discs_mis(graph_type="satlib")
    assert len(bench) == 1
    assert bench.problems[0].__class__.__name__ == "MaximumIndependentSet"
    assert bench.best_known[0] == 3.0


def test_discs_maxclique(fake_discs_root):
    bench = discs_maxclique(graph_type="rb")
    assert len(bench) == 1
    assert bench.problems[0].__class__.__name__ == "MaxClique"


def test_discs_normcut_no_best_known(fake_discs_root):
    bench = discs_normcut(graph_type="nets", subset="VGG")
    assert len(bench) == 1
    assert bench.problems[0].__class__.__name__ == "NormalizedCut"
    assert np.isnan(bench.best_known[0])


def test_discs_missing_root_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("QQA_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="discs.*does not exist"):
        discs_mis(graph_type="satlib")


def test_discs_missing_graph_type_raises(fake_discs_root):
    with pytest.raises(FileNotFoundError, match="DISCS subset directory"):
        discs_mis(graph_type="nope")


def test_discs_missing_subset_raises(fake_discs_root):
    with pytest.raises(FileNotFoundError, match="DISCS manifest not found"):
        discs_maxcut(graph_type="ba", subset="999")


def test_list_discs_subsets(fake_discs_root):
    catalog = list_discs_subsets()
    assert set(catalog.keys()) == {"maxcut", "mis", "maxclique", "normcut"}
    assert sorted(catalog["maxcut"]["ba"]) == ["200", "400"]
    assert catalog["mis"]["satlib"] == ["uf"]


def test_list_discs_subsets_empty_when_no_data(tmp_path, monkeypatch):
    monkeypatch.setenv("QQA_DATA_DIR", str(tmp_path))
    assert list_discs_subsets() == {}
