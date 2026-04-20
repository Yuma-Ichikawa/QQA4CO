"""Tests for ``qqa.NormalizedCut``."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import qqa


def _onehot(assignment: list[int], K: int) -> torch.Tensor:
    """Build a (1, N, K) one-hot tensor from a Python list."""
    x = torch.zeros((1, len(assignment), K))
    for i, k in enumerate(assignment):
        x[0, i, k] = 1.0
    return x


def test_normcut_K2_path_perfect_bisection():
    # A path 0-1-2-3-4 with the bisection {0,1} | {2,3,4}:
    # cut = 1 (edge 1-2);   vol_0 = deg(0)+deg(1) = 1 + 2 = 3
    # vol_1 = deg(2)+deg(3)+deg(4) = 2 + 2 + 1 = 5
    # Ncut = 1/3 + 1/5 = 8/15
    g = nx.path_graph(5)
    p = qqa.NormalizedCut(g, num_category=2, device="cpu", eps=0.0)
    x = _onehot([0, 0, 1, 1, 1], K=2)
    ncut = p.discrete_ncut(x).item()
    assert ncut == pytest.approx(1 / 3 + 1 / 5, rel=1e-6)


def test_normcut_K2_isolated_partition_uses_finite_value():
    # All nodes in partition 0 -> partition 1 has volume 0.
    # discrete_ncut should NOT NaN/inf — we replace empty-partition vol with 1
    # and cut with |E|, so the empty partition contributes |E|/1.
    g = nx.cycle_graph(4)
    p = qqa.NormalizedCut(g, num_category=2, device="cpu")
    x = _onehot([0, 0, 0, 0], K=2)
    ncut = p.discrete_ncut(x).item()
    assert torch.isfinite(torch.tensor(ncut)).item()


def test_normcut_loss_smooth_matches_discrete_on_one_hot():
    # On a balanced one-hot, the smooth and discrete forms must agree
    # (modulo the eps in the smooth denominator, which we make tiny).
    g = nx.cycle_graph(6)
    p = qqa.NormalizedCut(g, num_category=2, device="cpu", eps=1e-9)
    x = _onehot([0, 1, 0, 1, 0, 1], K=2)
    smooth = p.loss_fn(x).item()
    discrete = p.discrete_ncut(x).item()
    assert smooth == pytest.approx(discrete, rel=1e-4)


def test_normcut_K3_triangle_each_node_own_class():
    # Each node in its own class on K_3: every edge is cut.
    # cut_k = 2 for all k (each node has degree 2)? Let's compute:
    # Edges: (0,1), (0,2), (1,2). Assignment [0,1,2].
    # cut for class 0: edges incident to node 0 with the other endpoint
    # in a different class → both (0,1) and (0,2) → cut_0 = 2
    # similarly cut_1 = 2, cut_2 = 2.
    # vol_k = deg(k) = 2 for each.
    # Ncut = 3 * (2/2) = 3.
    g = nx.complete_graph(3)
    p = qqa.NormalizedCut(g, num_category=3, device="cpu", eps=0.0)
    x = _onehot([0, 1, 2], K=3)
    ncut = p.discrete_ncut(x).item()
    assert ncut == pytest.approx(3.0, abs=1e-6)


def test_normcut_score_summary_picks_best():
    # cycle_graph(8): bisection has Ncut=2 (cut=2, vol_each=8 -> 2/8 + 2/8 = 0.5)
    # vs. an unbalanced split where Ncut is *smaller still* due to the well-known
    # Ncut degeneracy on heavy-volume partitions. We just verify that
    # ``score_summary`` returns the argmin candidate and that all required
    # fields are populated; the actual minimum value depends on which split
    # happens to be the most extreme.
    g = nx.cycle_graph(8)
    p = qqa.NormalizedCut(g, num_category=2, device="cpu")
    x_a = _onehot([0, 0, 0, 0, 1, 1, 1, 1], K=2)[0]  # contiguous bisection
    x_b = _onehot([0, 1, 0, 1, 0, 1, 0, 1], K=2)[0]  # alternating
    x_disc = torch.stack([x_a, x_b], dim=0)
    summary = p.score_summary(x_disc)

    # Compute both Ncuts directly to know which is the argmin.
    ncuts = p.discrete_ncut(x_disc).tolist()
    expected_best = min(ncuts)

    assert summary["label"] == "Ncut"
    assert summary["value"] == pytest.approx(expected_best, rel=1e-5)
    assert summary["feasible"]
    assert "edge_cut" in summary["extra"]
    assert summary["extra"]["num_edges"] == g.number_of_edges()


def test_normcut_invalid_num_category():
    with pytest.raises(ValueError):
        qqa.NormalizedCut(nx.path_graph(3), num_category=1)


def test_normcut_rejects_empty_graph():
    # The DISCS NeurIPS-2023 release includes an empty TRANSFORMER pickle;
    # NormalizedCut must refuse to silently produce a 0-node Q matrix.
    with pytest.raises(ValueError, match="empty"):
        qqa.NormalizedCut(nx.empty_graph(0), num_category=2)


def test_normcut_rejects_no_edges():
    g = nx.empty_graph(5)  # 5 isolated vertices
    with pytest.raises(ValueError, match="at least one edge"):
        qqa.NormalizedCut(g, num_category=2)


def test_normcut_rejects_too_few_nodes():
    g = nx.complete_graph(2)
    with pytest.raises(ValueError, match=">= 3 nodes"):
        qqa.NormalizedCut(g, num_category=3)


def test_normcut_alias():
    # NormCut is the legacy alias matching DISCS' class name.
    assert qqa.NormCut is qqa.NormalizedCut


def test_normcut_anneal_smoke_runs():
    # End-to-end smoke: qqa.anneal accepts the new problem.
    g = nx.cycle_graph(8)
    p = qqa.NormalizedCut(g, num_category=2, device="cpu")
    out = qqa.anneal(
        p,
        sol_size=4,
        num_epochs=50,
        learning_rate=0.05,
        check_interval=10,
        verbose=False,
    )
    assert hasattr(out, "best_sol")
    assert torch.isfinite(torch.tensor(out.best_obj)).item()
