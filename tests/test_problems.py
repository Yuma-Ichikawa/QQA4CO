"""Unit tests for QUBO matrix construction and loss_fn correctness."""

from __future__ import annotations

import networkx as nx
import torch

import qqa


def test_mis_qubo_matrix_known_graph():
    # Triangle K_3: only the empty set and single vertices are independent.
    g = nx.complete_graph(3)
    problem = qqa.MaximumIndependentSet(g, penalty=5, device="cpu")
    # Diagonal = -1, off-diagonal = 5 on every pair.
    Q = problem.Q_mat
    assert Q.shape == (3, 3)
    for i in range(3):
        assert Q[i, i].item() == -1
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert Q[i, j].item() == 5
        assert Q[j, i].item() == 5


def test_mis_loss_on_indicator():
    g = nx.path_graph(4)
    problem = qqa.MaximumIndependentSet(g, penalty=3, device="cpu")
    # {0, 2} is independent -> loss = -2.
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    assert problem.loss_fn(x).item() == -2.0
    # {0, 1} violates one edge -> -2 + 3*2 = 4 (symmetric off-diag contributes twice).
    x = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert problem.loss_fn(x).item() == -2 + 3 * 2


def test_maxcut_loss_matches_cut_size():
    g = nx.cycle_graph(4)  # square
    problem = qqa.MaxCut(g, device="cpu")
    # Alternate 0/1 around the cycle cuts all 4 edges.
    x = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    cut_size = 4
    # Loss = x^T Q x = -(# cut edges) for unweighted max-cut.
    assert problem.loss_fn(x).item() == -cut_size


def test_maxclique_qubo_basic():
    g = nx.complete_graph(4)
    problem = qqa.MaxClique(g, penalty=3, device="cpu")
    Q = problem.Q_mat
    # All off-diagonals are edges -> should be 0; diagonal -1.
    for i in range(4):
        assert Q[i, i].item() == -1
        for j in range(4):
            if i != j:
                assert Q[i, j].item() == 0


def test_coloring_loss_zero_on_valid():
    g = nx.cycle_graph(4)
    problem = qqa.Coloring(g, num_category=2, device="cpu")
    # Valid 2-coloring.
    one_hot = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])
    assert problem.loss_fn(one_hot).item() == 0.0


def test_mis_instance_batch_shapes():
    graphs = [nx.random_regular_graph(d=2, n=6, seed=i) for i in range(3)]
    problem = qqa.MaximumIndependentSetInstance(graphs, max_node=6, penalty=2, device="cpu")
    x = torch.rand((4, 3, 6))
    out = problem.loss_fn(x)
    assert out.shape == (4, 3)
