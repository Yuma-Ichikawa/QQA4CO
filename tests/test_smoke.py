"""Fast smoke tests to verify the end-to-end pipeline runs."""

from __future__ import annotations

import networkx as nx
import torch

import qqa


def test_import_surface():
    assert callable(qqa.anneal)
    assert callable(qqa.fix_seed)
    assert hasattr(qqa, "__version__")


def test_mis_small_cpu():
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=20, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2, device="cpu")
    result = qqa.anneal(
        problem,
        sol_size=16,
        num_epochs=100,
        check_interval=1000,
        device="cpu",
        verbose=False,
    )
    # At the very least the best found must be feasible and non-positive.
    assert result.best_obj <= 0.0
    assert result.best_sol.shape[-1] == 20
    assert len(result.history["loss_mean"]) == 100


def test_mis_batch_instance_cpu():
    qqa.fix_seed(0)
    graphs = [nx.random_regular_graph(d=3, n=10, seed=i) for i in range(3)]
    problem = qqa.MaximumIndependentSetInstance(graphs, max_node=10, penalty=2, device="cpu")
    result = qqa.anneal(
        problem,
        sol_size=8,
        num_epochs=100,
        check_interval=1000,
        device="cpu",
        verbose=False,
    )
    assert result.best_obj.shape == (3,)
    # Each instance should have at least one node selected (loss < 0).
    assert (result.best_obj <= 0.0).all()


def test_coloring_cpu():
    qqa.fix_seed(0)
    g = nx.cycle_graph(6)  # 2-colorable
    problem = qqa.Coloring(g, num_category=2, device="cpu")
    result = qqa.anneal(
        problem,
        sol_size=16,
        num_epochs=200,
        check_interval=1000,
        device="cpu",
        verbose=False,
    )
    # A 2-colorable cycle should be solvable to zero conflicts here.
    assert result.best_obj <= 1.0


def test_anneal_result_runtime_positive():
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=10, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2, device="cpu")
    result = qqa.anneal(problem, sol_size=4, num_epochs=20, verbose=False)
    assert result.runtime > 0.0
    assert torch.is_tensor(result.best_sol)
