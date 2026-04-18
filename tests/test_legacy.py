"""Verify the legacy wrappers still produce valid results and raise warnings."""

from __future__ import annotations

import warnings

import networkx as nx

import qqa
from qqa import legacy


def test_batch_annealing_compat():
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=16, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2, device="cpu")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        best_sol, best_obj, runtime = legacy.batch_annealing(
            problem,
            sol_size=8,
            num_epochs=50,
            check_interval=1000,
            device="cpu",
            min_bg=-2,
            max_bg=0.1,
        )
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    assert best_obj <= 0.0
    assert runtime > 0.0
    assert best_sol.shape[-1] == 16


def test_batch_instance_annealing_compat():
    qqa.fix_seed(0)
    graphs = [nx.random_regular_graph(d=3, n=8, seed=i) for i in range(2)]
    problem = qqa.MaximumIndependentSetInstance(graphs, max_node=8, penalty=2, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_sol, best_obj, runtime = legacy.batch_instance_annealing(
            problem,
            sol_size=8,
            num_epochs=50,
            check_interval=1000,
            device="cpu",
        )
    assert best_obj.shape == (2,)
    assert runtime > 0.0


def test_batch_annealing_categorical_compat():
    qqa.fix_seed(0)
    g = nx.cycle_graph(6)
    problem = qqa.Coloring(g, num_category=2, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_string, best_loss, runtime = legacy.batch_annealing_categorical(
            problem,
            sol_size=8,
            num_epochs=50,
            check_interval=1000,
            device="cpu",
        )
    assert best_string.shape == (6,)
    assert runtime > 0.0


def test_batch_annealing_mis_trajectory_compat():
    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=10, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2, device="cpu")
    problem_p1 = qqa.MaximumIndependentSet(g, penalty=1, device="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_obj, runtime, dyn = legacy.batch_annealing_mis_trajectory(
            problem,
            problem_p1,
            sol_size=4,
            num_epochs=30,
            check_interval=1000,
            device="cpu",
        )
    assert runtime > 0.0
    assert "MIS_DYNAMICS" in dyn
    assert len(dyn["MIS_DYNAMICS"]) == 30
