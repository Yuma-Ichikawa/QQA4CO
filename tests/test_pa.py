"""Smoke + correctness tests for the Population Annealing backend.

Mirrors :mod:`tests.test_sa` so the two MCMC baselines stay symmetric.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import qqa


def test_population_annealing_is_exported():
    """PA entry point and result type live on the top-level qqa namespace."""
    assert hasattr(qqa, "population_annealing")
    assert hasattr(qqa, "PAResult")
    assert "population_annealing" in qqa.__all__
    assert "PAResult" in qqa.__all__


def test_pa_qubo_fast_path_finds_mis_on_path_graph():
    """A 6-node path has MIS size = 3; PA should reach it in a few temps."""
    qqa.fix_seed(0)
    g = nx.path_graph(6)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)

    result = qqa.population_annealing(
        problem,
        sol_size=32,
        num_temps=20,
        sweeps_per_temp=4,
        beta_start=0.1,
        beta_end=20.0,
        seed=0,
        device="cpu",
        verbose=False,
    )
    assert result.best_obj <= -3.0 + 1e-6, f"got best_obj={result.best_obj}"
    assert result.best_sol.shape == (6,)


def test_pa_population_size_preserved_after_resampling():
    """Resampling must keep the population at exactly ``sol_size`` replicas."""
    qqa.fix_seed(1)
    g = nx.erdos_renyi_graph(15, 0.4, seed=1)
    problem = qqa.MaxCut(g)
    R = 24
    res = qqa.population_annealing(
        problem,
        sol_size=R,
        num_temps=10,
        sweeps_per_temp=2,
        beta_start=0.1,
        beta_end=5.0,
        seed=1,
        verbose=False,
    )
    # ESS history has one entry per temperature step.
    assert len(res.history["ess"]) == 10
    for e in res.history["ess"]:
        assert 1.0 - 1e-6 <= e <= R + 1e-6


def test_pa_best_obj_history_monotone():
    """``best_obj`` history must be monotone non-increasing."""
    qqa.fix_seed(2)
    g = nx.erdos_renyi_graph(20, 0.4, seed=2)
    res = qqa.population_annealing(
        qqa.MaxCut(g),
        sol_size=32,
        num_temps=15,
        sweeps_per_temp=2,
        beta_start=0.1,
        beta_end=5.0,
        seed=2,
        verbose=False,
    )
    hist = res.history["best_obj"]
    for i in range(1, len(hist)):
        assert hist[i] <= hist[i - 1] + 1e-6


def test_pa_best_sol_reproduces_best_obj():
    """``loss_fn(best_sol)`` must equal ``best_obj`` to numerical precision."""
    qqa.fix_seed(3)
    g = nx.erdos_renyi_graph(20, 0.4, seed=3)
    prob = qqa.MaxCut(g)
    res = qqa.population_annealing(
        prob,
        sol_size=24,
        num_temps=10,
        sweeps_per_temp=3,
        beta_start=0.1,
        beta_end=5.0,
        seed=3,
        verbose=False,
    )
    recomputed = float(prob.loss_fn(res.best_sol.unsqueeze(0)).item())
    assert abs(recomputed - res.best_obj) < 1e-4


def test_pa_rejects_categorical():
    """Categorical relaxations must be rejected with NotImplementedError."""
    g = nx.erdos_renyi_graph(10, 0.3, seed=0)
    prob = qqa.Coloring(g, num_category=3)
    with pytest.raises(NotImplementedError):
        qqa.population_annealing(prob, sol_size=4, num_temps=3, sweeps_per_temp=1, verbose=False)


def test_pa_rejects_batched_instance():
    """Batched-instance problems are out of scope for the chain backends."""
    gs = [nx.erdos_renyi_graph(10, 0.3, seed=i) for i in range(3)]
    prob = qqa.MaxCutInstance(gs, max_node=10)
    with pytest.raises(NotImplementedError):
        qqa.population_annealing(prob, sol_size=4, num_temps=3, sweeps_per_temp=1, verbose=False)


def test_pa_initial_state_round_trip():
    """``initial_state`` is honoured and shape-validated."""
    qqa.fix_seed(4)
    g = nx.erdos_renyi_graph(12, 0.4, seed=4)
    prob = qqa.MaxCut(g)
    init = (torch.rand(8, 12) > 0.5).float()
    res = qqa.population_annealing(
        prob,
        sol_size=8,
        num_temps=4,
        sweeps_per_temp=2,
        beta_start=0.1,
        beta_end=2.0,
        initial_state=init,
        seed=4,
        verbose=False,
    )
    assert res.best_sol.shape == (12,)

    with pytest.raises(ValueError):
        qqa.population_annealing(
            prob,
            sol_size=8,
            num_temps=2,
            sweeps_per_temp=1,
            initial_state=torch.zeros(7, 12),
            verbose=False,
        )


def test_pa_resample_modes_both_run():
    """Both systematic and multinomial resampling produce sane results."""
    qqa.fix_seed(5)
    g = nx.erdos_renyi_graph(15, 0.4, seed=5)
    for mode in ("systematic", "multinomial"):
        res = qqa.population_annealing(
            qqa.MaxCut(g),
            sol_size=16,
            num_temps=8,
            sweeps_per_temp=2,
            beta_start=0.1,
            beta_end=4.0,
            resample=mode,
            seed=5,
            verbose=False,
        )
        assert res.best_sol.shape == (15,)
        assert res.best_obj < 0  # MaxCut loss is -|cut|


def test_pa_validates_arguments():
    """API guards must fire on bad arguments."""
    g = nx.path_graph(5)
    prob = qqa.MaxCut(g)

    with pytest.raises(ValueError):
        qqa.population_annealing(prob, sol_size=0, num_temps=2, verbose=False)
    with pytest.raises(ValueError):
        qqa.population_annealing(prob, sol_size=4, num_temps=0, verbose=False)
    with pytest.raises(ValueError):
        qqa.population_annealing(prob, sol_size=4, num_temps=2, sweeps_per_temp=-1, verbose=False)
    with pytest.raises(ValueError):
        qqa.population_annealing(
            prob, sol_size=4, num_temps=2, resample="not-a-mode", verbose=False
        )
    with pytest.raises(ValueError):
        qqa.population_annealing(
            prob,
            sol_size=4,
            num_temps=2,
            beta_schedule="exponential",  # invalid
            verbose=False,
        )
