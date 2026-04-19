"""Smoke + correctness tests for the Population Annealing backend.

Mirrors :mod:`tests.test_sa` so the two MCMC baselines stay symmetric.
"""

from __future__ import annotations

import math

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


def test_pa_returns_equilibrium_population_and_free_energy():
    """``final_x`` / ``final_loss`` mirror ``loss_fn``; free-energy fields populated."""
    qqa.fix_seed(6)
    g = nx.erdos_renyi_graph(12, 0.4, seed=6)
    prob = qqa.MaxCut(g)
    R = 16
    res = qqa.population_annealing(
        prob,
        sol_size=R,
        num_temps=8,
        sweeps_per_temp=2,
        beta_start=0.1,
        beta_end=4.0,
        seed=6,
        verbose=False,
    )
    assert res.final_x is not None
    assert res.final_loss is not None
    assert res.final_x.shape == (R, 12)
    assert res.final_loss.shape == (R,)
    # ``final_loss`` must be the loss of ``final_x``.
    recomputed = prob.loss_fn(res.final_x)
    assert torch.allclose(recomputed, res.final_loss, atol=1e-4)
    # All free-energy fields populated.
    assert res.log_z is not None
    assert res.free_energy is not None
    assert res.free_energy_density is not None
    # History has matching free-energy series.
    assert len(res.history["log_z_ratio"]) == 8
    assert len(res.history["log_z"]) == 8
    assert len(res.history["free_energy_density"]) == 8
    # ``log_z`` is the *absolute* ln Z(β_end); per-step ratios sum to
    # ln Z(β_end) − ln Z(0). Anchor: ln Z(0) = N · ln 2.
    log_z_zero = 12 * math.log(2.0)
    cum = sum(res.history["log_z_ratio"])
    assert math.isclose(log_z_zero + cum, res.history["log_z"][-1], rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(res.log_z, res.history["log_z"][-1], rel_tol=1e-6, abs_tol=1e-6)
    # F = -ln Z / β and F density = F / N must agree with the stored fields.
    beta_end = res.history["beta"][-1]
    assert math.isclose(res.free_energy, -res.log_z / beta_end, rel_tol=1e-6)
    assert math.isclose(res.free_energy_density, res.free_energy / 12, rel_tol=1e-6)


def test_pa_free_energy_recovers_one_spin_ising_exactly():
    """Single-spin Ising H = −h s has F(β) = −β⁻¹ ln(2 cosh βh).

    With R = 4096 replicas and a long anneal, the PA estimator should
    match within ~1% — this is the textbook closed-form check that
    catches sign / Δβ / log-space bugs in the reweighting code.
    """
    qqa.fix_seed(7)
    h = 0.7
    prob = qqa.Ising1D(N=1, h=h)
    beta_end = 3.0
    res = qqa.population_annealing(
        prob,
        sol_size=4096,
        num_temps=80,
        sweeps_per_temp=4,
        beta_start=0.05,
        beta_end=beta_end,
        seed=7,
        verbose=False,
    )
    # F(β) = −β⁻¹ ln(2 cosh βh) per spin (and N = 1 here)
    f_exact = -math.log(2.0 * math.cosh(beta_end * h)) / beta_end
    assert abs(res.free_energy_density - f_exact) < 0.02, (
        f"PA free-energy density {res.free_energy_density:.4f} vs exact {f_exact:.4f}"
    )


def test_pa_genealogy_matches_population_size():
    """``record_genealogy=True`` produces consistent parent / ancestor logs."""
    qqa.fix_seed(8)
    g = nx.erdos_renyi_graph(15, 0.4, seed=8)
    prob = qqa.MaxCut(g)
    R = 24
    res = qqa.population_annealing(
        prob,
        sol_size=R,
        num_temps=12,
        sweeps_per_temp=2,
        beta_start=0.1,
        beta_end=5.0,
        record_genealogy=True,
        seed=8,
        verbose=False,
    )
    assert res.genealogy is not None
    parents = res.genealogy["parents"]
    ancestors = res.genealogy["ancestors"]
    # First step (β_start) jumps from β=0 → β_start so it *does* resample,
    # so we should have one parent log per temperature step.
    assert len(parents) == 12
    for p in parents:
        assert p.shape == (R,)
        assert int(p.min().item()) >= 0
        assert int(p.max().item()) < R
    # Ancestor map has R entries, each in [0, R).
    assert ancestors.shape == (R,)
    assert int(ancestors.min().item()) >= 0
    assert int(ancestors.max().item()) < R
    # Number of distinct surviving founders must be <= R (and typically
    # much smaller after enough annealing).
    n_unique = int(torch.unique(ancestors).numel())
    assert 1 <= n_unique <= R


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
