"""Deep correctness tests: verify QQA actually solves each problem.

Each test instantiates a tiny version of a problem whose optimum (or a tight
lower bound on the optimum) is known analytically, runs :func:`qqa.anneal`,
and asserts that QQA finds it (or gets close enough) within a few hundred
epochs. They are deliberately small so the whole file runs in a few seconds
on a laptop CPU.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest
import torch

import qqa


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    """Seed every RNG before each test so QQA's stochastic loop produces the
    same result across Python invocations and (CPU) machines. Without this,
    tight thresholds such as ``E/N < -0.4`` for SK can flake on some seeds."""
    qqa.fix_seed(0)


# ---------------------------------------------------------------------------
# Combinatorial (graph-based) problems
# ---------------------------------------------------------------------------


def test_mis_on_path_graph_finds_optimum():
    """Independence number of P_n is ceil(n / 2). loss_fn = -|IS| at optimum."""
    n = 10
    g = nx.path_graph(n)
    problem = qqa.MaximumIndependentSet(g, penalty=2)
    result = qqa.anneal(problem, sol_size=64, num_epochs=500, verbose=False)
    assert result.best_obj <= -((n + 1) // 2) + 1e-6


def test_maxcut_on_k4_finds_optimum():
    """Max-cut of K_4 is 4. loss_fn = -cut at optimum."""
    g = nx.complete_graph(4)
    problem = qqa.MaxCut(g)
    result = qqa.anneal(problem, sol_size=64, num_epochs=400, verbose=False)
    assert result.best_obj <= -4 + 1e-6


def test_maxclique_finds_embedded_clique():
    """MaxClique of a graph with a planted K_4 ⇒ clique size >= 4."""
    g = nx.Graph()
    g.add_edges_from(itertools.combinations([0, 1, 2, 3], 2))
    g.add_edges_from([(4, 5), (5, 6), (4, 6)])
    problem = qqa.MaxClique(g, penalty=2)
    result = qqa.anneal(problem, sol_size=64, num_epochs=500, verbose=False)
    assert result.best_obj <= -4 + 1e-6


def test_coloring_on_bipartite_finds_zero_conflicts():
    """A bipartite graph with K=2 colours admits a proper colouring (loss 0)."""
    g = nx.complete_bipartite_graph(3, 3)
    problem = qqa.Coloring(g, num_category=2)
    result = qqa.anneal(problem, sol_size=64, num_epochs=400, verbose=False)
    assert result.best_obj <= 1e-5


# ---------------------------------------------------------------------------
# Spin problems
# ---------------------------------------------------------------------------


def test_ising1d_ferromagnetic_ground_state():
    """With J=1 and periodic boundaries, E_0 = -N ⇒ best_obj ≈ -N."""
    N = 12
    problem = qqa.Ising1D(N=N, J=1.0, h=0.0, periodic=True)
    result = qqa.anneal(problem, sol_size=64, num_epochs=400, verbose=False)
    assert result.best_obj <= -N + 1e-4


def test_ea_small_lattice_matches_exact_ground_state():
    """For a 3x3 EA lattice (N=9), enumerate all 2^9 configurations and check
    QQA finds the exact ground-state energy."""
    problem = qqa.EdwardsAnderson(L=3, dim=2, seed=1, periodic=False)
    N = problem.num_spins
    best_E = float("inf")
    J = problem.J
    for bits in itertools.product((-1.0, 1.0), repeat=N):
        s = torch.tensor(bits).unsqueeze(0)
        E = -0.5 * torch.einsum("bi,ij,bj->b", s, J, s).item()
        best_E = min(best_E, E)

    result = qqa.anneal(problem, sol_size=128, num_epochs=800, verbose=False)
    qqa_E = result.best_obj
    assert qqa_E <= best_E + 1e-4, (
        f"QQA energy {qqa_E:.6f} above brute-force ground state {best_E:.6f}"
    )


def test_sk_energy_is_substantially_negative():
    """On SK, QQA should reach E/N < -0.4 (true ≈ -0.7632)."""
    N = 60
    problem = qqa.SherringtonKirkpatrick(N=N, seed=0)
    result = qqa.anneal(problem, sol_size=128, num_epochs=800, verbose=False)
    e_per_spin = result.best_obj / N
    assert e_per_spin < -0.4, f"SK energy density {e_per_spin:.4f} not deeply negative enough."


def test_binary_perceptron_reaches_zero_errors():
    """Teacher is always a feasible student — QQA should find it."""
    N, alpha = 24, 0.4
    problem = qqa.BinaryPerceptron(N=N, alpha=alpha, seed=0, sharpness=8.0)
    result = qqa.anneal(problem, sol_size=128, num_epochs=800, verbose=False)
    # ``best_sol`` is the single winning replica of shape ``(N,)``; add a
    # leading batch dim so ``error_count`` (which expects ``(B, N)``) works.
    best_sol = problem.relaxation.project(result.best_sol.detach().cpu()).unsqueeze(0)
    errors = int(problem.error_count(best_sol).min().item())
    assert errors == 0, f"Perceptron left {errors} errors; expected 0."


def test_hopfield_recovers_stored_pattern():
    """Single stored pattern ⇒ ground state has overlap |m| ≈ 1."""
    N = 32
    problem = qqa.HopfieldMemory(N=N, patterns=1, seed=0)
    result = qqa.anneal(problem, sol_size=64, num_epochs=500, verbose=False)
    s_best = problem.relaxation.project(result.best_sol.detach().cpu()).unsqueeze(0)
    overlaps = problem.overlap(s_best.to(problem.J.device)).abs()
    m = overlaps.max().item()
    assert m >= 0.95, f"Max overlap {m:.3f} below stored-pattern threshold."


# ---------------------------------------------------------------------------
# UserProblem
# ---------------------------------------------------------------------------


def test_user_problem_matches_sk():
    """Plugging the SK loss via UserProblem reproduces the built-in class."""
    N = 30
    ref = qqa.SherringtonKirkpatrick(N=N, seed=7)
    J = ref.J.clone()

    user = qqa.UserProblem(
        num_vars=N,
        variable_kind="spin",
        loss_fn=lambda s: -0.5 * torch.einsum("bi,ij,bj->b", s, J, s),
    )
    r1 = qqa.anneal(ref, sol_size=64, num_epochs=500, verbose=False)
    r2 = qqa.anneal(user, sol_size=64, num_epochs=500, verbose=False)
    # Same energy function ⇒ both runs should reach comparable ground states
    # (both are minimisations; best_obj is negative for SK-like glass).
    assert abs(r1.best_obj - r2.best_obj) < 0.2 * abs(r1.best_obj) + 1.0


def test_user_problem_from_source_runs():
    src = """
import torch

N = 16
J = torch.eye(N) * 0.0
J = J + torch.ones(N, N) * 0.1
J.fill_diagonal_(0.0)

def loss_fn(s):
    return -0.5 * torch.einsum("bi,ij,bj->b", s, J, s)
"""
    problem = qqa.user_problem_from_source(src, num_vars=16, variable_kind="spin")
    result = qqa.anneal(problem, sol_size=32, num_epochs=200, verbose=False)
    assert np.isfinite(result.best_obj)


# ---------------------------------------------------------------------------
# PopulationTracker records correctly
# ---------------------------------------------------------------------------


def test_population_tracker_records_snapshots():
    from qqa.callbacks import PopulationTracker

    problem = qqa.Ising1D(N=12)
    tracker = PopulationTracker(stride=10, record_x=True)
    qqa.anneal(problem, sol_size=16, num_epochs=80, callbacks=[tracker], verbose=False)
    assert len(tracker.loss) == len(tracker.epochs) >= 8
    # Every snapshot has sol_size rows.
    assert tracker.loss[0].shape == (16,)
    assert tracker.x[0].shape[0] == 16
