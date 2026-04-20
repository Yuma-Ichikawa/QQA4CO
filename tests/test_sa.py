"""Smoke + correctness tests for the SA backend.

The QUBO fast path and the generic single-spin Metropolis path are exercised
on small problems where we can sanity-check the answer against the known
optimum.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import qqa


def test_simulated_annealing_is_exported():
    """SA entry point and result type live on the top-level qqa namespace."""
    assert hasattr(qqa, "simulated_annealing")
    assert hasattr(qqa, "SAResult")
    assert "simulated_annealing" in qqa.__all__
    assert "SAResult" in qqa.__all__


def test_sa_qubo_fast_path_finds_mis_on_path_graph():
    """A 6-node path has MIS size = 3 ({0,2,4} or {1,3,5}).

    The QUBO fast path should converge there in well under 200 sweeps with a
    modest batch — this is the smallest non-trivial sanity check.
    """
    qqa.fix_seed(0)
    g = nx.path_graph(6)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)

    result = qqa.simulated_annealing(
        problem,
        sol_size=64,
        num_sweeps=200,
        beta_start=0.1,
        beta_end=20.0,
        seed=0,
        device="cpu",
        verbose=False,
    )

    # MIS minimises -|S| + penalty * (#violated edges); for the path graph
    # the optimum independent set has size 3 → best_obj == -3.0 with no
    # violation. Allow tiny numerical slack.
    assert result.best_obj <= -3.0 + 1e-6, f"got best_obj={result.best_obj}"
    assert result.best_sol.shape == (6,)
    assert result.runtime > 0.0
    assert "loss_mean" in result.history
    assert len(result.history["loss_mean"]) == 200


def test_sa_qubo_fast_path_maxcut_complete_graph():
    """K_4 has Max-Cut = 4 (cut both pairs of opposite vertices)."""
    qqa.fix_seed(0)
    g = nx.complete_graph(4)
    problem = qqa.MaxCut(g)

    result = qqa.simulated_annealing(
        problem,
        sol_size=128,
        num_sweeps=300,
        beta_start=0.1,
        beta_end=10.0,
        seed=0,
        device="cpu",
        verbose=False,
    )

    # MaxCut QUBO minimises -cut, optimum = -4.
    assert result.best_obj <= -4.0 + 1e-6, f"got best_obj={result.best_obj}"


def test_sa_generic_path_on_knapsack():
    """Knapsack has no Q_mat → exercises the generic single-spin sweep.

    Ensures the fallback path works end-to-end, even if it is slower. We don't
    assert the global optimum because Knapsack's penalty surface makes random
    SA on a tiny budget noisy; we only check the loop ran and produced a
    valid {0,1} solution.
    """
    qqa.fix_seed(0)
    problem = qqa.Knapsack(N=10, capacity_ratio=0.5, seed=0)

    result = qqa.simulated_annealing(
        problem,
        sol_size=16,
        num_sweeps=20,
        beta_start=0.1,
        beta_end=10.0,
        seed=0,
        device="cpu",
        verbose=False,
    )

    assert result.best_sol.shape == (10,)
    # Discrete projection lives in {0, 1}.
    assert torch.all((result.best_sol == 0.0) | (result.best_sol == 1.0))
    assert isinstance(result.best_obj, float)


def test_sa_generic_path_runs_on_spin_problem():
    """SpinProblem (SK) goes through the single-spin path with sign flip.

    We don't assert optimality (small SK ground states are noisy), only that
    the loop completes and returns a sensible structure.
    """
    qqa.fix_seed(0)
    problem = qqa.SherringtonKirkpatrick(N=8, seed=0)

    result = qqa.simulated_annealing(
        problem,
        sol_size=16,
        num_sweeps=30,
        beta_start=0.1,
        beta_end=5.0,
        seed=0,
        device="cpu",
        verbose=False,
    )

    assert result.best_sol.shape == (8,)
    # Spins must be in {-1, +1}.
    assert torch.all((result.best_sol == 1.0) | (result.best_sol == -1.0))


def test_sa_zero_sweeps_returns_initial_eval():
    """num_sweeps=0 should still return a valid result with the random init."""
    qqa.fix_seed(0)
    g = nx.path_graph(4)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)

    result = qqa.simulated_annealing(
        problem,
        sol_size=8,
        num_sweeps=0,
        seed=0,
        device="cpu",
        verbose=False,
    )

    assert result.best_sol.shape == (4,)
    assert isinstance(result.best_obj, float)
    assert result.history["loss_mean"] == []


def test_sa_rejects_categorical_problem():
    """Coloring uses CategoricalRelaxation; SA isn't implemented for that yet."""
    g = nx.cycle_graph(5)
    problem = qqa.Coloring(g, num_category=3)
    with pytest.raises(NotImplementedError):
        qqa.simulated_annealing(problem, sol_size=2, num_sweeps=1, verbose=False)


def test_sa_validates_initial_state_shape():
    """Catching wrong-shape initial_state gives a clear error, not a deep crash."""
    g = nx.path_graph(5)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)
    with pytest.raises(ValueError):
        qqa.simulated_annealing(
            problem,
            sol_size=4,
            num_sweeps=1,
            initial_state=torch.zeros(4, 7),  # wrong N
            verbose=False,
        )


def test_sa_seed_reproducibility():
    """Same seed → identical best_obj on the QUBO fast path."""
    g = nx.path_graph(8)
    problem = qqa.MaximumIndependentSet(g, penalty=2.0)
    a = qqa.simulated_annealing(
        problem, sol_size=16, num_sweeps=50, seed=7, device="cpu", verbose=False
    )
    b = qqa.simulated_annealing(
        problem, sol_size=16, num_sweeps=50, seed=7, device="cpu", verbose=False
    )
    assert a.best_obj == b.best_obj


def test_sa_cli_smoke(tmp_path):
    """End-to-end CLI test: ``qqa solve --backend sa --problem mis``."""
    import shutil
    import subprocess

    qqa_bin = shutil.which("qqa")
    if qqa_bin is None:
        pytest.skip("qqa script not on PATH (install via `uv sync` or `pip install -e .`).")

    out = tmp_path / "sa_result.pkl"
    cmd = [
        qqa_bin,
        "solve",
        "--problem",
        "mis",
        "--size",
        "20",
        "--backend",
        "sa",
        "--epochs",
        "100",
        "--sol-size",
        "16",
        "--sa-beta-end",
        "5.0",
        "--seed",
        "0",
        "--quiet",
        "--output",
        str(out),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert "backend    : sa" in result.stdout


# ---------------------------------------------------------------------------
# Sampler-correctness regressions for the QUBO fast path.
#
# Earlier the QUBO fast path used a fully-parallel Metropolis sweep that
# proposed every bit independently against the same pre-sweep state. On a
# 3-regular MIS this oscillates deterministically between 0^N and 1^N at any
# temperature (see diagnose_pa_mis.py); the fix is the sequential single-bit
# sweep _qubo_seq_glauber_sweep which respects the Markov property bit by
# bit. These tests pin both the bug and the fix so a future "let's just
# parallelise it" refactor can't silently regress.
# ---------------------------------------------------------------------------


def test_qubo_parallel_metropolis_oscillates_on_3regular_mis():
    """Pin the documented failure mode of the legacy parallel sweep.

    On a regular-graph MIS every bit independently sees a favourable
    single-flip ΔE in the all-zeros and all-ones states, so a fully
    parallel proposal flips them all and bounces between the two
    extremes. We pin this behaviour so the deprecated routine cannot be
    re-introduced as a default by accident.
    """
    from qqa.sa import _qubo_parallel_metropolis_sweep  # noqa: PLC0415

    g = nx.random_regular_graph(3, 32, seed=0)
    problem = qqa.MaximumIndependentSet(g, device="cpu")
    q_sym = 0.5 * (problem.Q_mat + problem.Q_mat.t())
    q_diag = q_sym.diagonal().contiguous()
    rng = torch.Generator(device="cpu").manual_seed(0)
    x = torch.zeros((1, 32), dtype=torch.float32)

    sizes: list[int] = []
    for _ in range(8):
        x = _qubo_parallel_metropolis_sweep(x, q_sym, q_diag, beta=5.0, rng=rng)
        sizes.append(int(x.sum().item()))

    # Deterministic oscillation: 32, 0, 32, 0, ...
    assert sizes == [32, 0, 32, 0, 32, 0, 32, 0], (
        f"Parallel sweep is supposed to mode-lock on regular MIS; got {sizes}. "
        "If this test breaks, somebody fixed the buggy parallel sampler — "
        "great, but please update the docstring of _qubo_parallel_metropolis_sweep "
        "and remove the warning."
    )


def test_qubo_seq_glauber_sweep_solves_3regular_mis():
    """The current (sequential) sampler must reach the MIS optimum.

    Counterpart to the oscillation pin above. With the same graph and
    enough sweeps at moderate β, the sequential single-bit sampler must
    find a feasible IS with size in the textbook range (≥10 on 3-regular
    N=32). Catches any future refactor that accidentally re-introduces
    parallel updates inside the QUBO fast path.
    """
    from qqa.sa import _qubo_seq_glauber_sweep  # noqa: PLC0415

    g = nx.random_regular_graph(3, 32, seed=0)
    problem = qqa.MaximumIndependentSet(g, device="cpu")
    q_sym = 0.5 * (problem.Q_mat + problem.Q_mat.t())
    q_diag = q_sym.diagonal().contiguous()
    rng = torch.Generator(device="cpu").manual_seed(0)
    x = torch.zeros((4, 32), dtype=torch.float32)

    best = float("inf")
    for _ in range(50):
        x = _qubo_seq_glauber_sweep(x, q_sym, q_diag, beta=4.0, rng=rng)
        best = min(best, float(problem.loss_fn(x).min().item()))

    # On 3-regular N=32 a feasible IS of size 10 has loss = -10. Anything
    # weaker than -8 means we have either heavy infeasibility or are still
    # bouncing — both indicate the sampler regressed.
    assert best <= -10.0, (
        f"Sequential QUBO sweep should reach loss ≤ -10 on 3-regular MIS; got {best}."
    )


def test_sa_default_backend_solves_3regular_mis():
    """End-to-end SA must deliver a near-optimal MIS on 3-regular N=32.

    Under the buggy parallel sweep this returned ``best_obj ≥ -5``; the
    sequential fix routinely reaches ``-12`` or better. We pin a loose
    threshold of ``-10`` so the test stays stable across CI runners while
    still catching the historical regression.
    """
    g = nx.random_regular_graph(3, 32, seed=0)
    problem = qqa.MaximumIndependentSet(g, device="cpu")
    res = qqa.simulated_annealing(
        problem,
        sol_size=64,
        num_sweeps=500,
        beta_start=0.1,
        beta_end=8.0,
        beta_schedule="geometric",
        seed=0,
        verbose=False,
    )
    assert float(res.best_obj) <= -10.0, (
        f"SA on 3-regular MIS should reach best_obj ≤ -10; got {res.best_obj}. "
        "If this regresses look for accidental reintroduction of parallel "
        "single-bit proposals in the QUBO fast path (see lessons L29)."
    )
