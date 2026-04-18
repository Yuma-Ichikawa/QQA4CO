"""Unit tests for the spin-glass / physics problem classes."""

from __future__ import annotations

import numpy as np
import torch

import qqa


def test_spin_relaxation_roundtrip():
    relax = qqa.SpinRelaxation()
    x = torch.tensor([[0.0, 0.3, 0.7, 1.0]])
    s = relax.forward(x)
    assert torch.allclose(s, torch.tensor([[-1.0, -0.4, 0.4, 1.0]]))
    s_proj = relax.project(x)
    assert torch.allclose(s_proj, torch.tensor([[-1.0, -1.0, 1.0, 1.0]]))


def test_ising1d_ferromagnetic_ground_state():
    # FM chain, all spins aligned -> E = -J*N (PBC, periodic=True).
    problem = qqa.Ising1D(N=6, J=1.0, h=0.0, periodic=True)
    s_up = torch.ones((1, 6))
    assert float(problem.loss_fn(s_up)) == -6.0
    s_mixed = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
    assert float(problem.loss_fn(s_mixed)) == +6.0


def test_ising1d_anneal_finds_ground_state():
    qqa.fix_seed(0)
    problem = qqa.Ising1D(N=10, J=1.0, h=0.0, periodic=True)
    result = qqa.anneal(problem, sol_size=32, num_epochs=400, verbose=False, record_history=False)
    assert result.best_obj <= -10 + 1e-6


def test_edwards_anderson_small_consistency():
    problem = qqa.EdwardsAnderson(L=3, dim=2, seed=0, periodic=False)
    # N = 9, bonds = 2 * L * (L - 1) for open 2D -> 12 bonds.
    J = problem.J.cpu().numpy()
    assert J.shape == (9, 9)
    assert np.allclose(J, J.T)
    assert np.allclose(np.diag(J), 0.0)
    num_bonds = int((np.count_nonzero(J)) // 2)
    assert num_bonds == 2 * 3 * (3 - 1)


def test_sk_energy_negative_in_population():
    qqa.fix_seed(0)
    problem = qqa.SherringtonKirkpatrick(N=40, seed=0)
    s = torch.where(torch.rand((256, 40)) > 0.5, 1.0, -1.0)
    e = problem.loss_fn(s)
    # Some samples should produce negative energy.
    assert (e < 0).any()


def test_binary_perceptron_teacher_is_optimum():
    problem = qqa.BinaryPerceptron(N=30, alpha=0.5, seed=1, sharpness=10.0)
    teacher = problem.teacher.unsqueeze(0)
    errors = problem.error_count(teacher)
    assert int(errors) == 0


def test_hopfield_stored_pattern_recovery():
    problem = qqa.HopfieldMemory(N=40, patterns=1, seed=0)
    xi = problem.patterns[0:1]
    # Energy at stored pattern should be approx -N/2.
    e = float(problem.loss_fn(xi))
    assert e < -18.0  # loose bound, exact value ≈ -19.5 for N=40, P=1

    # Overlap with itself is 1.
    m = problem.overlap(xi)
    assert abs(float(m[0, 0]) - 1.0) < 1e-6


def test_ea_from_couplings_txt(tmp_path):
    # Create a tiny 2x2x2 random coupling file.
    rng = np.random.default_rng(0)
    lines = []
    # Manually build 2D 3x3 lattice with open BC.
    N = 9
    for i in range(3):
        for j in range(3):
            u = i * 3 + j
            if j + 1 < 3:
                v = i * 3 + (j + 1)
                lines.append(f"{u} {v} {rng.normal():.6f}")
            if i + 1 < 3:
                v = (i + 1) * 3 + j
                lines.append(f"{u} {v} {rng.normal():.6f}")
    path = tmp_path / "coup.txt"
    path.write_text("\n".join(lines))

    ea = qqa.EdwardsAnderson.from_couplings_txt(path, N=N)
    J = ea.J.cpu().numpy()
    assert J.shape == (N, N)
    assert np.allclose(J, J.T)
    assert np.allclose(np.diag(J), 0.0)
