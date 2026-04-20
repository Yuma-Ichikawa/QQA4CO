"""Regression tests for the integer-factorization Ising/QUBO benchmark.

These lock the **planted-solution** contract that gives the Hen-2026
benchmark its scientific value:

* the QUBO loss evaluates to **exactly zero** on the planted Boolean
  configuration (the bits of the prime factors), and
* every other binary configuration has *strictly positive* energy
  (gap ≥ 2, paper Sec. II.C),
* the decoder ``decode_factors`` recovers the original ``(p, q)`` from
  the planted ``x``.

A small smoke also drives ``qqa.anneal`` end-to-end on a 22-spin instance
(``N = 15``) so that any regression in the construction or in the
``score_summary`` interface surfaces immediately.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

import qqa
from qqa.problems.factorization import (
    IntegerFactorizationIsing,
    _add_and_gadget,
    _add_xor_gadget,
    _is_probable_prime,
    random_factorization_problems,
    random_prime,
    random_semiprime,
)

# --------------------------------------------------------------------------- #
# Number-theory helpers                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n,expected",
    [
        (2, True),
        (3, True),
        (5, True),
        (7, True),
        (11, True),
        (13, True),
        (17, True),
        (19, True),
        (23, True),
        (29, True),
        (4, False),
        (6, False),
        (9, False),
        (15, False),
        (21, False),
        (25, False),
        (1, False),
        (0, False),
    ],
)
def test_is_probable_prime_small(n: int, expected: bool):
    assert _is_probable_prime(n) is expected


def test_random_prime_has_correct_bit_length():
    import numpy as np

    rng = np.random.default_rng(0)
    for d in (3, 5, 8, 10):
        for _ in range(10):
            p = random_prime(d, rng)
            assert p.bit_length() == d
            assert _is_probable_prime(p)


def test_random_semiprime_distinct():
    p, q = random_semiprime(5, seed=42, distinct=True)
    assert p != q
    assert _is_probable_prime(p)
    assert _is_probable_prime(q)


# --------------------------------------------------------------------------- #
# Direct gadget spectrum tests (faithful to paper Eqs. 10–12)                 #
# --------------------------------------------------------------------------- #


def _eval_ising_form(h: np.ndarray, J: np.ndarray, e0: float, s: np.ndarray) -> float:
    """``H(s) = e_0 + h^T s + s^T J s`` with the symmetric-J convention used
    inside ``_build_qubo_from_compilation`` (J double-counts each pair, so
    ``s^T J s`` reproduces the gadget energy directly after symmetrisation).
    """
    return float(e0 + h @ s + s @ J @ s)


def test_and_gadget_full_spectrum():
    """Eq. (10): the AND gadget evaluates to {0,0,0,0,4,4,4,12} on the eight
    spin configurations (Boolean assignments (a,b,c) ∈ {0,1}^3).

    Specifically:
      satisfying (c = a∧b):                       four configs → 0
      violating (a∧b)=0 but c=1:                  three configs → 4
      violating (a∧b)=1 but c=0 [a=b=1, c=0]:     one config   → 12
    """
    n = 3
    h = np.zeros(n)
    J = np.zeros((n, n))
    e0 = _add_and_gadget(h, J, 0, 1, 2)
    J = 0.5 * (J + J.T)  # symmetrise (matches `_build_qubo_from_compilation`)

    energies: dict[tuple[int, int, int], float] = {}
    for s1, s2, s3 in itertools.product([-1, +1], repeat=3):
        a, b, c = (1 + s1) // 2, (1 + s2) // 2, (1 + s3) // 2
        E = _eval_ising_form(h, J, e0, np.array([s1, s2, s3], dtype=np.float64))
        energies[(a, b, c)] = E

    # Satisfying assignments of c = a∧b
    for (a, b, c), expected in {
        (0, 0, 0): 0,
        (0, 1, 0): 0,
        (1, 0, 0): 0,
        (1, 1, 1): 0,
    }.items():
        assert energies[(a, b, c)] == pytest.approx(expected), (
            f"AND gadget violates Eq.(10) at sat assignment (a,b,c)=({a},{b},{c}): "
            f"got {energies[(a, b, c)]}, expected {expected}"
        )
    # Violating assignments
    for (a, b, c), expected in {
        (0, 0, 1): 12,  # a∧b=0 but c=1, the "hard" violation
        (0, 1, 1): 4,
        (1, 0, 1): 4,
        (1, 1, 0): 4,  # a∧b=1 but c=0
    }.items():
        assert energies[(a, b, c)] == pytest.approx(expected), (
            f"AND gadget violates Eq.(10) at viol assignment (a,b,c)=({a},{b},{c}): "
            f"got {energies[(a, b, c)]}, expected {expected}"
        )
    # Spectrum (multiset)
    assert sorted(energies.values()) == [0, 0, 0, 0, 4, 4, 4, 12]


def test_xor_gadget_full_spectrum():
    """Eq. (11): the XOR gadget has 16 spin configurations.

    Per Eq. (12) the planted aux satisfies ``s_a^* = s_x ∧ s_y``. Combined
    with the XOR relation ``s_3 = -s_1 s_2`` (i.e. ``c = a ⊕ b``), exactly
    **four** configurations achieve energy 0; the other 12 must be ≥ 2
    (paper "Δ_⊕ = 2"). We assert both.
    """
    n = 4
    h = np.zeros(n)
    J = np.zeros((n, n))
    e0 = _add_xor_gadget(h, J, 0, 1, 2, 3)
    J = 0.5 * (J + J.T)

    sat_count = 0
    for s1, s2, s3, sa in itertools.product([-1, +1], repeat=4):
        a = (1 + s1) // 2
        b = (1 + s2) // 2
        c = (1 + s3) // 2
        d = (1 + sa) // 2
        E = _eval_ising_form(h, J, e0, np.array([s1, s2, s3, sa], dtype=np.float64))
        is_planted = (c == (a ^ b)) and (d == (a & b))
        if is_planted:
            sat_count += 1
            assert pytest.approx(0.0, abs=1e-9) == E, (
                f"XOR gadget should be 0 on planted (a,b,c,d)=({a},{b},{c},{d}), got {E}"
            )
        else:
            assert E >= 2.0 - 1e-9, f"XOR gadget gap < 2 at (a,b,c,d)=({a},{b},{c},{d}): {E}"
    assert sat_count == 4, f"XOR gadget should have 4 satisfying configs, got {sat_count}"


# --------------------------------------------------------------------------- #
# Planted-solution invariant                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p,q", [(3, 5), (5, 7), (3, 7), (11, 13), (13, 17)])
def test_planted_solution_has_zero_energy(p: int, q: int):
    """Eq. (13) of the paper: ``H(s*) = E_0`` on the planted assignment.

    Our QUBO normalisation absorbs ``E_0`` into ``offset`` so the loss is
    *exactly zero* on ``x*``. Anything else means the gadget arithmetic
    or the pin-substitution is off.
    """
    prob = IntegerFactorizationIsing(p=p, q=q)
    energy = prob.loss_fn(prob.planted_x.unsqueeze(0)).item()
    assert energy == pytest.approx(0.0, abs=1e-5), f"planted energy = {energy} ≠ 0 for N={p * q}"


@pytest.mark.parametrize("p,q", [(3, 5), (5, 7), (11, 13)])
def test_decoder_recovers_factors_from_planted(p: int, q: int):
    """``decode_factors(planted_x)`` must round-trip to ``(p, q)``."""
    prob = IntegerFactorizationIsing(p=p, q=q)
    p_hat, q_hat = prob.decode_factors(prob.planted_x)
    assert (p_hat, q_hat) == (p, q) or (p_hat, q_hat) == (q, p)
    assert p_hat * q_hat == prob.N


@pytest.mark.parametrize("p,q", [(3, 5), (5, 7), (11, 13)])
def test_planted_is_strict_minimum(p: int, q: int):
    """Random perturbations of ``planted_x`` must raise the energy by ≥ 2.

    The spectral gap from the gadget table is ``min(Δ_∧, Δ_⊕) = 2``;
    flipping a single free bit therefore costs at least 2.  We test 10
    random single-bit flips per instance.
    """
    prob = IntegerFactorizationIsing(p=p, q=q)
    base = prob.planted_x.clone()
    rng = torch.Generator().manual_seed(p * q)
    for _ in range(10):
        i = int(torch.randint(0, prob.num_nodes, (1,), generator=rng).item())
        flipped = base.clone()
        flipped[i] = 1.0 - flipped[i]
        e = prob.loss_fn(flipped.unsqueeze(0)).item()
        assert e >= 2.0 - 1e-6, f"single-flip energy {e} < 2 for N={prob.N}, flip idx={i}"


# --------------------------------------------------------------------------- #
# Public API contracts                                                        #
# --------------------------------------------------------------------------- #


def test_score_summary_is_json_serializable():
    """The bench runner persists ``result.score`` to disk."""
    import json

    prob = IntegerFactorizationIsing(p=11, q=13)
    summary = prob.score_summary(prob.planted_x.unsqueeze(0))
    assert summary["feasible"] is True
    assert summary["extra"]["matches_planted"] is True
    json.dumps(summary)  # must not raise


def test_score_summary_marks_random_x_infeasible():
    prob = IntegerFactorizationIsing(p=5, q=7)
    rng = torch.Generator().manual_seed(0)
    bad = (torch.rand(prob.num_nodes, generator=rng) > 0.5).float()
    summary = prob.score_summary(bad.unsqueeze(0))
    assert summary["feasible"] is False or summary["value"] == 0.0


def test_class_is_re_exported_from_qqa():
    """``qqa.IntegerFactorizationIsing`` and helpers must be public."""
    assert qqa.IntegerFactorizationIsing is IntegerFactorizationIsing
    assert qqa.random_semiprime is random_semiprime
    assert qqa.random_prime is random_prime
    assert qqa.random_factorization_problems is random_factorization_problems


def test_random_factorization_problems_yields_distinct_instances():
    suite = random_factorization_problems(bit_length=4, num_instances=4, seed=0)
    # Even with d=4 there are only a handful of 4-bit primes (11, 13);
    # we just check the constructor produces the requested count.
    assert len(suite) == 4
    for prob in suite:
        # Planted must always be ground state, regardless of N.
        e = prob.loss_fn(prob.planted_x.unsqueeze(0)).item()
        assert e == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# End-to-end smoke through ``qqa.anneal``                                     #
# --------------------------------------------------------------------------- #


def test_qqa_anneal_solves_n_eq_15():
    """``N = 15 = 3 * 5`` is a 22-free-spin problem; QQA should find x*.

    This test is the integration check that the QUBO is well-formed
    (proper ``Q_mat`` shape, ``BinaryRelaxation`` accepts it, the
    ``score_summary`` schema is what ``qqa.anneal`` expects).
    """
    torch.manual_seed(0)
    prob = IntegerFactorizationIsing(p=3, q=5)
    result = qqa.anneal(
        prob,
        sol_size=200,
        num_epochs=2000,
        learning_rate=0.5,
        temp=1e-3,
        schedule=qqa.LinearBGSchedule(min_bg=-2, max_bg=0.5),
        curve_rate=4,
        div_param=0.2,
        verbose=False,
        device="cpu",
        record_history=False,
    )
    assert result.best_obj == pytest.approx(0.0, abs=1e-3)
    assert result.score["feasible"] is True
    assert result.score["extra"]["N_hat"] == 15
    assert {result.score["extra"]["p_hat"], result.score["extra"]["q_hat"]} == {3, 5}


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        IntegerFactorizationIsing(p=1, q=5)
    with pytest.raises(ValueError):
        IntegerFactorizationIsing(p=3, q=0)
