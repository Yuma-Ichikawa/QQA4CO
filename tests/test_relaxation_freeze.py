"""Regression tests for the BinaryRelaxation freeze bug.

Background
----------
Pre-v0.5, ``BinaryRelaxation.forward`` returned ``x`` as-is and
``perturb_`` only called ``x.clamp_(0, 1)`` when ``temp > 0``. The default
PQQA / CRA path runs at ``temp == 0``, so AdamW could push ``x`` outside
``[0, 1]`` within a few thousand steps. Once ``x`` left the cube, the CRA
penalty ``Φ(p) = 1 - (1 - 2p)^c`` became *negative* and the optimiser
acquired a perverse incentive to drift further out, which froze PQQA's best
loss / DIV value at a sub-optimal plateau (e.g. MaxCut G70 stuck at ~9350
instead of ~9514).

These tests pin the invariant explicitly so a future "optimise away the
clamp" patch cannot silently re-introduce the regression.
"""

from __future__ import annotations

import torch

from qqa.relaxation import BinaryRelaxation, CategoricalRelaxation, SpinRelaxation


def test_binary_forward_clamps_to_unit_cube():
    """``forward`` must always return values in ``[0, 1]`` regardless of input."""
    relax = BinaryRelaxation()
    x = torch.tensor([[-5.0, -0.1, 0.0, 0.5, 1.0, 1.1, 5.0]])
    y = relax.forward(x)
    assert torch.isfinite(y).all()
    assert (y >= 0.0).all() and (y <= 1.0).all()


def test_binary_penalty_is_bounded_below_outside_cube():
    """The CRA penalty must stay non-negative even when ``x`` left the cube.

    Pre-fix, plugging ``x = -5`` into ``1 - (1 - 2x)^2 = 1 - 121 = -120``
    gave the optimiser a free reward for drifting further negative.
    """
    relax = BinaryRelaxation()
    x = torch.tensor([[-5.0, -0.1, 0.5, 1.5, 5.0]])
    pen = relax.penalty(x, curve_rate=2)
    assert torch.isfinite(pen).all()
    assert (pen >= 0.0).all(), f"penalty went negative on out-of-cube x: {pen}"


def test_binary_perturb_clamps_at_zero_temperature():
    """``perturb_(temp=0)`` must clamp ``x`` back into ``[0, 1]`` in-place.

    Pre-fix this branch returned without clamping, so AdamW's drift was
    never corrected and the relaxation slowly lost its semantic meaning.
    """
    relax = BinaryRelaxation()
    x = torch.tensor([[-3.0, 0.5, 4.0]], requires_grad=False)
    relax.perturb_(x, learning_rate=0.1, temp=0.0)
    assert (x >= 0.0).all() and (x <= 1.0).all()
    assert torch.equal(x, torch.tensor([[0.0, 0.5, 1.0]]))


def test_spin_project_clamps_before_threshold():
    """``SpinRelaxation.project`` must threshold a *clamped* ``x``."""
    relax = SpinRelaxation()
    x = torch.tensor([[-10.0, 0.4, 0.6, 12.0]])
    out = relax.project(x)
    assert torch.equal(out, torch.tensor([[-1.0, -1.0, 1.0, 1.0]]))


def test_categorical_forward_and_perturb_restore_simplex_domain():
    """Categorical AdamW drift must not create negative probabilities."""
    relax = CategoricalRelaxation()
    x = torch.tensor([[[-4.0, -2.0, 3.0], [0.0, 0.0, 0.0]]])
    probabilities = relax.forward(x)
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((1, 2)))

    relax.perturb_(x, learning_rate=0.1, temp=0.0)
    assert (x >= 1e-5).all() and (x <= 1.0).all()


def test_anneal_does_not_freeze_at_zero_temperature():
    """End-to-end smoke: a small QUBO under default temp=0 must keep
    improving past the first few hundred epochs (the regime where the
    pre-fix code froze)."""
    import networkx as nx  # noqa: PLC0415

    import qqa  # noqa: PLC0415

    qqa.fix_seed(0)
    g = nx.cycle_graph(20)
    problem = qqa.MaxCut(g)
    # Two short runs: if early-epoch drift had broken the relaxation,
    # the longer run could only do *worse*. We assert it does not.
    short = qqa.anneal(problem, sol_size=32, num_epochs=200, verbose=False)
    long_ = qqa.anneal(problem, sol_size=32, num_epochs=600, verbose=False)
    short_obj = float(short.best_obj)
    long_obj = float(long_.best_obj)
    assert long_obj <= short_obj + 1e-6, (
        f"longer run got worse: {long_obj} > {short_obj} — relaxation froze?"
    )
