"""Regression tests for the batched ``*Instance`` problem classes.

These lock the contract that the parallel-instance solve path relies on:

* the per-instance pad mask zeros out padded variables in ``loss_fn`` and
  in ``score_summary``,
* ``score_summary`` returns per-instance arrays (length ``num_instance``)
  with feasibility flags,
* heterogeneous-size batches do not corrupt feasible answers,
* the parallel ``qqa.anneal`` path produces a result whose feasibility-aware
  objective matches the sequential single-instance solves.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch

import qqa
from qqa.problems import (
    MaxCliqueInstance,
    MaxCut,
    MaxCutInstance,
    MaximumIndependentSet,
    MaximumIndependentSetInstance,
)

# --------------------------------------------------------------------------- #
# Pad mask: padded variables must be inert in loss_fn and score_summary       #
# --------------------------------------------------------------------------- #


def _line_graph(n: int) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from((i, i + 1) for i in range(n - 1))
    g.add_nodes_from(range(n))
    return g


def test_mis_instance_loss_ignores_padded_variables():
    g_small = _line_graph(3)  # 3 nodes
    g_large = _line_graph(6)  # 6 nodes  -> max_node = 6
    prob = MaximumIndependentSetInstance([g_small, g_large], penalty=2.0)

    # Force every variable to 1, including padded slots of g_small (idx 3..5).
    x_all_ones = torch.ones((1, 2, 6))
    x_real_only = x_all_ones * prob.pad_mask  # (I, N) broadcasts fine

    # Loss must not change when the padded ones are stripped — the pad mask
    # in loss_fn already does that.
    loss_with_pad = prob.loss_fn(x_all_ones)
    loss_without_pad = prob.loss_fn(x_real_only)
    torch.testing.assert_close(loss_with_pad, loss_without_pad)


def test_mis_instance_score_summary_counts_real_only():
    g_small = _line_graph(3)
    g_large = _line_graph(6)
    prob = MaximumIndependentSetInstance([g_small, g_large], penalty=2.0)

    # Both instances: the trivial all-ones discrete solution.
    x_disc = torch.ones((2, 6))
    summary = prob.score_summary(x_disc)

    # Real node counts are 3 and 6.
    np.testing.assert_array_equal(summary["value"], np.array([3, 6]))
    # All-ones violates every edge in a path graph (n-1 violations each).
    np.testing.assert_array_equal(summary["extra"]["violated_edges"], np.array([2, 5]))
    # Hence both are infeasible.
    np.testing.assert_array_equal(summary["feasible"], np.array([False, False]))


def test_mis_instance_alternating_is_feasible():
    """The alternating selection on a path graph is the maximum IS."""
    g_small = _line_graph(3)  # MIS = {0, 2}, size 2
    g_large = _line_graph(6)  # MIS = {0, 2, 4}, size 3
    prob = MaximumIndependentSetInstance([g_small, g_large], penalty=2.0)

    x_disc = torch.zeros((2, 6))
    x_disc[0, [0, 2]] = 1
    x_disc[1, [0, 2, 4]] = 1
    summary = prob.score_summary(x_disc)
    np.testing.assert_array_equal(summary["value"], np.array([2, 3]))
    np.testing.assert_array_equal(summary["feasible"], np.array([True, True]))
    assert summary["extra"]["feasible_count"] == 2


def test_mis_instance_rejects_too_small_max_node():
    g = _line_graph(5)
    with pytest.raises(ValueError):
        MaximumIndependentSetInstance([g], max_node=3)


def test_mis_instance_per_instance_penalty():
    g1, g2 = _line_graph(4), _line_graph(4)
    prob = MaximumIndependentSetInstance([g1, g2], penalty=[1.5, 4.0])
    # Q diag is -1 for real nodes; off-diag for the (0,1) edge of g1 is 1.5,
    # for g2 it is 4.0. Confirm that.
    assert prob.Q_tensor[0, 0, 1].item() == pytest.approx(1.5)
    assert prob.Q_tensor[1, 0, 1].item() == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# MaxClique / MaxCut analogues                                                #
# --------------------------------------------------------------------------- #


def test_maxclique_instance_pad_mask_zeroes_padded():
    g_tri = nx.complete_graph(3)
    g_k4 = nx.complete_graph(4)
    prob = MaxCliqueInstance([g_tri, g_k4], penalty=2.0)

    x = torch.ones((1, 2, 4))  # try to pick everything
    x_real = x * prob.pad_mask
    torch.testing.assert_close(prob.loss_fn(x), prob.loss_fn(x_real))

    # Both K_n are full cliques, so picking all real nodes is feasible.
    x_disc = torch.ones((2, 4))
    summary = prob.score_summary(x_disc)
    np.testing.assert_array_equal(summary["value"], np.array([3, 4]))
    np.testing.assert_array_equal(summary["feasible"], np.array([True, True]))


def test_maxcut_instance_pad_mask_and_score():
    # 4-cycle and 6-cycle. MaxCut on an even cycle is |E|.
    g_c4 = nx.cycle_graph(4)
    g_c6 = nx.cycle_graph(6)
    prob = MaxCutInstance([g_c4, g_c6])

    # Bisection alternating 0/1 reaches optimal cut.
    x_disc = torch.zeros((2, 6))
    x_disc[0, [0, 2]] = 1
    x_disc[1, [0, 2, 4]] = 1
    summary = prob.score_summary(x_disc)
    np.testing.assert_allclose(summary["value"], np.array([4.0, 6.0]))
    np.testing.assert_array_equal(summary["feasible"], np.array([True, True]))


# --------------------------------------------------------------------------- #
# End-to-end smoke: parallel anneal vs sequential single-instance anneal      #
# --------------------------------------------------------------------------- #


def _solve_single(g, *, sol_size, num_epochs):
    prob = MaximumIndependentSet(g, penalty=2.0)
    res = qqa.anneal(
        prob,
        sol_size=sol_size,
        learning_rate=1.0,
        temp=1e-3,
        schedule=qqa.LinearBGSchedule(min_bg=-3, max_bg=0.1),
        curve_rate=4,
        div_param=0.2,
        num_epochs=num_epochs,
        device="cpu",
        verbose=False,
        record_history=False,
    )
    return prob.score_summary(res.best_sol)


def test_parallel_mis_anneal_reaches_feasible_solutions_on_small_graphs():
    qqa.fix_seed(0)
    # Two small ER-ish graphs of different sizes.
    rng = np.random.default_rng(0)
    g1 = nx.erdos_renyi_graph(8, 0.25, seed=int(rng.integers(1e6)))
    g2 = nx.erdos_renyi_graph(12, 0.20, seed=int(rng.integers(1e6)))
    prob_inst = MaximumIndependentSetInstance([g1, g2], penalty=2.0)
    res = qqa.anneal(
        prob_inst,
        sol_size=16,
        learning_rate=1.0,
        temp=1e-3,
        schedule=qqa.LinearBGSchedule(min_bg=-3, max_bg=0.1),
        curve_rate=4,
        div_param=0.2,
        num_epochs=300,
        device="cpu",
        verbose=False,
        record_history=False,
    )
    assert res.best_sol.shape == (2, prob_inst.max_node)
    assert res.best_obj.shape == (2,)

    summary = prob_inst.score_summary(res.best_sol)
    # Both must be feasible IS at this small size with this many epochs.
    assert bool(summary["feasible"][0])
    assert bool(summary["feasible"][1])

    # And |IS| is at least as large as a trivial single-vertex set.
    assert int(summary["value"][0]) >= 1
    assert int(summary["value"][1]) >= 1


def test_parallel_mis_anneal_score_in_result_is_per_instance():
    """qqa.anneal now exports the per-instance score in result.score for
    batched-instance problems."""
    qqa.fix_seed(0)
    g1 = nx.erdos_renyi_graph(6, 0.3, seed=1)
    g2 = nx.erdos_renyi_graph(8, 0.3, seed=2)
    prob = MaximumIndependentSetInstance([g1, g2], penalty=2.0)
    res = qqa.anneal(
        prob,
        sol_size=8,
        learning_rate=1.0,
        temp=1e-3,
        schedule=qqa.LinearBGSchedule(min_bg=-3, max_bg=0.1),
        curve_rate=4,
        div_param=0.2,
        num_epochs=200,
        device="cpu",
        verbose=False,
        record_history=False,
    )
    score = res.score
    assert "value" in score and "feasible" in score
    assert len(score["value"]) == 2
    assert len(score["feasible"]) == 2


# --------------------------------------------------------------------------- #
# MaxCut single vs batched: sequential should never beat the batched solve    #
# by more than a small slack on the same seed and budget                      #
# --------------------------------------------------------------------------- #


def test_maxcut_single_and_instance_objectives_align_on_small_graph():
    qqa.fix_seed(0)
    g = nx.cycle_graph(8)  # optimal cut = 8
    prob_single = MaxCut(g)
    prob_inst = MaxCutInstance([g])
    res_single = qqa.anneal(
        prob_single,
        sol_size=16,
        learning_rate=1.0,
        temp=1e-3,
        schedule=qqa.LinearBGSchedule(min_bg=-3, max_bg=0.1),
        curve_rate=4,
        div_param=0.2,
        num_epochs=400,
        device="cpu",
        verbose=False,
        record_history=False,
    )
    res_inst = qqa.anneal(
        prob_inst,
        sol_size=16,
        learning_rate=1.0,
        temp=1e-3,
        schedule=qqa.LinearBGSchedule(min_bg=-3, max_bg=0.1),
        curve_rate=4,
        div_param=0.2,
        num_epochs=400,
        device="cpu",
        verbose=False,
        record_history=False,
    )
    cut_single = float(prob_single.score_summary(res_single.best_sol)["value"])
    cut_inst = float(prob_inst.score_summary(res_inst.best_sol)["value"][0])
    # Both runs should reach the optimal cut of 8 on an 8-cycle.
    assert cut_single == pytest.approx(8.0)
    assert cut_inst == pytest.approx(8.0)


# --------------------------------------------------------------------------- #
# JSON serializability — bench_discs.py dumps result.score into JSON          #
# --------------------------------------------------------------------------- #


def test_mis_instance_score_summary_json_serializable():
    """Bench runner serialises score['extra'] into JSON; lock that contract."""
    import json

    g1 = nx.cycle_graph(6)
    g2 = nx.cycle_graph(8)
    prob = MaximumIndependentSetInstance([g1, g2], penalty=2.0)
    x_disc = torch.zeros((2, 8))
    x_disc[0, [0, 2, 4]] = 1
    x_disc[1, [0, 2, 4, 6]] = 1
    summary = prob.score_summary(x_disc)
    # value/feasible are ndarray — convert at the call site (the bench does this);
    # extra must be plain python so the whole dict round-trips through json.
    payload = {
        "value": summary["value"].tolist(),
        "feasible": [bool(f) for f in summary["feasible"].tolist()],
        "extra": summary["extra"],
    }
    blob = json.dumps(payload)
    back = json.loads(blob)
    assert back["value"] == [3, 4]
    assert back["feasible"] == [True, True]
    assert back["extra"]["feasible_count"] == 2
    assert back["extra"]["num_instance"] == 2


def test_maxcut_instance_score_extra_json_safe():
    import json

    prob = MaxCutInstance([nx.cycle_graph(4), nx.cycle_graph(6)])
    x_disc = torch.zeros((2, 6))
    x_disc[0, [0, 2]] = 1
    x_disc[1, [0, 2, 4]] = 1
    summary = prob.score_summary(x_disc)
    json.dumps(summary["extra"])  # must not raise


def test_maxclique_instance_score_extra_json_safe():
    import json

    prob = MaxCliqueInstance([nx.complete_graph(3), nx.complete_graph(4)])
    x_disc = torch.ones((2, 4))
    summary = prob.score_summary(x_disc)
    json.dumps(summary["extra"])  # must not raise
