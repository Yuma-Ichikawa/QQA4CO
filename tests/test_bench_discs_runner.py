"""Regression tests for ``scripts.bench_discs`` helpers.

These lock the *feasibility-aware* objective contract: for penalised QUBO
problems (MIS, MaxClique) the runner must NOT use ``-best_obj`` blindly,
because that would over-count vertices when the best replica violates a
constraint. We therefore wire ``score_summary`` into the runner.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import networkx as nx
import torch

import qqa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
bench_discs = importlib.import_module("bench_discs")


class _FakeAnnealResult:
    def __init__(self, best_sol: torch.Tensor, best_obj: float):
        self.best_sol = best_sol
        self.best_obj = best_obj


def test_objective_and_feasibility_mis_clean_solution():
    # Path 0-1-2-3: max IS = 2 ({0,2} or {1,3}). The QQA loss for that
    # IS = -2 (no penalty). Runner should report value=2 and feasible=True.
    g = nx.path_graph(4)
    p = qqa.MaximumIndependentSet(g, penalty=3.0)
    x = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    result = _FakeAnnealResult(best_sol=x, best_obj=-2.0)
    obj, feas = bench_discs._objective_and_feasibility(p, result, "mis")
    assert obj == 2
    assert feas is True


def test_objective_and_feasibility_mis_infeasible_solution_caps_value():
    # The replica picks every vertex of an edge -> infeasible.
    g = nx.path_graph(4)
    p = qqa.MaximumIndependentSet(g, penalty=3.0)
    x = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    # raw loss = -|S| + penalty * #violated = -2 + 3 * 1 = 1.0
    result = _FakeAnnealResult(best_sol=x, best_obj=1.0)
    obj, feas = bench_discs._objective_and_feasibility(p, result, "mis")
    # Naïve  -best_obj  would say 'IS size = -1' (nonsense). The
    # feasibility-aware path uses score_summary which reports the actual
    # |S| of the (infeasible) replica AND flags it as infeasible.
    assert obj == 2  # chose 2 vertices, both endpoints of an edge
    assert feas is False


def test_objective_and_feasibility_maxcut_unconstrained():
    # K_3 is always cuttable; cut = 2 for any 2-1 partition.
    g = nx.complete_graph(3)
    p = qqa.MaxCut(g)
    x = torch.tensor([1, 0, 1], dtype=torch.float32)
    # qqa loss = -cut = -2
    result = _FakeAnnealResult(best_sol=x, best_obj=-2.0)
    obj, feas = bench_discs._objective_and_feasibility(p, result, "maxcut")
    assert obj == 2.0
    assert feas is True


def test_objective_and_feasibility_normcut_uses_discrete_projection():
    # 4-cycle bisected as {0,2}|{1,3}: every edge crosses, so each
    # partition's cut count == 4 (it appears as 'belongs to me' for all
    # 4 edges), volume == 2+2 == 4, Ncut = 4/4 + 4/4 = 2.0.
    g = nx.cycle_graph(4)
    p = qqa.NormalizedCut(g, num_category=2, eps=0.0)
    x = torch.zeros(4, 2)
    x[0, 0] = 1
    x[2, 0] = 1
    x[1, 1] = 1
    x[3, 1] = 1
    result = _FakeAnnealResult(best_sol=x, best_obj=999.9)  # ignored
    obj, feas = bench_discs._objective_and_feasibility(p, result, "normcut")
    assert abs(obj - 2.0) < 1e-6
    assert feas is True


def test_approx_ratio_signs():
    # Maximisation: ratio = obj / best
    assert bench_discs._approx_ratio(3.0, 4.0, "mis") == 0.75
    assert bench_discs._approx_ratio(3.0, 4.0, "maxcut") == 0.75
    assert bench_discs._approx_ratio(3.0, 4.0, "maxclique") == 0.75
    # Minimisation (NormCut): ratio = best / obj  (lower obj → ratio ≥ 1)
    assert bench_discs._approx_ratio(0.5, 1.0, "normcut") == 2.0
    # Edge cases
    assert bench_discs._approx_ratio(3.0, None, "mis") is None
    assert bench_discs._approx_ratio(3.0, 0.0, "mis") is None


def test_cli_rejects_parallel_with_sa_backend():
    """--parallel currently requires --backend qqa (sa/pa cannot batch)."""
    import pytest

    with pytest.raises(SystemExit, match="--parallel"):
        bench_discs.main(
            [
                "--suite", "mis-satlib-uf",
                "--backend", "sa",
                "--parallel",
                "--instances", "1",
            ]
        )


def test_cli_accepts_paper_hyperparameters(monkeypatch):
    """Argparse must accept all paper-level hp flags without erroring.

    We monkey-patch ``_resolve_triples`` to short-circuit before any
    data is loaded, so we only exercise the argparse contract.
    """
    monkeypatch.setattr(bench_discs, "_resolve_suite", lambda _suite: [])
    bench_discs.main(
        [
            "--suite", "mis-satlib-uf",
            "--backend", "qqa",
            "--instances", "1",
            "--sol-size", "100",
            "--num-epochs", "3000",
            "--learning-rate", "1.0",
            "--temp", "1e-3",
            "--curve-rate", "4",
            "--gamma-min", "-2",
            "--gamma-max", "0.1",
            "--div-param", "0.2",
            "--penalty", "2.0",
            "--device", "cpu",
        ]
    )
