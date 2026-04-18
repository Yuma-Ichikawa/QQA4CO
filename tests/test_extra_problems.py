"""Phase-A problem coverage: Knapsack, NumberPartitioning, VertexCover,
GraphBisection, MaxSAT3, TSP, QAP, NQueens, BalancedGraphPartition, plus the
``UserProblem`` / ``user_problem_from_source`` / ``load_problem_from_file``
helpers.

These problems were added in v0.3 and had minimal test coverage: this file
exercises every constructor + one ``qqa.anneal`` call per problem to catch
regressions in ``loss_fn`` / ``score_summary`` / relaxation wiring.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

import qqa


@pytest.fixture(autouse=True)
def _deterministic_seed() -> None:
    qqa.fix_seed(0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: qqa.Knapsack(N=12, seed=0),
        lambda: qqa.NumberPartitioning(N=12, seed=0),
        lambda: qqa.MaxSAT3(N=10, ratio=3.0, seed=0),
        lambda: qqa.VertexCover(nx.path_graph(8)),
        lambda: qqa.GraphBisection(nx.cycle_graph(8)),
        lambda: qqa.TSP(N=5, seed=0),
        lambda: qqa.QAP(N=4, seed=0),
        lambda: qqa.NQueens(N=5),
        lambda: qqa.BalancedGraphPartition(nx.cycle_graph(6), num_category=2),
    ],
)
def test_phase_a_problem_runs(factory):
    """Each Phase-A problem anneals without raising and produces a finite
    objective plus a populated ``score_summary`` dict."""
    problem = factory()
    result = qqa.anneal(problem, sol_size=16, num_epochs=80, verbose=False)
    assert torch.is_tensor(result.best_sol)
    obj = float(result.best_obj)
    assert obj == obj  # not NaN
    assert obj != float("inf") and obj != float("-inf")
    score = result.score
    assert "label" in score and "value" in score and "feasible" in score


def test_number_partitioning_perfect_split_is_zero_loss():
    """When the target is achievable exactly, the squared-diff loss is 0."""
    prob = qqa.NumberPartitioning(N=4, seed=0)
    # Build a plant: values = [3, 1, 1, 3], split {0,1} vs {2,3} has diff = 0.
    prob.values = torch.tensor([3.0, 1.0, 1.0, 3.0])
    s = torch.tensor([[+1.0, +1.0, -1.0, -1.0]])
    assert prob.loss_fn(s).item() == 0.0


def test_knapsack_respects_capacity_at_optimum():
    """A trivially packable instance: one tiny item, huge capacity."""
    prob = qqa.Knapsack(N=1, capacity_ratio=2.0, seed=0)
    x = torch.tensor([[1.0]])  # take the item
    loss = prob.loss_fn(x).item()
    # loss = -value (no overflow because capacity_ratio=2 > total weight).
    assert loss <= 0.0


def test_vertex_cover_zero_edges_is_empty_set_optimum():
    prob = qqa.VertexCover(nx.empty_graph(5))
    x = torch.zeros((1, 5))
    # No edges + empty set ⇒ loss = |S| = 0.
    assert prob.loss_fn(x).item() == 0.0


def test_nqueens_zero_conflicts_on_known_solution_n4():
    """The 4-queens placement {(0,1),(1,3),(2,0),(3,2)} has zero conflicts."""
    prob = qqa.NQueens(N=4)
    x = torch.zeros((1, 4, 4))
    cols = [1, 3, 0, 2]
    for r, c in enumerate(cols):
        x[0, r, c] = 1.0
    assert prob.loss_fn(x).item() == 0.0


def test_user_problem_binary_solves_trivial_quadratic():
    """Minimise ``Σ_i (x_i - 0.5)^2``: discrete projection is arbitrary so any
    binary vector is optimal; the continuous minimum is ``N * 0.25``."""

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return ((x - 0.5) ** 2).sum(dim=-1)

    prob = qqa.UserProblem(num_vars=8, variable_kind="binary", loss_fn=loss_fn)
    result = qqa.anneal(prob, sol_size=16, num_epochs=80, verbose=False)
    # Each component lands on 0 or 1 → per-component loss is 0.25 → total 2.0.
    assert result.best_obj == pytest.approx(2.0, abs=0.01)


def test_user_problem_categorical_requires_num_category():
    with pytest.raises(ValueError):
        qqa.UserProblem(num_vars=4, variable_kind="categorical", loss_fn=lambda x: x.sum(dim=-1))


def test_user_problem_rejects_unknown_kind():
    with pytest.raises(ValueError):
        qqa.UserProblem(num_vars=4, variable_kind="quaternion", loss_fn=lambda x: x.sum(dim=-1))  # type: ignore[arg-type]


def test_user_problem_from_source_requires_loss_fn():
    """Source without a ``loss_fn`` symbol must be rejected with a clear
    ValueError rather than a cryptic NameError downstream."""
    with pytest.raises(ValueError):
        qqa.user_problem_from_source("# empty\n", num_vars=4)


def test_load_problem_from_file_supports_problem_variable(tmp_path):
    path = tmp_path / "problem.py"
    path.write_text(
        "import torch\nimport qqa\n"
        "problem = qqa.UserProblem(\n"
        "    num_vars=8, variable_kind='binary',\n"
        "    loss_fn=lambda x: ((x - 0.5) ** 2).sum(dim=-1),\n"
        ")\n"
    )
    problem = qqa.load_problem_from_file(str(path))
    assert isinstance(problem, qqa.COProblem)
    assert problem.num_vars == 8


def test_load_problem_from_file_supports_factory(tmp_path):
    path = tmp_path / "factory.py"
    path.write_text(
        "import qqa\n"
        "def make_problem():\n"
        "    return qqa.UserProblem(\n"
        "        num_vars=6, variable_kind='binary',\n"
        "        loss_fn=lambda x: x.sum(dim=-1),\n"
        "    )\n"
    )
    problem = qqa.load_problem_from_file(str(path))
    assert isinstance(problem, qqa.COProblem)


def test_load_problem_from_file_rejects_non_problem(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text("problem = 42\n")
    # ``problem`` exists but is not a COProblem ⇒ no factory either ⇒
    # AttributeError (not silent success).
    with pytest.raises(AttributeError):
        qqa.load_problem_from_file(str(path))


# ---------------------------------------------------------------------------
# Non-contiguous graph labels (regression: constructors used to silently
# corrupt the QUBO or raise IndexError). Covers the ``normalize_graph`` path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (qqa.MaximumIndependentSet, {}),
        (qqa.MaxClique, {}),
        (qqa.MaxCut, {}),
        (qqa.VertexCover, {}),
        (qqa.GraphBisection, {}),
        (qqa.Coloring, {"num_category": 3}),
    ],
)
def test_non_contiguous_node_labels(cls, kwargs):
    g = nx.Graph()
    g.add_edges_from([(10, 20), (20, 30), (30, 10), (30, 40)])
    problem = cls(g, **kwargs)
    result = qqa.anneal(problem, sol_size=8, num_epochs=40, verbose=False)
    assert torch.is_tensor(result.best_sol)


# ---------------------------------------------------------------------------
# ``qqa.anneal`` edge cases
# ---------------------------------------------------------------------------


def test_anneal_rejects_zero_sol_size():
    prob = qqa.MaximumIndependentSet(nx.path_graph(4))
    with pytest.raises(ValueError):
        qqa.anneal(prob, sol_size=0, num_epochs=1, verbose=False)


def test_anneal_zero_epochs_returns_initial_sample():
    """Zero epochs: ``on_train_end`` must still see a valid CallbackState and
    the result should carry a finite ``best_obj``."""
    prob = qqa.MaximumIndependentSet(nx.path_graph(6))
    r = qqa.anneal(prob, sol_size=8, num_epochs=0, verbose=False)
    assert r.best_obj == r.best_obj  # not NaN
    assert r.runtime >= 0.0


# ---------------------------------------------------------------------------
# TSP penalty-method formulation
# ---------------------------------------------------------------------------


def test_tsp_uses_binary_relaxation_with_two_penalties():
    """v0.4 reformulated TSP as a true penalty method: BinaryRelaxation +
    explicit row + column penalties (instead of CategoricalRelaxation +
    one column penalty). This regression test pins both invariants."""
    from qqa.relaxation import BinaryRelaxation

    p = qqa.TSP(N=4, seed=0, row_penalty=7.0, col_penalty=11.0)
    assert isinstance(p.relaxation, BinaryRelaxation), "TSP must use BinaryRelaxation"
    assert p.row_penalty == 7.0
    assert p.col_penalty == 11.0
    # Latent shape is (sol_size, N, N) thanks to the shape_fn.
    assert p.relaxation.init(2, p, "cpu").shape == (2, 4, 4)


def test_tsp_loss_decomposes_into_three_additive_terms():
    """Independently compute the tour, row and column components and check
    ``loss_fn`` returns their weighted sum exactly. Catches sign/scaling
    drift in any of the three additive terms."""
    p = qqa.TSP(N=3, seed=42, row_penalty=2.0, col_penalty=3.0)
    # A non-permutation: position 0 picks city 0 twice (positions 0 and 1
    # both pick city 0; position 2 picks city 1). City 2 is missed.
    x = torch.zeros((1, 3, 3))
    x[0, 0, 0] = 1.0
    x[0, 1, 0] = 1.0
    x[0, 2, 1] = 1.0
    # row sums: [1, 1, 1] → row penalty 0
    # col sums: [2, 1, 0] → col penalty (2-1)^2 + (1-1)^2 + (0-1)^2 = 2
    # tour: pos 0 (city 0) → pos 1 (city 0): d[0,0]=0
    #       pos 1 (city 0) → pos 2 (city 1): d[0,1]
    #       pos 2 (city 1) → pos 0 (city 0): d[1,0]  (= d[0,1])
    d01 = float(p.distance[0, 1].item())
    expected = (2 * d01) + 2.0 * 0.0 + 3.0 * 2.0
    assert p.loss_fn(x).item() == pytest.approx(expected, rel=1e-5)


def test_tsp_score_summary_always_returns_a_valid_tour():
    """Even when the raw discrete projection violates row/col constraints
    the Hungarian snap must produce a feasible permutation. ``feasible``
    in the score is therefore always True; ``extra.raw_feasible`` reports
    whether the optimiser itself converged."""
    p = qqa.TSP(N=4, seed=0, row_penalty=5.0, col_penalty=5.0)
    x_disc = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ]
    )
    summary = p.score_summary(x_disc)
    assert summary["feasible"] is True
    assert summary["extra"]["raw_feasible"] is False
    assert summary["extra"]["snapped"] is True
    # ``best_sol`` must be left as a valid one-hot permutation in place.
    cleaned = x_disc[0]
    assert (cleaned.sum(dim=0) == 1).all()
    assert (cleaned.sum(dim=1) == 1).all()


# ---------------------------------------------------------------------------
# EA preview helper handles big lattices without OOM
# ---------------------------------------------------------------------------


def test_ea_preview_helper_uses_sparse_view_for_big_lattices(monkeypatch):
    """The Streamlit preview built a dense ``Heatmap(z=problem.J)`` which
    OOMed for the default EA setting (L=32, dim=3 ⇒ 32 768 spins). v0.4
    falls back to a sparse non-zero scatter once N exceeds the dense
    threshold. The chart payload must be ``Scatter`` (not ``Heatmap``)."""
    import sys as _sys

    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app"))
    import _common as common

    # L=16 dim=3 ⇒ N=4096 ⇒ matrix is 4096×4096 ≈ 67 M cells. Comfortably
    # above the dense threshold; small enough to construct in <1s.
    p = qqa.EdwardsAnderson(L=16, dim=3, seed=0)
    captured: dict[str, object] = {}

    def _fake_chart(fig, **_kwargs):
        captured["fig"] = fig

    monkeypatch.setattr(common.st, "plotly_chart", _fake_chart)
    monkeypatch.setattr(common.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(common.st, "info", lambda *a, **k: None)
    common.preview_problem(p, {"kind": "ea"})
    fig = captured["fig"]
    assert fig is not None
    # Big EA ⇒ sparse scatter, not dense heatmap.
    import plotly.graph_objects as _go

    assert isinstance(fig.data[0], _go.Scatter)


def test_anneal_cuda_requested_but_unavailable_raises():
    """Skip on hosts with CUDA; otherwise expect an informative RuntimeError."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; cannot exercise the fallback path.")
    prob = qqa.MaximumIndependentSet(nx.path_graph(4))
    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        qqa.anneal(prob, sol_size=4, num_epochs=1, device="cuda", verbose=False)
