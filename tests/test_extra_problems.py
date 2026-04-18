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


def test_anneal_cuda_requested_but_unavailable_raises():
    """Skip on hosts with CUDA; otherwise expect an informative RuntimeError."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; cannot exercise the fallback path.")
    prob = qqa.MaximumIndependentSet(nx.path_graph(4))
    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        qqa.anneal(prob, sol_size=4, num_epochs=1, device="cuda", verbose=False)
