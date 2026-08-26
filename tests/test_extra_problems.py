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
        lambda: qqa.MinimumDominatingSet(nx.path_graph(8)),
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


def test_mixed_type_node_labels_are_normalized_without_sorting():
    graph = nx.Graph()
    graph.add_edges_from([(1, "depot"), ("depot", ("customer", 2))])
    problem = qqa.MaxCut(graph)
    assert problem.Q_mat.shape == (3, 3)


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


def test_tsp_uses_sinkhorn_by_default_and_keeps_binary_opt_in():
    """Permutation structure is the default; the legacy penalty path remains opt-in."""
    from qqa.relaxation import BinaryRelaxation, SinkhornRelaxation

    p = qqa.TSP(N=4, seed=0, row_penalty=7.0, col_penalty=11.0)
    assert isinstance(p.relaxation, SinkhornRelaxation)
    assert p.row_penalty == 7.0
    assert p.col_penalty == 11.0
    assert p.relaxation.init(2, p, "cpu").shape == (2, 4, 4)
    binary = qqa.TSP(N=4, seed=0, relaxation="binary")
    assert isinstance(binary.relaxation, BinaryRelaxation)
    assert binary.relaxation.init(2, binary, "cpu").shape == (2, 4, 4)


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
    original = x_disc.clone()
    summary = p.score_summary(x_disc)
    assert summary["feasible"] is True
    assert summary["extra"]["raw_feasible"] is False
    assert summary["extra"]["snapped"] is True
    # Scoring is pure. Repair is explicit and returns a new valid assignment.
    assert torch.equal(x_disc, original)
    cleaned = p.repair_solution(x_disc)[0]
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


# ---------------------------------------------------------------------------
# Constructor-dispatch hardening (build_problem ↔ session-state robustness)
# ---------------------------------------------------------------------------


def _ensure_app_on_path() -> None:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "app"))


def test_build_problem_rejects_unknown_extra_kwargs(monkeypatch):
    """Displayed options must never be silently omitted by the dispatcher."""
    _ensure_app_on_path()
    import _common as common

    monkeypatch.setattr(common.st, "caption", lambda *a, **k: None)

    cfg = {
        "kind": "tsp",
        "size": 4,
        "seed": 0,
        "device": "cpu",
        "extra": {
            "col_penalty": 4.0,
            "mystery_kwarg": 999,
            "another_stale_one": "x",
        },
    }
    with pytest.raises(TypeError, match="another_stale_one, mystery_kwarg"):
        common.build_problem(cfg)


def test_build_problem_forwards_tsp_relaxation(monkeypatch):
    from qqa.relaxation import BinaryRelaxation

    _ensure_app_on_path()
    import _common as common

    monkeypatch.setattr(common.st, "caption", lambda *a, **k: None)
    cfg = {
        "kind": "tsp",
        "size": 4,
        "seed": 0,
        "device": "cpu",
        "extra": {"relaxation": "binary", "row_penalty": 3.0, "col_penalty": 4.0},
    }
    problem = common.build_problem(cfg)
    assert isinstance(problem.relaxation, BinaryRelaxation)
    assert problem.row_penalty == 3.0
    assert problem.col_penalty == 4.0


def test_tsp_back_compat_column_penalty_alias():
    """The legacy ``column_penalty`` kwarg sets *both* row and col
    penalties (old categorical-style behaviour) and emits a deprecation
    warning so callers can migrate."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = qqa.TSP(N=4, seed=0, column_penalty=7.5)
    assert p.row_penalty == 7.5
    assert p.col_penalty == 7.5
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "Expected a DeprecationWarning for legacy column_penalty kwarg"
    )


def test_tsp_penalty_weights_dict_overrides_scalars():
    """A structured ``penalty_weights`` dict is the future-proof way to
    declare an arbitrary number of penalty terms; when supplied it
    overrides the explicit scalar kwargs."""
    p = qqa.TSP(
        N=4,
        seed=0,
        row_penalty=99.0,
        col_penalty=99.0,
        penalty_weights={"row": 1.5, "col": 9.0},
    )
    assert p.row_penalty == 1.5
    assert p.col_penalty == 9.0
    assert p.penalty_weights == {"row": 1.5, "col": 9.0}


def test_qap_penalty_weights_dict_overrides_scalar():
    """Same structured-dict story for QAP — extending to additional
    QAP penalties later only needs more keys, not a signature change."""
    p = qqa.QAP(N=4, seed=0, column_penalty=99.0, penalty_weights={"column": 6.5})
    assert p.column_penalty == 6.5


def test_build_problem_modern_kwargs_win_over_legacy_alias(monkeypatch):
    """If both legacy and modern keys are present (e.g. saved config
    written by both old and new versions of the UI), the modern
    explicit keys must win — otherwise renaming a kwarg in the schema
    silently regresses the slider's effect on the next page reload."""
    _ensure_app_on_path()
    import _common as common

    monkeypatch.setattr(common.st, "caption", lambda *a, **k: None)

    cfg = {
        "kind": "tsp",
        "size": 4,
        "seed": 0,
        "device": "cpu",
        "extra": {
            "column_penalty": 2.0,  # legacy (would set both to 2.0)
            "row_penalty": 3.0,  # ← must win
            "col_penalty": 4.5,  # ← must win
        },
    }
    p = common.build_problem(cfg)
    assert p.row_penalty == 3.0
    assert p.col_penalty == 4.5


# ---------------------------------------------------------------------------
# 3-D EA cone visualisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        {"kind": "tsp", "size": 6, "seed": 0, "device": "cpu", "extra": {}},
        {"kind": "qap", "size": 4, "seed": 0, "device": "cpu", "extra": {}},
        {"kind": "knapsack", "size": 8, "seed": 0, "device": "cpu", "extra": {}},
        {"kind": "number_partition", "size": 10, "seed": 0, "device": "cpu", "extra": {}},
        {"kind": "maxsat3", "size": 8, "seed": 0, "device": "cpu", "extra": {"ratio": 3.0}},
        {"kind": "nqueens", "size": 6, "seed": 0, "device": "cpu", "extra": {}},
        {"kind": "hopfield", "size": 12, "seed": 0, "device": "cpu", "extra": {"patterns": 3}},
    ],
)
def test_preview_problem_renders_without_fallback_message(cfg, monkeypatch):
    """Every supported problem family must produce a real, branded
    preview rather than the generic "no preview available" placeholder
    of old. Captures every Streamlit element call and asserts:

    * at least one chart was emitted (or, for MaxSAT3, at least one
      formatted-clause markdown card, since its main preview is
      typographic);
    * none of the rendered text contains the legacy fallback message.
    """
    _ensure_app_on_path()
    import _common as common

    captured: dict[str, list] = {"charts": [], "markdowns": [], "captions": []}
    monkeypatch.setattr(common.st, "plotly_chart", lambda fig, **k: captured["charts"].append(fig))
    monkeypatch.setattr(
        common.st,
        "markdown",
        lambda *a, **k: captured["markdowns"].append(a[0] if a else ""),
    )
    monkeypatch.setattr(
        common.st,
        "caption",
        lambda *a, **k: captured["captions"].append(a[0] if a else ""),
    )
    monkeypatch.setattr(common.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(common.st, "warning", lambda *a, **k: None)
    monkeypatch.setattr(common.st, "latex", lambda *a, **k: None)

    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        common.st,
        "columns",
        lambda spec: [_Col() for _ in range(spec if isinstance(spec, int) else len(spec))],
    )

    p = common.build_problem(cfg)
    common.preview_problem(p, cfg)

    all_text = " ".join(captured["markdowns"]) + " ".join(captured["captions"])
    assert "No preview available" not in all_text, (
        f"Legacy fallback message leaked into the {cfg['kind']} preview."
    )
    # MaxSAT3's main canvas is its formatted-clause markdown card; for
    # every other kind we expect at least one Plotly chart trace.
    if cfg["kind"] != "maxsat3":
        assert captured["charts"], f"{cfg['kind']} produced zero Plotly charts — preview was empty."
    else:
        assert any("∨" in m for m in captured["markdowns"]), (
            "MaxSAT3 preview should render at least one formatted clause."
        )


def test_render_ea_3d_uses_cone_traces():
    """The 3-D Edwards–Anderson visualisation is the headline view for
    3-D Ising; pin that the figure contains at least one ``go.Cone``
    trace and a bond-line trace, so the renderer doesn't silently fall
    back to flat scatter markers."""
    _ensure_app_on_path()
    import _solution_viz as viz
    import numpy as _np
    import plotly.graph_objects as _go

    rng = _np.random.RandomState(0)
    s = rng.choice([-1, 1], size=4**3)
    fig = viz._ea_3d_cone_figure(s, 4, title="3D EA test")
    types = [type(t).__name__ for t in fig.data]
    assert any(isinstance(t, _go.Cone) for t in fig.data), f"Expected a Cone trace, got: {types}"
    assert any(isinstance(t, _go.Scatter3d) for t in fig.data), (
        f"Expected a Scatter3d bond trace, got: {types}"
    )


def test_anneal_cuda_requested_but_unavailable_raises():
    """Skip on hosts with CUDA; otherwise expect an informative RuntimeError."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; cannot exercise the fallback path.")
    prob = qqa.MaximumIndependentSet(nx.path_graph(4))
    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        qqa.anneal(prob, sol_size=4, num_epochs=1, device="cuda", verbose=False)


# ---------------------------------------------------------------------------
# MinimumDominatingSet — focused correctness checks
# ---------------------------------------------------------------------------


def test_min_dominating_set_path_graph_optimum_within_reach():
    """For a path P_n, γ(P_n) = ceil(n/3). On P_9 the minimum is 3.

    QQA does not have to *prove* the optimum, but on a small instance
    with the polish-on-by-default loop it should reliably hit it.
    """
    qqa.fix_seed(0)
    prob = qqa.MinimumDominatingSet(nx.path_graph(9), penalty=4.0)
    res = qqa.anneal(prob, sol_size=64, num_epochs=400, verbose=False)
    score = res.score
    assert score["feasible"], f"Expected a feasible dominating set, got {score}"
    assert score["value"] <= 3, f"Expected |S| <= 3 on P_9 (γ=3), got {score['value']}"


def test_min_dominating_set_loss_matches_discrete_definition():
    """The relaxed ``loss_fn`` must equal ``|S| + λ * #uncovered`` exactly
    when evaluated on a {0,1} bitstring (relaxation is tight at corners)."""
    g = nx.cycle_graph(6)
    prob = qqa.MinimumDominatingSet(g, penalty=4.0)
    # Pick a non-dominating set: {0} alone covers {0, 1, 5}, leaving
    # {2, 3, 4} uncovered → loss = 1 + 4 * 3 = 13.
    x = torch.zeros(1, 6)
    x[0, 0] = 1.0
    val = float(prob.loss_fn(x).item())
    assert val == pytest.approx(13.0, abs=1e-4)
    # And a true dominating set: {0, 3} covers everything → loss = 2.
    x = torch.zeros(1, 6)
    x[0, 0] = 1.0
    x[0, 3] = 1.0
    val = float(prob.loss_fn(x).item())
    assert val == pytest.approx(2.0, abs=1e-4)


# ---------------------------------------------------------------------------
# PSpinGlass / RandomFieldIsing — focused physics-correctness checks
# ---------------------------------------------------------------------------


def test_pspin_p2_matches_dense_gaussian_form():
    """For p = 2 the energy is a homogeneous degree-2 form in s. The total
    number of couplings must equal C(N, 2) and the loss must be invariant
    under a global spin flip s → -s (parity)."""
    qqa.fix_seed(0)
    prob = qqa.PSpinGlass(N=8, p=2, seed=0)
    assert int(prob.indices.shape[0]) == 28  # C(8, 2)
    s = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
    e_pos = float(prob.loss_fn(s).item())
    e_neg = float(prob.loss_fn(-s).item())
    assert e_pos == pytest.approx(e_neg, abs=1e-5), "p=2 must be parity-symmetric"


def test_pspin_p3_breaks_parity():
    """For odd p the energy flips sign under s → -s. This is the canonical
    distinguishing feature of p-spin models with odd interaction order."""
    prob = qqa.PSpinGlass(N=8, p=3, seed=1)
    s = torch.randn(1, 8).sign()
    e_pos = float(prob.loss_fn(s).item())
    e_neg = float(prob.loss_fn(-s).item())
    assert e_pos == pytest.approx(-e_neg, abs=1e-5), "p=3 must flip sign under s→-s"


def test_pspin_anneals_and_is_finite():
    prob = qqa.PSpinGlass(N=10, p=3, seed=0)
    res = qqa.anneal(prob, sol_size=32, num_epochs=120, verbose=False)
    assert torch.isfinite(torch.as_tensor(res.best_obj)).all()
    score = res.score
    assert score["feasible"] is True
    assert "p" in score["extra"] and score["extra"]["p"] == 3


def test_rfim_strong_field_aligns_with_field():
    """In the limit σ_h ≫ J the ground state is s_i = sign(h_i): each spin
    independently aligns with its local field. Run a tiny anneal at large
    h_std and check the resulting overlap is large and positive."""
    qqa.fix_seed(0)
    prob = qqa.RandomFieldIsing(L=4, dim=2, J=0.05, h_std=5.0, seed=0)
    res = qqa.anneal(prob, sol_size=64, num_epochs=300, verbose=False)
    s = res.best_sol.float().reshape(-1)
    h = prob.h.cpu().float()
    overlap = float((s * torch.sign(h)).mean().item())
    assert overlap > 0.7, f"strong-field RFIM should align with h: overlap={overlap}"


def test_rfim_lattice_sizes_and_field_shape():
    prob = qqa.RandomFieldIsing(L=3, dim=2, J=1.0, h_std=1.0, seed=0)
    assert prob.num_spins == 9
    assert prob.J.shape == (9, 9)
    assert prob.h.shape == (9,)
    # Symmetric ferromagnetic J with diag = 0.
    assert torch.allclose(prob.J, prob.J.t())
    assert float(prob.J.diag().abs().max().item()) == 0.0
