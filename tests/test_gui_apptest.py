"""Headless Streamlit ``AppTest`` coverage of the QQA dashboard.

These tests run the Streamlit script *in-process* (no subprocess, no browser)
via :mod:`streamlit.testing.v1`. They verify that:

* The Home page renders for a built-in problem.
* The Home page accepts the Custom problem flow and builds a UserProblem.
* The Solve page exposes hyper-parameter widgets and the Run button.
* The Visualize page handles the no-run state gracefully.

``AppTest`` ships with Streamlit ≥ 1.29, so the tests are skipped on older
versions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# The Home page hides the custom-problem toggle unless this env var is set.
os.environ.setdefault("QQA_ALLOW_CUSTOM", "1")

pytest.importorskip("streamlit", minversion="1.29.0")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
PAGE_DIR = APP.parent / "pages"


def test_home_page_renders_default_problem():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    titles = [h.body for h in at.title]
    assert any("Quasi-Quantum Annealing" in t for t in titles)
    cfg = at.session_state["problem_config"]
    assert cfg["kind"] == "mis"
    assert cfg["size"] >= 4


def test_home_page_custom_problem_flow():
    """Toggle Custom mode, supply a simple snippet, and confirm a UserProblem
    is built (the preview panel does not raise)."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    # Enable custom mode via the sidebar toggle.
    toggles = [t for t in at.sidebar.toggle if "custom" in t.label.lower()]
    assert toggles, "Custom-problem toggle missing from the sidebar"
    toggles[0].set_value(True)
    at.run()
    assert not at.exception
    cfg = at.session_state["problem_config"]
    assert cfg["kind"] == "custom"
    assert "source" in cfg["extra"]
    # The snippet executes without raising.

    # Fallback: import via the same mechanism streamlit_app uses.
    import sys

    sys.path.insert(0, str(APP.parent))
    from _common import build_problem as _build  # noqa: F811

    problem = _build(cfg)
    assert problem.num_vars == cfg["size"]


def test_solve_page_widgets_present():
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=60)
    # Seed the shared session state as if the user came from Home.
    at.session_state["problem_config"] = {
        "kind": "ising1d",
        "size": 8,
        "seed": 0,
        "device": "cpu",
        "extra": {},
    }
    at.run()
    assert not at.exception
    labels = [s.label for s in at.sidebar.slider]
    assert any("sol_size" in lab for lab in labels)
    assert any("epochs" in lab for lab in labels)
    run_buttons = [b for b in at.button if "Run" in b.label]
    assert run_buttons, "Run QQA button not rendered"


def test_visualize_page_handles_missing_run():
    """Visualize page without any run should warn, not crash."""
    at = AppTest.from_file(str(PAGE_DIR / "2_Visualize.py"), default_timeout=60)
    at.run()
    assert not at.exception
    warnings = [w.body for w in at.warning]
    assert any("Solve page" in w for w in warnings)


def _set_slider(at, label_fragment: str, value) -> None:
    """Set the slider whose label contains ``label_fragment`` to ``value``."""
    matches = [s for s in at.sidebar.slider if label_fragment in s.label]
    assert matches, f"No slider whose label contains {label_fragment!r}"
    matches[0].set_value(value)


def test_solve_page_end_to_end_run():
    """Full Solve flow: wire up a tiny problem and click Run.

    Exercises the live-callback path (Plotly fillcolor validation, metric
    tiles, population heatmap, diversity curve, and the final score card).
    A previous regression fed Plotly an 8-hex colour ('#RRGGBBAA') and only
    surfaced at runtime; this test pins the contract.
    """
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=90)
    at.session_state["problem_config"] = {
        "kind": "ising1d",
        "size": 8,
        "seed": 0,
        "device": "cpu",
        "extra": {},
    }
    at.run()
    assert not at.exception, at.exception

    # Shrink every slider so the test runs in a few seconds.
    _set_slider(at, "sol_size", 4)
    _set_slider(at, "epochs", 100)
    _set_slider(at, "UI update every", 10)
    at.run()
    assert not at.exception, at.exception

    runs = [b for b in at.button if "Run" in b.label]
    assert runs, "Run QQA button missing"
    runs[0].click()
    at.run()

    assert not at.exception, at.exception
    # Success is observable as either the score card or the raw-loss caption.
    texts = " ".join([m.value for m in at.markdown if m.value])
    assert "qqa-score" in texts or "energy" in texts.lower() or "raw loss" in texts


def test_solution_viz_smoke_across_problem_kinds(tmp_path):
    """Every registered renderer in ``_solution_viz`` must accept a real
    ``(problem, result, cfg)`` triple without raising.

    We drive this via a tiny Streamlit script so the renderer sees a proper
    script-run context. The script loops over one instance per problem
    family, runs 10 epochs of QQA, and calls ``render_solution_view``.
    AppTest's exception capture catches any failure on the way.
    """
    script = tmp_path / "soln_smoke.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(APP.parent)!r})\n"
        "import streamlit as st\n"
        "from _common import build_problem\n"
        "from _solution_viz import _RENDERERS, render_solution_view\n"
        "import qqa\n"
        "\n"
        "size_by_kind = {\n"
        "    'tsp': 6, 'qap': 5, 'nqueens': 5, 'ea': 8,\n"
        "    'maxsat3': 8, 'knapsack': 8, 'number_partition': 8,\n"
        "}\n"
        "extra_by_kind = {\n"
        "    'mis': {'graph_d': 3},\n"
        "    'maxcut': {'graph_d': 3},\n"
        "    'maxclique': {'graph_d': 3},\n"
        "    'vertex_cover': {'graph_d': 3},\n"
        "    'graph_bisection': {'graph_d': 3, 'balance_penalty': 2.0},\n"
        "    'coloring': {'num_category': 3, 'graph_d': 3},\n"
        "    'perceptron': {'alpha': 0.5},\n"
        "    'hopfield': {'patterns': 2},\n"
        "    'maxsat3': {'ratio': 3.0},\n"
        "    'ea': {'dim': 3},\n"
        "    'tsp': {'column_penalty': 3.0},\n"
        "    'qap': {'column_penalty': 10.0},\n"
        "    'number_partition': {'max_value': 50},\n"
        "    'knapsack': {'capacity_ratio': 0.5},\n"
        "}\n"
        "for kind in _RENDERERS:\n"
        "    cfg = {\n"
        "        'kind': kind,\n"
        "        'size': size_by_kind.get(kind, 16),\n"
        "        'seed': 0,\n"
        "        'device': 'cpu',\n"
        "        'extra': extra_by_kind.get(kind, {}),\n"
        "    }\n"
        "    problem = build_problem(cfg)\n"
        "    result = qqa.anneal(\n"
        "        problem, sol_size=4, num_epochs=10,\n"
        "        learning_rate=0.1, device='cpu', verbose=False,\n"
        "    )\n"
        "    st.markdown(f'### {kind}')\n"
        "    render_solution_view(problem, result, cfg)\n"
    )
    at = AppTest.from_file(str(script), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
