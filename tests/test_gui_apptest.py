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

import pytest

# The Home page hides the custom-problem toggle unless this env var is set.
# Using a plain assignment (not ``setdefault``) so that a CI / developer
# environment with ``QQA_ALLOW_CUSTOM=0`` still runs the custom-flow test.
os.environ["QQA_ALLOW_CUSTOM"] = "1"

pytest.importorskip("streamlit", minversion="1.29.0")

# Shared helpers — see ``tests/conftest.py``.
from conftest import APP, PAGE_DIR, make_problem_config  # noqa: E402
from conftest import set_slider as _set_slider
from streamlit.testing.v1 import AppTest  # noqa: E402


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
    """Toggle Custom mode, click Validate, and confirm a UserProblem is built
    (the preview panel does not raise)."""
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

    # Custom-problem preview is gated behind an explicit "Validate snippet"
    # click so we don't ``exec`` user code on every keystroke.  Click it
    # and then assert the preview ran cleanly.
    validate_btns = [b for b in at.button if "Validate" in b.label]
    assert validate_btns, "Validate snippet button missing"
    validate_btns[0].click()
    at.run()
    assert not at.exception

    # The preview panel must succeed end-to-end. Regression: previously the
    # preview called ``problem.relaxation(x)`` which raised
    # "'SpinRelaxation' object is not callable" and rendered as an st.error
    # under the Preview heading. AppTest does not raise on st.error, so we
    # have to inspect the rendered error elements explicitly.
    error_bodies = [getattr(e, "body", "") or getattr(e, "value", "") for e in at.error]
    assert not any("loss_fn raised" in (b or "") for b in error_bodies), (
        f"Custom problem preview surfaced an error: {error_bodies!r}"
    )
    success_bodies = [getattr(s, "body", "") or getattr(s, "value", "") for s in at.success]
    assert any("loss_fn output shape" in (b or "") for b in success_bodies), (
        "Custom problem preview did not display the expected success banner; "
        f"got success bodies={success_bodies!r}"
    )

    # Fallback: import via the same mechanism streamlit_app uses.
    import sys

    sys.path.insert(0, str(APP.parent))
    from _common import build_problem as _build  # noqa: F811

    problem = _build(cfg)
    assert problem.num_vars == cfg["size"]


def test_solve_page_widgets_present():
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=60)
    # Seed the shared session state as if the user came from Home.
    at.session_state["problem_config"] = make_problem_config("ising1d", 8)
    at.run()
    assert not at.exception
    labels = [s.label for s in at.sidebar.slider]
    assert any("sol_size" in lab for lab in labels)
    assert any("epochs" in lab for lab in labels)
    run_buttons = [b for b in at.button if "Run" in b.label]
    assert run_buttons, "Run QQA button not rendered"


def test_visualize_page_handles_missing_run():
    """Visualize page without any run should show a branded empty-state
    card with a CTA pointing at Solve, and must not crash."""
    at = AppTest.from_file(str(PAGE_DIR / "2_Visualize.py"), default_timeout=60)
    at.run()
    assert not at.exception
    # The card body is rendered through st.markdown — collect the html
    # blocks and assert the "Solve" CTA appears somewhere on the page.
    md_blobs = [m.value for m in at.markdown]
    assert any("Solve" in blob for blob in md_blobs), (
        "Empty-state card should mention Solve as the next step."
    )


def test_universal_studio_renders_all_workflows_without_running():
    at = AppTest.from_file(str(PAGE_DIR / "4_Universal.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("Universal Optimization Studio" in title.body for title in at.title)
    tab_labels = [tab.label for tab in at.tabs]
    for label in ("⚡ Mixed planning", "◎ Pareto studio", "◇ Black-box lab", "∑ TeX model"):
        assert label in tab_labels


def test_universal_studio_routes_reviewed_multiobjective_models_to_pareto():
    import qqa  # noqa: PLC0415

    spec = qqa.ModelSpec.from_dict(
        {
            "name": "ui-pareto",
            "variables": [{"name": "x", "kind": "integer", "lower": 0, "upper": 4, "size": 1}],
            "objectives": [
                {
                    "name": "cost",
                    "direction": "min",
                    "expression": "x",
                    "unit": "",
                },
                {
                    "name": "quality",
                    "direction": "max",
                    "expression": "x",
                    "unit": "",
                },
            ],
            "constraints": [],
            "notes": "",
        }
    )
    at = AppTest.from_file(str(PAGE_DIR / "4_Universal.py"), default_timeout=60)
    at.session_state["universal_spec"] = spec
    at.run()
    assert not at.exception, at.exception
    assert any("parallel Pareto solver" in info.body for info in at.info)
    assert not any(toggle.label == "SCIP proof phase" for toggle in at.toggle)


def test_solve_page_end_to_end_run():
    """Full Solve flow: wire up a tiny problem and click Run.

    Exercises the live-callback path (Plotly fillcolor validation, metric
    tiles, population heatmap, diversity curve, and the final score card).
    A previous regression fed Plotly an 8-hex colour ('#RRGGBBAA') and only
    surfaced at runtime; this test pins the contract.
    """
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=90)
    at.session_state["problem_config"] = make_problem_config("ising1d", 8)
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


def test_visualize_page_renders_pa_result():
    """After a PA run is stored in session state, Visualize must render
    the PA-specific tabs (ESS / β, Free energy, Equilibrium pop.) and the
    backend-aware Family tree tab without crashing or breaking the
    legacy tabs.
    """
    import networkx as nx  # noqa: PLC0415

    import qqa  # noqa: PLC0415

    g = nx.erdos_renyi_graph(8, 0.5, seed=0)
    prob = qqa.MaxCut(g)
    res = qqa.population_annealing(
        prob,
        sol_size=8,
        num_temps=6,
        sweeps_per_temp=1,
        beta_start=0.1,
        beta_end=2.0,
        record_genealogy=True,
        seed=0,
        verbose=False,
    )
    at = AppTest.from_file(str(PAGE_DIR / "2_Visualize.py"), default_timeout=90)
    at.session_state["last_result"] = res
    at.session_state["last_problem"] = prob
    at.session_state["last_pop_tracker"] = None
    at.session_state["problem_config"] = make_problem_config("maxcut", 8)
    at.run()
    assert not at.exception, at.exception
    # PA metric tiles populate (free-energy density + ln Z + R).
    metric_labels = [m.label for m in at.metric]
    assert any("F(β_end)" in lab for lab in metric_labels), (
        f"PA F(β_end)/N metric missing; got {metric_labels!r}"
    )
    # The backend-aware Family tree tab must be present (PA branch should
    # render Muller-plot output).
    tab_labels = [t.label for t in at.tabs]
    assert "Family tree" in tab_labels, (
        f"Family tree tab must always be present; got {tab_labels!r}"
    )
    # New PA-only analytic tabs must all render alongside the legacy trio.
    for extra_tab in (
        "PA: ESS / β",
        "PA: Free energy",
        "PA: Equilibrium pop.",
        "PA: Thermodynamics",
        "PA: Lineage vs energy",
        "PA: Ancestry Sankey",
    ):
        assert extra_tab in tab_labels, (
            f"PA-specific visualize tab {extra_tab!r} missing; got {tab_labels!r}"
        )
    # Backend-aware layout contract: PQQA-only tabs (driven by the
    # ``PopulationTracker`` that PA never produces) MUST NOT appear on a
    # PA run. The user explicitly asked for this — otherwise the tab bar
    # advertises six tabs that all just say "No population snapshots
    # recorded for this run." and clutter the UI.
    for pqqa_only_tab in (
        "Schedule",
        "Parallel population",
        "Solution-space PCA",
        "Diversity",
        "Loss spectrogram",
        "Ridgeline",
        "Replica fate",
    ):
        assert pqqa_only_tab not in tab_labels, (
            f"PQQA-only tab {pqqa_only_tab!r} must be hidden on a PA run; got tabs = {tab_labels!r}"
        )


def test_visualize_pqqa_family_tree_renders_dendrogram(tmp_path):
    """PQQA's Family tree view (dendrogram + per-clade energy trajectory)
    must render without exception when a population tracker is attached.

    This pins the new backend-aware "Family tree" tab so it doesn't
    regress to PA-only behaviour. The tab must be present for both
    backends and produce non-empty content for PQQA.
    """
    import sys

    sys.path.insert(0, str(APP.parent))
    from _common import build_problem as _build  # noqa: PLC0415

    import qqa  # noqa: PLC0415
    from qqa.callbacks import PopulationTracker  # noqa: PLC0415

    cfg = {"kind": "mis", "size": 24, "seed": 1, "device": "cpu", "extra": {}}
    problem = _build(cfg)
    tracker = PopulationTracker(stride=4)
    result = qqa.anneal(
        problem,
        sol_size=16,
        num_epochs=80,
        learning_rate=0.5,
        device="cpu",
        verbose=False,
        callbacks=[tracker],
    )
    at = AppTest.from_file(str(PAGE_DIR / "2_Visualize.py"), default_timeout=90)
    at.session_state["last_result"] = result
    at.session_state["last_problem"] = problem
    at.session_state["last_pop_tracker"] = tracker
    at.session_state["problem_config"] = cfg
    at.run()
    assert not at.exception, at.exception

    tab_labels = [t.label for t in at.tabs]
    assert "Family tree" in tab_labels, (
        f"Family tree tab must be present for PQQA results; got {tab_labels!r}"
    )
    # Backend-aware contract (mirror of the PA-run test): PA-only tabs
    # MUST NOT appear on a PQQA run. Otherwise we advertise six empty
    # tabs ("no free-energy history" / "no genealogy" etc).
    for pa_only_tab in (
        "PA: ESS / β",
        "PA: Free energy",
        "PA: Equilibrium pop.",
        "PA: Thermodynamics",
        "PA: Lineage vs energy",
        "PA: Ancestry Sankey",
    ):
        assert pa_only_tab not in tab_labels, (
            f"PA-only tab {pa_only_tab!r} must be hidden on a PQQA run; got tabs = {tab_labels!r}"
        )
    # PQQA-only tabs (what the Visualize page is designed around when
    # the run came from ``qqa.anneal``) must still be present.
    for pqqa_tab in (
        "Schedule",
        "Parallel population",
        "Solution-space PCA",
        "Diversity",
        "Loss spectrogram",
        "Ridgeline",
        "Replica fate",
    ):
        assert pqqa_tab in tab_labels, (
            f"PQQA tab {pqqa_tab!r} must be present on a PQQA run; got {tab_labels!r}"
        )


def test_solve_page_pa_backend_smoke_run():
    """The Solve page exposes PA as a backend and runs a tiny PA anneal end-to-end.

    Drives the new "Backend" radio onto Population Annealing, shrinks the
    PA hyper-parameter sliders, clicks Run, and asserts no exception. This
    catches the most common UI regressions: missing radio, mis-keyed
    sliders, callback signature mismatch with `qqa.population_annealing`,
    or a free-energy plot that breaks on a tiny problem.
    """
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=120)
    at.session_state["problem_config"] = make_problem_config("ising1d", 6)
    at.run()
    assert not at.exception, at.exception

    # Switch backend to PA.
    backend_radios = [r for r in at.sidebar.radio if "backend" in r.label.lower()]
    assert backend_radios, "Backend radio missing on Solve page"
    pa_options = [opt for opt in backend_radios[0].options if "PA" in opt]
    assert pa_options, f"PA option missing in {backend_radios[0].options!r}"
    backend_radios[0].set_value(pa_options[0])
    at.run()
    assert not at.exception, at.exception

    # PA sidebar should now expose its own knobs (and ONLY those — the
    # backend-aware refactor must hide PQQA-specific sliders).
    sidebar_labels = [s.label for s in at.sidebar.slider]
    assert any(lab.startswith("PA population") for lab in sidebar_labels), (
        f"PA hyper-parameter sliders missing; got {sidebar_labels!r}"
    )
    assert any(lab.startswith("num_temps") for lab in sidebar_labels), (
        f"PA num_temps slider missing; got {sidebar_labels!r}"
    )
    assert any(lab.startswith("sweeps_per_temp") for lab in sidebar_labels), (
        f"PA sweeps_per_temp slider missing; got {sidebar_labels!r}"
    )
    # Regression for the screenshot bug: PQQA-only sliders must be hidden.
    pqqa_only = {"sol_size", "epochs", "learning rate", "min bg", "max bg", "div_param"}
    leaked = [lab for lab in sidebar_labels if lab in pqqa_only]
    assert not leaked, (
        f"PQQA-only sliders leaked into PA sidebar: {leaked!r}; "
        f"all sidebar sliders = {sidebar_labels!r}"
    )
    # Greedy 1-flip polish is the symmetric knob to PQQA's post-processing;
    # the checkbox must be present and default ON so PA's reported quality
    # matches PQQA without the user having to remember.
    polish_boxes = [cb for cb in at.sidebar.checkbox if "polish" in cb.label.lower()]
    assert polish_boxes, (
        f"PA 'Greedy 1-flip polish' checkbox missing; "
        f"sidebar checkboxes = {[cb.label for cb in at.sidebar.checkbox]!r}"
    )
    assert polish_boxes[0].value is True, (
        "PA polish checkbox must default to True (symmetric with PQQA)"
    )

    _set_slider(at, "PA population", 8)
    _set_slider(at, "num_temps", 10)
    _set_slider(at, "sweeps_per_temp", 1)
    at.run()
    assert not at.exception, at.exception

    runs = [b for b in at.button if "Population Annealing" in b.label]
    assert runs, "Run Population Annealing button missing"
    runs[0].click()
    at.run()
    assert not at.exception, at.exception
    # Headline: PA's metric tiles (ESS, backend) should be on the page.
    metric_labels = [m.label for m in at.metric]
    assert "ESS" in metric_labels and any(lab == "backend" for lab in metric_labels), (
        f"PA-specific metric tiles missing; got labels={metric_labels!r}"
    )


def test_solve_page_blocks_pa_run_on_categorical_problem():
    """Regression: when the user picks PA and a categorical-relaxation
    problem (Coloring / QAP / NQueens / TSP / BGP) the Solve page must
    warn upfront and disable the "Run Population Annealing" button —
    never let them click Run only to eat a NotImplementedError traceback.

    Uses ``Coloring`` which uses a CategoricalRelaxation and is supported
    by PQQA. The assertion is two-fold: (a) the warning banner mentions
    PA being unavailable, (b) the Run button is disabled.
    """
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=120)
    at.session_state["problem_config"] = make_problem_config(
        "coloring", 10, num_category=3, degree=3
    )
    at.run()
    assert not at.exception, at.exception

    # Switch backend to PA.
    backend_radios = [r for r in at.sidebar.radio if "backend" in r.label.lower()]
    assert backend_radios, "Backend radio missing"
    pa_options = [opt for opt in backend_radios[0].options if "PA" in opt]
    if not pa_options:
        pytest.skip("PA backend not available in this build")
    backend_radios[0].set_value(pa_options[0])
    at.run()
    assert not at.exception, at.exception

    # A warning banner must explicitly mention PA being unavailable for
    # this problem. We look for the telltale phrase we added to the
    # capability probe; if the text shifts in future, update this match.
    warn_texts = [w.value for w in at.warning]
    matched = any(
        "Population Annealing is not available" in t
        or "Population Annealing is not available for this problem" in t
        for t in warn_texts
    )
    assert matched, (
        "PA capability warning banner missing on categorical problem + PA "
        f"backend; got warnings = {warn_texts!r}"
    )

    # The Run button is still rendered but must be disabled.
    runs = [b for b in at.button if "Population Annealing" in b.label]
    assert runs, "Run PA button missing entirely"
    assert runs[0].disabled, "Run PA button should be disabled on unsupported problem"


def test_solve_page_survives_old_qqa_without_population_annealing(monkeypatch):
    """Regression for ``module 'qqa' has no attribute 'population_annealing'``.

    Reproduces the production bug seen on Streamlit Community Cloud when
    pip resolved ``qqa==0.5.0`` from PyPI (which predates PA) instead of
    the in-tree wheel. The Solve page must:

    * not crash on import / first render,
    * not present PA as a selectable backend, and
    * surface a human-readable ``st.warning`` explaining the missing
      capability so the user knows to redeploy.
    """
    import qqa as _qqa  # noqa: PLC0415

    monkeypatch.delattr(_qqa, "population_annealing", raising=False)
    monkeypatch.delattr(_qqa, "PAResult", raising=False)

    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=60)
    at.session_state["problem_config"] = make_problem_config("ising1d", 6)
    at.run()
    assert not at.exception, at.exception

    backend_radios = [r for r in at.sidebar.radio if "backend" in r.label.lower()]
    assert backend_radios, "Backend radio missing"
    options = list(backend_radios[0].options)
    assert "PQQA" in options
    assert not any("PA" in opt for opt in options), (
        f"PA option must be hidden when qqa.population_annealing is absent; got {options!r}"
    )

    warnings = [getattr(w, "body", "") or getattr(w, "value", "") for w in at.warning]
    assert any("PA backend not available" in (b or "") for b in warnings), (
        f"Capability-missing warning not shown; warnings={warnings!r}"
    )


def test_visualize_page_survives_old_qqa_without_pa_result(monkeypatch):
    """Visualize page must not crash if the deployed qqa lacks ``PAResult``.

    Older wheels (pre-0.5.1) do not expose ``PAResult``; the page used to
    do a top-level ``from qqa import PAResult`` which would raise on
    import. The hardened page now degrades gracefully — it should render
    the empty state when no run is loaded, instead of crashing.
    """
    import qqa as _qqa  # noqa: PLC0415

    monkeypatch.delattr(_qqa, "PAResult", raising=False)

    at = AppTest.from_file(str(PAGE_DIR / "2_Visualize.py"), default_timeout=60)
    at.run()
    assert not at.exception, at.exception


def test_solve_runs_with_default_mis():
    """Regression for the duplicate-element-key crash on the default MIS
    problem.

    The previous "no-flash" tweak attached ``key="qqa_solve_dynamics"`` to
    a callback-driven ``st.empty().plotly_chart`` call. Each callback tick
    re-registered the same key inside the *same* script run and Streamlit
    raised ``StreamlitDuplicateElementKey``. This test reproduces the bug
    by driving Run on the **default** MIS configuration with a small
    epoch budget; if the duplicate-key crash returns it shows up here as
    ``at.exception``.
    """
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=120)
    # Mirror the Home page's default seeded session state.
    at.session_state["problem_config"] = make_problem_config("mis", 32, graph_d=3)
    at.run()
    assert not at.exception, at.exception

    _set_slider(at, "sol_size", 8)
    _set_slider(at, "epochs", 100)
    _set_slider(at, "UI update every", 10)
    at.run()
    assert not at.exception, at.exception

    runs = [b for b in at.button if "Run" in b.label]
    assert runs, "Run QQA button missing"
    runs[0].click()
    at.run()
    assert not at.exception, at.exception


def test_custom_problem_disabled_by_default_on_shared_deployments(monkeypatch):
    """Arbitrary Python execution must require an explicit operator opt-in."""
    monkeypatch.delenv("QQA_ALLOW_CUSTOM", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    toggles = [t for t in at.sidebar.toggle if "custom" in t.label.lower()]
    assert not toggles, "Custom-problem execution must be opt-in"
    captions = [caption.value for caption in at.sidebar.caption]
    assert any("disabled" in caption.lower() for caption in captions)


def test_visualize_tab_order_solution_first(tmp_path):
    """Solution must be the first tab on the Visualize page (user-requested
    information hierarchy: result first, dynamics second). We seed a tiny
    AnnealResult so the page renders past the early-return."""
    import sys

    sys.path.insert(0, str(APP.parent))
    from _common import build_problem as _build  # noqa: F811

    import qqa  # noqa: F811

    cfg = {"kind": "ising1d", "size": 6, "seed": 0, "device": "cpu", "extra": {}}
    problem = _build(cfg)
    result = qqa.anneal(
        problem, sol_size=4, num_epochs=10, learning_rate=0.1, device="cpu", verbose=False
    )

    at = AppTest.from_file(str(PAGE_DIR / "2_Visualize.py"), default_timeout=60)
    at.session_state["last_result"] = result
    at.session_state["last_problem"] = problem
    at.session_state["problem_config"] = cfg
    at.run()
    assert not at.exception, at.exception

    tab_labels = [t.label for t in at.tabs]
    assert tab_labels, "No tabs rendered on the Visualize page"
    assert tab_labels[0] == "Solution", f"Solution must be the first tab (got order: {tab_labels})"


def test_visualize_renders_3d_pca_and_diversity_with_population(tmp_path):
    """The new "3D PCA flow", "Diversity" and "Loss spectrogram" tabs must
    render without exception when a PopulationTracker is attached."""
    import sys

    sys.path.insert(0, str(APP.parent))
    from _common import build_problem as _build  # noqa: F811

    import qqa  # noqa: F811
    from qqa.callbacks import PopulationTracker  # noqa: F811

    cfg = {"kind": "mis", "size": 16, "seed": 0, "device": "cpu", "extra": {}}
    problem = _build(cfg)
    tracker = PopulationTracker(stride=2, record_x=True)
    result = qqa.anneal(
        problem,
        sol_size=8,
        num_epochs=12,
        learning_rate=0.1,
        device="cpu",
        verbose=False,
        callbacks=[tracker],
    )

    at = AppTest.from_file(str(PAGE_DIR / "2_Visualize.py"), default_timeout=60)
    at.session_state["last_result"] = result
    at.session_state["last_problem"] = problem
    at.session_state["last_pop_tracker"] = tracker
    at.session_state["problem_config"] = cfg
    at.run()
    assert not at.exception, at.exception
    labels = [t.label for t in at.tabs]
    for required in ("Solution-space PCA", "Diversity", "Loss spectrogram"):
        assert required in labels, f"missing tab {required!r} (got {labels})"

    # Belt and braces: the 3D PCA tab body wraps the PCA computation in a
    # try/except that surfaces failures via ``st.info``. If a future palette
    # / signature change reintroduces a KeyError, the exception is silently
    # rendered as "PCA could not be computed: ..." rather than failing the
    # smoke test. Walk every ``info`` message rendered on the page and make
    # sure no "could not be computed" string slipped through.
    info_texts = " ".join(getattr(m, "body", "") or "" for m in at.info)
    assert "could not be computed" not in info_texts.lower(), (
        f"a Visualize tab silently swallowed an error: {info_texts}"
    )


def test_solve_dynamics_separates_discrete_and_relaxed_best():
    """Regression: the per-replica chart used to plot the running discrete
    best (``state.best_obj``) on the same y-axis as the relaxed mean.  For
    losses spanning orders of magnitude the discrete line visually pinned
    to 0 (see ``tasks/lessons.md``).  We now plot the *relaxed* best of the
    current epoch on that line and surface the discrete best as its own
    metric tile.  This test asserts that both metric tiles are rendered
    after a tiny anneal run."""
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=120)
    at.session_state["problem_config"] = make_problem_config("ising1d", 8)
    at.run()
    assert not at.exception, at.exception

    _set_slider(at, "sol_size", 4)
    _set_slider(at, "epochs", 100)
    _set_slider(at, "UI update every", 10)
    at.run()

    runs = [b for b in at.button if "Run" in b.label]
    runs[0].click()
    at.run()
    assert not at.exception, at.exception

    metric_labels = [m.label for m in at.metric]
    assert any("best (discrete)" in lab for lab in metric_labels), (
        f"Expected a 'best (discrete)' metric tile; got {metric_labels!r}"
    )
    assert any("best (relaxed" in lab for lab in metric_labels), (
        f"Expected a 'best (relaxed, this epoch)' metric tile; got {metric_labels!r}"
    )


def test_plot_history_plotly_uses_qqa_theme():
    """``plot_history(backend='plotly')`` must produce a figure that matches
    the shared QQA plotly theme (transparent background, no fixed pixel
    width forcing horizontal scrolling, and a readable colorway)."""
    pytest.importorskip("plotly")
    from qqa.visualization import plot_history

    class _R:
        history = {
            "loss_mean": [0.0, -1.0, -2.0],
            "loss_std": [0.5, 0.4, 0.3],
            "penalty_mean": [1.0, 0.8, 0.6],
            "penalty_std": [0.1, 0.1, 0.1],
            "diversity": [0.2, 0.3, 0.4],
            "bg": [-2.0, -1.0, 0.1],
            "best_obj": [0.0, -0.5, -1.5],
        }

    fig = plot_history(_R(), backend="plotly", show=False)
    layout = fig.layout
    # Transparent canvas (so it composes onto any background).
    assert layout.paper_bgcolor in ("rgba(0,0,0,0)",)
    assert layout.plot_bgcolor in ("rgba(0,0,0,0)",)
    # No hard-coded pixel width — Streamlit's container should drive layout.
    assert layout.width is None, (
        f"plot_history figure pinned width={layout.width}; let containers size it"
    )
    # Centered title.
    assert layout.title.x == 0.5


def test_compare_page_renders_with_default_problem():
    """The Compare page must render in the default 'sweep' mode without
    crashing when a problem is already in session_state."""
    at = AppTest.from_file(str(PAGE_DIR / "3_Compare.py"), default_timeout=60)
    at.session_state["problem_config"] = make_problem_config("ising1d", 8)
    at.run()
    assert not at.exception, at.exception
    # Top-of-sidebar mode selector must be present.
    radio_labels = [r.label for r in at.sidebar.radio]
    assert any("Compare mode" in lab for lab in radio_labels), (
        f"Mode selector missing; got radios={radio_labels!r}"
    )


def test_compare_page_shootout_mode_runs_pqqa_vs_sa_vs_pa():
    """Switch to the shootout mode, click Run, and confirm that the
    summary metrics for all three backends (PQQA, SA, PA) render."""
    at = AppTest.from_file(str(PAGE_DIR / "3_Compare.py"), default_timeout=180)
    at.session_state["problem_config"] = make_problem_config("ising1d", 8)
    at.run()
    assert not at.exception, at.exception
    # Flip the mode selector to shootout. The radio label includes "PA"
    # since the v0.5.x rename — match by substring rather than literal.
    radios = [r for r in at.sidebar.radio if "Compare mode" in r.label]
    assert radios, "Compare mode radio missing"
    shootout_options = [opt for opt in radios[0].options if "shootout" in opt.lower()]
    assert shootout_options, f"shootout option missing; got {radios[0].options!r}"
    radios[0].set_value(shootout_options[0])
    at.run()
    assert not at.exception, at.exception

    for label, value in (
        ("PQQA epochs", 100),
        ("PQQA sol_size", 8),
        ("SA num_sweeps", 100),
        ("SA chains", 8),
        ("PA num_temps", 10),
        ("PA sweeps_per_temp", 2),
        ("PA population", 8),
    ):
        matches = [s for s in at.sidebar.slider if label in s.label]
        if matches:
            matches[0].set_value(value)
    at.run()

    runs = [b for b in at.button if "shootout" in b.label.lower() or "Run" in b.label]
    assert runs, f"Run shootout button missing; buttons={[b.label for b in at.button]!r}"
    runs[0].click()
    at.run()
    assert not at.exception, at.exception

    metric_labels = [m.label for m in at.metric]
    for needle in ("PQQA best_obj", "SA best_obj", "PA best_obj"):
        assert any(needle in lab for lab in metric_labels), (
            f"{needle} metric tile missing; got {metric_labels!r}"
        )


def test_solve_page_exposes_polish_and_warmstart_toggles():
    """The Solve page must surface the post-processing & warm-start toggles
    introduced in the v0.5 release. Polish defaults to ON; warm-start
    defaults to OFF (only useful for graph problems)."""
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=60)
    at.session_state["problem_config"] = make_problem_config("maxcut", 16, graph_d=3)
    at.run()
    assert not at.exception, at.exception
    toggle_labels = [t.label for t in at.sidebar.toggle]
    assert any("polish" in lab.lower() for lab in toggle_labels), (
        f"Polish toggle missing; got {toggle_labels!r}"
    )
    assert any(
        "warm-start" in lab.lower() or "warm start" in lab.lower() for lab in toggle_labels
    ), f"Warm-start toggle missing; got {toggle_labels!r}"
    polish_toggles = [t for t in at.sidebar.toggle if "polish" in t.label.lower()]
    assert polish_toggles[0].value is True, "Polish toggle should default to ON"
    warm_toggles = [
        t
        for t in at.sidebar.toggle
        if "warm-start" in t.label.lower() or "warm start" in t.label.lower()
    ]
    assert warm_toggles[0].value is False, "Warm-start toggle should default to OFF"


def test_home_page_lists_min_dominating_set_and_bgp():
    """The new problem catalog entries (MinimumDominatingSet,
    BalancedGraphPartition) must be selectable from the Home page so the
    UI exercises the same registry as the CLI."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    selectboxes = at.sidebar.selectbox
    family_select = next((s for s in selectboxes if "family" in s.label.lower()), None)
    problem_select = next((s for s in selectboxes if s.label == "Problem"), None)
    assert family_select is not None and problem_select is not None, (
        f"Family/Problem selectboxes missing; got {[s.label for s in selectboxes]!r}"
    )
    # The dropdown is populated with human-readable labels via
    # ``format_func``, so search the rendered strings rather than the raw
    # kind keys. Default family is "Graph (binary QUBO)".
    graph_options = [str(o).lower() for o in (problem_select.options or [])]
    assert any("dominating" in o for o in graph_options), (
        f"Minimum Dominating Set missing from Graph problems; got {graph_options!r}"
    )
    family_select.set_value("Categorical / assignment")
    at.run()
    assert not at.exception, at.exception
    problem_select = next(s for s in at.sidebar.selectbox if s.label == "Problem")
    cat_options = [str(o).lower() for o in (problem_select.options or [])]
    assert any("balanced" in o or "partition" in o for o in cat_options), (
        f"Balanced Graph Partition missing from Categorical problems; got {cat_options!r}"
    )
    # Physics catalog must list the v0.5 additions (PSpinGlass, RFIM).
    family_select = next(s for s in at.sidebar.selectbox if "family" in s.label.lower())
    family_select.set_value("Physics / spin")
    at.run()
    assert not at.exception, at.exception
    problem_select = next(s for s in at.sidebar.selectbox if s.label == "Problem")
    phys_options = [str(o).lower() for o in (problem_select.options or [])]
    assert any("p-spin" in o or "pspin" in o for o in phys_options), (
        f"p-spin glass missing from Physics problems; got {phys_options!r}"
    )
    assert any("rfim" in o or "random field" in o for o in phys_options), (
        f"Random Field Ising missing from Physics problems; got {phys_options!r}"
    )


def test_solve_runs_with_min_dominating_set_default():
    """End-to-end smoke for the new MinimumDominatingSet problem under
    the same UI flow used by every other graph QUBO."""
    at = AppTest.from_file(str(PAGE_DIR / "1_Solve.py"), default_timeout=120)
    at.session_state["problem_config"] = make_problem_config("min_dominating_set", 16, graph_d=3)
    at.run()
    assert not at.exception, at.exception
    _set_slider(at, "sol_size", 8)
    _set_slider(at, "epochs", 100)
    _set_slider(at, "UI update every", 10)
    at.run()
    assert not at.exception, at.exception
    runs = [b for b in at.button if "Run" in b.label]
    assert runs, "Run QQA button missing"
    runs[0].click()
    at.run()
    assert not at.exception, at.exception


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
