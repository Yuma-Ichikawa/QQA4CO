"""Shared test fixtures / helpers for the QQA4CO test suite.

Anything living here is auto-loaded by pytest for every test module under
``tests/``. We deliberately keep the surface small: two path constants
and two pure-Python helpers that centralise the patterns repeated across
``test_gui_apptest.py`` (problem-config dicts, slider wiring).

Streamlit is *not* imported at module import time — any test that needs
``AppTest`` still guards it with ``pytest.importorskip("streamlit")``
at the top of its own file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Repository root resolved relative to this file. All tests are run from
#: the repo root so the absolute path works equally well in CI and locally.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Streamlit entry points — exported so individual tests no longer re-derive
#: them from ``__file__``.
APP = REPO_ROOT / "app" / "streamlit_app.py"
PAGE_DIR = APP.parent / "pages"


def make_problem_config(
    kind: str,
    size: int,
    *,
    seed: int = 0,
    device: str = "cpu",
    **extra: Any,
) -> dict[str, Any]:
    """Build a ``problem_config`` dict the way the Streamlit pages expect.

    Every ``AppTest`` that jumps straight to a page (Solve / Visualize /
    Compare) has to seed ``session_state["problem_config"]``; this helper
    trims 7-line literals down to a single call::

        at.session_state["problem_config"] = make_problem_config(
            "mis", 32, graph_d=3
        )

    Any ``extra`` kwargs are forwarded into the nested ``extra`` dict, which
    is where every page-specific parameter (``graph_d``, ``num_category``,
    ``source``, …) lives.
    """
    return {
        "kind": kind,
        "size": size,
        "seed": seed,
        "device": device,
        "extra": dict(extra),
    }


def set_slider(at, label_fragment: str, value) -> None:
    """Set the sidebar slider whose label contains ``label_fragment``.

    Lifted from ``test_gui_apptest.py`` so other AppTest-based suites can
    reuse it. Raises ``AssertionError`` (not a silent noop) when no slider
    matches — each Solve-page slider is load-bearing and a missing match
    almost always means a UI regression rather than a flaky selector.
    """
    matches = [s for s in at.sidebar.slider if label_fragment in s.label]
    assert matches, f"No slider whose label contains {label_fragment!r}"
    matches[0].set_value(value)
