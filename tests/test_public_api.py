"""Regression guard for the top-level ``qqa`` package surface.

The point of these tests is to catch silent breakage of the *advertised*
import paths. Every name in :data:`qqa.__all__` is documented in the
README / mkdocs site and used by external scripts, so a removal would
be a breaking change.
"""

from __future__ import annotations

import importlib

import pytest

import qqa

# Names that have been part of the public API since at least 0.3.0.
# Removing one of these from ``qqa`` (without a deprecation cycle) is a
# breaking change.
LEGACY_PUBLIC_NAMES = [
    "AnnealResult",
    "anneal",
    "fix_seed",
    "generate_graph",
    "MaximumIndependentSet",
    "MaxClique",
    "MaxCut",
    "VertexCover",
    "GraphBisection",
    "Coloring",
    "BalancedGraphPartition",
    "Ising1D",
    "EdwardsAnderson",
    "SherringtonKirkpatrick",
    "BinaryPerceptron",
    "HopfieldMemory",
    "Knapsack",
    "NumberPartitioning",
    "MaxSAT3",
    "TSP",
    "QAP",
    "NQueens",
    "UserProblem",
    "user_problem_from_source",
    "load_problem_from_file",
    "BinaryRelaxation",
    "BinaryInstanceRelaxation",
    "SpinRelaxation",
    "CategoricalRelaxation",
    "LinearBGSchedule",
    "COProblem",
    "QUBOProblem",
    "SpinProblem",
]

# Names re-exported in the post-0.3.0 cleanup. They are *additive* on
# top of ``LEGACY_PUBLIC_NAMES`` and must stay reachable from the top
# level so that contributor docs can write ``from qqa import Callback``.
NEW_PUBLIC_NAMES = [
    "Callback",
    "CallbackState",
    "HistoryRecorder",
    "AutoDivTuner",
    "PopulationTracker",
    "TrajectoryTracker",
    "Relaxation",
]


@pytest.mark.parametrize("name", LEGACY_PUBLIC_NAMES + NEW_PUBLIC_NAMES)
def test_top_level_export_exists(name: str) -> None:
    assert hasattr(qqa, name), f"qqa.{name} must be importable from the top level"
    assert name in qqa.__all__, f"qqa.{name} must be listed in qqa.__all__"


def test_callback_re_export_is_the_same_object() -> None:
    """The re-exports must be identical objects, not look-alike copies."""
    from qqa.callbacks import (
        AutoDivTuner,
        Callback,
        CallbackState,
        HistoryRecorder,
        PopulationTracker,
        TrajectoryTracker,
    )

    assert qqa.Callback is Callback
    assert qqa.CallbackState is CallbackState
    assert qqa.HistoryRecorder is HistoryRecorder
    assert qqa.AutoDivTuner is AutoDivTuner
    assert qqa.PopulationTracker is PopulationTracker
    assert qqa.TrajectoryTracker is TrajectoryTracker


def test_relaxation_protocol_re_export() -> None:
    from qqa.relaxation import Relaxation

    assert qqa.Relaxation is Relaxation


def test_version_is_string_and_matches_metadata() -> None:
    """``qqa.__version__`` must come from ``importlib.metadata`` so it
    cannot drift from the wheel metadata in ``pyproject.toml``."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _metadata_version

    assert isinstance(qqa.__version__, str)
    assert qqa.__version__  # non-empty

    try:
        meta = _metadata_version("qqa")
    except PackageNotFoundError:
        # Editable install without metadata — accept the documented fallback.
        assert qqa.__version__ == "0.0.0+unknown"
    else:
        assert qqa.__version__ == meta, (
            f"qqa.__version__={qqa.__version__!r} does not match "
            f"importlib.metadata.version('qqa')={meta!r}; the two have "
            "drifted, which means a hand-edited string slipped into "
            "src/qqa/__init__.py. Restore the importlib.metadata source."
        )


def test_logging_module_imports_and_returns_qqa_logger() -> None:
    """The opt-in logging helper does not configure handlers but must
    return a logger rooted at ``qqa``."""
    mod = importlib.import_module("qqa._logging")
    log = mod.get_logger()
    assert log.name == "qqa"
    # Child loggers via dotted name.
    child = mod.get_logger("qqa.subpackage")
    assert child.name == "qqa.subpackage"
