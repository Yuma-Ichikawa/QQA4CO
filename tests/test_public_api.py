"""Regression guard for the top-level ``qqa`` package surface.

The point of these tests is to catch silent breakage of the *advertised*
import paths. Every name in :data:`qqa.__all__` is documented in the
README / mkdocs site and used by external scripts, so a removal would
be a breaking change.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

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
    # Post-0.4.0 additions.
    "MinimumDominatingSet",
    "PSpinGlass",
    "RandomFieldIsing",
    "PAResult",
    "SAResult",
    "enable_tf32",
    "resolve_device",
    "polish",
    "population_annealing",
    "simulated_annealing",
    "warmstart",
    "MaxCliqueInstance",
    "MaxCutInstance",
    "MaximumIndependentSetInstance",
    # Mixed binary/integer/real modelling API.
    "Binary",
    "Integer",
    "Real",
    "BinaryVariable",
    "IntegerVariable",
    "RealVariable",
    "VariableSpace",
    "Constraint",
    "MixedProblem",
    "MixedRelaxation",
    "solve_mixed",
    "save_html_report",
    "build_portfolio_pareto",
    "plot_pareto_diagnostics",
]


def test_metadata_only_import_does_not_load_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import qqa; print(qqa.__version__); print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert completed.stdout.splitlines()[-1] == "False"


def test_tex_submodule_remains_available_from_package_root() -> None:
    """Preserve the historical ``qqa.tex.DEFAULT_*`` access pattern."""
    assert isinstance(qqa.tex.DEFAULT_BASE_URL, str)
    assert isinstance(qqa.tex.DEFAULT_MODEL, str)


@pytest.mark.parametrize("name", LEGACY_PUBLIC_NAMES + NEW_PUBLIC_NAMES)
def test_top_level_export_exists(name: str) -> None:
    assert hasattr(qqa, name), f"qqa.{name} must be importable from the top level"
    assert name in qqa.__all__, f"qqa.{name} must be listed in qqa.__all__"


def test_public_api_set_matches_dunder_all() -> None:
    """``__all__`` must agree with what is reachable as ``qqa.<name>``.

    Catches the common slip of adding a class to ``__all__`` while
    forgetting to import it (or vice versa).
    """

    advertised = set(qqa.__all__) - {"__version__"}
    actually_present = {n for n in advertised if hasattr(qqa, n)}
    missing = sorted(advertised - actually_present)
    assert not missing, (
        f"{missing!r} is listed in qqa.__all__ but not importable from "
        "the top level — fix the import in src/qqa/__init__.py."
    )


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


def test_auto_device_is_a_valid_public_solver_device() -> None:
    resolved = qqa.resolve_device("auto")
    assert str(resolved).split(":")[0] in {"cpu", "cuda", "mps"}
    assert qqa.resolve_device("cpu") == "cpu"


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


def test_plain_import_does_not_activate_optional_solver_modules() -> None:
    code = """
import json
import sys
import qqa

optional = sorted(
    name for name in sys.modules
    if name.startswith(("qqa.hybrid", "qqa.benchmarking", "qqa.io"))
    or name.startswith(("pyscipopt", "pyqplib"))
)
print(json.dumps(optional))
"""
    environment = dict(os.environ)
    source = str((Path(__file__).parents[1] / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source, environment.get("PYTHONPATH", "")) if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == []


def test_optional_legacy_exports_are_lazy_and_not_in_pure_dunder_all() -> None:
    assert "solve_qqa_scip" not in qqa.__all__
    assert qqa.solve_qqa_scip.__module__ == "qqa.hybrid.scip"


def test_hybrid_configuration_does_not_load_scip_plugin() -> None:
    code = """
import json
import sys
from qqa.hybrid import QQAHeuristicConfig

QQAHeuristicConfig()
loaded = sorted(
    name for name in sys.modules
    if name == "qqa.hybrid.scip_heuristic" or name.startswith("pyscipopt")
)
print(json.dumps(loaded))
"""
    environment = dict(os.environ)
    source = str((Path(__file__).parents[1] / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source, environment.get("PYTHONPATH", "")) if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == []


def test_scip_heuristic_registration_module_does_not_import_torch() -> None:
    code = """
import json
import sys
import qqa.hybrid.scip_heuristic

loaded = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch."))
print(json.dumps(loaded))
"""
    environment = dict(os.environ)
    source = str((Path(__file__).parents[1] / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source, environment.get("PYTHONPATH", "")) if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == []


def test_lightweight_runtime_io_and_presolve_facades_do_not_import_torch() -> None:
    code = """
import json
import sys
import qqa.io
import qqa.presolve
import qqa.runtime

loaded = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch."))
print(json.dumps(loaded))
"""
    environment = dict(os.environ)
    source = str((Path(__file__).parents[1] / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source, environment.get("PYTHONPATH", "")) if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == []


def test_benchmark_facade_is_lazy() -> None:
    code = """
import json
import sys
import qqa.benchmarking

loaded = sorted(
    name for name in sys.modules
    if name.startswith(("qqa.benchmarking.algebraic_runner", "qqa.benchmarking.metrics"))
    or name.startswith(("pyscipopt", "pyqplib"))
)
print(json.dumps(loaded))
"""
    environment = dict(os.environ)
    source = str((Path(__file__).parents[1] / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source, environment.get("PYTHONPATH", "")) if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == []


def test_benchmark_cli_registration_does_not_load_solver_integrations() -> None:
    code = """
import json
import sys
from qqa.cli import build_parser

build_parser().parse_args(["benchmark", "fetch", "miplib", "--output", "public-data"])
loaded = sorted(
    name for name in sys.modules
    if name.startswith(("qqa.benchmarking.algebraic_runner", "qqa.hybrid.scip_heuristic"))
    or name.startswith(("pyscipopt", "pyqplib"))
)
print(json.dumps(loaded))
"""
    environment = dict(os.environ)
    source = str((Path(__file__).parents[1] / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source, environment.get("PYTHONPATH", "")) if item
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(completed.stdout) == []
