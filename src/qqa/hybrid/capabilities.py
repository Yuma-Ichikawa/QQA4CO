"""Runtime capability probes for optional exact solvers."""

from __future__ import annotations

import subprocess
import sys
from functools import cache

_PROBES = {
    "scip": (
        "from pyscipopt import Model, quicksum; "
        "from pyscipopt.recipes.nonlinear import set_nonlinear_objective; "
        "assert callable(Model) and callable(quicksum) and callable(set_nonlinear_objective)"
    ),
    "highs": "import highspy; assert hasattr(highspy, 'Highs')",
    "cpsat": "from ortools.sat.python import cp_model; assert hasattr(cp_model, 'CpModel')",
    "cuopt": "import cuopt",
}


@cache
def _available(backend: str) -> bool:
    """Probe a native backend without loading its shared libraries here.

    Some optional solver wheels bundle different builds of the same native
    library (notably HiGHS). Loading both into one interpreter can therefore
    make the second import fail even though each backend works independently.
    Exact solves already use isolated workers; capability discovery follows
    the same boundary and caches its result.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBES[backend]],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30.0,
        )
    except (KeyError, OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def scip_available() -> bool:
    """Return whether the complete PySCIPOpt integration can be imported.

    Looking up only the package spec is insufficient: an incompatible binary
    wheel can be installed yet fail while importing SCIP or its nonlinear
    recipe. Auto-routing must never select a backend that cannot start.
    """
    return _available("scip")


def highs_available() -> bool:
    return _available("highs")


def cpsat_available() -> bool:
    return _available("cpsat")


def cuopt_available() -> bool:
    return _available("cuopt")


__all__ = ["cpsat_available", "cuopt_available", "highs_available", "scip_available"]
