"""Runtime capability probes for optional exact solvers."""

from __future__ import annotations


def scip_available() -> bool:
    """Return whether the complete PySCIPOpt integration can be imported.

    Looking up only the package spec is insufficient: an incompatible binary
    wheel can be installed yet fail while importing SCIP or its nonlinear
    recipe. Auto-routing must never select a backend that cannot start.
    """
    try:
        from pyscipopt import Model, quicksum
        from pyscipopt.recipes.nonlinear import set_nonlinear_objective
    except (ImportError, OSError):
        return False
    return callable(Model) and callable(quicksum) and callable(set_nonlinear_objective)


__all__ = ["scip_available"]
