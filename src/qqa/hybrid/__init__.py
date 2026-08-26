"""Explicitly opt-in hybrid solvers combining QQA with exact optimisation.

The package facade is lazy so lightweight capability/configuration access does
not import PySCIPOpt-facing implementations before a solver is requested.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CoreSurrogate": ("qqa.hybrid.surrogate", "CoreSurrogate"),
    "QQAHeuristic": ("qqa.hybrid.scip_heuristic", "QQAHeuristic"),
    "QQAHeuristicConfig": ("qqa.hybrid.heuristic_types", "QQAHeuristicConfig"),
    "QQAHeuristicStats": ("qqa.hybrid.heuristic_types", "QQAHeuristicStats"),
    "SCIPExpressionError": ("qqa.hybrid.scip_model", "SCIPExpressionError"),
    "SCIPHybridResult": ("qqa.hybrid.scip", "SCIPHybridResult"),
    "SCIPModelResult": ("qqa.hybrid.scip_model", "SCIPModelResult"),
    "build_core_surrogate": ("qqa.hybrid.surrogate", "build_core_surrogate"),
    "generate_surrogate_candidates": (
        "qqa.hybrid.surrogate",
        "generate_surrogate_candidates",
    ),
    "include_qqa_heuristic": ("qqa.hybrid.scip_heuristic", "include_qqa_heuristic"),
    "scip_available": ("qqa.hybrid.capabilities", "scip_available"),
    "solve_qqa_scip": ("qqa.hybrid.scip", "solve_qqa_scip"),
    "solve_spec_scip": ("qqa.hybrid.scip_model", "solve_spec_scip"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
