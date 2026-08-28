"""Explicitly opt-in hybrid solvers combining QQA with exact optimisation.

The package facade is lazy so lightweight capability/configuration access does
not import PySCIPOpt-facing implementations before a solver is requested.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ExactFeedback": ("qqa.hybrid.feedback", "ExactFeedback"),
    "ExactFeedbackBus": ("qqa.hybrid.feedback", "ExactFeedbackBus"),
    "LinearCut": ("qqa.hybrid.feedback", "LinearCut"),
    "ExactBackendResult": ("qqa.hybrid.exact", "ExactBackendResult"),
    "CoreSurrogate": ("qqa.hybrid.surrogate", "CoreSurrogate"),
    "QQAHeuristic": ("qqa.hybrid.scip_heuristic", "QQAHeuristic"),
    "QQAHeuristicConfig": ("qqa.hybrid.heuristic_types", "QQAHeuristicConfig"),
    "QQAHeuristicStats": ("qqa.hybrid.heuristic_types", "QQAHeuristicStats"),
    "NeighborhoodBudget": ("qqa.hybrid.neighborhood_portfolio", "NeighborhoodBudget"),
    "GraphInducedNeighborhoodGenerator": (
        "qqa.hybrid.neighborhood_portfolio",
        "GraphInducedNeighborhoodGenerator",
    ),
    "LocalBranchingNeighborhoodGenerator": (
        "qqa.hybrid.neighborhood_portfolio",
        "LocalBranchingNeighborhoodGenerator",
    ),
    "NeighborhoodPortfolio": (
        "qqa.hybrid.neighborhood_portfolio",
        "NeighborhoodPortfolio",
    ),
    "NeighborhoodStats": ("qqa.hybrid.neighborhood_portfolio", "NeighborhoodStats"),
    "ScoredNeighborhoodGenerator": (
        "qqa.hybrid.neighborhood_portfolio",
        "ScoredNeighborhoodGenerator",
    ),
    "TrustRegionNeighborhoodGenerator": (
        "qqa.hybrid.neighborhood_portfolio",
        "TrustRegionNeighborhoodGenerator",
    ),
    "SCIPExpressionError": ("qqa.hybrid.scip_model", "SCIPExpressionError"),
    "SCIPHybridResult": ("qqa.hybrid.scip", "SCIPHybridResult"),
    "SCIPModelResult": ("qqa.hybrid.scip_model", "SCIPModelResult"),
    "build_core_surrogate": ("qqa.hybrid.surrogate", "build_core_surrogate"),
    "cpsat_available": ("qqa.hybrid.capabilities", "cpsat_available"),
    "cuopt_available": ("qqa.hybrid.capabilities", "cuopt_available"),
    "generate_surrogate_candidates": (
        "qqa.hybrid.surrogate",
        "generate_surrogate_candidates",
    ),
    "include_qqa_heuristic": ("qqa.hybrid.scip_heuristic", "include_qqa_heuristic"),
    "highs_available": ("qqa.hybrid.capabilities", "highs_available"),
    "scip_available": ("qqa.hybrid.capabilities", "scip_available"),
    "solve_cpsat_algebraic": ("qqa.hybrid.exact", "solve_cpsat_algebraic"),
    "solve_exact_algebraic": ("qqa.hybrid.exact", "solve_exact_algebraic"),
    "solve_highs_algebraic": ("qqa.hybrid.exact", "solve_highs_algebraic"),
    "solve_qqa_scip": ("qqa.hybrid.scip", "solve_qqa_scip"),
    "solve_scip_algebraic": ("qqa.hybrid.exact", "solve_scip_algebraic"),
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
