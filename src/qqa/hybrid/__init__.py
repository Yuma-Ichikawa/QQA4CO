"""Hybrid solvers that combine QQA exploration with exact optimisation."""

from qqa.hybrid.capabilities import scip_available
from qqa.hybrid.scip import SCIPHybridResult, solve_qqa_scip
from qqa.hybrid.scip_heuristic import (
    QQAHeuristic,
    QQAHeuristicConfig,
    QQAHeuristicStats,
    include_qqa_heuristic,
)
from qqa.hybrid.scip_model import SCIPExpressionError, SCIPModelResult, solve_spec_scip
from qqa.hybrid.surrogate import CoreSurrogate, build_core_surrogate, generate_surrogate_candidates

__all__ = [
    "SCIPExpressionError",
    "SCIPHybridResult",
    "SCIPModelResult",
    "QQAHeuristicConfig",
    "QQAHeuristic",
    "QQAHeuristicStats",
    "CoreSurrogate",
    "build_core_surrogate",
    "generate_surrogate_candidates",
    "include_qqa_heuristic",
    "scip_available",
    "solve_qqa_scip",
    "solve_spec_scip",
]
