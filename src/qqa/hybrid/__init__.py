"""Hybrid solvers that combine QQA exploration with exact optimisation."""

from qqa.hybrid.capabilities import scip_available
from qqa.hybrid.scip import SCIPHybridResult, solve_qqa_scip
from qqa.hybrid.scip_model import SCIPExpressionError, SCIPModelResult, solve_spec_scip

__all__ = [
    "SCIPExpressionError",
    "SCIPHybridResult",
    "SCIPModelResult",
    "scip_available",
    "solve_qqa_scip",
    "solve_spec_scip",
]
