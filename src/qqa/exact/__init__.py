"""Optional CP/SAT and global-optimisation runtimes."""

from qqa.exact.cp import CPResult, solve_cp_model_ir
from qqa.exact.sat import SATResult, solve_sat_model_ir
from qqa.exact.spatial import (
    BilinearTerm,
    McCormickEnvelope,
    SpatialBranchAndBoundResult,
    spatial_branch_and_bound,
)

__all__ = [
    "BilinearTerm",
    "CPResult",
    "McCormickEnvelope",
    "SATResult",
    "SpatialBranchAndBoundResult",
    "solve_sat_model_ir",
    "solve_cp_model_ir",
    "spatial_branch_and_bound",
]
