"""Hybrid solvers that combine QQA exploration with exact optimisation."""

from qqa.hybrid.scip import SCIPHybridResult, solve_qqa_scip

__all__ = ["SCIPHybridResult", "solve_qqa_scip"]
