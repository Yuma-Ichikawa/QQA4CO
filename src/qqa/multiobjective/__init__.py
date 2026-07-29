"""Parallel multi-objective optimisation and Pareto visualisation."""

from qqa.multiobjective.problem import MultiObjectiveProblem, Objective
from qqa.multiobjective.solver import ParetoResult, pareto_anneal
from qqa.multiobjective.visualization import plot_pareto, plot_pareto_diagnostics

__all__ = [
    "MultiObjectiveProblem",
    "Objective",
    "ParetoResult",
    "pareto_anneal",
    "plot_pareto",
    "plot_pareto_diagnostics",
]
