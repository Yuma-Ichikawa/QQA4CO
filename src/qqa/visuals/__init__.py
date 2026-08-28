"""Advanced, responsibility-separated visualisation implementation."""

from qqa.visuals._data import (
    constraint_rows,
    serialisable_summary,
    solution_rows,
    trajectory,
)
from qqa.visuals.cockpit import plot_optimization_cockpit
from qqa.visuals.dashboard import (
    plot_constraint_diagnostics,
    plot_result_dashboard,
    plot_variable_solution,
)
from qqa.visuals.explorer import DecisionRecord, decision_explorer

__all__ = [
    "DecisionRecord",
    "constraint_rows",
    "decision_explorer",
    "plot_constraint_diagnostics",
    "plot_optimization_cockpit",
    "plot_result_dashboard",
    "plot_variable_solution",
    "serialisable_summary",
    "solution_rows",
    "trajectory",
]
