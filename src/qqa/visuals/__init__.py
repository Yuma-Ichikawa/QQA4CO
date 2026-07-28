"""Advanced, responsibility-separated visualisation implementation."""

from qqa.visuals._data import (
    constraint_rows,
    serialisable_summary,
    solution_rows,
    trajectory,
)
from qqa.visuals.dashboard import (
    plot_constraint_diagnostics,
    plot_result_dashboard,
    plot_variable_solution,
)

__all__ = [
    "constraint_rows",
    "plot_constraint_diagnostics",
    "plot_result_dashboard",
    "plot_variable_solution",
    "serialisable_summary",
    "solution_rows",
    "trajectory",
]
