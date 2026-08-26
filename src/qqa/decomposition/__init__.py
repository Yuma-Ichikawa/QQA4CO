"""Discrete proposal and continuous-completion decomposition."""

from qqa.decomposition.completion import (
    CompletionResult,
    complete_integer_assignment,
    complete_integer_assignment_dive,
    create_completion_template,
)

__all__ = [
    "CompletionResult",
    "complete_integer_assignment",
    "complete_integer_assignment_dive",
    "create_completion_template",
]
