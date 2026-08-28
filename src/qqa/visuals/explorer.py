"""Counterfactual and archive-stability data for a Decision Explorer UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from qqa.model.ir import ModelIR, VariableDomain


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    index: int
    value: float
    archive_stability: float | None
    alternative_value: float | None
    objective_delta: float | None
    violation_delta: float | None


def _total_violation(model: ModelIR, values: torch.Tensor) -> float:
    return float(
        sum(
            violation.reshape(-1)[0].item()
            for violation in model.constraint_violations(values).values()
        )
    )


def decision_explorer(result: Any, model: ModelIR | None = None) -> list[dict[str, Any]]:
    """Return JSON-ready stability and one-coordinate counterfactual records."""
    if result.solution is None:
        raise ValueError("Decision exploration requires a primal solution.")
    solution = result.solution.detach().to(torch.float64).reshape(-1)
    archive = getattr(result, "archive", None)
    archive_solutions = None if archive is None else archive.solutions()
    if archive_solutions is not None:
        archive_solutions = archive_solutions.to(solution).reshape(len(archive_solutions), -1)
    domains = []
    if model is not None:
        for block in model.variables:
            domains.extend([block.domain] * block.size)
    baseline_objective = (
        float(model.objective_values(solution)[0].item()) if model is not None else None
    )
    baseline_violation = _total_violation(model, solution) if model is not None else None
    records = []
    for index, value in enumerate(solution.tolist()):
        stability = (
            None
            if archive_solutions is None
            else float((archive_solutions[:, index] == value).to(torch.float64).mean().item())
        )
        alternative = None
        objective_delta = None
        violation_delta = None
        if model is not None and index < len(domains):
            if domains[index] is VariableDomain.BINARY:
                alternative = 1.0 - value
            elif domains[index] is VariableDomain.SPIN:
                alternative = -value
            if alternative is not None:
                assert baseline_objective is not None and baseline_violation is not None
                counterfactual = solution.clone()
                counterfactual[index] = alternative
                objective_delta = (
                    float(model.objective_values(counterfactual)[0].item()) - baseline_objective
                )
                violation_delta = _total_violation(model, counterfactual) - baseline_violation
        records.append(
            asdict(
                DecisionRecord(
                    index,
                    float(value),
                    stability,
                    alternative,
                    objective_delta,
                    violation_delta,
                )
            )
        )
    return records


__all__ = ["DecisionRecord", "decision_explorer"]
