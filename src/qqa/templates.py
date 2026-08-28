"""Validated domain templates that compile to portable typed ModelIR models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from qqa.model import ConstraintIR, LinearFactor, ModelIR, ModelMetadata, ObjectiveIR, VariableBlock
from qqa.model.native import NoOverlapFactor


@dataclass(frozen=True, slots=True)
class DomainTemplate:
    name: str
    version: str
    description: str
    tags: tuple[str, ...]
    builder: Callable[..., ModelIR]


def build_facility_location_template(
    opening_costs: Sequence[float], assignment_costs: Sequence[Sequence[float]]
) -> ModelIR:
    opening = torch.as_tensor(opening_costs, dtype=torch.float64).reshape(-1)
    assignment = torch.as_tensor(assignment_costs, dtype=torch.float64)
    if assignment.ndim != 2 or assignment.shape[1] != len(opening) or not len(opening):
        raise ValueError("assignment_costs must have shape (customers, facilities).")
    if not torch.isfinite(opening).all() or not torch.isfinite(assignment).all():
        raise ValueError("Template costs must be finite.")
    facilities = len(opening)
    customers = assignment.shape[0]
    assignment_indices = torch.arange(
        facilities, facilities + customers * facilities, dtype=torch.long
    ).reshape(customers, facilities)
    objective_indices = torch.arange(facilities + customers * facilities)
    objective_weights = torch.cat((opening, assignment.reshape(-1)))
    constraints = []
    for customer, row in enumerate(assignment_indices):
        constraints.append(
            ConstraintIR(
                f"customer_{customer}_assigned",
                ObjectiveIR((LinearFactor(row, torch.ones(facilities)),)),
                "==",
                1.0,
            )
        )
        for facility, assignment_index in enumerate(row.tolist()):
            constraints.append(
                ConstraintIR(
                    f"customer_{customer}_facility_{facility}_open",
                    ObjectiveIR(
                        (
                            LinearFactor(
                                torch.tensor([assignment_index, facility]),
                                torch.tensor([1.0, -1.0]),
                            ),
                        )
                    ),
                    "<=",
                    0.0,
                )
            )
    return ModelIR(
        (
            VariableBlock("open", "binary", (facilities,)),
            VariableBlock("assign", "binary", (customers, facilities)),
        ),
        ObjectiveIR((LinearFactor(objective_indices, objective_weights),)),
        tuple(constraints),
        metadata=ModelMetadata(
            "facility-location",
            problem_class="facility-location",
            source_format="qqa-template-v1",
        ),
    )


def build_single_machine_template(
    durations: Sequence[int], *, horizon: int | None = None
) -> ModelIR:
    lengths = torch.as_tensor(durations, dtype=torch.float64).reshape(-1)
    if not len(lengths) or torch.any(lengths <= 0) or not torch.equal(lengths, lengths.round()):
        raise ValueError("durations must contain positive integers.")
    default_horizon = int(lengths.sum().item())
    horizon = default_horizon if horizon is None else int(horizon)
    if horizon < int(lengths.max().item()):
        raise ValueError("horizon is too short for at least one job.")
    jobs = len(lengths)
    makespan = jobs
    constraints = [
        ConstraintIR(
            "machine_no_overlap",
            ObjectiveIR((NoOverlapFactor(torch.arange(jobs), lengths),)),
            "<=",
            0.0,
        )
    ]
    for job, duration in enumerate(lengths.tolist()):
        constraints.append(
            ConstraintIR(
                f"job_{job}_completion",
                ObjectiveIR(
                    (
                        LinearFactor(
                            torch.tensor([job, makespan]),
                            torch.tensor([1.0, -1.0]),
                        ),
                    ),
                    constant=float(duration),
                ),
                "<=",
                0.0,
            )
        )
    return ModelIR(
        (VariableBlock("schedule", "integer", (jobs + 1,), 0, horizon),),
        ObjectiveIR((LinearFactor(torch.tensor([makespan]), torch.tensor([1.0])),)),
        tuple(constraints),
        metadata=ModelMetadata(
            "single-machine-scheduling",
            problem_class="scheduling",
            source_format="qqa-template-v1",
        ),
    )


_TEMPLATES = {
    "facility-location": DomainTemplate(
        "facility-location",
        "1",
        "Uncapacitated facility opening and customer assignment.",
        ("facility", "assignment", "binary"),
        build_facility_location_template,
    ),
    "single-machine-scheduling": DomainTemplate(
        "single-machine-scheduling",
        "1",
        "Non-preemptive single-machine scheduling with makespan minimisation.",
        ("scheduling", "cp", "integer"),
        build_single_machine_template,
    ),
}


def available_templates() -> tuple[DomainTemplate, ...]:
    return tuple(_TEMPLATES[name] for name in sorted(_TEMPLATES))


def build_template(name: str, /, **parameters: Any) -> ModelIR:
    try:
        template = _TEMPLATES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown template {name!r}; choose from {sorted(_TEMPLATES)}.") from exc
    return template.builder(**parameters)


__all__ = [
    "DomainTemplate",
    "available_templates",
    "build_facility_location_template",
    "build_single_machine_template",
    "build_template",
]
