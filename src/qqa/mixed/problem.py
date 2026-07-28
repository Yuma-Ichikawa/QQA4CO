"""Declarative mixed-integer/nonlinear optimisation problems."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from qqa.mixed.relaxation import MixedRelaxation
from qqa.mixed.variables import VariableSpace, VariableSpec
from qqa.problems.base import COProblem

NamedTensors = Mapping[str, torch.Tensor]
BatchFunction = Callable[[NamedTensors], torch.Tensor]
ConstraintSense = Literal["<=", ">=", "=="]


def _batch_vector(value, *, batch_size: int, label: str, like: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=like.device, dtype=like.dtype)
    if tensor.ndim == 0 and batch_size == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(
            f"{label} must return shape ({batch_size},), got {tuple(tensor.shape)}. "
            "Do not reduce across the leading population dimension."
        )
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{label} returned NaN or infinity.")
    return tensor


@dataclass(frozen=True, slots=True)
class Constraint:
    """A differentiable scalar constraint evaluated for every replica.

    Args:
        function: Callable receiving named batched tensors.
        sense: One of ``"<="``, ``">="``, or ``"=="``.
        rhs: Right-hand side.
        weight: Squared-violation weight added to the optimisation loss.
        scale: Characteristic unit used to normalise the violation.
        tolerance: Raw-unit feasibility tolerance used for reporting.
        name: Stable label shown in diagnostics and reports.
    """

    function: BatchFunction
    sense: ConstraintSense = "<="
    rhs: float = 0.0
    weight: float = 100.0
    scale: float = 1.0
    tolerance: float = 1e-4
    name: str = "constraint"

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("Constraint function must be callable.")
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(f"Unknown constraint sense {self.sense!r}.")
        for field_name in ("rhs", "weight", "scale", "tolerance"):
            try:
                value = float(getattr(self, field_name))
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Constraint {field_name} must be a real number.") from exc
            if not math.isfinite(value):
                raise ValueError(f"Constraint {field_name} must be finite, got {value}.")
            object.__setattr__(self, field_name, value)
        if self.weight <= 0:
            raise ValueError(f"Constraint weight must be > 0, got {self.weight}.")
        if self.scale <= 0:
            raise ValueError(f"Constraint scale must be > 0, got {self.scale}.")
        if self.tolerance < 0:
            raise ValueError(f"Constraint tolerance must be >= 0, got {self.tolerance}.")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Constraint name must be a non-empty string.")

    def violation(self, lhs: torch.Tensor) -> torch.Tensor:
        """Return non-negative raw violation in the constraint's units."""
        residual = lhs - self.rhs
        if self.sense == "<=":
            return residual.clamp_min(0.0)
        if self.sense == ">=":
            return (-residual).clamp_min(0.0)
        return residual.abs()


class MixedProblem(COProblem):
    """A GPU-vectorised optimisation model with heterogeneous variables.

    ``objective`` and constraint functions receive a mapping from declared
    names to tensors. The leading dimension is always the parallel population.
    All objectives are minimised.
    """

    def __init__(
        self,
        variables: Sequence[VariableSpec],
        objective: BatchFunction,
        *,
        constraints: Sequence[Constraint] = (),
        name: str = "mixed-problem",
        objective_label: str = "objective",
        objective_unit: str = "",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if not callable(objective):
            raise TypeError("objective must be callable.")
        if not name:
            raise ValueError("name must not be empty.")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"dtype must be torch.float32 or torch.float64, got {dtype}.")
        self.space = VariableSpace(tuple(variables))
        self.variables = self.space.variables
        self.constraints = tuple(constraints)
        if any(not isinstance(constraint, Constraint) for constraint in self.constraints):
            raise TypeError("constraints must contain Constraint instances.")
        constraint_names = [constraint.name for constraint in self.constraints]
        if len(set(constraint_names)) != len(constraint_names):
            raise ValueError("Constraint names must be unique.")

        self.objective = objective
        self.name = name
        self.objective_label = objective_label
        self.objective_unit = objective_unit
        self.dtype = dtype
        # ``solve_mixed`` may calibrate this once before annealing so
        # constraint units cannot be dwarfed by a large monetary/scientific
        # objective. Direct ``loss_fn`` users retain the exact declared
        # weights because the default is one.
        self.penalty_multiplier = 1.0
        self.num_vars = self.space.dimension
        self.num_nodes = self.space.dimension
        self.relaxation = MixedRelaxation(self.space)

    def objective_values(self, values: torch.Tensor) -> torch.Tensor:
        values = self._ensure_batched(values)
        return _batch_vector(
            self.objective(self.space.unpack(values)),
            batch_size=values.shape[0],
            label="objective",
            like=values,
        )

    def constraint_values(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self._ensure_batched(values)
        named = self.space.unpack(values)
        return {
            constraint.name: _batch_vector(
                constraint.function(named),
                batch_size=values.shape[0],
                label=f"constraint {constraint.name!r}",
                like=values,
            )
            for constraint in self.constraints
        }

    def constraint_violations(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        lhs = self.constraint_values(values)
        return {
            constraint.name: constraint.violation(lhs[constraint.name])
            for constraint in self.constraints
        }

    def constraint_penalty(self, values: torch.Tensor) -> torch.Tensor:
        """Return the weighted squared constraint penalty per replica."""
        values = self._ensure_batched(values)
        total = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
        lhs = self.constraint_values(values)
        for constraint in self.constraints:
            normalised = constraint.violation(lhs[constraint.name]) / constraint.scale
            total = total + constraint.weight * normalised.square()
        return total

    def loss_fn(self, values: torch.Tensor) -> torch.Tensor:
        values = self._ensure_batched(values)
        return self.objective_values(values) + self.penalty_multiplier * self.constraint_penalty(
            values
        )

    def pack(self, values: dict, **kwargs) -> torch.Tensor:
        """Pack and validate one named solution."""
        kwargs.setdefault("dtype", self.dtype)
        return self.space.pack(values, **kwargs)

    def unpack(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return named tensor views of one or more packed solutions."""
        return self.space.unpack(values)

    def score_summary(self, x_disc: torch.Tensor) -> dict:
        values = self._ensure_batched(x_disc)
        with torch.no_grad():
            objective = self.objective_values(values)[0]
            lhs = self.constraint_values(values)
            named = self.space.unpack(values)

        constraint_report: dict[str, dict[str, float | str | bool]] = {}
        feasible = True
        for constraint in self.constraints:
            lhs_value = float(lhs[constraint.name][0].item())
            violation = max(0.0, float(constraint.violation(lhs[constraint.name])[0].item()))
            ok = violation <= constraint.tolerance
            feasible = feasible and ok
            constraint_report[constraint.name] = {
                "lhs": lhs_value,
                "sense": constraint.sense,
                "rhs": float(constraint.rhs),
                "violation": violation,
                "tolerance": float(constraint.tolerance),
                "feasible": ok,
            }

        solution: dict[str, float | list[float]] = {}
        for variable in self.variables:
            value = named[variable.name][0].detach().cpu()
            solution[variable.name] = float(value.item()) if variable.size == 1 else value.tolist()

        return {
            "label": self.objective_label,
            "value": float(objective.item()),
            "unit": self.objective_unit,
            "feasible": feasible,
            "extra": {
                "penalized_loss": float(self.loss_fn(values)[0].item()),
                "penalty_multiplier": float(self.penalty_multiplier),
                "variables": solution,
                "constraints": constraint_report,
            },
        }

    def solve(self, **kwargs):
        """Solve this model with mixed-friendly defaults."""
        from qqa.mixed.solve import solve_mixed

        return solve_mixed(self, **kwargs)

    def _ensure_batched(self, values: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(values):
            raise TypeError(f"Expected a torch.Tensor, got {type(values).__name__}.")
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.num_vars:
            raise ValueError(
                f"Expected values with shape (B, {self.num_vars}), got {tuple(values.shape)}."
            )
        return values

    def __repr__(self) -> str:
        return (
            f"MixedProblem(name={self.name!r}, num_vars={self.num_vars}, "
            f"num_constraints={len(self.constraints)})"
        )
