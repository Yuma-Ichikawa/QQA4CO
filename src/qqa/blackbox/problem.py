"""User-facing black-box problem declarations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import torch

from qqa.mixed.variables import VariableSpace, VariableSpec
from qqa.runtime.security import validate_portable_payload

ScalarPoint = Mapping[str, float | int | list[float] | list[int]]
ScalarFunction = Callable[[ScalarPoint], float]
ConstraintSense = Literal["<=", ">=", "=="]
Direction = Literal["min", "max"]


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _scalar_result(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must return a real scalar, not bool.")
    try:
        tensor = torch.as_tensor(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TypeError(f"{label} must return a real scalar.") from exc
    if tensor.numel() != 1 or tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"{label} must return exactly one real scalar.")
    number = float(tensor.item())
    if not math.isfinite(number):
        raise FloatingPointError(f"{label} returned NaN or infinity.")
    return number


@dataclass(frozen=True, slots=True)
class BlackBoxConstraint:
    """A possibly non-differentiable constraint evaluated point by point."""

    function: ScalarFunction
    sense: ConstraintSense = "<="
    rhs: float = 0.0
    tolerance: float = 1e-8
    scale: float = 1.0
    name: str = "constraint"

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("BlackBoxConstraint function must be callable.")
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError("constraint sense must be '<=', '>=', or '=='.")
        for field_name in ("rhs", "tolerance", "scale"):
            value = _finite_number(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
        if self.tolerance < 0:
            raise ValueError("tolerance must be >= 0.")
        if self.scale <= 0:
            raise ValueError("scale must be > 0.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("constraint name must be a non-empty string.")

    def violation(self, value: float) -> float:
        residual = _finite_number(value, "constraint value") - self.rhs
        if self.sense == "<=":
            return max(0.0, residual)
        if self.sense == ">=":
            return max(0.0, -residual)
        return abs(residual)


class BlackBoxProblem:
    """An expensive objective over binary, integer, real, or mixed variables.

    The objective receives one plain-Python named point at a time, making it
    suitable for simulators, remote services, subprocesses, and legacy code.
    Independent points can be evaluated concurrently with ``workers > 1``.
    """

    def __init__(
        self,
        variables: Sequence[VariableSpec],
        objective: ScalarFunction,
        *,
        constraints: Sequence[BlackBoxConstraint] = (),
        direction: Direction = "min",
        name: str = "black-box-problem",
        evaluator_version: str = "1",
    ):
        if not callable(objective):
            raise TypeError("objective must be callable.")
        if direction not in ("min", "max"):
            raise ValueError("direction must be 'min' or 'max'.")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        if not isinstance(evaluator_version, str) or not evaluator_version.strip():
            raise ValueError("evaluator_version must be a non-empty string.")
        validate_portable_payload({"name": name, "evaluator_version": evaluator_version})
        self.space = VariableSpace(tuple(variables))
        self.variables = self.space.variables
        self.objective = objective
        self.constraints = tuple(constraints)
        if any(not isinstance(item, BlackBoxConstraint) for item in self.constraints):
            raise TypeError("constraints must contain BlackBoxConstraint instances.")
        names = [constraint.name for constraint in self.constraints]
        if len(names) != len(set(names)):
            raise ValueError("constraint names must be unique.")
        self.direction = direction
        self.name = name
        self.evaluator_version = evaluator_version

    def _named_point(self, values: torch.Tensor) -> dict:
        named = self.space.unpack(values)
        point: dict[str, float | int | list[float] | list[int]] = {}
        for variable in self.variables:
            tensor = named[variable.name].detach().cpu()
            if variable.kind == "real":
                point[variable.name] = (
                    float(tensor.item()) if variable.size == 1 else tensor.tolist()
                )
            else:
                ints = tensor.round().to(torch.int64)
                point[variable.name] = int(ints.item()) if variable.size == 1 else ints.tolist()
        return point

    def evaluate_one(self, values: torch.Tensor) -> tuple[float, list[float], dict]:
        """Evaluate one packed physical point with strict finite checks."""
        if values.ndim != 1:
            raise ValueError("values must be a one-dimensional packed point.")
        self.space.validate(values)
        point = self._named_point(values)
        objective = _scalar_result(self.objective(point), "black-box objective")
        violations: list[float] = []
        for constraint in self.constraints:
            lhs = _scalar_result(
                constraint.function(point),
                f"black-box constraint {constraint.name!r}",
            )
            violations.append(
                max(0.0, constraint.violation(lhs) - constraint.tolerance) / constraint.scale
            )
        return objective, violations, point

    def evaluate_batch(
        self, values: torch.Tensor, *, workers: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        """Evaluate packed points sequentially or with a thread pool."""
        if values.ndim != 2 or values.shape[1] != self.space.dimension:
            raise ValueError(
                f"values must have shape (B, {self.space.dimension}), got {tuple(values.shape)}."
            )
        if values.shape[0] == 0:
            raise ValueError("values must contain at least one point.")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer.")
        cpu_values = values.detach().to(device="cpu", dtype=torch.float64)
        if workers == 1:
            rows = [self.evaluate_one(row) for row in cpu_values]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(self.evaluate_one, list(cpu_values)))
        objectives = torch.tensor([row[0] for row in rows], dtype=torch.float64)
        if self.constraints:
            violations = torch.tensor([row[1] for row in rows], dtype=torch.float64)
        else:
            violations = torch.zeros((len(rows), 0), dtype=torch.float64)
        return objectives, violations, [row[2] for row in rows]

    def solve(self, **kwargs):
        """Optimise this black-box model."""
        from qqa.blackbox.solver import blackbox_optimize

        return blackbox_optimize(self, **kwargs)


__all__ = ["BlackBoxConstraint", "BlackBoxProblem", "Direction", "ScalarPoint"]
