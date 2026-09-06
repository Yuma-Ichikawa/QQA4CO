"""Constraint-wise adaptive Powell--Hestenes--Rockafellar penalties."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from qqa.callbacks import Callback, CallbackState


def normalised_residuals(problem, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed ``g(x) <= 0`` residuals and equality flags."""
    if not problem.constraints:
        return (
            values.new_zeros((values.shape[0], 0)),
            torch.zeros(0, dtype=torch.bool, device=values.device),
        )
    custom = getattr(problem, "normalised_constraint_residuals", None)
    if callable(custom):
        return custom(values)
    lhs = problem.constraint_values(values)
    residuals = []
    equality = []
    for constraint in problem.constraints:
        residual = (lhs[constraint.name] - constraint.rhs) / constraint.scale
        if constraint.sense == ">=":
            residual = -residual
        residuals.append(residual)
        equality.append(constraint.sense == "==")
    return (
        torch.stack(residuals, dim=1),
        torch.tensor(equality, dtype=torch.bool, device=values.device),
    )


class ConstraintArchive:
    """Device-resident feasibility-first and feasible-objective incumbents."""

    def __init__(self) -> None:
        self.feasibility_solution: torch.Tensor | None = None
        self._objective_solution: torch.Tensor | None = None
        self._feasibility_key: torch.Tensor | None = None
        self._objective_key: torch.Tensor | None = None
        self._has_feasible: torch.Tensor | None = None
        self.observations = 0

    @property
    def objective_solution(self) -> torch.Tensor | None:
        """Return the feasible incumbent, synchronising only on explicit access."""
        if self._objective_solution is None or self._has_feasible is None:
            return None
        return self._objective_solution if bool(self._has_feasible.item()) else None

    @property
    def maximum_violation(self) -> float:
        return math.inf if self._feasibility_key is None else float(self._feasibility_key[0].item())

    @property
    def total_violation(self) -> float:
        return math.inf if self._feasibility_key is None else float(self._feasibility_key[1].item())

    @property
    def feasibility_objective(self) -> float:
        return math.inf if self._feasibility_key is None else float(self._feasibility_key[2].item())

    @property
    def objective(self) -> float:
        if self._objective_key is None or self.objective_solution is None:
            return math.inf
        return float(self._objective_key.item())

    @staticmethod
    def _objective(problem, values: torch.Tensor) -> torch.Tensor:
        ranking = getattr(problem, "ranking_objective", None)
        return ranking(values) if callable(ranking) else problem.objective_values(values)

    def update(self, problem, values: torch.Tensor) -> None:
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if len(values) == 0:
            return
        with torch.no_grad():
            violations = problem.constraint_violations(values)
            matrix = torch.stack(
                [violations[row.name] / row.scale for row in problem.constraints],
                dim=1,
            )
            maximum = matrix.amax(dim=1)
            total = matrix.sum(dim=1)
            objective = self._objective(problem, values)
            keys = torch.stack((maximum, total, objective), dim=1)
            order = torch.arange(len(values), device=values.device)
            for column in range(keys.shape[1] - 1, -1, -1):
                order = order[torch.argsort(keys[order, column], stable=True)]
            feasibility_index = order[0]
            feasibility_key = keys[feasibility_index]
            if self.feasibility_solution is None:
                self.feasibility_solution = values[feasibility_index].detach().clone()
                self._feasibility_key = feasibility_key.detach().clone()
            else:
                if self._feasibility_key is None:
                    raise RuntimeError("Feasibility archive key was not initialized.")
                current = self._feasibility_key.to(feasibility_key)
                better = torch.zeros((), dtype=torch.bool, device=values.device)
                equal = torch.ones((), dtype=torch.bool, device=values.device)
                for column in range(len(feasibility_key)):
                    better |= equal & (feasibility_key[column] < current[column])
                    equal &= feasibility_key[column] == current[column]
                self.feasibility_solution = torch.where(
                    better, values[feasibility_index], self.feasibility_solution.to(values)
                ).detach()
                self._feasibility_key = torch.where(better, feasibility_key, current).detach()

            feasible = torch.ones(len(values), dtype=torch.bool, device=values.device)
            for row in problem.constraints:
                feasible &= violations[row.name] <= row.tolerance
            feasible_objective = torch.where(
                feasible,
                objective,
                torch.full_like(objective, torch.inf),
            )
            objective_index = torch.argmin(feasible_objective)
            objective_value = feasible_objective[objective_index]
            any_feasible = feasible.any()
            if self._objective_solution is None:
                self._objective_solution = values[objective_index].detach().clone()
                self._objective_key = objective_value.detach().clone()
                self._has_feasible = any_feasible.detach().clone()
            else:
                if self._objective_key is None or self._has_feasible is None:
                    raise RuntimeError("Objective archive state was not initialized.")
                old_key = self._objective_key.to(objective_value)
                old_has = self._has_feasible.to(any_feasible)
                better = any_feasible & (~old_has | (objective_value < old_key))
                self._objective_solution = torch.where(
                    better, values[objective_index], self._objective_solution.to(values)
                ).detach()
                self._objective_key = torch.where(better, objective_value, old_key).detach()
                self._has_feasible = (old_has | any_feasible).detach()
        self.observations += len(values)

    def candidates(self) -> list[torch.Tensor]:
        result = []
        if self.feasibility_solution is not None:
            result.append(self.feasibility_solution)
        objective_solution = self.objective_solution
        if objective_solution is not None:
            result.append(objective_solution)
        return result

    def diagnostics(self) -> dict[str, object]:
        return {
            "observations": self.observations,
            "maximum_violation": self.maximum_violation,
            "total_violation": self.total_violation,
            "has_feasible": self.objective_solution is not None,
            "best_feasible_objective": (
                self.objective if self.objective_solution is not None else None
            ),
        }

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        if self.feasibility_solution is None or self._feasibility_key is None:
            return {}
        result = {
            "constraint_archive_feasibility_solution": self.feasibility_solution,
            "constraint_archive_feasibility_key": self._feasibility_key,
            "constraint_archive_observations": self._feasibility_key.new_tensor(
                self.observations, dtype=torch.int64
            ),
        }
        if (
            self._objective_solution is not None
            and self._objective_key is not None
            and self._has_feasible is not None
        ):
            result.update(
                {
                    "constraint_archive_objective_solution": self._objective_solution,
                    "constraint_archive_objective_key": self._objective_key,
                    "constraint_archive_has_feasible": self._has_feasible,
                }
            )
        return result

    def restore_checkpoint_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        required = {
            "constraint_archive_feasibility_solution",
            "constraint_archive_feasibility_key",
            "constraint_archive_observations",
        }
        present = required & tensors.keys()
        if not present:
            return
        if present != required:
            raise ValueError("Checkpoint contains an incomplete constraint archive.")
        self.feasibility_solution = tensors["constraint_archive_feasibility_solution"].clone()
        self._feasibility_key = tensors["constraint_archive_feasibility_key"].clone()
        self.observations = int(tensors["constraint_archive_observations"].cpu().item())
        optional = {
            "constraint_archive_objective_solution",
            "constraint_archive_objective_key",
            "constraint_archive_has_feasible",
        }
        optional_present = optional & tensors.keys()
        if optional_present and optional_present != optional:
            raise ValueError("Checkpoint contains an incomplete feasible-objective archive.")
        if optional_present:
            self._objective_solution = tensors["constraint_archive_objective_solution"].clone()
            self._objective_key = tensors["constraint_archive_objective_key"].clone()
            self._has_feasible = tensors["constraint_archive_has_feasible"].clone()


@dataclass(slots=True)
class AdaptiveAugmentedLagrangian:
    """Mutable, per-solve augmented-Lagrangian state."""

    multipliers: torch.Tensor
    rho: torch.Tensor
    previous_violation: torch.Tensor
    previous_residual: torch.Tensor
    rho_growth: float = 2.0
    minimum_rho: float = 1e-8
    maximum_rho: float = 1e10
    improvement_ratio: float = 0.9
    balance_mu: float = 10.0
    updates: int = 0
    rho_increases: int | torch.Tensor = 0
    rho_decreases: int | torch.Tensor = 0

    @classmethod
    def for_problem(
        cls,
        problem,
        *,
        penalty_multiplier: float = 1.0,
        rho_growth: float = 2.0,
        maximum_rho: float = 1e10,
        improvement_ratio: float = 0.9,
    ) -> AdaptiveAugmentedLagrangian:
        if not rho_growth > 1 or not math.isfinite(rho_growth):
            raise ValueError("rho_growth must be finite and > 1.")
        if maximum_rho < 1 or not math.isfinite(maximum_rho):
            raise ValueError("maximum_rho must be finite and >= 1.")
        if not 0 < improvement_ratio < 1:
            raise ValueError("improvement_ratio must be in (0, 1).")
        # 0.5*rho*r^2 initially matches declared weight*multiplier*r^2.
        rho = torch.tensor(
            [2.0 * row.weight * penalty_multiplier for row in problem.constraints],
            dtype=torch.float64,
        ).clamp_(max=maximum_rho)
        return cls(
            multipliers=torch.zeros(len(problem.constraints), dtype=torch.float64),
            rho=rho,
            previous_violation=torch.full(
                (len(problem.constraints),),
                float("inf"),
                dtype=torch.float64,
            ),
            previous_residual=torch.zeros(len(problem.constraints), dtype=torch.float64),
            rho_growth=float(rho_growth),
            maximum_rho=float(maximum_rho),
            improvement_ratio=float(improvement_ratio),
        )

    def penalty(self, problem, values: torch.Tensor) -> torch.Tensor:
        residuals, equality = normalised_residuals(problem, values)
        if residuals.shape[1] == 0:
            return values.new_zeros(values.shape[0])
        multipliers = self.multipliers.to(device=values.device, dtype=values.dtype)
        rho = self.rho.to(device=values.device, dtype=values.dtype)
        equality_loss = multipliers * residuals + 0.5 * rho * residuals.square()
        shifted = (residuals + multipliers / rho).clamp_min(0.0)
        inequality_loss = 0.5 * rho * shifted.square() - 0.5 * multipliers.square() / rho
        return torch.where(equality, equality_loss, inequality_loss).sum(dim=1)

    def update(self, problem, values: torch.Tensor) -> None:
        residuals, equality = normalised_residuals(problem, values)
        if residuals.shape[1] == 0:
            return
        detached = residuals.detach().to(dtype=torch.float64)
        equality = equality.to(device=detached.device)
        self.multipliers = self.multipliers.to(detached)
        self.rho = self.rho.to(detached)
        self.previous_violation = self.previous_violation.to(detached)
        self.previous_residual = self.previous_residual.to(detached)
        violations = torch.where(equality, detached.abs(), detached.clamp_min(0.0))
        maximum = violations.amax(dim=1)
        total = violations.sum(dim=1)
        keys = torch.stack((maximum, total), dim=1)
        order = torch.arange(len(detached), device=detached.device)
        for column in range(keys.shape[1] - 1, -1, -1):
            order = order[torch.argsort(keys[order, column], stable=True)]
        representative_index = order[0]
        representative = detached[representative_index]
        current = violations.median(dim=0).values

        self.multipliers[equality] += self.rho[equality] * representative[equality]
        inequality = ~equality
        self.multipliers[inequality] = torch.clamp_min(
            self.multipliers[inequality] + self.rho[inequality] * representative[inequality],
            0.0,
        )
        if self.updates:
            primal_residual = current
            dual_residual = self.rho * (representative - self.previous_residual).abs()
            stalled = current > self.improvement_ratio * self.previous_violation
            increase = stalled & (primal_residual > self.balance_mu * dual_residual)
            decrease = dual_residual > self.balance_mu * primal_residual
            self.rho[increase] = torch.clamp(
                self.rho[increase] * self.rho_growth,
                max=self.maximum_rho,
            )
            self.rho[decrease] = torch.clamp(
                self.rho[decrease] / self.rho_growth,
                min=self.minimum_rho,
            )
            self.rho_increases = (
                torch.as_tensor(self.rho_increases, device=detached.device, dtype=torch.int64)
                + increase.sum()
            )
            self.rho_decreases = (
                torch.as_tensor(self.rho_decreases, device=detached.device, dtype=torch.int64)
                + decrease.sum()
            )
        self.previous_violation = current
        self.previous_residual = representative
        self.updates += 1

    def diagnostics(self) -> dict[str, object]:
        return {
            "updates": self.updates,
            "multipliers": self.multipliers.detach().cpu().tolist(),
            "rho": self.rho.detach().cpu().tolist(),
            "previous_violation": self.previous_violation.detach().cpu().tolist(),
            "primal_dual_balance_mu": self.balance_mu,
            "rho_increases": int(torch.as_tensor(self.rho_increases).cpu().item()),
            "rho_decreases": int(torch.as_tensor(self.rho_decreases).cpu().item()),
        }

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        reference = self.multipliers
        return {
            "al_multipliers": self.multipliers,
            "al_rho": self.rho,
            "al_previous_violation": self.previous_violation,
            "al_previous_residual": self.previous_residual,
            "al_updates": reference.new_tensor(self.updates, dtype=torch.int64),
            "al_rho_increases": torch.as_tensor(
                self.rho_increases, device=reference.device, dtype=torch.int64
            ),
            "al_rho_decreases": torch.as_tensor(
                self.rho_decreases, device=reference.device, dtype=torch.int64
            ),
        }

    def restore_checkpoint_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        names = {
            "multipliers",
            "rho",
            "previous_violation",
            "previous_residual",
            "updates",
            "rho_increases",
            "rho_decreases",
        }
        keys = {f"al_{name}" for name in names}
        present = keys & tensors.keys()
        if not present:
            return
        if present != keys:
            raise ValueError("Checkpoint contains incomplete augmented-Lagrangian state.")
        expected_shape = self.multipliers.shape
        for name in ("multipliers", "rho", "previous_violation", "previous_residual"):
            value = tensors[f"al_{name}"].to(torch.float64)
            if value.shape != expected_shape:
                raise ValueError(f"Checkpoint augmented-Lagrangian {name} shape is invalid.")
            setattr(self, name, value.clone())
        self.updates = int(tensors["al_updates"].cpu().item())
        self.rho_increases = tensors["al_rho_increases"].to(torch.int64).clone()
        self.rho_decreases = tensors["al_rho_decreases"].to(torch.int64).clone()


class AdaptiveALCallback(Callback):
    """Update augmented-Lagrangian state on relaxed training points."""

    def __init__(self, *, update_interval: int = 50, controller=None) -> None:
        if (
            isinstance(update_interval, bool)
            or not isinstance(update_interval, int)
            or update_interval < 1
        ):
            raise ValueError("update_interval must be a positive integer.")
        self.update_interval = update_interval
        self.controller = controller

    def on_epoch_end(self, state: CallbackState) -> None:
        if (state.epoch + 1) % self.update_interval != 0 and state.epoch != state.num_epochs - 1:
            return
        controller = self.controller or getattr(state.problem, "_augmented_lagrangian", None)
        if controller is None:
            return
        with torch.no_grad():
            values = state.relaxation.forward(state.x)
            controller.update(state.problem, values)

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        return {} if self.controller is None else self.controller.checkpoint_tensors()

    def restore_checkpoint_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        if self.controller is not None:
            self.controller.restore_checkpoint_tensors(tensors)


class ConstraintArchiveCallback(Callback):
    """Capture feasibility and objective incumbents throughout training."""

    def __init__(self, archive: ConstraintArchive, *, update_interval: int = 10) -> None:
        if (
            isinstance(update_interval, bool)
            or not isinstance(update_interval, int)
            or update_interval < 1
        ):
            raise ValueError("update_interval must be a positive integer.")
        self.archive = archive
        self.update_interval = update_interval

    def on_epoch_end(self, state: CallbackState) -> None:
        if (state.epoch + 1) % self.update_interval != 0 and state.epoch != state.num_epochs - 1:
            return
        with torch.no_grad():
            self.archive.update(state.problem, state.relaxation.project(state.x))

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        return self.archive.checkpoint_tensors()

    def restore_checkpoint_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        self.archive.restore_checkpoint_tensors(tensors)


__all__ = [
    "AdaptiveALCallback",
    "AdaptiveAugmentedLagrangian",
    "ConstraintArchive",
    "ConstraintArchiveCallback",
    "normalised_residuals",
]
