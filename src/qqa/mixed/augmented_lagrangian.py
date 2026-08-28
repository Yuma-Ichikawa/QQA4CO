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


@dataclass(slots=True)
class ConstraintArchive:
    """Separate feasibility-first and feasible-objective incumbents."""

    feasibility_solution: torch.Tensor | None = None
    objective_solution: torch.Tensor | None = None
    maximum_violation: float = math.inf
    total_violation: float = math.inf
    feasibility_objective: float = math.inf
    objective: float = math.inf
    observations: int = 0

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
            feasibility_index = min(
                range(len(values)),
                key=lambda index: (
                    float(maximum[index]),
                    float(total[index]),
                    float(objective[index]),
                    index,
                ),
            )
            feasibility_key = (
                float(maximum[feasibility_index]),
                float(total[feasibility_index]),
                float(objective[feasibility_index]),
            )
            current_key = (
                self.maximum_violation,
                self.total_violation,
                self.feasibility_objective,
            )
            if feasibility_key < current_key:
                self.feasibility_solution = values[feasibility_index].detach().clone()
                (
                    self.maximum_violation,
                    self.total_violation,
                    self.feasibility_objective,
                ) = feasibility_key

            feasible = torch.ones(len(values), dtype=torch.bool, device=values.device)
            for row in problem.constraints:
                feasible &= violations[row.name] <= row.tolerance
            feasible_indices = torch.nonzero(feasible, as_tuple=False).reshape(-1).tolist()
            if feasible_indices:
                objective_index = min(
                    feasible_indices,
                    key=lambda index: (float(objective[index]), index),
                )
                objective_value = float(objective[objective_index])
                if objective_value < self.objective:
                    self.objective = objective_value
                    self.objective_solution = values[objective_index].detach().clone()
        self.observations += len(values)

    def candidates(self) -> list[torch.Tensor]:
        result = []
        if self.feasibility_solution is not None:
            result.append(self.feasibility_solution)
        if self.objective_solution is not None:
            result.append(self.objective_solution)
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
    rho_increases: int = 0
    rho_decreases: int = 0

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
        detached = residuals.detach().to(device="cpu", dtype=torch.float64)
        violations = torch.where(equality.cpu(), detached.abs(), detached.clamp_min(0.0))
        maximum = violations.amax(dim=1)
        total = violations.sum(dim=1)
        # Exact lexicographic feasibility-first selection. A weighted scalar
        # can reverse the order when two maximum violations are very close.
        representative_index = min(
            range(len(detached)),
            key=lambda index: (
                float(maximum[index]),
                float(total[index]),
                index,
            ),
        )
        representative = detached[representative_index]
        current = violations.median(dim=0).values

        self.multipliers[equality.cpu()] += (
            self.rho[equality.cpu()] * representative[equality.cpu()]
        )
        inequality = ~equality.cpu()
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
            self.rho_increases += int(increase.sum().item())
            self.rho_decreases += int(decrease.sum().item())
        self.previous_violation = current
        self.previous_residual = representative
        self.updates += 1

    def diagnostics(self) -> dict[str, object]:
        return {
            "updates": self.updates,
            "multipliers": self.multipliers.tolist(),
            "rho": self.rho.tolist(),
            "previous_violation": self.previous_violation.tolist(),
            "primal_dual_balance_mu": self.balance_mu,
            "rho_increases": self.rho_increases,
            "rho_decreases": self.rho_decreases,
        }


class AdaptiveALCallback(Callback):
    """Update augmented-Lagrangian state on relaxed training points."""

    def __init__(self, *, update_interval: int = 50) -> None:
        if (
            isinstance(update_interval, bool)
            or not isinstance(update_interval, int)
            or update_interval < 1
        ):
            raise ValueError("update_interval must be a positive integer.")
        self.update_interval = update_interval

    def on_epoch_end(self, state: CallbackState) -> None:
        if (state.epoch + 1) % self.update_interval != 0 and state.epoch != state.num_epochs - 1:
            return
        controller = getattr(state.problem, "_augmented_lagrangian", None)
        if controller is None:
            return
        with torch.no_grad():
            values = state.relaxation.forward(state.x)
            controller.update(state.problem, values)


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


__all__ = [
    "AdaptiveALCallback",
    "AdaptiveAugmentedLagrangian",
    "ConstraintArchive",
    "ConstraintArchiveCallback",
    "normalised_residuals",
]
