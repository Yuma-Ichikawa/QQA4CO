"""Bidirectional, versioned feedback bus between primal and exact engines."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any

import torch

from qqa.model.ir import ClauseFactor, ConstraintIR, LinearFactor, ObjectiveIR


@dataclass(frozen=True, slots=True)
class LinearCut:
    indices: torch.Tensor
    coefficients: torch.Tensor
    sense: str
    rhs: float
    name: str = "dynamic-cut"

    def __post_init__(self) -> None:
        indices = torch.as_tensor(self.indices, dtype=torch.long).reshape(-1).detach().clone()
        coefficients = (
            torch.as_tensor(self.coefficients, dtype=torch.float64).reshape(-1).detach().clone()
        )
        if indices.shape != coefficients.shape or torch.any(indices < 0):
            raise ValueError("Cut indices and coefficients must align.")
        if self.sense not in {"<=", ">=", "=="}:
            raise ValueError("Cut sense must be <=, >=, or ==.")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "coefficients", coefficients)

    def as_constraint(self) -> ConstraintIR:
        return ConstraintIR(
            self.name,
            ObjectiveIR((LinearFactor(self.indices, self.coefficients),)),
            self.sense,
            self.rhs,
        )


@dataclass(frozen=True, slots=True)
class ExactFeedback:
    version: int = 0
    lp_primal: torch.Tensor | None = None
    dual_multipliers: torch.Tensor | None = None
    reduced_costs: torch.Tensor | None = None
    branch_scores: torch.Tensor | None = None
    fractionalities: torch.Tensor | None = None
    local_lower: torch.Tensor | None = None
    local_upper: torch.Tensor | None = None
    incumbent: torch.Tensor | None = None
    cuts: tuple[LinearCut, ...] = ()
    no_goods: tuple[tuple[int, ...], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def _clone_optional(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


class ExactFeedbackBus:
    """Thread-safe latest-state bus with monotone versions and bounded history."""

    def __init__(self, *, maximum_cuts: int = 256, maximum_no_goods: int = 256) -> None:
        if maximum_cuts < 1 or maximum_no_goods < 1:
            raise ValueError("Feedback bounds must be positive.")
        self.maximum_cuts = maximum_cuts
        self.maximum_no_goods = maximum_no_goods
        self._state = ExactFeedback()
        self._lock = RLock()

    def publish(self, **updates: Any) -> ExactFeedback:
        with self._lock:
            updates = dict(updates)
            if "cuts" in updates:
                updates["cuts"] = tuple(updates["cuts"])[-self.maximum_cuts :]
            if "no_goods" in updates:
                updates["no_goods"] = tuple(updates["no_goods"])[-self.maximum_no_goods :]
            for name, value in list(updates.items()):
                if torch.is_tensor(value):
                    updates[name] = value.detach().clone()
            self._state = replace(self._state, version=self._state.version + 1, **updates)
            return self.snapshot()

    def snapshot(self) -> ExactFeedback:
        with self._lock:
            return ExactFeedback(
                version=self._state.version,
                lp_primal=_clone_optional(self._state.lp_primal),
                dual_multipliers=_clone_optional(self._state.dual_multipliers),
                reduced_costs=_clone_optional(self._state.reduced_costs),
                branch_scores=_clone_optional(self._state.branch_scores),
                fractionalities=_clone_optional(self._state.fractionalities),
                local_lower=_clone_optional(self._state.local_lower),
                local_upper=_clone_optional(self._state.local_upper),
                incumbent=_clone_optional(self._state.incumbent),
                cuts=self._state.cuts,
                no_goods=self._state.no_goods,
                metadata=dict(self._state.metadata),
            )

    def constraints(self) -> tuple[ConstraintIR, ...]:
        return tuple(cut.as_constraint() for cut in self.snapshot().cuts)

    def no_good_factors(self) -> tuple[ClauseFactor, ...]:
        factors = []
        for assignment in self.snapshot().no_goods:
            indices = torch.arange(len(assignment), dtype=torch.long).unsqueeze(0)
            # A no-good clause negates the complete forbidden assignment.
            signs = torch.tensor([[-1 if value else 1 for value in assignment]], dtype=torch.int8)
            factors.append(ClauseFactor(indices, signs))
        return tuple(factors)


__all__ = ["ExactFeedback", "ExactFeedbackBus", "LinearCut"]
