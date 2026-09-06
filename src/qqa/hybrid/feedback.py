"""Bidirectional, versioned feedback bus between primal and exact engines."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from threading import RLock
from typing import Any

import torch

from qqa.model.ir import ClauseFactor, ConstraintIR, LinearFactor, ObjectiveIR

_TENSOR_FIELDS = frozenset(
    {
        "lp_primal",
        "dual_multipliers",
        "reduced_costs",
        "branch_scores",
        "fractionalities",
        "local_lower",
        "local_upper",
        "incumbent",
    }
)


@dataclass(frozen=True, slots=True)
class LinearCut:
    indices: torch.Tensor
    coefficients: torch.Tensor
    sense: str
    rhs: float
    name: str = "dynamic-cut"

    def __post_init__(self) -> None:
        raw_indices = torch.as_tensor(self.indices)
        if (
            raw_indices.dtype == torch.bool
            or raw_indices.is_floating_point()
            or raw_indices.is_complex()
        ):
            raise ValueError("Cut indices must be integers.")
        indices = raw_indices.to(dtype=torch.long).reshape(-1).detach().clone()
        coefficients = (
            torch.as_tensor(self.coefficients, dtype=torch.float64).reshape(-1).detach().clone()
        )
        if indices.shape != coefficients.shape or torch.any(indices < 0):
            raise ValueError("Cut indices and coefficients must align.")
        if (
            isinstance(self.rhs, bool)
            or not isinstance(self.rhs, Real)
            or not torch.isfinite(coefficients).all()
            or not math.isfinite(self.rhs)
        ):
            raise ValueError("Cut coefficients and rhs must be finite.")
        if self.sense not in {"<=", ">=", "=="}:
            raise ValueError("Cut sense must be <=, >=, or ==.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Cut name must be a non-empty string.")
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


def _clone_cut(cut: LinearCut) -> LinearCut:
    if not isinstance(cut, LinearCut):
        raise TypeError("cuts must contain LinearCut values.")
    return LinearCut(cut.indices, cut.coefficients, cut.sense, cut.rhs, cut.name)


def _canonical_no_goods(values: Any) -> tuple[tuple[int, ...], ...]:
    no_goods = []
    for assignment in values:
        canonical = tuple(assignment)
        if not canonical:
            raise ValueError("No-good assignments must be non-empty.")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value not in {0, 1}
            for value in canonical
        ):
            raise ValueError("No-good assignments must contain only integer zeros and ones.")
        no_goods.append(tuple(int(value) for value in canonical))
    return tuple(no_goods)


class ExactFeedbackBus:
    """Thread-safe latest-state bus with monotone versions and bounded history."""

    def __init__(self, *, maximum_cuts: int = 256, maximum_no_goods: int = 256) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (maximum_cuts, maximum_no_goods)
        ):
            raise ValueError("Feedback bounds must be positive.")
        self.maximum_cuts = maximum_cuts
        self.maximum_no_goods = maximum_no_goods
        self._state = ExactFeedback()
        self._lock = RLock()

    def publish(self, **updates: Any) -> ExactFeedback:
        with self._lock:
            updates = dict(updates)
            if "version" in updates:
                raise ValueError("Feedback versions are managed by the bus.")
            if "cuts" in updates:
                updates["cuts"] = tuple(_clone_cut(cut) for cut in updates["cuts"])[
                    -self.maximum_cuts :
                ]
            if "no_goods" in updates:
                updates["no_goods"] = _canonical_no_goods(updates["no_goods"])[
                    -self.maximum_no_goods :
                ]
            if "metadata" in updates:
                if not isinstance(updates["metadata"], dict):
                    raise TypeError("metadata must be a dictionary.")
                updates["metadata"] = copy.deepcopy(updates["metadata"])
            for name in _TENSOR_FIELDS & updates.keys():
                value = updates[name]
                if value is not None and not torch.is_tensor(value):
                    raise TypeError(f"{name} must be a tensor or None.")
                updates[name] = _clone_optional(value)
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
                cuts=tuple(_clone_cut(cut) for cut in self._state.cuts),
                no_goods=self._state.no_goods,
                metadata=copy.deepcopy(self._state.metadata),
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
