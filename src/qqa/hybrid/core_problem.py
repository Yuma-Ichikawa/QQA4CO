"""Vectorised QQA core model built from SCIP LP state and active rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from qqa.hybrid.heuristic_types import QQAHeuristicConfig
from qqa.hybrid.surrogate import CoreSurrogate
from qqa.mixed import Binary, Constraint, Integer, MixedProblem
from qqa.mixed.variables import VariableSpec


class CoreRowProblem(MixedProblem):
    """MixedProblem with one vectorised evaluation of all active LP rows.

    Exposing the rows as real :class:`Constraint` objects activates QQA's
    constraint-wise augmented Lagrangian and feasibility archive.  The base
    ``MixedProblem`` API normally evaluates one Python callable per row; this
    specialised core keeps the callback cheap by stacking the population once
    and evaluating every lower/upper side in one matrix multiplication.
    """

    def __init__(
        self,
        declarations: list[VariableSpec],
        objective: Any,
        names: list[str],
        surrogate: CoreSurrogate,
        *,
        row_penalty: float,
        dtype: torch.dtype,
    ) -> None:
        constraints: list[Constraint] = []
        source_rows: list[int] = []
        weight = row_penalty / max(1, surrogate.num_rows)

        def placeholder(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return 0.0 * values[names[0]]

        for row in range(surrogate.num_rows):
            lower = float(surrogate.row_lower[row])
            upper = float(surrogate.row_upper[row])
            scale = float(surrogate.row_scale[row])
            tolerance = 1e-7 * scale
            if (
                math.isfinite(lower)
                and math.isfinite(upper)
                and math.isclose(
                    lower,
                    upper,
                    rel_tol=1e-12,
                    abs_tol=1e-12 * scale,
                )
            ):
                constraints.append(
                    Constraint(
                        placeholder,
                        sense="==",
                        rhs=lower,
                        weight=weight,
                        scale=scale,
                        tolerance=tolerance,
                        name=f"lp_row_{row}_equality",
                    )
                )
                source_rows.append(row)
                continue
            if math.isfinite(lower):
                constraints.append(
                    Constraint(
                        placeholder,
                        sense=">=",
                        rhs=lower,
                        weight=weight,
                        scale=scale,
                        tolerance=tolerance,
                        name=f"lp_row_{row}_lower",
                    )
                )
                source_rows.append(row)
            if math.isfinite(upper):
                constraints.append(
                    Constraint(
                        placeholder,
                        sense="<=",
                        rhs=upper,
                        weight=weight,
                        scale=scale,
                        tolerance=tolerance,
                        name=f"lp_row_{row}_upper",
                    )
                )
                source_rows.append(row)
        super().__init__(
            declarations,
            objective,
            constraints=constraints,
            name="scip-guided-integer-core",
            dtype=dtype,
        )
        self._core_names = tuple(names)
        self._row_matrix = torch.as_tensor(surrogate.row_matrix, dtype=dtype)
        self._row_offset = torch.as_tensor(surrogate.row_offset, dtype=dtype)
        self._constraint_source_rows = tuple(source_rows)
        self._row_device_cache: dict[
            tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def constraint_values(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        values = self._ensure_batched(values)
        named = self.space.unpack(values)
        stacked = torch.stack([named[name] for name in self._core_names], dim=1)
        cache_key = (stacked.device, stacked.dtype)
        cached = self._row_device_cache.get(cache_key)
        if cached is None:
            cached = (self._row_matrix.to(stacked), self._row_offset.to(stacked))
            self._row_device_cache[cache_key] = cached
        matrix, offset = cached
        activity = stacked @ matrix.T + offset
        return {
            constraint.name: activity[:, source]
            for constraint, source in zip(
                self.constraints,
                self._constraint_source_rows,
                strict=True,
            )
        }


def build_core_problem(
    state,
    selection,
    positions: list[int],
    surrogate: CoreSurrogate,
    config: QQAHeuristicConfig,
    *,
    adaptive_rows: bool = False,
) -> tuple[MixedProblem, list[str]]:
    dtype = torch.float32 if config.core_dtype == "float32" else torch.float64
    declarations: list[VariableSpec] = []
    names = []
    targets = []
    spans = []
    reduced = []
    weights = []
    for output_index, position in enumerate(positions):
        variable_index = int(selection.core_indices[position])
        name = f"z_{output_index}"
        lower = int(math.ceil(selection.local_lower[position] - 1e-9))
        upper = int(math.floor(selection.local_upper[position] + 1e-9))
        if state.variable_types[variable_index] == "BINARY":
            declarations.append(Binary(name))
        else:
            declarations.append(Integer(name, lower=lower, upper=upper))
        names.append(name)
        targets.append(float(state.lp_values[variable_index]))
        spans.append(float(max(1, upper - lower)))
        reduced.append(float(state.reduced_costs[variable_index]))
        weights.append(float(max(0.1, selection.scores[position])))

    target_vector = torch.tensor(targets, dtype=dtype)
    span_vector = torch.tensor(spans, dtype=dtype)
    reduced_vector = torch.tensor(reduced, dtype=dtype)
    reduced_vector /= reduced_vector.abs().amax().clamp_min(1.0)
    weight_vector = torch.tensor(weights, dtype=dtype)
    quadratic = torch.tensor(surrogate.quadratic, dtype=dtype)
    linear = torch.tensor(surrogate.linear, dtype=dtype)
    adaptive_rows = bool(adaptive_rows and surrogate.num_rows)
    row_matrix = torch.tensor(surrogate.row_matrix, dtype=dtype)
    row_offset = torch.tensor(surrogate.row_offset, dtype=dtype)
    row_lower = torch.tensor(surrogate.row_lower, dtype=dtype)
    row_upper = torch.tensor(surrogate.row_upper, dtype=dtype)
    row_scale = torch.tensor(surrogate.row_scale, dtype=dtype)
    constant_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, ...]] = {}

    def objective(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack([values[name] for name in names], dim=1)
        cache_key = (stacked.device, stacked.dtype)
        constants = constant_cache.get(cache_key)
        if constants is None:
            constants = tuple(
                tensor.to(stacked)
                for tensor in (
                    target_vector,
                    span_vector,
                    reduced_vector,
                    weight_vector,
                    quadratic,
                    linear,
                    row_matrix,
                    row_offset,
                    row_lower,
                    row_upper,
                    row_scale,
                )
            )
            constant_cache[cache_key] = constants
        (
            target,
            span,
            redcost,
            weight,
            local_quadratic,
            local_linear,
            local_row_matrix,
            local_row_offset,
            local_row_lower,
            local_row_upper,
            local_row_scale,
        ) = constants
        original = (
            0.5 * torch.einsum("bi,ij,bj->b", stacked, local_quadratic, stacked)
            + stacked @ local_linear
        )
        original = config.objective_weight * original / surrogate.objective_scale
        proximity = config.proximity_weight * (weight * ((stacked - target) / span).square()).mean(
            dim=1
        )
        direction = config.reduced_cost_weight * (redcost * (stacked - target) / span).mean(dim=1)
        if adaptive_rows or not surrogate.num_rows:
            row_loss = torch.zeros(
                stacked.shape[0],
                dtype=stacked.dtype,
                device=stacked.device,
            )
        else:
            activity = stacked @ local_row_matrix.T + local_row_offset
            violation = torch.relu(local_row_lower - activity) + torch.relu(
                activity - local_row_upper
            )
            row_loss = config.row_penalty * (violation / local_row_scale).square().mean(dim=1)
        return original + proximity + direction + row_loss

    if adaptive_rows:
        problem: MixedProblem = CoreRowProblem(
            declarations,
            objective,
            names,
            surrogate,
            row_penalty=config.row_penalty,
            dtype=dtype,
        )
    else:
        problem = MixedProblem(
            declarations,
            objective,
            name="scip-guided-integer-core",
            dtype=dtype,
        )
    return problem, names


__all__ = ["CoreRowProblem", "build_core_problem"]
