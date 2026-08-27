"""Mixed and multi-objective microgrid capacity/dispatch models."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from qqa.mixed import Binary, Constraint, Integer, MixedProblem, Real
from qqa.multiobjective import MultiObjectiveProblem, Objective

_CAPACITY = torch.tensor([80.0, 60.0, 45.0, 35.0])
_MINIMUM = torch.tensor([18.0, 12.0, 8.0, 4.0])
_FIXED_COST = torch.tensor([185.0, 142.0, 105.0, 62.0])
_MARGINAL_COST = torch.tensor([48.0, 57.0, 71.0, 93.0])
_EMISSIONS = torch.tensor([0.72, 0.49, 0.28, 0.06])
_DEMAND_MW = 170.0
_RESERVE_MW = 24.0


def _data_like(values: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return source.to(device=values.device, dtype=values.dtype)


def _cost(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    power = v["power"]
    fixed = _data_like(power, _FIXED_COST)
    marginal = _data_like(power, _MARGINAL_COST)
    # The small convex heat-rate term makes this a genuine mixed nonlinear
    # dispatch problem instead of a disguised linear knapsack.
    generation = (fixed * v["commit"] + marginal * power + 0.055 * power.square()).sum(-1)
    storage = 96.0 * v["storage_units"] + 7.5 * v["storage_mw"].square()
    demand_response = 125.0 * v["demand_response"] + 3.0 * v["demand_response"].square()
    return generation + storage + demand_response


def _emissions(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    power = v["power"]
    factors = _data_like(power, _EMISSIONS)
    # Storage is assigned a small lifecycle intensity.
    return (factors * power).sum(-1) + 0.035 * v["storage_mw"]


def _available_reserve(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
    power = v["power"]
    capacity = _data_like(power, _CAPACITY)
    return (capacity * v["commit"] - power).sum(-1) + 5.0 * v["storage_units"]


def _constraints() -> list[Constraint]:
    def maximum_output(index: int, capacity: float):
        def constraint(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return v["power"][:, index] - capacity * v["commit"][:, index]

        return constraint

    def minimum_output(index: int, minimum: float):
        def constraint(v: Mapping[str, torch.Tensor]) -> torch.Tensor:
            return minimum * v["commit"][:, index] - v["power"][:, index]

        return constraint

    constraints = [
        Constraint(
            lambda v: v["power"].sum(-1) + v["storage_mw"] + v["demand_response"],
            sense=">=",
            rhs=_DEMAND_MW,
            weight=100_000_000.0,
            scale=_DEMAND_MW,
            tolerance=0.2,
            name="demand_balance",
        ),
        Constraint(
            _available_reserve,
            sense=">=",
            rhs=_RESERVE_MW,
            weight=20_000_000.0,
            scale=_RESERVE_MW,
            tolerance=0.2,
            name="spinning_reserve",
        ),
        Constraint(
            lambda v: v["storage_mw"] - 15.0 * v["storage_units"],
            sense="<=",
            rhs=0.0,
            weight=20_000_000.0,
            scale=15.0,
            tolerance=0.05,
            name="storage_link",
        ),
    ]
    for index in range(4):
        capacity = float(_CAPACITY[index])
        minimum = float(_MINIMUM[index])
        constraints.extend(
            [
                Constraint(
                    maximum_output(index, capacity),
                    sense="<=",
                    rhs=0.0,
                    weight=20_000_000.0,
                    scale=capacity,
                    tolerance=0.05,
                    name=f"unit_{index}_maximum",
                ),
                Constraint(
                    minimum_output(index, minimum),
                    sense="<=",
                    rhs=0.0,
                    weight=20_000_000.0,
                    scale=minimum,
                    tolerance=0.05,
                    name=f"unit_{index}_minimum",
                ),
            ]
        )
    return constraints


def _variables():
    return [
        Binary("commit", size=4),
        Real("power", 0.0, 80.0, size=4),
        Integer("storage_units", 0, 6),
        Real("storage_mw", 0.0, 45.0),
        Real("demand_response", 0.0, 20.0),
    ]


def build_microgrid_dispatch() -> MixedProblem:
    """Return a constrained mixed unit-commitment and dispatch model."""
    return MixedProblem(
        _variables(),
        _cost,
        constraints=_constraints(),
        name="microgrid-dispatch",
        objective_label="operating cost",
        objective_unit="USD/h",
    )


def build_microgrid_pareto() -> MultiObjectiveProblem:
    """Return cost/emissions/resilience trade-offs for the same microgrid."""
    return MultiObjectiveProblem(
        _variables(),
        [
            Objective(_cost, "cost", direction="min", unit="USD/h"),
            Objective(_emissions, "emissions", direction="min", unit="tCO2/h"),
            Objective(_available_reserve, "resilience", direction="max", unit="MW"),
        ],
        constraints=_constraints(),
        name="microgrid-pareto",
    )


__all__ = ["build_microgrid_dispatch", "build_microgrid_pareto"]
