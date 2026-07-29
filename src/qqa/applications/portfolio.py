"""Cardinality-constrained multi-objective portfolio allocation."""

from __future__ import annotations

import torch

from qqa.mixed import Binary, Constraint, Real
from qqa.multiobjective import MultiObjectiveProblem, Objective

_EXPECTED_RETURN = torch.tensor([0.055, 0.072, 0.083, 0.061, 0.095, 0.068])
_CURRENT_WEIGHT = torch.tensor([0.25, 0.20, 0.15, 0.10, 0.20, 0.10])
_FACTOR_LOADINGS = torch.tensor(
    [
        [0.14, 0.02, 0.01],
        [0.17, -0.01, 0.03],
        [0.20, 0.04, -0.02],
        [0.11, 0.07, 0.02],
        [0.23, -0.03, 0.06],
        [0.15, 0.05, 0.04],
    ]
)
_IDIOSYNCRATIC_VOL = torch.tensor([0.10, 0.12, 0.15, 0.09, 0.17, 0.11])


def _like(values: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    return data.to(device=values.device, dtype=values.dtype)


def _risk(v: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = v["weight"]
    loadings = _like(weights, _FACTOR_LOADINGS)
    idiosyncratic = _like(weights, _IDIOSYNCRATIC_VOL)
    factor_exposure = weights @ loadings
    return factor_exposure.square().sum(dim=-1) + (weights * idiosyncratic).square().sum(dim=-1)


def _return(v: dict[str, torch.Tensor]) -> torch.Tensor:
    expected = _like(v["weight"], _EXPECTED_RETURN)
    return (v["weight"] * expected).sum(dim=-1)


def _turnover(v: dict[str, torch.Tensor]) -> torch.Tensor:
    current = _like(v["weight"], _CURRENT_WEIGHT)
    return (v["weight"] - current).abs().sum(dim=-1)


def _constraints() -> list[Constraint]:
    constraints = [
        Constraint(
            lambda v: v["weight"].sum(dim=-1),
            sense="==",
            rhs=1.0,
            weight=1_000_000.0,
            scale=1.0,
            tolerance=0.003,
            name="fully_invested",
        ),
        Constraint(
            lambda v: v["select"].sum(dim=-1),
            sense="<=",
            rhs=4.0,
            weight=100_000.0,
            scale=4.0,
            tolerance=0.0,
            name="maximum_cardinality",
        ),
        Constraint(
            lambda v: v["select"].sum(dim=-1),
            sense=">=",
            rhs=2.0,
            weight=100_000.0,
            scale=2.0,
            tolerance=0.0,
            name="minimum_cardinality",
        ),
    ]
    for index in range(6):
        constraints.extend(
            [
                Constraint(
                    lambda v, i=index: v["weight"][:, i] - 0.50 * v["select"][:, i],
                    sense="<=",
                    rhs=0.0,
                    weight=250_000.0,
                    scale=0.50,
                    tolerance=0.002,
                    name=f"asset_{index}_maximum_link",
                ),
                Constraint(
                    lambda v, i=index: 0.05 * v["select"][:, i] - v["weight"][:, i],
                    sense="<=",
                    rhs=0.0,
                    weight=250_000.0,
                    scale=0.05,
                    tolerance=0.002,
                    name=f"asset_{index}_minimum_link",
                ),
            ]
        )
    return constraints


def build_portfolio_pareto() -> MultiObjectiveProblem:
    """Return risk/return/turnover allocation with cardinality constraints."""
    return MultiObjectiveProblem(
        [
            Binary("select", size=6),
            Real("weight", 0.0, 0.50, size=6),
        ],
        [
            Objective(_risk, "risk", direction="min", unit="variance"),
            Objective(_return, "expected_return", direction="max", unit="fraction/year"),
            Objective(_turnover, "turnover", direction="min", unit="fraction"),
        ],
        constraints=_constraints(),
        name="portfolio-pareto",
    )


__all__ = ["build_portfolio_pareto"]
