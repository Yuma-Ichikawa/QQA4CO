"""A constrained mixed-variable black-box process-tuning benchmark."""

from __future__ import annotations

import math
from typing import cast

from qqa.blackbox import BlackBoxConstraint, BlackBoxProblem
from qqa.blackbox.problem import ScalarPoint
from qqa.mixed import Binary, Integer, Real


def _scalar(point: ScalarPoint, name: str) -> float | int:
    """Return a scalar field guaranteed by this model's declarations."""
    value = point[name]
    if isinstance(value, list):
        raise TypeError(f"{name} must be scalar.")
    return cast(float | int, value)


def _process_response(point: ScalarPoint) -> tuple[float, float, float]:
    """Return yield, heat load, and hourly profit for one plant setting."""
    catalyst = int(_scalar(point, "catalyst"))
    reactors = int(_scalar(point, "reactors"))
    temperature = float(_scalar(point, "temperature"))
    residence = float(_scalar(point, "residence_time"))
    recycle = float(_scalar(point, "recycle"))

    preferred_temperature = 372.0 + 9.0 * catalyst
    yield_fraction = (
        0.835
        + 0.055 * catalyst
        - 7.0e-5 * (temperature - preferred_temperature) ** 2
        - 0.025 * (residence - 2.1) ** 2
        + 0.045 * recycle
        - 0.006 * (reactors - 3) ** 2
    )
    yield_fraction = min(0.96, max(0.45, yield_fraction))
    throughput = reactors * (1.15 / residence) * (1.0 + 0.28 * recycle)
    heat_load = reactors * (0.11 * (temperature - 300.0) ** 1.35) * (1.0 + recycle)
    revenue = 1_850.0 * yield_fraction * throughput
    utility_cost = 8.5 * heat_load + 110.0 * recycle * throughput
    catalyst_cost = 280.0 * catalyst + 45.0 * reactors
    profit = revenue - utility_cost - catalyst_cost
    return yield_fraction, heat_load, profit


def build_process_blackbox() -> BlackBoxProblem:
    """Return a realistic simulator-like profit optimisation problem.

    The objective deliberately uses Python branching and ``math``-style scalar
    operations. It therefore has no autograd graph and exercises the black-box
    path rather than the differentiable mixed solver.
    """

    return BlackBoxProblem(
        [
            Binary("catalyst"),
            Integer("reactors", 1, 5),
            Real("temperature", 320.0, 420.0),
            Real("residence_time", 0.7, 4.0),
            Real("recycle", 0.0, 0.6),
        ],
        lambda point: _process_response(point)[2],
        direction="max",
        constraints=[
            BlackBoxConstraint(
                lambda point: _process_response(point)[0],
                sense=">=",
                rhs=0.82,
                tolerance=1e-4,
                scale=0.05,
                name="minimum_yield",
            ),
            BlackBoxConstraint(
                lambda point: _process_response(point)[1],
                sense="<=",
                rhs=135.0,
                tolerance=1e-3,
                scale=25.0,
                name="heat_capacity",
            ),
            BlackBoxConstraint(
                lambda point: (
                    math.ceil(_scalar(point, "reactors") / 2) + _scalar(point, "catalyst")
                ),
                sense="<=",
                rhs=3.0,
                scale=1.0,
                name="operator_crews",
            ),
        ],
        name="process-blackbox",
    )


__all__ = ["build_process_blackbox"]
