"""Rolling-horizon solving with explicit solution-state transfer."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch

from qqa.result import SolveResult


@dataclass(frozen=True, slots=True)
class RollingHorizonResult:
    stages: tuple[SolveResult, ...]

    @property
    def final(self) -> SolveResult:
        if not self.stages:
            raise RuntimeError("Rolling-horizon result has no stages.")
        return self.stages[-1]


def solve_rolling_horizon(
    models: Iterable[Any],
    *,
    transfer: Callable[[torch.Tensor, Any, int], torch.Tensor | None] | None = None,
    **solve_kwargs: Any,
) -> RollingHorizonResult:
    """Solve an ordered model stream and warm-start each stage from its predecessor."""
    from qqa.api import solve  # noqa: PLC0415

    results = []
    initial = None
    for stage, model in enumerate(models):
        result = solve(model, initial_solution=initial, **solve_kwargs)
        results.append(result)
        candidate = result.solution.detach().clone()
        initial = transfer(candidate, model, stage) if transfer is not None else candidate
    if not results:
        raise ValueError("models must contain at least one stage.")
    return RollingHorizonResult(tuple(results))


__all__ = ["RollingHorizonResult", "solve_rolling_horizon"]
