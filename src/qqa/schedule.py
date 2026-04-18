"""Annealing schedules for the penalty coefficient ``bg``.

A schedule is any callable ``schedule(epoch, num_epochs) -> float``.
The default linear schedule matches the original QQA paper.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LinearBGSchedule:
    """Linear schedule ``bg(t) = min_bg + (max_bg - min_bg) * t / T``.

    When ``min_bg < 0`` and ``max_bg > 0`` the landscape transitions from
    "convex, half-integer minima" (the quasi-quantum regime) to the discrete
    regime where binary corners are favoured.

    .. note::
       The annealing loop calls this with ``epoch ∈ {0, ..., T-1}``, so the
       final value is ``min_bg + (T-1)/T * (max_bg - min_bg)`` rather than
       exactly ``max_bg`` — the schedule approaches ``max_bg`` as ``T → ∞``.
       For short runs (small ``num_epochs``) the last ``bg`` can therefore be
       visibly smaller than ``max_bg``. Increase ``num_epochs`` if you need to
       reach ``max_bg`` (matching the published QQA curves), or supply a
       custom callable for a different endpoint convention.
    """

    min_bg: float = -2.0
    max_bg: float = 0.1

    def __call__(self, epoch: int, num_epochs: int) -> float:
        if num_epochs <= 1:
            return self.max_bg
        return self.min_bg + (self.max_bg - self.min_bg) * epoch / num_epochs
