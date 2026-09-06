"""One monotonic wall-clock budget shared by every solve stage."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import perf_counter


@dataclass(slots=True)
class SolveContext:
    """Track an end-to-end deadline without inventing stage time.

    The clock starts at the public API boundary, so loading, inspection,
    presolve, warm-start generation, search, repair, and certification all
    consume the same user budget.
    """

    budget: float | None = None
    started: float = field(default_factory=perf_counter)

    def __post_init__(self) -> None:
        if self.budget is not None and (
            isinstance(self.budget, bool) or not math.isfinite(self.budget) or self.budget <= 0
        ):
            raise ValueError("SolveContext budget must be finite and positive or None.")

    @property
    def elapsed(self) -> float:
        return max(0.0, perf_counter() - self.started)

    @property
    def remaining(self) -> float | None:
        if self.budget is None:
            return None
        return max(0.0, self.budget - self.elapsed)

    @property
    def expired(self) -> bool:
        remaining = self.remaining
        return remaining is not None and remaining <= 0.0

    def allocation(self, fraction: float) -> float | None:
        """Cap one planned stage by its fraction and the actual remainder."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Stage budget fraction must be in [0, 1].")
        remaining = self.remaining
        if self.budget is None:
            return None
        if remaining is None:
            raise RuntimeError("A bounded solve context has no remaining-time value.")
        return min(remaining, self.budget * fraction)


__all__ = ["SolveContext"]
