"""Annealing schedules for the QQA discretisation coefficient.

Every schedule implements ``schedule(epoch, num_epochs) -> float`` and uses
an endpoint-inclusive convention: for a run with two or more epochs, epoch
zero returns ``minimum`` and epoch ``num_epochs - 1`` returns ``maximum``.
Keeping that convention in one module prevents subtle differences between
the Python API, CLI, UI, and benchmark runner.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


def _progress(epoch: int, num_epochs: int) -> float:
    """Return clamped endpoint-inclusive progress in ``[0, 1]``."""
    if num_epochs <= 1:
        return 1.0
    return min(1.0, max(0.0, epoch / (num_epochs - 1)))


@runtime_checkable
class Schedule(Protocol):
    """Structural protocol shared by all QQA schedules."""

    def __call__(self, epoch: int, num_epochs: int) -> float: ...


@dataclass
class LinearBGSchedule:
    """Endpoint-inclusive linear schedule.

    When ``min_bg < 0`` and ``max_bg > 0``, the penalty transitions from a
    soft-centre curvature contribution to the discrete regime where binary
    corners are favoured. Negative BG does not imply that an arbitrary
    combined objective is globally convex or has a unique minimiser.

    A one-epoch run returns ``max_bg`` so that even a smoke run performs a
    discrete-facing update.
    """

    min_bg: float = -2.0
    max_bg: float = 0.1

    def __call__(self, epoch: int, num_epochs: int) -> float:
        return self.min_bg + (self.max_bg - self.min_bg) * _progress(epoch, num_epochs)


@dataclass(frozen=True, slots=True)
class CosineBGSchedule:
    """Cosine easing with zero slope at both endpoints."""

    min_bg: float = -2.0
    max_bg: float = 0.1

    def __call__(self, epoch: int, num_epochs: int) -> float:
        t = _progress(epoch, num_epochs)
        weight = 0.5 - 0.5 * math.cos(math.pi * t)
        return self.min_bg + (self.max_bg - self.min_bg) * weight


@dataclass(frozen=True, slots=True)
class ExponentialBGSchedule:
    """Exponentially weighted interpolation that also supports negative BG."""

    min_bg: float = -2.0
    max_bg: float = 0.1
    rate: float = 5.0

    def __call__(self, epoch: int, num_epochs: int) -> float:
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise ValueError("rate must be finite and > 0.")
        t = _progress(epoch, num_epochs)
        weight = math.expm1(self.rate * t) / math.expm1(self.rate)
        return self.min_bg + (self.max_bg - self.min_bg) * weight


@dataclass(frozen=True, slots=True)
class SigmoidBGSchedule:
    """Normalised logistic schedule."""

    min_bg: float = -2.0
    max_bg: float = 0.1
    steepness: float = 10.0

    def __call__(self, epoch: int, num_epochs: int) -> float:
        if not math.isfinite(self.steepness) or self.steepness <= 0:
            raise ValueError("steepness must be finite and > 0.")
        t = _progress(epoch, num_epochs)
        low = 1.0 / (1.0 + math.exp(self.steepness / 2.0))
        high = 1.0 / (1.0 + math.exp(-self.steepness / 2.0))
        raw = 1.0 / (1.0 + math.exp(-self.steepness * (t - 0.5)))
        weight = (raw - low) / (high - low)
        return self.min_bg + (self.max_bg - self.min_bg) * weight


@dataclass(frozen=True, slots=True)
class PolynomialBGSchedule:
    """Polynomial interpolation; powers above one delay discretisation."""

    min_bg: float = -2.0
    max_bg: float = 0.1
    power: float = 2.0

    def __call__(self, epoch: int, num_epochs: int) -> float:
        if not math.isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be finite and > 0.")
        weight = _progress(epoch, num_epochs) ** self.power
        return self.min_bg + (self.max_bg - self.min_bg) * weight


@dataclass(frozen=True, slots=True)
class PiecewiseBGSchedule:
    """Piecewise-linear schedule through normalised ``(progress, value)`` knots."""

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("points must contain at least two knots.")
        progress = tuple(point[0] for point in self.points)
        if progress[0] != 0.0 or progress[-1] != 1.0:
            raise ValueError("piecewise progress must start at 0 and end at 1.")
        if any(not 0 <= value <= 1 for value in progress):
            raise ValueError("piecewise progress must lie in [0, 1].")
        if any(right <= left for left, right in zip(progress, progress[1:], strict=False)):
            raise ValueError("piecewise progress must be strictly increasing.")
        if any(not math.isfinite(value) for _, value in self.points):
            raise ValueError("piecewise values must be finite.")

    def __call__(self, epoch: int, num_epochs: int) -> float:
        t = _progress(epoch, num_epochs)
        for (left_t, left), (right_t, right) in zip(self.points, self.points[1:], strict=False):
            if t <= right_t:
                weight = (t - left_t) / (right_t - left_t)
                return left + (right - left) * weight
        return self.points[-1][1]


@dataclass(frozen=True, slots=True)
class CyclicBGSchedule:
    """Triangular cycles around a linear trend for periodic exploration."""

    min_bg: float = -2.0
    max_bg: float = 0.1
    cycles: int = 2
    amplitude: float = 0.2

    def __call__(self, epoch: int, num_epochs: int) -> float:
        if isinstance(self.cycles, bool) or self.cycles < 1:
            raise ValueError("cycles must be a positive integer.")
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("amplitude must be finite and >= 0.")
        t = _progress(epoch, num_epochs)
        base = self.min_bg + (self.max_bg - self.min_bg) * t
        if t in {0.0, 1.0}:
            return base
        triangular = 1.0 - 2.0 * abs((t * self.cycles) % 1.0 - 0.5)
        return base - self.amplitude * triangular


@dataclass(frozen=True, slots=True)
class ReheatBGSchedule:
    """Linear schedule with bounded reheating drops at configured progress."""

    min_bg: float = -2.0
    max_bg: float = 0.1
    reheats: tuple[float, ...] = (0.5, 0.75)
    strength: float = 0.25
    width: float = 0.1

    def __call__(self, epoch: int, num_epochs: int) -> float:
        if not 0 <= self.strength <= 1 or not 0 < self.width <= 1:
            raise ValueError("strength must be in [0, 1] and width in (0, 1].")
        if any(not 0 < point < 1 for point in self.reheats):
            raise ValueError("reheat points must lie strictly in (0, 1).")
        t = _progress(epoch, num_epochs)
        progress = t
        for point in self.reheats:
            if point <= t < point + self.width:
                local = (t - point) / self.width
                progress = min(progress, t - self.strength * (1.0 - local))
        progress = max(0.0, progress)
        return self.min_bg + (self.max_bg - self.min_bg) * progress


@dataclass(slots=True)
class AdaptiveBGSchedule:
    """Cosine schedule with bounded, observation-driven reheating.

    The annealer calls :meth:`observe` at a low-frequency control interval.
    Stagnation or collapsed diversity temporarily delays discretisation;
    successful windows gradually return to the base cosine path.  The class
    remains opt-in and never changes the default QQA dynamics.
    """

    min_bg: float = -2.0
    max_bg: float = 0.1
    diversity_floor: float = 0.02
    reheat_strength: float = 0.15
    recovery: float = 0.5
    _offset: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.min_bg,
            self.max_bg,
            self.diversity_floor,
            self.reheat_strength,
            self.recovery,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Adaptive schedule parameters must be finite.")
        if self.max_bg <= self.min_bg:
            raise ValueError("max_bg must exceed min_bg.")
        if not 0 <= self.diversity_floor <= 1:
            raise ValueError("diversity_floor must lie in [0, 1].")
        if not 0 < self.reheat_strength <= 1 or not 0 < self.recovery <= 1:
            raise ValueError("reheat_strength and recovery must lie in (0, 1].")

    def __call__(self, epoch: int, num_epochs: int) -> float:
        base = CosineBGSchedule(self.min_bg, self.max_bg)(epoch, num_epochs)
        value = base + self._offset * (self.max_bg - self.min_bg)
        # Keep exact endpoint semantics even after an earlier reheat.
        if _progress(epoch, num_epochs) >= 1.0:
            return self.max_bg
        return min(self.max_bg, max(self.min_bg, value))

    def observe(self, *, improved: bool, diversity_ratio: float | None = None) -> None:
        """Update the bounded reheat offset from one control window."""
        if diversity_ratio is not None and (
            not math.isfinite(diversity_ratio) or diversity_ratio < 0
        ):
            raise ValueError("diversity_ratio must be finite and non-negative or None.")
        collapsed = diversity_ratio is not None and diversity_ratio < self.diversity_floor
        if not improved or collapsed:
            self._offset = max(-0.75, self._offset - self.reheat_strength)
        else:
            self._offset = min(0.0, self._offset + self.reheat_strength * self.recovery)


def make_schedule(
    name: str,
    *,
    minimum: float = -2.0,
    maximum: float = 0.1,
) -> Schedule:
    """Build a validated standard schedule by a stable public name."""
    factories: dict[str, Callable[..., Schedule]] = {
        "linear": LinearBGSchedule,
        "cosine": CosineBGSchedule,
        "exponential": ExponentialBGSchedule,
        "sigmoid": SigmoidBGSchedule,
        "polynomial": PolynomialBGSchedule,
        "cyclic": CyclicBGSchedule,
        "reheat": ReheatBGSchedule,
        "adaptive": AdaptiveBGSchedule,
    }
    try:
        factory = factories[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown schedule {name!r}; choose from {sorted(factories)}.") from exc
    return factory(min_bg=minimum, max_bg=maximum)


__all__ = [
    "AdaptiveBGSchedule",
    "CosineBGSchedule",
    "CyclicBGSchedule",
    "ExponentialBGSchedule",
    "LinearBGSchedule",
    "PiecewiseBGSchedule",
    "PolynomialBGSchedule",
    "ReheatBGSchedule",
    "Schedule",
    "SigmoidBGSchedule",
    "make_schedule",
]
