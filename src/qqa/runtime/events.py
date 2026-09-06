"""Versioned, environment-neutral solver event stream."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from time import perf_counter
from typing import Any

import torch

from qqa.callbacks import Callback, CallbackState


class EventKind(str, Enum):
    SOLVE_STARTED = "SolveStarted"
    PRESOLVE_REDUCED = "PresolveReduced"
    RELAXATION_UPDATED = "RelaxationUpdated"
    CANDIDATE_GENERATED = "CandidateGenerated"
    CANDIDATE_REPAIRED = "CandidateRepaired"
    INCUMBENT_IMPROVED = "IncumbentImproved"
    SEARCH_MERIT_IMPROVED = "SearchMeritImproved"
    DUAL_BOUND_IMPROVED = "DualBoundImproved"
    CUT_ADDED = "CutAdded"
    REPLICA_RESTARTED = "ReplicaRestarted"
    REPLICA_EXCHANGED = "ReplicaExchanged"
    CONSTRAINT_VIOLATION_UPDATED = "ConstraintViolationUpdated"
    KERNEL_PROFILED = "KernelProfiled"
    SOLVE_FINISHED = "SolveFinished"


def _portable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Event payloads must contain finite numbers.")
        return value
    if torch.is_tensor(value):
        if value.numel() > 4096:
            raise ValueError("Event tensors are limited to 4096 values; emit an aggregate instead.")
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    raise TypeError(f"Unsupported event payload value: {type(value).__name__}.")


@dataclass(frozen=True, slots=True)
class SolveEvent:
    sequence: int
    kind: EventKind
    elapsed_seconds: float
    payload: dict[str, Any]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("Event sequence must be a non-negative integer.")
        object.__setattr__(self, "kind", EventKind(self.kind))
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        object.__setattr__(self, "payload", _portable(self.payload))
        if self.schema_version != 2:
            raise ValueError("Only event schema version 2 is supported.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


class EventRecorder(Callback):
    """Record annealing telemetry without a per-epoch device synchronisation.

    Epoch metrics are written to one preallocated device tensor and transferred
    only from ``on_train_end``.  External engines may also call :meth:`emit` at
    their existing control points.
    """

    def __init__(
        self,
        *,
        stride: int = 10,
        maximum_events: int = 10_000,
        started_at: float | None = None,
    ) -> None:
        if isinstance(stride, bool) or stride < 1:
            raise ValueError("stride must be a positive integer.")
        if isinstance(maximum_events, bool) or maximum_events < 2:
            raise ValueError("maximum_events must be an integer >= 2.")
        self.stride = int(stride)
        self.maximum_events = int(maximum_events)
        self.events: list[SolveEvent] = []
        self._started = perf_counter() if started_at is None else float(started_at)
        if not math.isfinite(self._started):
            raise ValueError("started_at must be a finite monotonic timestamp or None.")
        self._buffer: torch.Tensor | None = None
        self._records = 0
        self._elapsed: list[float] = []

    def emit(
        self,
        kind: EventKind | str,
        payload: dict[str, Any] | None = None,
        *,
        elapsed_seconds: float | None = None,
    ) -> None:
        if len(self.events) >= self.maximum_events:
            return
        elapsed = perf_counter() - self._started if elapsed_seconds is None else elapsed_seconds
        self.events.append(SolveEvent(len(self.events), EventKind(kind), elapsed, payload or {}))

    def on_train_begin(self, state: CallbackState) -> None:
        records = max(1, (state.num_epochs + self.stride - 1) // self.stride + 1)
        self._buffer = torch.empty((records, 7), device=state.x.device, dtype=torch.float64)
        self._records = 0
        self._elapsed = []
        if not any(event.kind is EventKind.SOLVE_STARTED for event in self.events):
            self.emit(
                EventKind.SOLVE_STARTED,
                {"epochs": state.num_epochs, "replicas": int(state.x.shape[0])},
                elapsed_seconds=0.0,
            )

    def on_epoch_end(self, state: CallbackState) -> None:
        if state.epoch % self.stride != 0 and state.epoch != state.num_epochs - 1:
            return
        if self._buffer is None:
            self.on_train_begin(state)
        assert self._buffer is not None
        best = torch.as_tensor(state.best_obj, device=state.x.device, dtype=torch.float64)
        self._buffer[self._records] = torch.stack(
            (
                best.reshape(-1).amin(),
                state.losses.detach().to(torch.float64).mean(),
                state.losses.detach().to(torch.float64).amin(),
                state.penalties.detach().to(torch.float64).mean(),
                torch.as_tensor(state.diversity, device=state.x.device, dtype=torch.float64),
                torch.tensor(float(state.bg), device=state.x.device, dtype=torch.float64),
                torch.tensor(float(state.epoch), device=state.x.device, dtype=torch.float64),
            )
        )
        self._elapsed.append(max(0.0, perf_counter() - self._started))
        self._records += 1

    def on_train_end(self, state: CallbackState) -> None:
        rows = (
            torch.empty((0, 7), dtype=torch.float64)
            if self._buffer is None
            else self._buffer[: self._records].detach().cpu()
        )
        previous = math.inf
        if len(self._elapsed) != len(rows):
            raise RuntimeError("Event timestamps and device telemetry are misaligned.")
        for (best, mean, minimum, penalty, diversity, bg, epoch), elapsed in zip(
            rows.tolist(), self._elapsed, strict=True
        ):
            self.emit(
                EventKind.RELAXATION_UPDATED,
                {
                    "epoch": int(epoch),
                    "loss_mean": mean,
                    "loss_min": minimum,
                    "penalty_mean": penalty,
                    "diversity": diversity,
                    "beta": bg,
                },
                elapsed_seconds=elapsed,
            )
            if best < previous:
                self.emit(
                    EventKind.SEARCH_MERIT_IMPROVED,
                    {"epoch": int(epoch), "search_merit": best},
                    elapsed_seconds=elapsed,
                )
                previous = best

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]


__all__ = ["EventKind", "EventRecorder", "SolveEvent"]
