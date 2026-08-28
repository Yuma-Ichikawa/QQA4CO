"""Correctly synchronised kernel timing, counters, and bounded autotuning."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class KernelProfile:
    name: str
    elapsed_seconds: float
    device: str
    bytes_read: int = 0
    bytes_written: int = 0
    operations: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        transferred = self.bytes_read + self.bytes_written
        payload["bandwidth_bytes_per_second"] = transferred / max(self.elapsed_seconds, 1e-30)
        payload["operations_per_second"] = self.operations / max(self.elapsed_seconds, 1e-30)
        return payload


def profile_kernel(
    name: str,
    function: Callable[[], Any],
    *,
    device: str | torch.device,
    bytes_read: int = 0,
    bytes_written: int = 0,
    operations: int = 0,
) -> tuple[Any, KernelProfile]:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end)) / 1000.0
    else:
        started = perf_counter()
        result = function()
        elapsed = perf_counter() - started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("Kernel timer returned an invalid duration.")
    return result, KernelProfile(
        name,
        elapsed,
        str(resolved),
        int(bytes_read),
        int(bytes_written),
        int(operations),
    )


class KernelAutotuner:
    def __init__(self) -> None:
        self._choices: dict[str, str] = {}
        self.profiles: dict[str, tuple[KernelProfile, ...]] = {}

    def choose(
        self,
        key: str,
        candidates: dict[str, Callable[[], Any]],
        *,
        device: str | torch.device,
    ) -> str:
        if key in self._choices:
            return self._choices[key]
        if not candidates:
            raise ValueError("At least one autotuning candidate is required.")
        rows = []
        for name, candidate in candidates.items():
            _, profile = profile_kernel(name, candidate, device=device)
            rows.append(profile)
        selected = min(rows, key=lambda row: row.elapsed_seconds).name
        self._choices[key] = selected
        self.profiles[key] = tuple(rows)
        return selected


__all__ = ["KernelAutotuner", "KernelProfile", "profile_kernel"]
