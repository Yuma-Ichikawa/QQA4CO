"""Bounded feasibility-first archive for diverse solver incumbents."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from qqa.gpu.bitpack import pack_binary, packed_hamming_distance


@dataclass(frozen=True, slots=True)
class EliteEntry:
    solution: torch.Tensor
    objective: float
    feasible: bool
    violation: float
    provenance: str = "qqa"


class EliteArchive:
    """Deduplicated archive balancing quality and Hamming diversity."""

    def __init__(
        self, maximum_size: int = 64, minimum_distance: float = 0.01, *, bitpack: bool = True
    ) -> None:
        if isinstance(maximum_size, bool) or maximum_size < 1:
            raise ValueError("maximum_size must be a positive integer.")
        if not 0 <= minimum_distance <= 1:
            raise ValueError("minimum_distance must lie in [0, 1].")
        self.maximum_size = maximum_size
        self.minimum_distance = minimum_distance
        self.bitpack = bool(bitpack)
        self._entries: list[EliteEntry] = []

    def _distance(self, left: torch.Tensor, right: torch.Tensor) -> float:
        if (
            self.bitpack
            and torch.all((left == 0) | (left == 1))
            and torch.all((right == 0) | (right == 1))
        ):
            packed_left = pack_binary(left.reshape(-1))
            packed_right = pack_binary(right.reshape(-1))
            return float(packed_hamming_distance(packed_left, packed_right).item()) / left.numel()
        return float((left != right).to(torch.float32).mean().item())

    @staticmethod
    def _rank(entry: EliteEntry) -> tuple[bool, float, float]:
        return (not entry.feasible, 0.0 if entry.feasible else entry.violation, entry.objective)

    def add(self, entry: EliteEntry) -> bool:
        candidate = entry.solution.detach().clone()
        entry = EliteEntry(
            candidate,
            float(entry.objective),
            bool(entry.feasible),
            float(entry.violation),
            entry.provenance,
        )
        replace_index: int | None = None
        for index, known in enumerate(self._entries):
            if known.solution.shape != candidate.shape:
                continue
            distance = self._distance(known.solution, candidate)
            if distance < self.minimum_distance:
                if self._rank(entry) < self._rank(known):
                    replace_index = index
                else:
                    return False
                break
        if replace_index is not None:
            self._entries[replace_index] = entry
        else:
            self._entries.append(entry)
        self._entries.sort(key=self._rank)
        del self._entries[self.maximum_size :]
        return any(item is entry for item in self._entries)

    @property
    def entries(self) -> tuple[EliteEntry, ...]:
        return tuple(self._entries)

    def solutions(self) -> torch.Tensor | None:
        if not self._entries:
            return None
        return torch.stack([entry.solution for entry in self._entries])


__all__ = ["EliteArchive", "EliteEntry"]
