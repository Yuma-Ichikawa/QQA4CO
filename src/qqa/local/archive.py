"""Bounded feasibility-first archive for diverse solver incumbents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qqa.callbacks import Callback, CallbackState
from qqa.gpu.bitpack import pack_binary, packed_hamming_distance


@dataclass(frozen=True, slots=True)
class EliteEntry:
    solution: torch.Tensor
    objective: float
    feasible: bool
    violation: float
    provenance: str = "qqa"
    objective_vector: tuple[float, ...] = ()
    violation_pattern: tuple[bool, ...] = ()
    lineage: str | None = None
    node_id: int | None = None
    core_id: str | None = None
    epoch: int | None = None


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
            tuple(float(value) for value in entry.objective_vector),
            tuple(bool(value) for value in entry.violation_pattern),
            entry.lineage,
            entry.node_id,
            entry.core_id,
            entry.epoch,
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

    @property
    def best_feasible(self) -> EliteEntry | None:
        return next((entry for entry in self._entries if entry.feasible), None)

    @property
    def best_infeasible(self) -> EliteEntry | None:
        infeasible = [entry for entry in self._entries if not entry.feasible]
        return min(infeasible, key=self._rank, default=None)

    def restart_centres(self, count: int) -> torch.Tensor | None:
        """Return quality/diversity centres for restart, RINS, or relinking."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer.")
        if not self._entries:
            return None
        selected = [self._entries[0]]
        while len(selected) < min(count, len(self._entries)):
            selected_ids = {id(entry) for entry in selected}
            candidate = max(
                (entry for entry in self._entries if id(entry) not in selected_ids),
                key=lambda entry: min(
                    self._distance(entry.solution, known.solution) for known in selected
                ),
            )
            selected.append(candidate)
        return torch.stack([entry.solution for entry in selected])

    def diagnostics(self) -> dict[str, Any]:
        return {
            "size": len(self._entries),
            "feasible": sum(entry.feasible for entry in self._entries),
            "infeasible": sum(not entry.feasible for entry in self._entries),
            "lineages": sorted({entry.lineage for entry in self._entries if entry.lineage}),
            "epochs": sorted({entry.epoch for entry in self._entries if entry.epoch is not None}),
        }


class HistoricalEliteCallback(Callback):
    """Capture bounded historical quality/diversity candidates on-device.

    Candidate selection performs no ``item()`` or host transfer in the hot
    loop.  The bounded snapshots are copied once at solve completion and then
    folded into :class:`EliteArchive`.
    """

    def __init__(
        self,
        *,
        maximum_size: int = 64,
        interval: int = 25,
        candidates_per_snapshot: int = 4,
    ) -> None:
        if maximum_size < 1 or interval < 1 or candidates_per_snapshot < 1:
            raise ValueError("Archive size, interval, and snapshot size must be positive.")
        self.archive = EliteArchive(maximum_size=maximum_size)
        self.interval = int(interval)
        self.candidates_per_snapshot = int(candidates_per_snapshot)
        self._solutions: list[torch.Tensor] = []
        self._objectives: list[torch.Tensor] = []
        self._violations: list[torch.Tensor] = []
        self._epochs: list[int] = []
        self._maximum_snapshots = max(2, (maximum_size * 4) // candidates_per_snapshot)

    def on_epoch_end(self, state: CallbackState) -> None:
        if (state.epoch + 1) % self.interval != 0 and state.epoch != state.num_epochs - 1:
            return
        with torch.no_grad():
            solutions = state.relaxation.project(state.x)
            objective_fn = getattr(state.problem, "ranking_objective", state.problem.loss_fn)
            objectives = objective_fn(solutions)
            merit = objectives.reshape(objectives.shape[0], -1).mean(dim=1)
            count = min(self.candidates_per_snapshot, len(solutions))
            quality = torch.argsort(merit)[:count]
            flattened = solutions.reshape(len(solutions), -1)
            centre = flattened[quality[:1]]
            distance = (flattened != centre).to(torch.float32).mean(dim=1)
            diverse = torch.argsort(distance, descending=True)[:count]
            selected = torch.unique(torch.cat((quality, diverse)), sorted=False)[: 2 * count]
            violations = torch.zeros(len(selected), device=solutions.device, dtype=torch.float64)
            violation_fn = getattr(state.problem, "constraint_violations", None)
            constraints = tuple(getattr(state.problem, "constraints", ()))
            if callable(violation_fn) and constraints:
                rows = violation_fn(solutions[selected])
                violations = torch.stack(
                    [rows[row.name].to(torch.float64) / float(row.scale) for row in constraints],
                    dim=1,
                ).sum(dim=1)
            self._solutions.append(solutions[selected].detach().clone())
            self._objectives.append(merit[selected].detach().clone())
            self._violations.append(violations.detach().clone())
            self._epochs.append(state.epoch)
            if len(self._solutions) > self._maximum_snapshots:
                self._solutions.pop(0)
                self._objectives.pop(0)
                self._violations.pop(0)
                self._epochs.pop(0)

    def on_train_end(self, state: CallbackState) -> None:  # noqa: ARG002
        for solutions, objectives, violations, epoch in zip(
            self._solutions, self._objectives, self._violations, self._epochs, strict=True
        ):
            solutions_cpu = solutions.detach().cpu()
            objectives_cpu = objectives.detach().cpu().tolist()
            violations_cpu = violations.detach().cpu().tolist()
            for index, (objective, violation) in enumerate(
                zip(objectives_cpu, violations_cpu, strict=True)
            ):
                self.archive.add(
                    EliteEntry(
                        solutions_cpu[index],
                        float(objective),
                        float(violation) <= 1e-8,
                        float(violation),
                        provenance="historical-qqa",
                        lineage=f"replica-snapshot-{epoch}",
                        epoch=epoch,
                    )
                )

    def device_restart_centres(self, count: int) -> torch.Tensor | None:
        """Return recent quality/diversity centres without a host transfer."""
        if count < 1 or not self._solutions:
            return None
        candidates = torch.cat(self._solutions[-4:], dim=0)
        return candidates[: min(count, len(candidates))]

    def checkpoint_tensors(self) -> dict[str, torch.Tensor]:
        """Return bounded device snapshots for pickle-free continuation."""
        if not self._solutions:
            return {}
        device = self._solutions[0].device
        return {
            "archive_solutions": torch.cat(self._solutions),
            "archive_objectives": torch.cat(self._objectives),
            "archive_violations": torch.cat(self._violations),
            "archive_snapshot_sizes": torch.tensor(
                [len(values) for values in self._solutions], device=device, dtype=torch.int64
            ),
            "archive_snapshot_epochs": torch.tensor(self._epochs, device=device, dtype=torch.int64),
        }

    def restore_checkpoint_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Restore archive snapshots after checkpoint checksums were verified."""
        required = {
            "archive_solutions",
            "archive_objectives",
            "archive_violations",
            "archive_snapshot_sizes",
            "archive_snapshot_epochs",
        }
        present = required & tensors.keys()
        if not present:
            return
        if present != required:
            raise ValueError("Checkpoint contains an incomplete historical archive.")
        sizes = tensors["archive_snapshot_sizes"].to(torch.int64).cpu().tolist()
        epochs = tensors["archive_snapshot_epochs"].to(torch.int64).cpu().tolist()
        if len(sizes) != len(epochs) or any(size < 1 for size in sizes):
            raise ValueError("Checkpoint historical archive boundaries are invalid.")
        total = sum(sizes)
        solutions = tensors["archive_solutions"]
        objectives = tensors["archive_objectives"].reshape(-1)
        violations = tensors["archive_violations"].reshape(-1)
        if len(solutions) != total or len(objectives) != total or len(violations) != total:
            raise ValueError("Checkpoint historical archive tensors do not align.")
        self._solutions = list(solutions.split(sizes))[-self._maximum_snapshots :]
        self._objectives = list(objectives.split(sizes))[-self._maximum_snapshots :]
        self._violations = list(violations.split(sizes))[-self._maximum_snapshots :]
        self._epochs = [int(epoch) for epoch in epochs][-self._maximum_snapshots :]


__all__ = ["EliteArchive", "EliteEntry", "HistoricalEliteCallback"]
