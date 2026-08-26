"""Replica-island execution across one or more explicitly selected devices."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from qqa.annealing import AnnealResult, anneal


@dataclass(slots=True)
class IslandResult:
    best_result: AnnealResult
    island_results: tuple[AnnealResult, ...]
    devices: tuple[str, ...]
    migrations: int


def _island_call(
    problem_factory: Callable[[str], Any],
    device: str,
    kwargs: dict[str, Any],
    initial_state: torch.Tensor | None,
) -> AnnealResult:
    problem = problem_factory(device)
    options = {**kwargs, "device": device, "return_population": True, "verbose": False}
    if initial_state is not None:
        replicas = int(options.get("sol_size", len(initial_state)))
        repeats = (replicas + len(initial_state) - 1) // len(initial_state)
        expanded = torch.cat([initial_state] * repeats, dim=0)[:replicas]
        options["initial_state"] = expanded.to(device)
    return anneal(problem, **options)


def run_replica_islands(
    problem_factory: Callable[[str], Any],
    *,
    devices: tuple[str, ...] | list[str],
    rounds: int = 2,
    migration_size: int = 4,
    **anneal_kwargs: Any,
) -> IslandResult:
    """Run independent QQA islands and rotate elites between rounds.

    A factory is required because many legacy problems own device tensors at
    construction time. It prevents unsafe implicit cross-device moves.
    """
    selected_devices = tuple(str(torch.device(item)) for item in devices)
    if not selected_devices:
        raise ValueError("At least one device is required.")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer.")
    if (
        isinstance(migration_size, bool)
        or not isinstance(migration_size, int)
        or migration_size < 1
    ):
        raise ValueError("migration_size must be a positive integer.")
    epochs = int(anneal_kwargs.pop("num_epochs", 1500))
    per_round = max(1, (epochs + rounds - 1) // rounds)
    options = {**anneal_kwargs, "num_epochs": per_round}
    initial: list[torch.Tensor | None] = [None] * len(selected_devices)
    latest: list[AnnealResult] = []
    migrations = 0
    for round_index in range(rounds):
        with ThreadPoolExecutor(max_workers=len(selected_devices)) as executor:
            futures = [
                executor.submit(_island_call, problem_factory, device, options, state)
                for device, state in zip(selected_devices, initial, strict=True)
            ]
            latest = [future.result() for future in futures]
        if round_index == rounds - 1:
            break
        elites = []
        for result in latest:
            population = result.final_population
            elites.append(
                result.best_sol.reshape(1, *result.best_sol.shape)
                if population is None
                else population[:migration_size]
            )
        initial = [elites[(index - 1) % len(elites)].detach().cpu() for index in range(len(elites))]
        migrations += sum(len(item) for item in elites)
    best = min(latest, key=lambda result: float(result.best_obj))
    return IslandResult(best, tuple(latest), selected_devices, migrations)


__all__ = ["IslandResult", "run_replica_islands"]
