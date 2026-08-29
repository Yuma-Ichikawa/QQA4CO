"""Replica-island execution across one or more explicitly selected devices."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from qqa.annealing import AnnealResult, anneal
from qqa.engines.distributed import select_diverse_migrants


@dataclass(slots=True)
class IslandResult:
    best_result: AnnealResult
    island_results: tuple[AnnealResult, ...]
    devices: tuple[str, ...]
    migrations: int
    round_results: tuple[tuple[AnnealResult, ...], ...] = ()
    deadline_reached: bool = False


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
    total_time = anneal_kwargs.pop("time_limit", None)
    if total_time is not None and (
        isinstance(total_time, bool)
        or not isinstance(total_time, (int, float))
        or total_time <= 0
    ):
        raise ValueError("time_limit must be positive or None.")
    effective_rounds = min(rounds, max(1, epochs))
    deadline = None if total_time is None else perf_counter() + float(total_time)
    initial: list[torch.Tensor | None] = [None] * len(selected_devices)
    latest: list[AnnealResult] = []
    all_results: list[AnnealResult] = []
    round_results: list[tuple[AnnealResult, ...]] = []
    migrations = 0
    for round_index in range(effective_rounds):
        remaining = None if deadline is None else deadline - perf_counter()
        if remaining is not None and remaining <= 0 and latest:
            break
        round_epochs = epochs // effective_rounds + int(round_index < epochs % effective_rounds)
        options = {
            **anneal_kwargs,
            "num_epochs": round_epochs,
            "time_limit": (
                None
                if remaining is None
                else max(1e-6, remaining / max(1, effective_rounds - round_index))
            ),
        }
        with ThreadPoolExecutor(max_workers=len(selected_devices)) as executor:
            futures = [
                executor.submit(_island_call, problem_factory, device, options, state)
                for device, state in zip(selected_devices, initial, strict=True)
            ]
            latest = [future.result() for future in futures]
        round_results.append(tuple(latest))
        all_results.extend(latest)
        if round_index == effective_rounds - 1 or (
            deadline is not None and perf_counter() >= deadline
        ):
            break
        elites = []
        for device, result in zip(selected_devices, latest, strict=True):
            population = result.final_population
            if population is None:
                elites.append(result.best_sol.reshape(1, *result.best_sol.shape))
                continue
            with torch.no_grad():
                objectives = problem_factory(device).loss_fn(population).reshape(-1)
            count = min(migration_size, len(population))
            elites.append(select_diverse_migrants(population, objectives, count=count))
        initial = [elites[(index - 1) % len(elites)].detach().cpu() for index in range(len(elites))]
        migrations += sum(len(item) for item in elites)
    if not all_results:
        raise RuntimeError("Island deadline elapsed before any QQA round completed.")
    best = min(all_results, key=lambda result: float(result.best_obj))
    return IslandResult(
        best,
        tuple(latest),
        selected_devices,
        migrations,
        tuple(round_results),
        deadline is not None and perf_counter() >= deadline,
    )


__all__ = ["IslandResult", "run_replica_islands"]
