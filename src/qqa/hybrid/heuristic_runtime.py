"""Pure numerical helpers for the opt-in SCIP-guided QQA heuristic.

This module deliberately has no PySCIPOpt dependency.  Keeping population
construction and candidate scoring here makes them independently testable and
keeps Python/Torch work out of the SCIP callback orchestration.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager

import numpy as np
import torch

from qqa.mixed import MixedProblem


@contextmanager
def torch_thread_budget(threads: int):
    """Temporarily cap Torch without expanding the pool after a solver call."""
    previous = torch.get_num_threads()
    if previous == threads:
        yield
        return
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        # Expanded pools can keep workers alive between benchmark instances.
        if previous < threads:
            torch.set_num_threads(previous)


@contextmanager
def torch_seed(seed: int, device: str):
    """Isolate QQA RNG state, including the selected CUDA device when used."""
    target = torch.device(device)
    devices: list[int] = []
    if target.type == "cuda" and torch.cuda.is_available():
        devices = [target.index if target.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def build_initial_population(
    seeds: Iterable[np.ndarray],
    *,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sol_size: int,
    seed: int,
) -> np.ndarray:
    """Create one bounded, deduplicated population with vectorised random fill."""
    target = np.asarray(target, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if target.shape != lower.shape or target.shape != upper.shape or target.ndim != 1:
        raise ValueError("target, lower, and upper must be aligned one-dimensional arrays.")
    if sol_size < 1:
        raise ValueError("sol_size must be positive.")

    retained: list[np.ndarray] = []
    signatures: set[bytes] = set()
    for values in seeds:
        bounded = np.clip(np.asarray(values, dtype=np.float64), lower, upper)
        if bounded.shape != target.shape:
            raise ValueError("Every initial population seed must match target shape.")
        signature = np.round(bounded, decimals=10).tobytes()
        if signature not in signatures:
            signatures.add(signature)
            retained.append(bounded)
            if len(retained) == sol_size:
                break

    population = np.empty((sol_size, target.size), dtype=np.float64)
    populated = len(retained)
    if populated:
        population[:populated] = retained
    remaining = sol_size - populated
    if not remaining:
        return population

    rng = np.random.default_rng(seed)
    probability = np.clip((target - lower) / np.maximum(1.0, upper - lower), 0.0, 1.0)
    row_numbers = np.arange(populated, sol_size)
    probabilities = np.where((row_numbers % 2 == 0)[:, None], probability, 0.5)
    draws = rng.random((remaining, target.size))
    population[populated:] = np.where(draws < probabilities, upper, lower)
    return population


def select_repair_positions(
    problem: MixedProblem,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    positions: Sequence[int],
    *,
    max_changes: int,
    beam_width: int,
) -> list[int]:
    """Choose a jointly improving change set with batched beam evaluation."""
    incumbent = np.asarray(incumbent, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if incumbent.shape != candidate.shape or incumbent.shape != (len(positions),):
        raise ValueError("incumbent, candidate, and positions must align.")
    if max_changes < 1 or beam_width < 1:
        return []
    differing = np.flatnonzero(np.abs(candidate - incumbent) > 0.5).tolist()
    if not differing:
        return []

    with torch.no_grad():
        base_loss = float(problem.loss_fn(torch.as_tensor(incumbent, dtype=torch.float64))[0])
    beam: list[tuple[tuple[int, ...], np.ndarray]] = [((), incumbent.copy())]
    best_loss = base_loss
    best_selection: tuple[int, ...] = ()
    depth_limit = min(max_changes, len(differing))
    for _ in range(depth_limit):
        selectors: list[tuple[int, ...]] = []
        points: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for selected, point in beam:
            selected_set = set(selected)
            for local_position in differing:
                if local_position in selected_set:
                    continue
                expanded = tuple(sorted((*selected, local_position)))
                if expanded in seen:
                    continue
                seen.add(expanded)
                trial = point.copy()
                trial[local_position] = candidate[local_position]
                selectors.append(expanded)
                points.append(trial)
        if not points:
            break
        with torch.no_grad():
            losses = (
                problem.loss_fn(torch.as_tensor(np.stack(points), dtype=torch.float64))
                .detach()
                .cpu()
                .numpy()
            )
        order = np.argsort(losses, kind="stable")
        beam = [(selectors[index], points[index]) for index in order[:beam_width]]
        leading = int(order[0])
        leading_loss = float(losses[leading])
        if leading_loss < best_loss:
            best_loss = leading_loss
            best_selection = selectors[leading]

    tolerance = 1e-8 * max(1.0, abs(base_loss), abs(best_loss))
    if not best_selection or best_loss >= base_loss - tolerance:
        return []
    return [int(positions[local_position]) for local_position in best_selection]


def rank_repair_candidates(
    problem: MixedProblem,
    ranked: Sequence[np.ndarray],
    *,
    reference: np.ndarray,
    positions: Sequence[int],
    lower: np.ndarray,
    upper: np.ndarray,
    max_changes: int,
    beam_width: int,
    selector: Callable[..., list[int]] = select_repair_positions,
) -> tuple[list[np.ndarray], dict[bytes, list[int]]]:
    """Rank QQA candidates by partial-repair loss using one batched loss call."""
    if not ranked:
        return [], {}
    by_full_position = {
        int(full_position): local_position
        for local_position, full_position in enumerate(positions)
    }
    plans: dict[bytes, list[int]] = {}
    evaluation_points: list[np.ndarray] = []
    classes: list[int] = []
    for reduced in ranked:
        rounded = np.clip(np.rint(np.asarray(reduced, dtype=np.float64)), lower, upper)
        fixes = selector(
            problem,
            reference,
            rounded,
            positions,
            max_changes=max_changes,
            beam_width=beam_width,
        )
        plans[np.asarray(rounded, dtype=np.int64).tobytes()] = fixes
        if fixes:
            partial = reference.copy()
            for full_position in fixes:
                partial[by_full_position[full_position]] = rounded[
                    by_full_position[full_position]
                ]
            evaluation_points.append(partial)
            classes.append(0)
        else:
            evaluation_points.append(rounded)
            classes.append(1)

    with torch.no_grad():
        losses = (
            problem.loss_fn(torch.as_tensor(np.stack(evaluation_points), dtype=torch.float64))
            .detach()
            .cpu()
            .numpy()
        )
    order = sorted(range(len(ranked)), key=lambda index: (classes[index], losses[index], index))
    return [ranked[index] for index in order], plans


__all__ = [
    "build_initial_population",
    "rank_repair_candidates",
    "select_repair_positions",
    "torch_seed",
    "torch_thread_budget",
]
