"""SAT/weighted-MaxSAT runtime with an explicit optional dependency boundary."""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from time import perf_counter

import torch

from qqa.model.ir import ClauseFactor, ModelIR, VariableDomain


@dataclass(frozen=True, slots=True)
class SATResult:
    solution: torch.Tensor
    objective: float
    status: str
    runtime: float
    proven: bool
    backend: str


def _clauses(factor: ClauseFactor) -> list[list[int]]:
    return [
        [
            (int(index) + 1) * (1 if int(sign) > 0 else -1)
            for index, sign in zip(indices, signs, strict=True)
        ]
        for indices, signs in zip(factor.indices.tolist(), factor.signs.tolist(), strict=True)
    ]


def _solve_rc2_payload(
    hard: list[list[int]], soft: list[tuple[list[int], int]]
) -> list[int] | None:
    try:
        from pysat.examples.rc2 import RC2
        from pysat.formula import WCNF
    except ImportError as exc:
        raise ImportError(
            "Install `qqa[discs]` (python-sat) to use the SAT/MaxSAT runtime."
        ) from exc
    formula = WCNF()
    for clause in hard:
        formula.append(clause)
    for clause, weight in soft:
        formula.append(clause, weight=weight)
    with RC2(formula, adapt=True, exhaust=True) as solver:
        return solver.compute()


def _rc2_worker(
    hard: list[list[int]],
    soft: list[tuple[list[int], int]],
    output: mp.Queue,
) -> None:
    try:
        output.put(("ok", _solve_rc2_payload(hard, soft)))
    except Exception as exc:  # noqa: BLE001 - explicit process boundary
        output.put(("error", type(exc).__name__))


def _solve_with_deadline(
    hard: list[list[int]], soft: list[tuple[list[int], int]], time_limit: float | None
) -> tuple[list[int] | None, bool]:
    if time_limit is None:
        return _solve_rc2_payload(hard, soft), False
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else "spawn")
    output = context.Queue(maxsize=1)
    process = context.Process(target=_rc2_worker, args=(hard, soft, output), daemon=True)
    process.start()
    process.join(timeout=time_limit)
    if process.is_alive():
        process.terminate()
        process.join()
        output.close()
        return None, True
    try:
        envelope = output.get(timeout=0.25)
    except Empty as exc:
        raise RuntimeError("SAT worker terminated without a result.") from exc
    finally:
        output.close()
    if envelope[0] == "error":
        if envelope[1] == "ImportError":
            raise ImportError(
                "Install `qqa[discs]` (python-sat) to use the SAT/MaxSAT runtime."
            )
        raise RuntimeError(f"SAT worker failed ({envelope[1]}).")
    return envelope[1], False


def solve_sat_model_ir(model: ModelIR, *, time_limit: float | None = None) -> SATResult:
    """Solve native clause factors with PySAT RC2/SAT and proof-safe semantics."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
    if time_limit is not None and (
        isinstance(time_limit, bool) or not math.isfinite(time_limit) or time_limit <= 0
    ):
        raise ValueError("time_limit must be finite and positive or None.")
    if any(block.domain is not VariableDomain.BINARY for block in model.variables):
        raise NotImplementedError("SAT runtime requires binary variable blocks.")
    objective = [factor for factor in model.objective.factors if isinstance(factor, ClauseFactor)]
    if len(objective) != len(model.objective.factors):
        raise NotImplementedError("SAT runtime accepts clause objective factors only.")
    hard = []
    for row in model.constraints:
        if (
            row.sense != "<="
            or row.rhs != 0
            or any(not isinstance(factor, ClauseFactor) for factor in row.expression.factors)
        ):
            raise NotImplementedError("Hard SAT constraints must be clause penalties <= 0.")
        for factor in row.expression.factors:
            if not isinstance(factor, ClauseFactor):
                raise NotImplementedError("Hard SAT constraints require clause factors.")
            hard.extend(_clauses(factor))
    soft: list[tuple[list[int], int]] = []
    for factor in objective:
        assert factor.weights is not None
        for clause, weight in zip(_clauses(factor), factor.weights.tolist(), strict=True):
            rounded = int(round(float(weight)))
            if abs(rounded - float(weight)) > 1e-9 or rounded <= 0:
                raise ValueError("PySAT RC2 requires positive integral clause weights.")
            soft.append((clause, rounded))
    started = perf_counter()
    assignment, timed_out = _solve_with_deadline(hard, soft, time_limit)
    if timed_out:
        return SATResult(
            torch.empty(0),
            float("inf"),
            "limit_reached_no_incumbent",
            perf_counter() - started,
            False,
            "pysat-rc2-isolated",
        )
    if assignment is None:
        return SATResult(
            torch.empty(0),
            float("inf"),
            "infeasible_proven",
            perf_counter() - started,
            True,
            "pysat-rc2-isolated" if time_limit is not None else "pysat-rc2",
        )
    values = torch.zeros(model.num_variables, dtype=torch.float64)
    for literal in assignment:
        if 1 <= abs(literal) <= model.num_variables:
            values[abs(literal) - 1] = float(literal > 0)
    objective_value = float(model.objective_values(values)[0].item())
    return SATResult(
        values,
        objective_value,
        "optimal",
        perf_counter() - started,
        True,
        "pysat-rc2-isolated" if time_limit is not None else "pysat-rc2",
    )


__all__ = ["SATResult", "solve_sat_model_ir"]
