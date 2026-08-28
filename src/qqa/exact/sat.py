"""SAT/weighted-MaxSAT runtime with an explicit optional dependency boundary."""

from __future__ import annotations

from dataclasses import dataclass
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


def solve_sat_model_ir(model: ModelIR, *, time_limit: float | None = None) -> SATResult:
    """Solve native clause factors with PySAT RC2/SAT and proof-safe semantics."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
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
    started = perf_counter()
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
    for factor in objective:
        assert factor.weights is not None
        for clause, weight in zip(_clauses(factor), factor.weights.tolist(), strict=True):
            rounded = int(round(float(weight)))
            if abs(rounded - float(weight)) > 1e-9 or rounded <= 0:
                raise ValueError("PySAT RC2 requires positive integral clause weights.")
            formula.append(clause, weight=rounded)
    with RC2(formula, adapt=True, exhaust=True) as solver:
        assignment = solver.compute()
        if assignment is None:
            return SATResult(
                torch.empty(0),
                float("inf"),
                "infeasible_proven",
                perf_counter() - started,
                True,
                "pysat-rc2",
            )
        values = torch.zeros(model.num_variables, dtype=torch.float64)
        for literal in assignment:
            if 1 <= abs(literal) <= model.num_variables:
                values[abs(literal) - 1] = float(literal > 0)
        objective_value = float(model.objective_values(values)[0].item())
    return SATResult(
        values, objective_value, "optimal", perf_counter() - started, True, "pysat-rc2"
    )


__all__ = ["SATResult", "solve_sat_model_ir"]
