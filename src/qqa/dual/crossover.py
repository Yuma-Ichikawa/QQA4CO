"""CPU dual-simplex crossover and portable LP basis-status reporting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

import numpy as np
import torch
from scipy import optimize, sparse

from qqa.algebraic import AlgebraicModel
from qqa.dual.pdhg import PDHGResult


class BasisStatus(str, Enum):
    BASIC = "basic"
    AT_LOWER = "at_lower"
    AT_UPPER = "at_upper"
    FIXED = "fixed"
    FREE = "free"
    INACTIVE = "inactive"


@dataclass(frozen=True, slots=True)
class LPCrossoverResult:
    solution: torch.Tensor
    objective: float
    variable_status: tuple[BasisStatus, ...]
    row_status: tuple[BasisStatus, ...]
    maximum_infeasibility: float
    iterations: int
    runtime: float
    proven_optimal: bool
    backend: str = "scipy-highs-ds"


def _status(value: float, lower: float, upper: float, tolerance: float) -> BasisStatus:
    at_lower = math.isfinite(lower) and abs(value - lower) <= tolerance
    at_upper = math.isfinite(upper) and abs(value - upper) <= tolerance
    if at_lower and at_upper:
        return BasisStatus.FIXED
    if at_lower:
        return BasisStatus.AT_LOWER
    if at_upper:
        return BasisStatus.AT_UPPER
    if not math.isfinite(lower) and not math.isfinite(upper):
        return BasisStatus.FREE
    return BasisStatus.BASIC


def crossover_lp(
    model: AlgebraicModel,
    relaxation: PDHGResult | torch.Tensor | None = None,
    *,
    time_limit: float | None = None,
    tolerance: float = 1e-7,
) -> LPCrossoverResult:
    """Crossover a linear relaxation to a basic solution with dual simplex.

    SciPy's public HiGHS interface does not accept an LP warm start, so the
    optional PDHG point is checked for model alignment and retained as the
    semantic hand-off, while HiGHS performs a clean dual-simplex crossover on
    the identical sparse model. No coefficient or infinite bound is changed.
    """
    if not isinstance(model, AlgebraicModel):
        raise TypeError("model must be an AlgebraicModel.")
    if not model.objective.is_linear or any(
        not row.expression.is_linear for row in model.constraints
    ):
        raise NotImplementedError("LP crossover requires a linear objective and linear rows.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be finite and positive or None.")
    if relaxation is not None:
        warm = relaxation.primal_solution if isinstance(relaxation, PDHGResult) else relaxation
        warm = torch.as_tensor(warm).reshape(-1)
        if warm.numel() != model.num_variables or not torch.isfinite(warm).all():
            raise ValueError("The crossover warm point must be finite and model-aligned.")

    objective = model.objective.linear_dense().astype(np.float64)
    sign = 1.0 if model.objective_sense == "minimize" else -1.0
    c = sign * objective
    inequalities: list[sparse.csr_matrix] = []
    inequality_rhs: list[float] = []
    equalities: list[sparse.csr_matrix] = []
    equality_rhs: list[float] = []
    for row in model.constraints:
        matrix_row = row.expression.linear_csr
        lower = row.lower - row.expression.constant
        upper = row.upper - row.expression.constant
        if math.isfinite(lower) and math.isfinite(upper) and abs(lower - upper) <= tolerance:
            equalities.append(matrix_row)
            equality_rhs.append(0.5 * (lower + upper))
            continue
        if math.isfinite(upper):
            inequalities.append(matrix_row)
            inequality_rhs.append(upper)
        if math.isfinite(lower):
            inequalities.append(-matrix_row)
            inequality_rhs.append(-lower)
    options: dict[str, bool | float] = {"presolve": True}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    started = perf_counter()
    result = optimize.linprog(
        c,
        A_ub=sparse.vstack(inequalities, format="csr") if inequalities else None,
        b_ub=np.asarray(inequality_rhs) if inequalities else None,
        A_eq=sparse.vstack(equalities, format="csr") if equalities else None,
        b_eq=np.asarray(equality_rhs) if equalities else None,
        bounds=list(zip(model.lower_array, model.upper_array, strict=True)),
        method="highs-ds",
        options=options,
    )
    runtime = perf_counter() - started
    if result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError(f"LP crossover returned no primal solution (status={result.status}).")
    evaluation = model.evaluate(result.x)
    variable_status = tuple(
        _status(value, lower, upper, tolerance)
        for value, lower, upper in zip(result.x, model.lower_array, model.upper_array, strict=True)
    )
    row_status = []
    for row, activity in zip(model.constraints, evaluation.constraint_values, strict=True):
        active = (math.isfinite(row.lower) and abs(activity - row.lower) <= tolerance) or (
            math.isfinite(row.upper) and abs(activity - row.upper) <= tolerance
        )
        row_status.append(BasisStatus.BASIC if active else BasisStatus.INACTIVE)
    objective_value = float(evaluation.objective)
    return LPCrossoverResult(
        torch.as_tensor(result.x.copy(), dtype=torch.float64),
        objective_value,
        variable_status,
        tuple(row_status),
        float(evaluation.maximum_infeasibility),
        int(getattr(result, "nit", 0) or 0),
        runtime,
        bool(result.success and evaluation.maximum_infeasibility <= tolerance),
    )


__all__ = ["BasisStatus", "LPCrossoverResult", "crossover_lp"]
