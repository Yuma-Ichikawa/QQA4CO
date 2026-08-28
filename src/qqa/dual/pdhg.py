"""GPU/CPU PDHG solver for sparse linear programming relaxations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from scipy import sparse

from qqa.algebraic import AlgebraicModel


@dataclass(frozen=True, slots=True)
class PDHGResult:
    primal_solution: torch.Tensor
    dual_solution: torch.Tensor
    reduced_costs: torch.Tensor
    primal_objective: float
    dual_bound: float | None
    canonical_lower_bound: float | None
    primal_residual: float
    dual_residual: float
    kkt_residual: float
    iterations: int
    runtime: float
    converged: bool
    infeasibility_ray_candidate: torch.Tensor | None = None


def _torch_sparse(
    matrix: sparse.csr_matrix, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.as_tensor(np.vstack((coo.row, coo.col)), device=device, dtype=torch.long)
    values = torch.as_tensor(coo.data, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(
        indices,
        values,
        coo.shape,
        device=device,
        dtype=dtype,
        check_invariants=False,
    ).coalesce()


def _matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(matrix, vector.unsqueeze(1)).squeeze(1)


def _transpose_matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(matrix.transpose(0, 1), vector.unsqueeze(1)).squeeze(1)


def _dual_bound(
    c: torch.Tensor,
    matrix: torch.Tensor,
    row_lower: torch.Tensor,
    row_upper: torch.Tensor,
    variable_lower: torch.Tensor,
    variable_upper: torch.Tensor,
    dual: torch.Tensor,
) -> torch.Tensor | None:
    # d(y) = -h*(-c-A^T y) - g*(y), where h/g are box indicators.
    row_term = torch.where(
        dual > 0,
        row_upper * dual,
        torch.where(dual < 0, row_lower * dual, torch.zeros_like(dual)),
    )
    if not torch.isfinite(row_term).all():
        return None
    z = -c - _transpose_matvec(matrix, dual)
    variable_term = torch.where(
        z > 0,
        variable_upper * z,
        torch.where(z < 0, variable_lower * z, torch.zeros_like(z)),
    )
    if not torch.isfinite(variable_term).all():
        return None
    return -variable_term.sum() - row_term.sum()


def solve_lp_relaxation(
    model: AlgebraicModel,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float64,
    max_iterations: int = 10_000,
    tolerance: float = 1e-6,
    restart_interval: int = 200,
    time_limit: float | None = None,
) -> PDHGResult:
    """Solve the continuous linear relaxation and return primal/dual/KKT data.

    Integrality is intentionally relaxed.  Nonlinear rows are rejected rather
    than linearised silently, and a dual bound is returned only when every
    conjugate term is finite.
    """
    if not isinstance(model, AlgebraicModel):
        raise TypeError("model must be an AlgebraicModel.")
    if not model.objective.is_linear or any(
        not row.expression.is_linear for row in model.constraints
    ):
        raise NotImplementedError("PDHG currently requires a linear objective and linear rows.")
    if max_iterations < 1 or restart_interval < 1:
        raise ValueError("max_iterations and restart_interval must be positive.")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
        raise ValueError("time_limit must be finite and positive or None.")
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for PDHG but is unavailable.")

    size = model.num_variables
    if model.constraints:
        scipy_matrix = sparse.vstack(
            [row.expression.linear_csr for row in model.constraints], format="csr"
        )
        row_lower_np = np.asarray(
            [row.lower - row.expression.constant for row in model.constraints], dtype=np.float64
        )
        row_upper_np = np.asarray(
            [row.upper - row.expression.constant for row in model.constraints], dtype=np.float64
        )
    else:
        scipy_matrix = sparse.csr_matrix((0, size), dtype=np.float64)
        row_lower_np = np.empty(0, dtype=np.float64)
        row_upper_np = np.empty(0, dtype=np.float64)
    matrix = _torch_sparse(scipy_matrix, resolved, dtype)
    sign = 1.0 if model.objective_sense == "minimize" else -1.0
    c = torch.as_tensor(sign * model.objective.linear_dense(), device=resolved, dtype=dtype)
    variable_lower = torch.as_tensor(model.lower_array.copy(), device=resolved, dtype=dtype)
    variable_upper = torch.as_tensor(model.upper_array.copy(), device=resolved, dtype=dtype)
    row_lower = torch.as_tensor(row_lower_np, device=resolved, dtype=dtype)
    row_upper = torch.as_tensor(row_upper_np, device=resolved, dtype=dtype)

    finite_lower = torch.where(
        torch.isfinite(variable_lower), variable_lower, torch.zeros_like(variable_lower)
    )
    finite_upper = torch.where(
        torch.isfinite(variable_upper), variable_upper, torch.zeros_like(variable_upper)
    )
    midpoint = torch.where(
        torch.isfinite(variable_lower) & torch.isfinite(variable_upper),
        0.5 * (finite_lower + finite_upper),
        torch.where(torch.isfinite(variable_lower), finite_lower, finite_upper),
    )
    x = midpoint.clone()
    y = torch.zeros(len(model.constraints), device=resolved, dtype=dtype)
    x_average = x.clone()
    y_average = y.clone()
    if scipy_matrix.nnz:
        column_norm = torch.as_tensor(
            np.asarray(abs(scipy_matrix).sum(axis=0)).reshape(-1), device=resolved, dtype=dtype
        ).clamp_min(1e-6)
        row_norm = torch.as_tensor(
            np.asarray(abs(scipy_matrix).sum(axis=1)).reshape(-1), device=resolved, dtype=dtype
        ).clamp_min(1e-6)
    else:
        column_norm = torch.ones(size, device=resolved, dtype=dtype)
        row_norm = torch.ones(0, device=resolved, dtype=dtype)
    tau = 0.95 / column_norm
    sigma = 0.95 / row_norm
    started = perf_counter()
    converged = False
    iterations = 0
    primal_residual_tensor = torch.tensor(torch.inf, device=resolved, dtype=dtype)
    dual_residual_tensor = torch.tensor(torch.inf, device=resolved, dtype=dtype)
    for iteration in range(max_iterations):
        if time_limit is not None and perf_counter() - started >= time_limit:
            break
        previous = x
        x = torch.clamp(
            x - tau * (c + _transpose_matvec(matrix, y)), variable_lower, variable_upper
        )
        extrapolated = 2 * x - previous
        if len(y):
            argument = y / sigma + _matvec(matrix, extrapolated)
            projection = torch.clamp(argument, row_lower, row_upper)
            y = y + sigma * (_matvec(matrix, extrapolated) - projection)
        weight = 1.0 / (iteration + 1)
        x_average = x_average.lerp(x, weight)
        y_average = y_average.lerp(y, weight)
        iterations = iteration + 1
        if iterations % 25 == 0 or iterations == max_iterations:
            activity = _matvec(matrix, x_average)
            row_violation = torch.maximum(
                (row_lower - activity).clamp_min(0),
                (activity - row_upper).clamp_min(0),
            )
            primal_residual_tensor = row_violation.amax() if len(row_violation) else x.new_zeros(())
            gradient = c + _transpose_matvec(matrix, y_average)
            projected = x_average - torch.clamp(
                x_average - gradient, variable_lower, variable_upper
            )
            dual_residual_tensor = projected.abs().amax()
            if bool(torch.maximum(primal_residual_tensor, dual_residual_tensor) <= tolerance):
                converged = True
                break
        if iterations % restart_interval == 0:
            x = x_average.clone()
            y = y_average.clone()

    runtime = perf_counter() - started
    activity = _matvec(matrix, x_average)
    row_violation = torch.maximum(
        (row_lower - activity).clamp_min(0),
        (activity - row_upper).clamp_min(0),
    )
    primal_residual_tensor = row_violation.amax() if len(row_violation) else x.new_zeros(())
    reduced_costs = c + _transpose_matvec(matrix, y_average)
    projected = x_average - torch.clamp(x_average - reduced_costs, variable_lower, variable_upper)
    dual_residual_tensor = projected.abs().amax()
    bound = _dual_bound(
        c,
        matrix,
        row_lower,
        row_upper,
        variable_lower,
        variable_upper,
        y_average,
    )
    constant = sign * model.objective.constant
    canonical_primal = float((c @ x_average).item() + constant)
    canonical_bound = None if bound is None else float(bound.item() + constant)
    primal_objective = (
        canonical_primal if model.objective_sense == "minimize" else -canonical_primal
    )
    original_bound = (
        None
        if canonical_bound is None
        else canonical_bound
        if model.objective_sense == "minimize"
        else -canonical_bound
    )
    ray = None
    if not converged and len(y_average) and float(primal_residual_tensor.item()) > 1e3 * tolerance:
        ray = y_average / y_average.norm().clamp_min(1e-30)
    return PDHGResult(
        x_average.detach(),
        y_average.detach(),
        reduced_costs.detach(),
        primal_objective,
        original_bound,
        canonical_bound,
        float(primal_residual_tensor.item()),
        float(dual_residual_tensor.item()),
        float(torch.maximum(primal_residual_tensor, dual_residual_tensor).item()),
        iterations,
        runtime,
        converged,
        None if ray is None else ray.detach(),
    )


__all__ = ["PDHGResult", "solve_lp_relaxation"]
