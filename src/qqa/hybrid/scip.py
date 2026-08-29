"""QQA + SCIP hybrid refinement for binary quadratic models.

QQA explores many basins on the selected Torch device.  Diverse projected
replicas are then installed as SCIP primal starts for an exact MIQP solve.
SCIP can improve the incumbent and, when time permits, certify optimality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from time import perf_counter
from typing import Any

import torch

from qqa.annealing import AnnealResult, anneal
from qqa.problems.base import QUBOProblem
from qqa.utils import safe_score_summary


@dataclass(slots=True)
class SCIPHybridResult:
    """Unified result from the QQA exploration and SCIP proof phases."""

    best_sol: torch.Tensor
    best_obj: float
    runtime: float
    qqa_result: AnnealResult
    scip_runtime: float
    scip_status: str
    dual_bound: float | None
    gap: float | None
    n_warm_starts: int
    score: dict = field(default_factory=dict)

    @property
    def proven_optimal(self) -> bool:
        """Whether SCIP certified global optimality."""
        return self.scip_status.lower() == "optimal"

    @property
    def history(self) -> dict:
        """QQA exploration history, for report/CLI compatibility."""
        return self.qqa_result.history


def _require_pyscipopt():
    try:
        from pyscipopt import Model, quicksum
        from pyscipopt.recipes.nonlinear import set_nonlinear_objective
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "SCIP hybrid solving requires the optional dependency. "
            "Install it with `pip install 'qqa[scip]'`."
        ) from exc
    return Model, quicksum, set_nonlinear_objective


def _validate_qubo(problem: QUBOProblem) -> torch.Tensor:
    if not isinstance(problem, QUBOProblem):
        raise TypeError(
            "solve_qqa_scip currently accepts a QUBOProblem with a square Q_mat; "
            f"got {type(problem).__name__}."
        )
    q_mat = getattr(problem, "Q_mat", None)
    if not torch.is_tensor(q_mat) or q_mat.ndim != 2 or q_mat.shape[0] != q_mat.shape[1]:
        raise TypeError("problem.Q_mat must be a square torch.Tensor.")
    if not torch.isfinite(q_mat).all():
        raise ValueError("problem.Q_mat contains NaN or infinity.")
    return q_mat.detach().to(device="cpu", dtype=torch.float64)


def _rank_unique_starts(
    q_mat: torch.Tensor,
    qqa_result: AnnealResult,
    max_starts: int,
) -> list[torch.Tensor]:
    candidates = [qqa_result.best_sol.detach().cpu().reshape(-1)]
    if qqa_result.final_population is not None:
        candidates.extend(row for row in qqa_result.final_population.detach().cpu())

    unique: dict[bytes, torch.Tensor] = {}
    for candidate in candidates:
        bits = candidate.round().clamp(0, 1).to(torch.uint8).contiguous()
        unique.setdefault(bytes(bits.numpy()), bits.to(torch.float64))

    ranked = sorted(
        unique.values(),
        key=lambda x: float(torch.dot(x, q_mat.mv(x)).item()),
    )
    return ranked[:max_starts]


def solve_qqa_scip(
    problem: QUBOProblem,
    *,
    qqa_kwargs: dict[str, Any] | None = None,
    time_limit: float = 60.0,
    relative_gap: float = 0.0,
    max_warm_starts: int = 32,
    threads: int = 1,
    verbose: bool = False,
) -> SCIPHybridResult:
    """Explore a QUBO with QQA and refine/certify it with SCIP.

    The exact SCIP model preserves QQA's convention ``x.T @ Q @ x`` even
    when ``Q`` is not symmetric: off-diagonal coefficients ``Q[i,j]`` and
    ``Q[j,i]`` are combined into one binary-product term.
    """
    if (
        isinstance(time_limit, bool)
        or not isinstance(time_limit, Real)
        or not math.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError(f"time_limit must be finite and > 0, got {time_limit}.")
    if (
        isinstance(relative_gap, bool)
        or not isinstance(relative_gap, Real)
        or not math.isfinite(relative_gap)
        or relative_gap < 0
    ):
        raise ValueError(f"relative_gap must be finite and >= 0, got {relative_gap}.")
    if (
        isinstance(max_warm_starts, bool)
        or not isinstance(max_warm_starts, int)
        or max_warm_starts < 1
    ):
        raise ValueError("max_warm_starts must be a positive integer.")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")

    q_mat = _validate_qubo(problem)
    Model, quicksum, set_nonlinear_objective = _require_pyscipopt()
    started = perf_counter()

    defaults: dict[str, Any] = {
        "sol_size": max(64, max_warm_starts * 2),
        "num_epochs": 1000,
        "verbose": verbose,
        "return_population": True,
    }
    if qqa_kwargs:
        defaults.update(qqa_kwargs)
    defaults["return_population"] = True
    qqa_budget = float(time_limit) - (perf_counter() - started)
    if qqa_budget <= 0:
        defaults["num_epochs"] = 0
    else:
        requested_qqa_budget = defaults.get("time_limit")
        defaults["time_limit"] = (
            qqa_budget
            if requested_qqa_budget is None
            else min(float(requested_qqa_budget), qqa_budget)
        )
    qqa_result = anneal(problem, **defaults)
    starts = _rank_unique_starts(q_mat, qqa_result, max_warm_starts)

    model = Model(f"qqa-scip-{type(problem).__name__}")
    if not verbose:
        model.hideOutput()
    model.setRealParam("limits/gap", float(relative_gap))
    model.setIntParam("parallel/maxnthreads", threads)
    model.setIntParam("lp/threads", threads)

    n = q_mat.shape[0]
    x_vars = [model.addVar(vtype="B", name=f"x_{index}") for index in range(n)]
    linear = quicksum(float(q_mat[i, i]) * x_vars[i] for i in range(n))
    quadratic_terms = [
        float(q_mat[i, j] + q_mat[j, i]) * x_vars[i] * x_vars[j]
        for i in range(n)
        for j in range(i + 1, n)
        if float(q_mat[i, j] + q_mat[j, i]) != 0.0
    ]
    if quadratic_terms:
        set_nonlinear_objective(model, linear + quicksum(quadratic_terms), "minimize")
    else:
        model.setObjective(linear, "minimize")

    accepted = 0
    for start in starts:
        solution = model.createSol()
        for variable, value in zip(x_vars, start.tolist(), strict=True):
            model.setSolVal(solution, variable, value)
        if model.addSol(solution):
            accepted += 1

    remaining = float(time_limit) - (perf_counter() - started)
    scip_ran = remaining > 1e-3
    if scip_ran:
        model.setRealParam("limits/time", remaining)
        scip_started = perf_counter()
        model.optimize()
        scip_runtime = perf_counter() - scip_started
        status = str(model.getStatus())
        best = model.getBestSol()
    else:
        scip_runtime = 0.0
        status = "timelimit"
        best = None

    best_sol = qqa_result.best_sol.detach().clone()
    best_obj = float(qqa_result.best_obj)
    if best is not None:
        scip_sol = torch.tensor(
            [model.getSolVal(best, variable) for variable in x_vars],
            device=problem.Q_mat.device,
            dtype=problem.Q_mat.dtype,
        ).round()
        scip_obj = float(problem.loss_fn(scip_sol.unsqueeze(0))[0].item())
        if scip_obj <= best_obj + 1e-8:
            best_sol = scip_sol
            best_obj = scip_obj

    dual_bound: float | None = None
    gap: float | None = None
    if scip_ran:
        try:
            dual_bound = float(model.getDualbound())
            if not math.isfinite(dual_bound):
                dual_bound = None
        except Exception:  # pragma: no cover - SCIP status dependent
            dual_bound = None
        try:
            gap = float(model.getGap())
            if not math.isfinite(gap):
                gap = None
        except Exception:  # pragma: no cover - SCIP status dependent
            gap = None

    score = safe_score_summary(problem, best_sol, fallback_obj=best_obj)
    return SCIPHybridResult(
        best_sol=best_sol,
        best_obj=best_obj,
        runtime=perf_counter() - started,
        qqa_result=qqa_result,
        scip_runtime=scip_runtime,
        scip_status=status,
        dual_bound=dual_bound,
        gap=gap,
        n_warm_starts=accepted,
        score=score,
    )


__all__ = ["SCIPHybridResult", "solve_qqa_scip"]
