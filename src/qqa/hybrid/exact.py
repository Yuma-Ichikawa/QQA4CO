"""Lazy, optional exact-backend adapters for the common algebraic model."""

from __future__ import annotations

import math
import pickle
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy import sparse

from qqa.algebraic import AlgebraicConstraint, AlgebraicModel, SparseQuadratic, VariableType

_ISOLATED_WORKER_STARTUP_TIMEOUT_SECONDS = 600.0
_ISOLATED_WORKER_SHUTDOWN_GRACE_SECONDS = 120.0


@dataclass(slots=True)
class ExactBackendResult:
    """Legacy-compatible exact result consumed by the stable API adapter."""

    best_sol: torch.Tensor
    best_obj: float
    runtime: float
    scip_status: str
    dual_bound: float | None = None
    gap: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    final_population: torch.Tensor | None = None


def _validate_request(model: AlgebraicModel, time_limit: float, threads: int) -> None:
    if not isinstance(model, AlgebraicModel):
        raise TypeError("model must be an AlgebraicModel.")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and > 0.")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")


def _canonical(model: AlgebraicModel, objective: float) -> float:
    return objective if model.objective_sense == "minimize" else -objective


def solve_scip_algebraic(
    model: AlgebraicModel,
    *,
    time_limit: float,
    threads: int = 1,
    relative_gap: float = 0.0,
    warm_start: torch.Tensor | None = None,
    verbose: bool = False,
) -> ExactBackendResult:
    """Solve an algebraic LP/MIP/QP with SCIP and an optional QQA incumbent."""
    _validate_request(model, time_limit, threads)
    from qqa.presolve.scip_bridge import build_scip_model

    started = perf_counter()
    scip, variables = build_scip_model(model, verbose=verbose)
    scip.setRealParam("limits/time", float(time_limit))
    scip.setRealParam("limits/gap", float(relative_gap))
    scip.setIntParam("parallel/maxnthreads", threads)
    scip.setIntParam("lp/threads", threads)
    accepted_warm_start = False
    if warm_start is not None and warm_start.numel() == model.num_variables:
        candidate = warm_start.detach().reshape(-1).cpu().to(torch.float64)
        if torch.isfinite(candidate).all():
            solution = scip.createSol()
            for variable, value in zip(variables, candidate.tolist(), strict=True):
                scip.setSolVal(solution, variable, float(value))
            accepted_warm_start = bool(scip.addSol(solution))
    scip.optimize()
    status = str(scip.getStatus())
    best = scip.getBestSol()
    if best is None:
        raise RuntimeError(f"{status}: exact backend returned no primal solution.")
    values = np.asarray([scip.getSolVal(best, variable) for variable in variables])
    evaluation = model.evaluate(values)
    bound: float | None
    try:
        bound = float(scip.getDualbound())
        if not math.isfinite(bound):
            bound = None
    except Exception:
        bound = None
    gap: float | None
    try:
        gap = float(scip.getGap())
        if not math.isfinite(gap):
            gap = None
    except Exception:
        gap = None
    return ExactBackendResult(
        torch.as_tensor(values, dtype=torch.float64),
        _canonical(model, evaluation.objective),
        perf_counter() - started,
        status,
        dual_bound=bound,
        gap=gap,
        diagnostics={
            "backend": "scip",
            "accepted_warm_start": accepted_warm_start,
            "maximum_infeasibility": evaluation.maximum_infeasibility,
        },
    )


def _require_linear(model: AlgebraicModel, backend: str) -> None:
    if not model.objective.is_linear or any(
        not constraint.expression.is_linear for constraint in model.constraints
    ):
        raise NotImplementedError(f"{backend} adapter currently accepts linear models only.")


def solve_cpsat_algebraic(
    model: AlgebraicModel,
    *,
    time_limit: float,
    threads: int = 1,
    relative_gap: float = 0.0,  # noqa: ARG001 - shared adapter signature
    warm_start: torch.Tensor | None = None,
    verbose: bool = False,
) -> ExactBackendResult:
    """Solve a bounded integer linear model with OR-Tools CP-SAT.

    CP-SAT is integer-only. Coefficients must already be integral so the
    adapter never changes model semantics through hidden rounding.
    """
    _validate_request(model, time_limit, threads)
    _require_linear(model, "CP-SAT")
    if any(kind is VariableType.CONTINUOUS for kind in model.variable_type_values):
        raise NotImplementedError("CP-SAT does not support continuous variables.")
    finite = np.isfinite(model.lower_bounds) & np.isfinite(model.upper_bounds)
    if not finite.all():
        raise ValueError("CP-SAT requires finite variable bounds.")
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install `qqa[cpsat]` to use the CP-SAT adapter.") from exc

    def integers(values: np.ndarray, label: str) -> np.ndarray:
        rounded = np.rint(values)
        if not np.allclose(values, rounded, atol=1e-10, rtol=0.0):
            raise ValueError(f"CP-SAT requires integral {label}; explicit scaling is required.")
        return rounded.astype(np.int64)

    started = perf_counter()
    cp = cp_model.CpModel()
    variables = [
        cp.new_int_var(int(math.ceil(lower)), int(math.floor(upper)), name)
        for name, lower, upper in zip(
            model.variable_names, model.lower_bounds, model.upper_bounds, strict=True
        )
    ]
    objective_coo = model.objective.linear_csr.tocoo()
    objective_coefficients = integers(objective_coo.data, "objective coefficients")
    objective = sum(
        int(value) * variables[int(index)]
        for index, value in zip(objective_coo.col, objective_coefficients, strict=True)
    ) + int(integers(np.asarray([model.objective.constant]), "objective constant")[0])
    (cp.minimize if model.objective_sense == "minimize" else cp.maximize)(objective)
    for row in model.constraints:
        coo = row.expression.linear_csr.tocoo()
        coefficients = integers(coo.data, f"constraint {row.name!r} coefficients")
        expression = sum(
            int(value) * variables[int(index)]
            for index, value in zip(coo.col, coefficients, strict=True)
        ) + int(integers(np.asarray([row.expression.constant]), "constraint constant")[0])
        if math.isfinite(row.lower):
            lower = int(integers(np.asarray([row.lower]), f"constraint {row.name!r} lower")[0])
            cp.add(expression >= lower)
        if math.isfinite(row.upper):
            upper = int(integers(np.asarray([row.upper]), f"constraint {row.name!r} upper")[0])
            cp.add(expression <= upper)
    if warm_start is not None and warm_start.numel() == model.num_variables:
        for variable, value in zip(
            variables, warm_start.detach().reshape(-1).cpu().tolist(), strict=True
        ):
            cp.add_hint(variable, int(round(value)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = threads
    solver.parameters.log_search_progress = bool(verbose)
    code = solver.solve(cp)
    status = solver.status_name(code)
    if code not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise RuntimeError(f"{status}: CP-SAT returned no primal solution.")
    values = np.asarray([solver.value(variable) for variable in variables], dtype=np.float64)
    evaluation = model.evaluate(values)
    bound = float(solver.best_objective_bound)
    objective_value = evaluation.objective
    denominator = max(1.0, abs(objective_value), abs(bound))
    gap = abs(objective_value - bound) / denominator
    return ExactBackendResult(
        torch.as_tensor(values, dtype=torch.float64),
        _canonical(model, objective_value),
        perf_counter() - started,
        status.lower(),
        dual_bound=bound,
        gap=gap,
        diagnostics={"backend": "cpsat", "maximum_infeasibility": evaluation.maximum_infeasibility},
    )


def solve_highs_algebraic(
    model: AlgebraicModel,
    *,
    time_limit: float,
    threads: int = 1,
    relative_gap: float = 0.0,
    warm_start: torch.Tensor | None = None,
    verbose: bool = False,
) -> ExactBackendResult:
    """Solve a sparse linear LP/MIP using the optional HiGHS Python API."""
    _validate_request(model, time_limit, threads)
    _require_linear(model, "HiGHS")
    try:
        import highspy
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install `qqa[highs]` to use the HiGHS adapter.") from exc
    started = perf_counter()
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", bool(verbose))
    highs.setOptionValue("time_limit", float(time_limit))
    highs.setOptionValue("threads", threads)
    highs.setOptionValue("mip_rel_gap", float(relative_gap))
    objective = model.objective.linear_dense().astype(np.float64)
    if model.objective_sense == "maximize":
        objective = -objective
    size = model.num_variables
    highs.addCols(
        size,
        objective,
        model.lower_array,
        model.upper_array,
        0,
        np.zeros(size + 1, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )
    if model.constraints:
        csr = sparse.vstack([row.expression.linear_csr for row in model.constraints], format="csr")
        lower = np.asarray(
            [row.lower - row.expression.constant for row in model.constraints], dtype=np.float64
        )
        upper = np.asarray(
            [row.upper - row.expression.constant for row in model.constraints], dtype=np.float64
        )
        highs.addRows(
            len(model.constraints),
            lower,
            upper,
            int(csr.nnz),
            csr.indptr.astype(np.int32),
            csr.indices.astype(np.int32),
            csr.data,
        )
    integer = model.integer_indices
    if integer.size:
        highs.changeColsIntegrality(
            len(integer),
            integer.astype(np.int32),
            np.full(len(integer), highspy.HighsVarType.kInteger),
        )
    if warm_start is not None and warm_start.numel() == size:
        candidate = warm_start.detach().reshape(-1).cpu().numpy().astype(np.float64)
        with suppress(TypeError, RuntimeError):
            highs.setSolution(size, np.arange(size, dtype=np.int32), candidate)
    highs.run()
    solution = highs.getSolution()
    values = np.asarray(solution.col_value, dtype=np.float64)
    if values.shape != (size,) or not np.isfinite(values).all():
        raise RuntimeError(f"{highs.getModelStatus()}: HiGHS returned no primal solution.")
    evaluation = model.evaluate(values)
    info = highs.getInfo()
    bound = float(info.mip_dual_bound) if integer.size else evaluation.objective
    if integer.size:
        bound = (
            bound + model.objective.constant
            if model.objective_sense == "minimize"
            else -bound + model.objective.constant
        )
    gap = float(info.mip_gap) if integer.size else 0.0
    return ExactBackendResult(
        torch.as_tensor(values, dtype=torch.float64),
        _canonical(model, evaluation.objective),
        perf_counter() - started,
        str(highs.modelStatusToString(highs.getModelStatus())).lower(),
        dual_bound=bound,
        gap=gap,
        diagnostics={"backend": "highs", "maximum_infeasibility": evaluation.maximum_infeasibility},
    )


def _expression_payload(expression: SparseQuadratic) -> dict[str, Any]:
    return {
        "quadratic": expression.quadratic,
        "linear": expression.linear,
        "constant": expression.constant,
    }


def _model_payload(model: AlgebraicModel) -> dict[str, Any]:
    """Return a pickle-safe algebraic payload without environment metadata."""
    return {
        "name": model.name,
        "variable_names": model.variable_names,
        "variable_types": tuple(item.value for item in model.variable_type_values),
        "lower_bounds": model.lower_bounds,
        "upper_bounds": model.upper_bounds,
        "objective": _expression_payload(model.objective),
        "constraints": tuple(
            {
                "name": row.name,
                "expression": _expression_payload(row.expression),
                "lower": row.lower,
                "upper": row.upper,
            }
            for row in model.constraints
        ),
        "objective_sense": model.objective_sense,
        "problem_type": model.problem_type,
        "source_format": model.source_format,
    }


def _model_from_payload(payload: dict[str, Any]) -> AlgebraicModel:
    def expression(record: dict[str, Any]) -> SparseQuadratic:
        # Pickle protocol 5 preserves the source buffer's read-only flag.
        # SparseQuadratic canonicalisation removes explicit zeros in-place, so
        # reconstruct with owned writable arrays at this process boundary.
        return SparseQuadratic(
            record["quadratic"].copy(),
            record["linear"].copy(),
            record["constant"],
        )

    return AlgebraicModel(
        payload["name"],
        payload["variable_names"],
        payload["variable_types"],
        payload["lower_bounds"],
        payload["upper_bounds"],
        expression(payload["objective"]),
        tuple(
            AlgebraicConstraint(
                row["name"],
                expression(row["expression"]),
                row["lower"],
                row["upper"],
            )
            for row in payload["constraints"]
        ),
        payload["objective_sense"],
        payload["problem_type"],
        payload["source_format"],
    )


def _backend_functions():
    return {
        "scip": solve_scip_algebraic,
        "cpsat": solve_cpsat_algebraic,
        "highs": solve_highs_algebraic,
    }


def _result_payload(result: ExactBackendResult) -> dict[str, Any]:
    return {
        "best_sol": result.best_sol.detach().cpu().numpy(),
        "best_obj": result.best_obj,
        "runtime": result.runtime,
        "scip_status": result.scip_status,
        "dual_bound": result.dual_bound,
        "gap": result.gap,
        "diagnostics": result.diagnostics,
    }


def _result_from_payload(payload: dict[str, Any]) -> ExactBackendResult:
    return ExactBackendResult(
        torch.as_tensor(payload["best_sol"], dtype=torch.float64),
        payload["best_obj"],
        payload["runtime"],
        payload["scip_status"],
        payload["dual_bound"],
        payload["gap"],
        payload["diagnostics"],
    )


def _safe_error(exc: Exception) -> tuple[str, str, str]:
    """Classify a backend failure without publishing a machine path."""
    message = str(exc)
    if "/" in message or "\\" in message:
        message = "backend initialisation or execution failed"
    return ("error", type(exc).__name__, message)


def _run_backend_payload(
    payload: dict[str, Any], backend: str, kwargs: dict[str, Any]
) -> tuple[Any, ...]:
    """Run one backend and return a pickle-safe, path-free envelope."""
    try:
        result = _backend_functions()[backend](_model_from_payload(payload), **kwargs)
        return ("ok", _result_payload(result))
    except Exception as exc:  # noqa: BLE001 - deliberate process boundary taxonomy
        return _safe_error(exc)


def _raise_remote(error_type: str, message: str) -> None:
    exceptions = {
        "ImportError": ImportError,
        "NotImplementedError": NotImplementedError,
        "TypeError": TypeError,
        "ValueError": ValueError,
    }
    exception = exceptions.get(error_type, RuntimeError)
    raise exception(f"{error_type}: {message}")


def solve_exact_algebraic(
    model: AlgebraicModel,
    backend: str,
    *,
    isolated: bool = True,
    worker_startup_timeout: float = _ISOLATED_WORKER_STARTUP_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> ExactBackendResult:
    """Dispatch an exact backend, isolated from native-library ABI conflicts.

    ``worker_startup_timeout`` covers interpreter/package loading and payload
    restoration only. The backend receives ``time_limit`` unchanged after the
    worker signals that it is ready, so cold filesystems do not consume the
    optimisation budget.
    """
    solvers = {
        "scip": solve_scip_algebraic,
        "cpsat": solve_cpsat_algebraic,
        "highs": solve_highs_algebraic,
    }
    if backend == "cuopt":
        try:
            import cuopt  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install a compatible NVIDIA cuOpt package to use this adapter."
            ) from exc
        raise NotImplementedError(
            "The cuOpt Python model API is version-specific; use the documented cuOpt file/SDK "
            "bridge rather than silently changing algebraic model semantics."
        )
    try:
        function = solvers[backend]
    except KeyError as exc:
        raise ValueError(f"Unknown exact backend {backend!r}.") from exc
    if not isolated:
        return function(model, **kwargs)
    _validate_request(model, float(kwargs.get("time_limit", 0.0)), int(kwargs.get("threads", 1)))
    if not math.isfinite(worker_startup_timeout) or worker_startup_timeout <= 0:
        raise ValueError("worker_startup_timeout must be finite and > 0.")
    with tempfile.TemporaryDirectory(prefix="qqa-exact-") as temporary:
        directory = Path(temporary)
        request = directory / "request.bin"
        response = directory / "response.bin"
        ready = directory / "ready"
        with request.open("wb") as stream:
            pickle.dump((_model_payload(model), backend, kwargs), stream, protocol=5)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "qqa.hybrid._exact_worker",
                str(request),
                str(response),
                str(ready),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            startup_deadline = perf_counter() + worker_startup_timeout
            while not ready.is_file() and process.poll() is None:
                if perf_counter() >= startup_deadline:
                    raise TimeoutError(f"{backend} backend worker failed to become ready in time.")
                time.sleep(0.05)
            if process.poll() is None:
                runtime_timeout = (
                    float(kwargs["time_limit"]) + _ISOLATED_WORKER_SHUTDOWN_GRACE_SECONDS
                )
                try:
                    process.wait(timeout=runtime_timeout)
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError(
                        f"{backend} backend worker exceeded its isolated runtime budget."
                    ) from exc
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        if process.returncode != 0 or not response.is_file():
            raise RuntimeError(f"{backend} backend worker terminated without a result envelope.")
        try:
            with response.open("rb") as stream:
                envelope = pickle.load(stream)
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise RuntimeError(
                f"{backend} backend worker returned an invalid result envelope."
            ) from exc
    if not isinstance(envelope, tuple) or not envelope:
        raise RuntimeError(f"{backend} backend worker returned an invalid result envelope.")
    if envelope[0] == "error":
        _raise_remote(envelope[1], envelope[2])
    result = _result_from_payload(envelope[1])
    result.diagnostics["process_isolated"] = True
    return result


__all__ = [
    "ExactBackendResult",
    "solve_cpsat_algebraic",
    "solve_exact_algebraic",
    "solve_highs_algebraic",
    "solve_scip_algebraic",
]
