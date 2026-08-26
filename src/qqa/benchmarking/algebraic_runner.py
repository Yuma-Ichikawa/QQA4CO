"""Unified MIPLIB/QPLIB execution through SCIP and conditional QQA."""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from sys import version_info
from time import perf_counter

import numpy as np

from qqa.algebraic import AlgebraicModel
from qqa.benchmarking.metrics import (
    BenchmarkComparisonResult,
    BenchmarkFailure,
    BenchmarkResult,
    BenchmarkSuiteResult,
    SCIPProgressTracker,
    normalised_primal_error,
    primal_integral,
    relative_gap,
    summarise_benchmarks,
    summarise_comparison,
)
from qqa.hybrid.heuristic_types import QQAHeuristicConfig
from qqa.hybrid.scip_heuristic import include_qqa_heuristic
from qqa.io import load_mps, load_qplib
from qqa.presolve import build_scip_model

_RETAINED_NATIVE_MODELS: list[object] = []


def detect_format(path: str | Path) -> str:
    name = Path(path).name.lower()
    if name.endswith(".qplib"):
        return "qplib"
    if any(name.endswith(suffix) for suffix in (".mps", ".mps.gz", ".mps.bz2", ".lp", ".lp.gz")):
        return "miplib"
    raise ValueError("Cannot detect format; use a .qplib, .mps[.gz], or .lp[.gz] file.")


def load_reference_values(path: str | Path) -> dict[str, tuple[str, float | None]]:
    """Parse MIPLIB/QPLIB ``.solu`` records."""
    records: dict[str, tuple[str, float | None]] = {}
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2 or not parts[0].startswith("="):
                continue
            status = parts[0].strip("=")
            value = None
            if len(parts) >= 3:
                try:
                    value = float(parts[2])
                except ValueError:
                    value = None
            records[parts[1]] = (status, value)
    return records


def _load_algebraic(path: Path, format: str) -> AlgebraicModel:
    # SCIP itself remains the feasibility authority for MIPLIB runs and the
    # QQA surrogate reads active LP rows directly. Avoid duplicating every MPS
    # row into Python sparse objects merely for objective/provenance metrics.
    return load_qplib(path) if format == "qplib" else load_mps(path, include_constraints=False)


def _load_model(
    path: Path,
    format: str,
    *,
    algebraic: AlgebraicModel | None = None,
):
    algebraic = _load_algebraic(path, format) if algebraic is None else algebraic
    if format == "qplib":
        scip, variables = build_scip_model(algebraic)
        return algebraic, scip, variables
    try:
        from pyscipopt import Model
    except (ImportError, OSError) as exc:  # pragma: no cover
        raise ImportError("Benchmark execution requires `qqa[scip]`.") from exc
    scip = Model(algebraic.name)
    scip.hideOutput()
    scip.readProblem(str(path))
    variables = tuple(scip.getVars(transformed=False))
    return algebraic, scip, variables


def _solution_in_algebraic_order(
    model, algebraic: AlgebraicModel, variables, solution
) -> np.ndarray:
    by_name = {variable.name: variable for variable in variables}
    missing = [name for name in algebraic.variable_names if name not in by_name]
    if missing:
        raise ValueError(f"SCIP solution is missing original variables: {missing[:3]}")
    return np.asarray(
        [model.getSolVal(solution, by_name[name]) for name in algebraic.variable_names],
        dtype=np.float64,
    )


def _software_versions(*, qplib: bool) -> dict[str, str]:
    packages = ["qqa", "pyscipopt", "torch"]
    if qplib:
        packages.append("pyqplib")
    versions = {"python": f"{version_info.major}.{version_info.minor}.{version_info.micro}"}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def _configure_scip_threads(model, threads: int) -> None:
    """Apply one auditable thread limit to SCIP and its LP solver.

    ``parallel/maxnthreads`` alone does not constrain the LP solver: the
    default ``lp/threads=0`` delegates to the linked LP implementation and may
    use every available core. Benchmark comparisons therefore set both.
    """
    model.setIntParam("parallel/maxnthreads", int(threads))
    model.setIntParam("lp/threads", int(threads))


def _qqa_is_applicable(
    algebraic: AlgebraicModel,
    config: QQAHeuristicConfig,
) -> bool:
    """Apply cheap original-model gates before registering a SCIP plugin."""
    return bool(
        algebraic.integer_indices.size > 0
        and (
            algebraic.problem_type is None
            or config.allowed_qplib_problem_types is None
            or algebraic.problem_type.upper() in config.allowed_qplib_problem_types
        )
        and (
            config.maximum_integer_variables is None
            or algebraic.integer_indices.size <= config.maximum_integer_variables
        )
        and (
            config.maximum_problem_variables is None
            or algebraic.num_variables <= config.maximum_problem_variables
        )
    )


def _qqa_applicability_hint(
    source: Path,
    resolved_format: str,
    config: QQAHeuristicConfig,
    *,
    algebraic: AlgebraicModel | None,
) -> bool | None:
    """Return a cheap structural decision without loading a QPLIB model.

    QPLIB's second ``PROBTYPE`` character is the variable class and the fourth
    header line is the variable count.  Returning ``None`` on malformed input
    deliberately falls back to independent solver runs and the normal parser
    error boundary.
    """
    if algebraic is not None:
        return _qqa_is_applicable(algebraic, config)
    if resolved_format != "qplib":
        return None
    try:
        with source.open(encoding="utf-8") as stream:
            stream.readline()
            problem_type = stream.readline().strip().upper()
            stream.readline()
            num_variables = int(stream.readline().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    if len(problem_type) != 3 or problem_type[1] not in {"B", "C", "G", "I", "M"}:
        return None
    if (
        config.allowed_qplib_problem_types is not None
        and problem_type not in config.allowed_qplib_problem_types
    ):
        return False
    variable_class = problem_type[1]
    if variable_class == "C":
        return False
    if (
        config.maximum_problem_variables is not None
        and num_variables > config.maximum_problem_variables
    ):
        return False
    # Pure-binary and pure-integer QPLIB models expose the exact integer
    # count in the header. Mixed classes need their sparse type exceptions to
    # be parsed, so an integer-count gate remains deliberately undecided here
    # and is applied exactly after loading the algebraic model.
    return not (
        variable_class in {"B", "I"}
        and config.maximum_integer_variables is not None
        and num_variables > config.maximum_integer_variables
    )


def _reuse_equivalent_aggressive_result(result: BenchmarkResult) -> BenchmarkResult:
    """Represent SG-CQQA's exact structural bypass without a noisy rerun."""
    return replace(
        result,
        solver="sg-cqqa",
        qqa=None,
        run_config={
            **result.run_config,
            "torch_threads": None,
            "qqa_applicable": False,
            "qqa_plugin_active": False,
            "equivalent_baseline_reuse": True,
        },
    )


def run_benchmark_instance(
    path: str | Path,
    *,
    format: str = "auto",
    solver: str = "sg-cqqa",
    time_limit: float = 60.0,
    relative_gap_limit: float = 0.0,
    threads: int = 1,
    seed: int = 0,
    reference_file: str | Path | None = None,
    qqa_config: QQAHeuristicConfig | None = None,
    verbose: bool = False,
    _algebraic: AlgebraicModel | None = None,
    _reference_records: dict[str, tuple[str, float | None]] | None = None,
    _defer_cleanup: bool = False,
    _isolated_worker: bool = False,
) -> BenchmarkResult:
    """Run one public benchmark with a single total SCIP/QQA deadline."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Benchmark instance does not exist: {source}")
    resolved_format = detect_format(source) if format == "auto" else format.lower()
    if resolved_format not in {"miplib", "qplib"}:
        raise ValueError("format must be auto, miplib, or qplib.")
    if solver not in {"scip", "scip-aggressive", "sg-cqqa"}:
        raise ValueError("solver must be 'scip', 'scip-aggressive', or 'sg-cqqa'.")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and > 0.")
    if not math.isfinite(relative_gap_limit) or relative_gap_limit < 0:
        raise ValueError("relative_gap_limit must be finite and >= 0.")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    if resolved_format == "qplib" and _algebraic is None and not _isolated_worker:
        return _run_isolated_benchmark_instance(
            source,
            resolved_format=resolved_format,
            solver=solver,
            seed=seed,
            qqa_config=qqa_config or QQAHeuristicConfig(),
            reference_records=None,
            run_kwargs={
                "format": resolved_format,
                "time_limit": time_limit,
                "relative_gap_limit": relative_gap_limit,
                "threads": threads,
                "reference_file": reference_file,
                "verbose": verbose,
            },
            common_import=False,
        )

    started = perf_counter()
    algebraic, model, variables = _load_model(
        source,
        resolved_format,
        algebraic=_algebraic,
    )
    if not verbose:
        model.hideOutput()
    model.setRealParam("limits/gap", float(relative_gap_limit))
    _configure_scip_threads(model, threads)
    for parameter in (
        "randomization/randomseedshift",
        "randomization/permutationseed",
        "randomization/lpseed",
    ):
        with suppress(Exception):
            model.setIntParam(parameter, seed)
    if solver in {"scip-aggressive", "sg-cqqa"}:
        from pyscipopt import SCIP_PARAMSETTING

        model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)

    def evaluate_original_solution(active_model, solution):
        values = _solution_in_algebraic_order(active_model, algebraic, variables, solution)
        return algebraic.evaluate(values), values

    tracker = SCIPProgressTracker(
        solution_evaluator=evaluate_original_solution,
        objective_sense=algebraic.objective_sense,
        time_horizon=float(time_limit),
    )
    tracker.attach(model)
    resolved_qqa_config = qqa_config or QQAHeuristicConfig()
    qqa_structurally_applicable = _qqa_is_applicable(algebraic, resolved_qqa_config)
    qqa_applicable = solver == "sg-cqqa" and qqa_structurally_applicable
    heuristic = (
        include_qqa_heuristic(
            model,
            resolved_qqa_config,
            algebraic=algebraic,
            incumbent_provider=lambda: tracker.best_values,
            completion_template_factory=(
                lambda: _load_model(source, resolved_format, algebraic=algebraic)[1]
            ),
        )
        if qqa_applicable
        else None
    )
    # Solver-model construction, plugin setup, QQA calls and SCIP share one
    # wall-clock budget. A paired campaign performs its common algebraic import
    # once before entering this solver-specific clock.
    tracker.time_offset = perf_counter() - started
    remaining = float(time_limit) - tracker.time_offset
    optimised = remaining > 1e-3
    if optimised:
        model.setRealParam("limits/time", remaining)
        model.optimize()
    runtime = perf_counter() - started
    status = str(model.getStatus()) if optimised else "setup-time-limit"
    best = model.getBestSol() if optimised else None
    objective = None
    evaluation = None
    if best is not None and runtime <= time_limit:
        values = _solution_in_algebraic_order(model, algebraic, variables, best)
        evaluation = algebraic.evaluate(values)
        objective = evaluation.objective
    tracked = tracker.best_evaluation
    if tracked is not None and (
        evaluation is None
        or (algebraic.objective_sense == "maximize" and tracked.objective > evaluation.objective)
        or (algebraic.objective_sense == "minimize" and tracked.objective < evaluation.objective)
    ):
        evaluation = tracked
        objective = tracked.objective
    try:
        dual_bound = float(model.getDualbound()) if optimised else None
        if dual_bound is not None and not math.isfinite(dual_bound):
            dual_bound = None
    except Exception:
        dual_bound = None

    reference = None
    reference_status = None
    if reference_file is not None:
        records = (
            load_reference_values(reference_file)
            if _reference_records is None
            else _reference_records
        )
        keys = (algebraic.name, source.name.removesuffix(".gz").removesuffix(".mps"))
        for key in keys:
            if key in records:
                reference_status, reference = records[key]
                break
    solving_time = float(model.getSolvingTime()) if optimised else 0.0
    integral = (
        primal_integral(
            tracker.trajectory,
            reference=reference,
            horizon=float(time_limit),
            objective_sense=algebraic.objective_sense,
        )
        if reference is not None
        else None
    )
    provenance = {
        **dict(algebraic.metadata),
        "reference_name": Path(reference_file).name if reference_file is not None else None,
        "reference_status": reference_status,
        "software": _software_versions(qplib=resolved_format == "qplib"),
    }
    result = BenchmarkResult(
        instance=algebraic.name,
        format=resolved_format,
        solver=solver,
        objective_sense=algebraic.objective_sense,
        status=status,
        runtime=runtime,
        solving_time=solving_time,
        nodes=int(model.getNNodes()) if optimised else 0,
        objective=objective,
        dual_bound=dual_bound,
        gap=relative_gap(objective, dual_bound),
        feasible=evaluation is not None and evaluation.maximum_infeasibility <= 1e-6,
        maximum_infeasibility=(
            evaluation.maximum_infeasibility if evaluation is not None else None
        ),
        time_to_first_feasible=tracker.time_to_first_feasible,
        primal_integral=integral,
        reference_objective=reference,
        primal_error=normalised_primal_error(
            objective,
            reference,
            objective_sense=algebraic.objective_sense,
        ),
        problem_type=algebraic.problem_type,
        trajectory=tuple(tracker.trajectory),
        qqa=heuristic.stats.as_dict() if heuristic is not None else None,
        run_config={
            "time_limit": float(time_limit),
            "relative_gap_limit": float(relative_gap_limit),
            "threads": threads,
            "scip_parallel_threads": threads,
            "scip_lp_threads": threads,
            "torch_threads": threads if qqa_applicable else None,
            "qqa_applicable": qqa_structurally_applicable,
            "qqa_plugin_active": qqa_applicable,
            "metric_clock": (
                "total_wall_clock"
                if _algebraic is None
                else "solver_wall_clock_after_common_import"
            ),
            "seed": seed,
        },
        provenance=provenance,
    )
    if _defer_cleanup:
        # Keep every Python wrapper alive until the isolated worker calls
        # ``os._exit``.  Dropping the last wrapper here would invoke SCIP's
        # native destructor before the worker can persist the result.
        _RETAINED_NATIVE_MODELS.append((model, variables, best, tracker, heuristic))
    else:
        if heuristic is not None and heuristic.completion_template is not None:
            with suppress(Exception):
                heuristic.completion_template.free()
            heuristic.completion_template = None
        with suppress(Exception):
            model.free()
        gc.collect()
    return result


class _IsolatedBenchmarkError(RuntimeError):
    def __init__(self, error_type: str):
        super().__init__(error_type)
        self.error_type = error_type


def _isolated_benchmark_worker(
    output_path: str,
    source_path: str,
    resolved_format: str,
    solver: str,
    seed: int,
    qqa_config: QQAHeuristicConfig,
    reference_records: dict[str, tuple[str, float | None]] | None,
    run_kwargs: dict,
    common_import: bool = True,
) -> None:
    """Run one native solver in a disposable process and persist its result.

    Some large nonlinear SCIP models retain native copy/plugin state whose
    teardown is platform dependent.  The worker deliberately exits without
    Python-level teardown after durably writing the portable result; the OS
    then reclaims all native state without exposing a long campaign to one
    model's allocator failure.
    """
    try:
        source = Path(source_path)
        algebraic = _load_algebraic(source, resolved_format) if common_import else None
        result = run_benchmark_instance(
            source,
            solver=solver,
            seed=seed,
            qqa_config=qqa_config,
            _algebraic=algebraic,
            _reference_records=reference_records,
            _defer_cleanup=True,
            _isolated_worker=True,
            **run_kwargs,
        )
        payload = {"result": result.to_dict()}
    except BaseException as exc:  # noqa: BLE001 - cross-process error boundary
        payload = {"error_type": type(exc).__name__}
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os._exit(0)


def _run_isolated_benchmark_instance(
    source: Path,
    *,
    resolved_format: str,
    solver: str,
    seed: int,
    qqa_config: QQAHeuristicConfig,
    reference_records: dict[str, tuple[str, float | None]] | None,
    run_kwargs: dict,
    common_import: bool,
) -> BenchmarkResult:
    """Execute one QPLIB run behind a native-process fault boundary."""
    with tempfile.TemporaryDirectory(prefix="qqa-benchmark-") as directory:
        request = Path(directory) / "request.json"
        output = Path(directory) / "result.json"
        serializable_kwargs = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in run_kwargs.items()
        }
        request.write_text(
            json.dumps(
                {
                    "source_path": str(source),
                    "resolved_format": resolved_format,
                    "solver": solver,
                    "seed": seed,
                    "qqa_config": asdict(qqa_config),
                    "reference_records": reference_records,
                    "run_kwargs": serializable_kwargs,
                    "common_import": common_import,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "qqa.benchmarking._worker",
                str(request),
                str(output),
            ]
        )
        time_limit = float(run_kwargs.get("time_limit", 60.0))
        try:
            process.wait(timeout=max(300.0, time_limit + 120.0))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise _IsolatedBenchmarkError("WorkerTimeout") from None
        if not output.is_file():
            raise _IsolatedBenchmarkError("NativeSolverProcessError")
        payload = json.loads(output.read_text(encoding="utf-8"))
    if "error_type" in payload:
        raise _IsolatedBenchmarkError(str(payload["error_type"]))
    return BenchmarkResult.from_dict(payload["result"])


def run_benchmark_suite(
    paths,
    **kwargs,
) -> BenchmarkSuiteResult:
    """Run several instances and aggregate overall/PROBTYPE metrics."""
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be a sequence of instance paths.")
    instances = tuple(paths)
    if not instances:
        raise ValueError("paths must contain at least one instance.")
    results = tuple(run_benchmark_instance(path, **kwargs) for path in instances)
    return BenchmarkSuiteResult(results, summarise_benchmarks(results))


def _comparison_solver_order(
    solvers: tuple[str, ...],
    *,
    execution_order: str,
    seed: int,
    instance_index: int,
) -> tuple[str, ...]:
    """Return a deterministic order while balancing first-run cache effects."""
    if execution_order == "fixed" or (seed + instance_index) % 2 == 0:
        return solvers
    return tuple(reversed(solvers))


def compare_benchmark_solvers(
    paths,
    *,
    solvers=("scip", "scip-aggressive", "sg-cqqa"),
    seeds=(0,),
    baseline_solver: str = "scip",
    execution_order: str = "balanced",
    qqa_config: QQAHeuristicConfig | None = None,
    checkpoint_file: str | Path | None = None,
    resume: bool = False,
    continue_on_error: bool = False,
    retry_failures: bool = False,
    **kwargs,
) -> BenchmarkComparisonResult:
    """Run a resumable paired campaign with portable configuration metadata."""
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be a sequence of instance paths.")
    instances = tuple(paths)
    if not instances:
        raise ValueError("paths must contain at least one instance.")
    instance_names = tuple(Path(instance).name for instance in instances)
    if len(set(instance_names)) != len(instance_names):
        raise ValueError("paths must have unique basenames for portable checkpoints.")
    solver_names = tuple(solvers)
    if not solver_names or len(set(solver_names)) != len(solver_names):
        raise ValueError("solvers must be a non-empty sequence without duplicates.")
    allowed = {"scip", "scip-aggressive", "sg-cqqa"}
    if any(solver not in allowed for solver in solver_names):
        raise ValueError("Unknown comparison solver.")
    if baseline_solver not in solver_names:
        raise ValueError("baseline_solver must be included in solvers.")
    if execution_order not in {"fixed", "balanced"}:
        raise ValueError("execution_order must be 'fixed' or 'balanced'.")
    seed_values = tuple(seeds)
    if not seed_values or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seed_values
    ):
        raise ValueError("seeds must contain non-negative integers.")

    base_config = qqa_config or QQAHeuristicConfig()
    qqa_config_metadata = asdict(base_config)
    # Per-run seeds come from the explicit campaign ``seeds`` axis below.
    # Keeping the constructor seed here duplicates that axis and prevents
    # independently executed seed shards from validating as one campaign.
    qqa_config_metadata.pop("seed", None)
    comparison_config = {
        "instances": list(instance_names),
        "solvers": list(solver_names),
        "seeds": list(seed_values),
        "baseline_solver": baseline_solver,
        "execution_order": execution_order,
        "format": kwargs.get("format", "auto"),
        "time_limit": float(kwargs.get("time_limit", 60.0)),
        "relative_gap_limit": float(kwargs.get("relative_gap_limit", 0.0)),
        "threads": int(kwargs.get("threads", 1)),
        "thread_policy": {
            "scip_parallel": int(kwargs.get("threads", 1)),
            "scip_lp": int(kwargs.get("threads", 1)),
            "torch_sg_cqqa": int(kwargs.get("threads", 1)),
        },
        "metric_clock": "solver_wall_clock_after_common_import",
        "equivalent_bypass_reuse": True,
        "reference_name": (
            Path(kwargs["reference_file"]).name if kwargs.get("reference_file") else None
        ),
        "qqa_config": qqa_config_metadata,
    }
    checkpoint = Path(checkpoint_file).expanduser() if checkpoint_file is not None else None
    results: list[BenchmarkResult] = []
    failures: list[BenchmarkFailure] = []
    if resume:
        if checkpoint is None:
            raise ValueError("resume requires checkpoint_file.")
        if checkpoint.is_file():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("comparison_config") != comparison_config:
                raise ValueError("Checkpoint configuration does not match this campaign.")
            results.extend(BenchmarkResult.from_dict(row) for row in payload.get("results", ()))
            failures.extend(BenchmarkFailure.from_dict(row) for row in payload.get("failures", ()))
    elif checkpoint is not None and checkpoint.exists():
        raise FileExistsError("checkpoint_file already exists; use resume=True to continue it.")

    completed = {
        (
            str(result.provenance.get("source_name", result.instance)),
            result.solver,
            int(result.run_config.get("seed", 0)),
        )
        for result in results
    }
    failed = {(row.instance, row.solver, row.seed) for row in failures}
    if retry_failures:
        failures.clear()
        failed.clear()
    reference_records = (
        load_reference_values(kwargs["reference_file"])
        if kwargs.get("reference_file") is not None
        else None
    )

    def campaign_result() -> BenchmarkComparisonResult:
        rows = tuple(results)
        summary = summarise_comparison(rows, baseline_solver=baseline_solver)
        by_solver = {
            solver: sum(failure.solver == solver for failure in failures) for solver in solver_names
        }
        summary["campaign"] = {
            "requested_runs": len(instances) * len(solver_names) * len(seed_values),
            "completed_runs": len(rows),
            "failed_runs": len(failures),
            "failures_by_solver": by_solver,
        }
        return BenchmarkComparisonResult(
            rows,
            summary,
            comparison_config,
            tuple(failures),
        )

    def save_checkpoint() -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_name(f".{checkpoint.name}.tmp")
        temporary.write_text(
            json.dumps(
                campaign_result().to_dict(),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(checkpoint)

    for seed in seed_values:
        seeded_config = replace(base_config, seed=seed)
        for instance_index, instance in enumerate(instances):
            source = Path(instance).expanduser()
            name = source.name
            requested_format = kwargs.get("format", "auto")
            resolved_format = (
                detect_format(source) if requested_format == "auto" else requested_format
            )
            pending_solvers = [
                solver
                for solver in solver_names
                if (name, solver, seed) not in completed and (name, solver, seed) not in failed
            ]
            if not pending_solvers:
                continue
            algebraic = None
            if resolved_format != "qplib":
                try:
                    algebraic = _load_algebraic(source, resolved_format)
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    for solver in pending_solvers:
                        failures.append(
                            BenchmarkFailure(
                                name,
                                resolved_format,
                                solver,
                                seed,
                                type(exc).__name__,
                            )
                        )
                    save_checkpoint()
                    continue
            ordered_solvers = _comparison_solver_order(
                solver_names,
                execution_order=execution_order,
                seed=seed,
                instance_index=instance_index,
            )
            applicability_hint = _qqa_applicability_hint(
                source,
                resolved_format,
                seeded_config,
                algebraic=algebraic,
            )
            if applicability_hint is False and "scip-aggressive" in ordered_solvers:
                ordered_solvers = (
                    "scip-aggressive",
                    *(solver for solver in ordered_solvers if solver != "scip-aggressive"),
                )
            for solver in ordered_solvers:
                key = (name, solver, seed)
                if key in completed or key in failed:
                    continue
                if solver == "sg-cqqa" and applicability_hint is False:
                    equivalent = next(
                        (
                            row
                            for row in results
                            if str(row.provenance.get("source_name", row.instance)) == name
                            and row.solver == "scip-aggressive"
                            and int(row.run_config.get("seed", 0)) == seed
                        ),
                        None,
                    )
                    if equivalent is not None:
                        results.append(_reuse_equivalent_aggressive_result(equivalent))
                        completed.add(key)
                        save_checkpoint()
                        continue
                    equivalent_failure = next(
                        (
                            row
                            for row in failures
                            if row.instance == name
                            and row.solver == "scip-aggressive"
                            and row.seed == seed
                        ),
                        None,
                    )
                    if equivalent_failure is not None:
                        failures.append(
                            BenchmarkFailure(
                                name,
                                equivalent_failure.format,
                                "sg-cqqa",
                                seed,
                                equivalent_failure.error_type,
                            )
                        )
                        failed.add(key)
                        save_checkpoint()
                        continue
                try:
                    if resolved_format == "qplib":
                        result = _run_isolated_benchmark_instance(
                            source,
                            resolved_format=resolved_format,
                            solver=solver,
                            seed=seed,
                            qqa_config=seeded_config,
                            reference_records=reference_records,
                            run_kwargs=dict(kwargs),
                            common_import=True,
                        )
                    else:
                        result = run_benchmark_instance(
                            source,
                            solver=solver,
                            seed=seed,
                            qqa_config=seeded_config,
                            _algebraic=algebraic,
                            _reference_records=reference_records,
                            **kwargs,
                        )
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    error_type = (
                        exc.error_type
                        if isinstance(exc, _IsolatedBenchmarkError)
                        else type(exc).__name__
                    )
                    failures.append(
                        BenchmarkFailure(
                            name,
                            resolved_format,
                            solver,
                            seed,
                            error_type,
                        )
                    )
                else:
                    results.append(result)
                    completed.add(key)
                save_checkpoint()
    return campaign_result()


__all__ = [
    "compare_benchmark_solvers",
    "detect_format",
    "load_reference_values",
    "run_benchmark_instance",
    "run_benchmark_suite",
]
