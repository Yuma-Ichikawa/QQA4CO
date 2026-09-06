"""Command-line adapter for the optional MIPLIB/QPLIB integration.

This module contains only argument registration and command orchestration.
Heavy readers and solver integrations remain local imports, so commands such
as ``qqa version`` do not activate SCIP or QPLIB dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _optional_positive_int(value: str) -> int | None:
    """Parse a positive integer or an explicit unbounded sentinel."""
    normalized = value.strip().lower()
    if normalized in {"none", "unbounded"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a positive integer, 'none', or 'unbounded'"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer, 'none', or 'unbounded'")
    return parsed


def _add_heuristic_options(parser: argparse.ArgumentParser) -> None:
    """Register the shared, explicitly opt-in SG-CQQA tuning surface."""
    from qqa.hybrid.heuristic_types import QQAHeuristicConfig

    defaults = QQAHeuristicConfig()
    parser.add_argument("--core-size", type=int, default=defaults.core_size)
    parser.add_argument(
        "--maximum-problem-variables",
        type=_optional_positive_int,
        default=defaults.maximum_problem_variables,
        metavar="N|none",
        help="Skip larger models before plugin setup; 'none' enables bounded-core use at any size.",
    )
    parser.add_argument(
        "--maximum-integer-variables",
        type=_optional_positive_int,
        default=defaults.maximum_integer_variables,
        metavar="N|none",
        help="Skip models with more integer variables; 'none' removes this structural gate.",
    )
    parser.add_argument("--qplib-problem-types", nargs="+", default=None)
    parser.add_argument("--minimum-core-size", type=int, default=defaults.minimum_core_size)
    parser.add_argument(
        "--maximum-core-saturation", type=float, default=defaults.maximum_core_saturation
    )
    parser.add_argument("--sol-size", type=int, default=defaults.sol_size)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--max-calls", type=int, default=defaults.max_calls)
    parser.add_argument("--max-candidates", type=int, default=defaults.max_candidates)
    parser.add_argument("--completion-time", type=float, default=defaults.completion_time)
    parser.add_argument("--completion-nodes", type=int, default=defaults.completion_nodes)
    parser.add_argument("--dive-lp-iterations", type=int, default=defaults.dive_lp_iterations)
    parser.add_argument("--qqa-fix-fraction", type=float, default=defaults.qqa_fix_fraction)
    parser.add_argument("--repair-beam-width", type=int, default=defaults.repair_beam_width)
    parser.add_argument("--reference-pool-size", type=int, default=defaults.reference_pool_size)
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=defaults.minimum_relative_improvement,
    )
    parser.add_argument("--min-call-time", type=float, default=defaults.minimum_call_time)
    parser.add_argument("--min-qqa-time", type=float, default=defaults.minimum_qqa_time)
    parser.add_argument("--maximum-call-time", type=float, default=defaults.maximum_call_time)
    parser.add_argument(
        "--maximum-call-time-fraction",
        type=float,
        default=defaults.maximum_call_time_fraction,
    )
    parser.add_argument(
        "--minimum-runtime-startup-time",
        type=float,
        default=defaults.minimum_runtime_startup_time,
    )
    parser.add_argument(
        "--min-nodes-between-calls", type=int, default=defaults.min_nodes_between_calls
    )
    parser.add_argument("--fast-candidates", type=int, default=defaults.fast_candidates)
    parser.add_argument("--no-subscip-repair", action="store_true")
    parser.add_argument("--local-branching-radius", type=int, default=None)
    parser.add_argument("--max-lp-rows", type=int, default=defaults.max_lp_rows)
    parser.add_argument("--objective-weight", type=float, default=defaults.objective_weight)
    parser.add_argument("--row-penalty", type=float, default=defaults.row_penalty)
    parser.add_argument("--proximity-weight", type=float, default=defaults.proximity_weight)
    parser.add_argument("--reduced-cost-weight", type=float, default=defaults.reduced_cost_weight)
    parser.add_argument("--allow-nonimproving-candidates", action="store_true")
    parser.add_argument("--allow-no-incumbent", action="store_true")
    parser.add_argument("--no-adaptive-row-lagrangian", action="store_true")
    parser.add_argument("--continue-qqa-without-improvement", action="store_true")
    parser.add_argument(
        "--maximum-overhead-fraction", type=float, default=defaults.maximum_overhead_fraction
    )
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument(
        "--core-dtype",
        choices=("float32", "float64"),
        default=defaults.core_dtype,
        help="Numerical dtype for the bounded QQA surrogate search.",
    )


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    """Register process-boundary controls shared by run and compare."""
    parser.add_argument(
        "--worker-timeout",
        type=float,
        default=None,
        help="Hard timeout for each isolated native worker (defaults to budget plus grace).",
    )
    parser.add_argument(
        "--include-solution-values",
        action="store_true",
        help="Store the verified final solution in original variable order (can be large).",
    )
    parser.add_argument(
        "--implementation-revision",
        default=None,
        help="Public 7-64 character lowercase hexadecimal source revision.",
    )


def _named_paths(values: list[str], *, option: str) -> dict[str, Path]:
    """Parse repeatable ``NAME=PATH`` arguments without resolving private paths."""
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip().lower()
        if not separator or not name or not raw_path.strip():
            raise ValueError(f"{option} entries must use NAME=PATH.")
        if name in result:
            raise ValueError(f"{option} contains duplicate name {name!r}.")
        result[name] = Path(raw_path).expanduser()
    return result


def add_benchmark_parser(subparsers) -> argparse.ArgumentParser:
    """Add the complete ``qqa benchmark`` command tree."""
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Fetch, inspect, and solve MIPLIB/QPLIB instances.",
    )
    commands = benchmark.add_subparsers(dest="benchmark_command", required=True)

    fetch = commands.add_parser(
        "fetch",
        help="Download an official benchmark snapshot or one public instance.",
    )
    fetch.add_argument("library", choices=("miplib", "qplib"))
    fetch.add_argument("--output", required=True)
    fetch.add_argument(
        "--instance",
        default=None,
        help="Fetch one instance id/name instead of the complete public archive.",
    )
    fetch.add_argument("--no-extract", action="store_true")
    fetch.add_argument("--overwrite", action="store_true")

    inspect = commands.add_parser(
        "inspect",
        help="Read an MPS/QPLIB instance and print sparse model metadata.",
    )
    inspect.add_argument("instance")
    inspect.add_argument("--format", choices=("auto", "miplib", "qplib"), default="auto")

    merge = commands.add_parser(
        "merge",
        help="Validate and merge disjoint benchmark comparison shards.",
    )
    merge.add_argument("campaign", nargs="+")
    merge.add_argument("--output", required=True)
    merge.add_argument("--quiet", action="store_true")

    publish = commands.add_parser(
        "publish",
        help="Create path-free compact/full campaign artifacts and a checksum manifest.",
    )
    publish.add_argument(
        "--campaign",
        action="append",
        required=True,
        metavar="LIBRARY=PATH",
        help="Merged campaign JSON; repeat once per library.",
    )
    publish.add_argument(
        "--snapshot",
        action="append",
        required=True,
        metavar="LIBRARY=PATH",
        help="Matching public snapshot JSON; repeat once per library.",
    )
    publish.add_argument("--output", required=True, help="Destination directory.")
    publish.add_argument(
        "--implementation-revision",
        default=None,
        help="Public 7-64 character lowercase hexadecimal source revision.",
    )
    publish.add_argument("--quiet", action="store_true")

    run = commands.add_parser(
        "run",
        help="Solve one MIPLIB/QPLIB instance with SCIP or SG-CQQA.",
    )
    run.add_argument("instance", nargs="+")
    run.add_argument("--format", choices=("auto", "miplib", "qplib"), default="auto")
    run.add_argument(
        "--solver",
        choices=("scip", "scip-aggressive", "sg-cqqa"),
        default="sg-cqqa",
    )
    run.add_argument("--time-limit", type=float, default=60.0)
    run.add_argument("--gap", type=float, default=0.0)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--reference-file", default=None)
    run.add_argument("--output", default=None, help="Write machine-readable JSON.")
    run.add_argument("--quiet", action="store_true")
    _add_runtime_options(run)
    _add_heuristic_options(run)
    run.add_argument("--seed", type=int, default=0)

    compare = commands.add_parser(
        "compare",
        help="Run paired SCIP/native-heuristic/SG-CQQA comparisons.",
    )
    compare.add_argument("instance", nargs="+")
    compare.add_argument("--format", choices=("auto", "miplib", "qplib"), default="auto")
    compare.add_argument(
        "--solvers",
        nargs="+",
        choices=("scip", "scip-aggressive", "sg-cqqa"),
        default=("scip-aggressive", "sg-cqqa"),
    )
    compare.add_argument("--baseline-solver", default="scip-aggressive")
    compare.add_argument("--execution-order", choices=("balanced", "fixed"), default="balanced")
    compare.add_argument("--seeds", nargs="+", type=int, default=(0,))
    compare.add_argument("--time-limit", type=float, default=60.0)
    compare.add_argument("--gap", type=float, default=0.0)
    compare.add_argument("--threads", type=int, default=1)
    compare.add_argument("--reference-file", default=None)
    compare.add_argument("--output", required=True)
    compare.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching incremental JSON checkpoint at --output.",
    )
    compare.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a path-free failure and continue with the remaining runs.",
    )
    compare.add_argument(
        "--retry-failures",
        action="store_true",
        help="When resuming, retry runs recorded as failures.",
    )
    compare.add_argument(
        "--no-equivalent-baseline-reuse",
        action="store_true",
        help="Execute structurally inapplicable hybrid cells independently for audit accounting.",
    )
    compare.add_argument(
        "--isolate-all",
        action="store_true",
        help="Run every solver cell in a fresh process for ABI and memory isolation.",
    )
    compare.add_argument(
        "--include-import-in-budget",
        action="store_true",
        help="Start each solver clock before parsing the original model.",
    )
    compare.add_argument("--quiet", action="store_true")
    _add_runtime_options(compare)
    _add_heuristic_options(compare)
    return benchmark


def _heuristic_config(args: argparse.Namespace, *, seed: int):
    from qqa.hybrid import QQAHeuristicConfig

    return QQAHeuristicConfig(
        core_size=args.core_size,
        maximum_problem_variables=args.maximum_problem_variables,
        maximum_integer_variables=args.maximum_integer_variables,
        allowed_qplib_problem_types=(
            tuple(args.qplib_problem_types) if args.qplib_problem_types is not None else None
        ),
        minimum_core_size=args.minimum_core_size,
        maximum_core_saturation=args.maximum_core_saturation,
        sol_size=args.sol_size,
        epochs=args.epochs,
        max_calls=args.max_calls,
        max_candidates=args.max_candidates,
        completion_time=args.completion_time,
        completion_nodes=args.completion_nodes,
        dive_lp_iterations=args.dive_lp_iterations,
        subscip_repair=not args.no_subscip_repair,
        qqa_fix_fraction=args.qqa_fix_fraction,
        repair_beam_width=args.repair_beam_width,
        reference_pool_size=args.reference_pool_size,
        minimum_relative_improvement=args.minimum_relative_improvement,
        minimum_call_time=args.min_call_time,
        minimum_qqa_time=args.min_qqa_time,
        maximum_call_time=args.maximum_call_time,
        maximum_call_time_fraction=args.maximum_call_time_fraction,
        minimum_runtime_startup_time=args.minimum_runtime_startup_time,
        fast_candidates=args.fast_candidates,
        min_nodes_between_calls=args.min_nodes_between_calls,
        local_branching_radius=args.local_branching_radius,
        max_lp_rows=args.max_lp_rows,
        objective_weight=args.objective_weight,
        row_penalty=args.row_penalty,
        proximity_weight=args.proximity_weight,
        reduced_cost_weight=args.reduced_cost_weight,
        require_surrogate_improvement=not args.allow_nonimproving_candidates,
        require_incumbent=not args.allow_no_incumbent,
        adaptive_row_lagrangian=not args.no_adaptive_row_lagrangian,
        stop_qqa_after_nonimproving_call=not args.continue_qqa_without_improvement,
        maximum_overhead_fraction=args.maximum_overhead_fraction,
        threads=args.threads,
        seed=seed,
        device=args.device,
        core_dtype=args.core_dtype,
        verbose=not args.quiet,
    )


def _write_result(result, output_name: str | None, *, print_output: bool = True) -> None:
    rendered = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if print_output:
        print(rendered)
    if output_name:
        output = Path(output_name).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def run_benchmark_command(args: argparse.Namespace) -> int:
    """Execute one parsed ``qqa benchmark`` command."""
    if args.benchmark_command == "fetch":
        from qqa.benchmarking.download import fetch_benchmark, fetch_instance

        try:
            payload = (
                fetch_instance(
                    args.library,
                    args.instance,
                    args.output,
                    overwrite=args.overwrite,
                )
                if args.instance
                else fetch_benchmark(
                    args.library,
                    args.output,
                    extract=not args.no_extract,
                    overwrite=args.overwrite,
                )
            )
        except RuntimeError as exc:
            print(f"[qqa benchmark] {exc}", file=sys.stderr)
            return 1
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.benchmark_command == "inspect":
        from qqa.benchmarking.algebraic_runner import detect_format

        instance = Path(args.instance).expanduser()
        resolved_format = detect_format(instance) if args.format == "auto" else args.format
        if resolved_format == "qplib":
            from qqa.io import load_qplib

            model = load_qplib(instance)
        else:
            from qqa.io import load_mps

            model = load_mps(instance)
        print(json.dumps(model.summary(), ensure_ascii=False, indent=2))
        return 0

    if args.benchmark_command == "merge":
        from qqa.benchmarking.merge import merge_benchmark_campaigns

        merged = merge_benchmark_campaigns(args.campaign)
        rendered = json.dumps(merged.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        if not args.quiet:
            print(rendered)
        return 0

    if args.benchmark_command == "publish":
        from qqa.benchmarking.publication import publish_benchmark_campaigns

        try:
            campaigns = _named_paths(args.campaign, option="--campaign")
            snapshots = _named_paths(args.snapshot, option="--snapshot")
            manifest = publish_benchmark_campaigns(
                campaigns,
                snapshots,
                args.output,
                implementation_revision=args.implementation_revision,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"[qqa benchmark] {exc}", file=sys.stderr)
            return 2
        if not args.quiet:
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    from qqa.benchmarking.algebraic_runner import (
        compare_benchmark_solvers,
        run_benchmark_instance,
        run_benchmark_suite,
    )

    comparison = args.benchmark_command == "compare"
    seed = args.seeds[0] if comparison else args.seed
    config = _heuristic_config(args, seed=seed)
    instances = tuple(Path(value).expanduser() for value in args.instance)
    common = {
        "format": args.format,
        "time_limit": args.time_limit,
        "relative_gap_limit": args.gap,
        "threads": args.threads,
        "reference_file": args.reference_file,
        "worker_timeout": args.worker_timeout,
        "implementation_revision": args.implementation_revision,
        "verbose": not args.quiet,
    }
    result: Any
    if comparison:
        result = compare_benchmark_solvers(
            instances,
            solvers=args.solvers,
            seeds=args.seeds,
            baseline_solver=args.baseline_solver,
            execution_order=args.execution_order,
            qqa_config=config,
            checkpoint_file=args.output,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
            retry_failures=args.retry_failures,
            reuse_equivalent_baseline=not args.no_equivalent_baseline_reuse,
            isolate_all=args.isolate_all,
            include_import_in_budget=args.include_import_in_budget,
            include_solution_values=args.include_solution_values,
            **common,
        )
    else:
        common.update(
            solver=args.solver,
            seed=args.seed,
            qqa_config=config,
            include_solution_values=args.include_solution_values,
        )
        result = (
            run_benchmark_instance(instances[0], **common)
            if len(instances) == 1
            else run_benchmark_suite(instances, **common)
        )
    _write_result(result, args.output, print_output=not args.quiet)
    return 0


__all__ = ["add_benchmark_parser", "run_benchmark_command"]
