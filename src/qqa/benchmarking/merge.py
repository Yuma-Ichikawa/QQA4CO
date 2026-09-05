"""Validated merging for independently executed benchmark shards."""

from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

from qqa.benchmarking.metrics import (
    BenchmarkComparisonResult,
    BenchmarkFailure,
    BenchmarkResult,
    summarise_comparison,
)


def merge_benchmark_campaigns(paths) -> BenchmarkComparisonResult:
    """Merge disjoint portable checkpoints and recompute every aggregate.

    All comparison settings except ``instances`` and ``seeds`` must match
    exactly. Shards may partition either axis (or both), but their requested
    ``(instance, seed)`` cells must be disjoint and form one complete Cartesian
    campaign. Duplicate records are rejected instead of silently selecting one.
    """
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be a sequence of campaign JSON files.")
    sources = tuple(Path(path).expanduser() for path in paths)
    if not sources:
        raise ValueError("paths must contain at least one campaign JSON file.")

    base_config = None
    all_instances: set[str] = set()
    all_seeds: set[int] = set()
    requested_cells: set[tuple[str, int]] = set()
    results: list[BenchmarkResult] = []
    failures: list[BenchmarkFailure] = []
    result_keys: set[tuple[str, str, int]] = set()
    failure_keys: set[tuple[str, str, int]] = set()

    for source in sources:
        opener = gzip.open if source.suffix == ".gz" else Path.open
        with opener(source, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        config = deepcopy(payload.get("comparison_config"))
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("instances"), list)
            or not isinstance(config.get("seeds"), list)
        ):
            raise ValueError(f"Invalid comparison configuration in {source.name!r}.")
        instances = config.pop("instances")
        seeds = config.pop("seeds")
        if not instances or any(not isinstance(name, str) or not name for name in instances):
            raise ValueError(f"Invalid instance list in {source.name!r}.")
        if len(set(instances)) != len(instances):
            raise ValueError(f"Duplicate instance within {source.name!r}.")
        if not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds
        ):
            raise ValueError(f"Invalid seed list in {source.name!r}.")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seed within {source.name!r}.")
        qqa_config = config.get("qqa_config")
        if isinstance(qqa_config, dict):
            configured_seed = qqa_config.pop("seed", None)
            if configured_seed is not None and configured_seed not in seeds:
                raise ValueError(f"QQA seed is outside {source.name!r}.")
        source_cells = {(name, seed) for name in instances for seed in seeds}
        overlap = requested_cells.intersection(source_cells)
        if overlap:
            raise ValueError(f"Campaign shards duplicate request cell {min(overlap)!r}.")
        requested_cells.update(source_cells)
        if base_config is None:
            base_config = config
        elif config != base_config:
            raise ValueError("Campaign shard comparison configurations do not match.")
        all_instances.update(instances)
        all_seeds.update(seeds)

        for raw in payload.get("results", ()):
            result = BenchmarkResult.from_dict(raw)
            source_name = str(result.provenance.get("source_name", result.instance))
            if source_name not in instances:
                raise ValueError(f"Campaign record {source_name!r} is outside {source.name!r}.")
            result_seed = int(result.run_config.get("seed", 0))
            if result_seed not in seeds:
                raise ValueError(
                    f"Campaign record seed {result_seed!r} is outside {source.name!r}."
                )
            key = (
                source_name,
                result.solver,
                result_seed,
            )
            if key in result_keys or key in failure_keys:
                raise ValueError(f"Duplicate campaign record for {key!r}.")
            result_keys.add(key)
            results.append(result)
        for raw in payload.get("failures", ()):
            failure = BenchmarkFailure.from_dict(raw)
            if failure.instance not in instances:
                raise ValueError(
                    f"Campaign failure {failure.instance!r} is outside {source.name!r}."
                )
            if failure.seed not in seeds:
                raise ValueError(
                    f"Campaign failure seed {failure.seed!r} is outside {source.name!r}."
                )
            key = (failure.instance, failure.solver, failure.seed)
            if key in result_keys or key in failure_keys:
                raise ValueError(f"Duplicate campaign record for {key!r}.")
            failure_keys.add(key)
            failures.append(failure)

    assert base_config is not None
    instance_names = sorted(all_instances)
    seed_values = sorted(all_seeds)
    expected_cells = {(name, seed) for name in instance_names for seed in seed_values}
    if requested_cells != expected_cells:
        raise ValueError("Campaign shards do not form a complete instance/seed Cartesian grid.")
    instance_order = {name: index for index, name in enumerate(instance_names)}
    solver_names = list(base_config["solvers"])
    solver_order = {name: index for index, name in enumerate(solver_names)}
    seed_order = {seed: index for index, seed in enumerate(seed_values)}
    results.sort(
        key=lambda row: (
            instance_order[str(row.provenance.get("source_name", row.instance))],
            seed_order[int(row.run_config.get("seed", 0))],
            solver_order[row.solver],
        )
    )
    failures.sort(
        key=lambda row: (
            instance_order[row.instance],
            seed_order[row.seed],
            solver_order[row.solver],
        )
    )
    comparison_config = {
        "instances": instance_names,
        "seeds": seed_values,
        **base_config,
    }
    baseline = str(base_config["baseline_solver"])
    summary = summarise_comparison(results, baseline_solver=baseline)
    summary["campaign"] = {
        "requested_runs": len(requested_cells) * len(solver_names),
        "completed_runs": len(results),
        "failed_runs": len(failures),
        "failures_by_solver": {
            solver: sum(row.solver == solver for row in failures) for solver in solver_names
        },
        "failures_by_type": {
            error_type: sum(row.error_type == error_type for row in failures)
            for error_type in sorted({row.error_type for row in failures})
        },
    }
    return BenchmarkComparisonResult(
        tuple(results),
        summary,
        comparison_config,
        tuple(failures),
    )


__all__ = ["merge_benchmark_campaigns"]
