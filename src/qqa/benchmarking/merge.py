"""Validated merging for independently executed benchmark shards."""

from __future__ import annotations

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

    All comparison settings except ``instances`` must match exactly. Duplicate
    instances and duplicate solver/instance/seed records are rejected instead
    of silently selecting one shard.
    """
    if isinstance(paths, (str, Path)):
        raise TypeError("paths must be a sequence of campaign JSON files.")
    sources = tuple(Path(path).expanduser() for path in paths)
    if not sources:
        raise ValueError("paths must contain at least one campaign JSON file.")

    base_config = None
    all_instances: list[str] = []
    results: list[BenchmarkResult] = []
    failures: list[BenchmarkFailure] = []
    result_keys: set[tuple[str, str, int]] = set()
    failure_keys: set[tuple[str, str, int]] = set()

    for source in sources:
        payload = json.loads(source.read_text(encoding="utf-8"))
        config = deepcopy(payload.get("comparison_config"))
        if not isinstance(config, dict) or not isinstance(config.get("instances"), list):
            raise ValueError(f"Invalid comparison configuration in {source.name!r}.")
        instances = config.pop("instances")
        if not instances or any(not isinstance(name, str) or not name for name in instances):
            raise ValueError(f"Invalid instance list in {source.name!r}.")
        if len(set(instances)) != len(instances):
            raise ValueError(f"Duplicate instance within {source.name!r}.")
        overlap = set(all_instances).intersection(instances)
        if overlap:
            raise ValueError(f"Campaign shards contain duplicate instance {min(overlap)!r}.")
        if base_config is None:
            base_config = config
        elif config != base_config:
            raise ValueError("Campaign shard comparison configurations do not match.")
        all_instances.extend(instances)

        for raw in payload.get("results", ()):
            result = BenchmarkResult.from_dict(raw)
            source_name = str(result.provenance.get("source_name", result.instance))
            if source_name not in instances:
                raise ValueError(f"Campaign record {source_name!r} is outside {source.name!r}.")
            key = (
                source_name,
                result.solver,
                int(result.run_config.get("seed", 0)),
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
            key = (failure.instance, failure.solver, failure.seed)
            if key in result_keys or key in failure_keys:
                raise ValueError(f"Duplicate campaign record for {key!r}.")
            failure_keys.add(key)
            failures.append(failure)

    assert base_config is not None
    instance_names = sorted(all_instances)
    instance_order = {name: index for index, name in enumerate(instance_names)}
    solver_names = list(base_config["solvers"])
    solver_order = {name: index for index, name in enumerate(solver_names)}
    seed_values = list(base_config["seeds"])
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
    comparison_config = {"instances": instance_names, **base_config}
    baseline = str(base_config["baseline_solver"])
    summary = summarise_comparison(results, baseline_solver=baseline)
    summary["campaign"] = {
        "requested_runs": len(instance_names) * len(solver_names) * len(seed_values),
        "completed_runs": len(results),
        "failed_runs": len(failures),
        "failures_by_solver": {
            solver: sum(row.solver == solver for row in failures) for solver in solver_names
        },
    }
    return BenchmarkComparisonResult(
        tuple(results),
        summary,
        comparison_config,
        tuple(failures),
    )


__all__ = ["merge_benchmark_campaigns"]
