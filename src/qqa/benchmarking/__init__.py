"""Lazy public facade for the opt-in benchmark integration.

Importing :mod:`qqa.benchmarking` exposes the portable result types and
MIPLIB/QPLIB helpers without eagerly loading parsers, SCIP-facing runners, or
plotting dependencies. A concrete implementation is imported only when its
attribute is first requested.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BenchmarkComparisonResult": ("qqa.benchmarking.metrics", "BenchmarkComparisonResult"),
    "BenchmarkFailure": ("qqa.benchmarking.metrics", "BenchmarkFailure"),
    "BenchmarkResult": ("qqa.benchmarking.metrics", "BenchmarkResult"),
    "BenchmarkSuiteResult": ("qqa.benchmarking.metrics", "BenchmarkSuiteResult"),
    "IncumbentPoint": ("qqa.benchmarking.metrics", "IncumbentPoint"),
    "compare_benchmark_solvers": (
        "qqa.benchmarking.algebraic_runner",
        "compare_benchmark_solvers",
    ),
    "detect_format": ("qqa.benchmarking.algebraic_runner", "detect_format"),
    "fetch_benchmark": ("qqa.benchmarking.download", "fetch_benchmark"),
    "fetch_instance": ("qqa.benchmarking.download", "fetch_instance"),
    "load_reference_values": (
        "qqa.benchmarking.algebraic_runner",
        "load_reference_values",
    ),
    "merge_benchmark_campaigns": (
        "qqa.benchmarking.merge",
        "merge_benchmark_campaigns",
    ),
    "normalised_primal_error": (
        "qqa.benchmarking.metrics",
        "normalised_primal_error",
    ),
    "publish_benchmark_campaigns": (
        "qqa.benchmarking.publication",
        "publish_benchmark_campaigns",
    ),
    "run_benchmark_instance": (
        "qqa.benchmarking.algebraic_runner",
        "run_benchmark_instance",
    ),
    "run_benchmark_suite": (
        "qqa.benchmarking.algebraic_runner",
        "run_benchmark_suite",
    ),
    "run_miplib": ("qqa.benchmarking.miplib", "run_miplib"),
    "run_qplib": ("qqa.benchmarking.qplib", "run_qplib"),
    "summarise_benchmarks": ("qqa.benchmarking.metrics", "summarise_benchmarks"),
    "summarise_comparison": ("qqa.benchmarking.metrics", "summarise_comparison"),
    "validate_portable_payload": (
        "qqa.benchmarking.publication",
        "validate_portable_payload",
    ),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = list(_EXPORTS)
