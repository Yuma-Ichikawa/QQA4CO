"""Wheel-shipped benchmark runner and report renderer.

The implementation lives in the package so :mod:`qqa.bench` and the
``qqa bench-*`` commands behave identically from a source checkout, wheel,
or sdist install.  Files under ``scripts/`` are deliberately thin command
wrappers around these modules.
"""

from __future__ import annotations

from qqa.benchmarking.algebraic_runner import (
    compare_benchmark_solvers,
    detect_format,
    load_reference_values,
    run_benchmark_instance,
    run_benchmark_suite,
)
from qqa.benchmarking.download import fetch_benchmark, fetch_instance
from qqa.benchmarking.merge import merge_benchmark_campaigns
from qqa.benchmarking.metrics import (
    BenchmarkComparisonResult,
    BenchmarkFailure,
    BenchmarkResult,
    BenchmarkSuiteResult,
    IncumbentPoint,
    normalised_primal_error,
    summarise_benchmarks,
    summarise_comparison,
)
from qqa.benchmarking.miplib import run_miplib
from qqa.benchmarking.publication import (
    publish_benchmark_campaigns,
    validate_portable_payload,
)
from qqa.benchmarking.qplib import run_qplib

__all__ = [
    "BenchmarkResult",
    "BenchmarkComparisonResult",
    "BenchmarkFailure",
    "BenchmarkSuiteResult",
    "IncumbentPoint",
    "compare_benchmark_solvers",
    "detect_format",
    "fetch_benchmark",
    "fetch_instance",
    "load_reference_values",
    "merge_benchmark_campaigns",
    "normalised_primal_error",
    "publish_benchmark_campaigns",
    "run_benchmark_instance",
    "run_benchmark_suite",
    "run_miplib",
    "run_qplib",
    "summarise_benchmarks",
    "summarise_comparison",
    "validate_portable_payload",
]
