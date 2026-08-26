"""Convenience entry points for MIPLIB instances."""

from qqa.benchmarking.algebraic_runner import run_benchmark_instance
from qqa.io.mps import load_mps


def run_miplib(path, **kwargs):
    kwargs["format"] = "miplib"
    return run_benchmark_instance(path, **kwargs)


__all__ = ["load_mps", "run_miplib"]
