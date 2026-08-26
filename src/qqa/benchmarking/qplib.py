"""Convenience entry points for QPLIB instances."""

from qqa.benchmarking.algebraic_runner import run_benchmark_instance
from qqa.io.qplib import load_qplib


def run_qplib(path, **kwargs):
    kwargs["format"] = "qplib"
    return run_benchmark_instance(path, **kwargs)


__all__ = ["load_qplib", "run_qplib"]
