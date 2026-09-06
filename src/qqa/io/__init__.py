"""Lazy importers for public optimisation benchmark formats."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "load_dimacs": ("qqa.io.formats", "load_dimacs"),
    "load_ising_text": ("qqa.io.formats", "load_ising_text"),
    "load_model_ir_json": ("qqa.io.formats", "load_model_ir_json"),
    "load_mps": ("qqa.io.mps", "load_mps"),
    "load_opb": ("qqa.io.formats", "load_opb"),
    "load_portable_model": ("qqa.io.formats", "load_portable_model"),
    "load_qplib": ("qqa.io.qplib", "load_qplib"),
    "load_qubo_text": ("qqa.io.formats", "load_qubo_text"),
    "model_ir_from_dict": ("qqa.io.formats", "model_ir_from_dict"),
    "qplib_available": ("qqa.io.qplib", "qplib_available"),
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
