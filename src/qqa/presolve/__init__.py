"""Lazy presolve state, reduction, and scaling exports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BoundTighteningInfeasibleError": ("qqa.model.bounds", "BoundTighteningInfeasibleError"),
    "BoundTighteningResult": ("qqa.model.bounds", "BoundTighteningResult"),
    "PersistencyResult": ("qqa.presolve.qubo", "PersistencyResult"),
    "SCIPState": ("qqa.presolve.scip_bridge", "SCIPState"),
    "SCIPVariableMap": ("qqa.presolve.scip_bridge", "SCIPVariableMap"),
    "ScalingFactors": ("qqa.presolve.scaling", "ScalingFactors"),
    "build_scip_model": ("qqa.presolve.scip_bridge", "build_scip_model"),
    "compute_scaling": ("qqa.presolve.scaling", "compute_scaling"),
    "detect_qubo_symmetries": ("qqa.presolve.qubo", "detect_qubo_symmetries"),
    "dominance_fixings": ("qqa.presolve.qubo", "dominance_fixings"),
    "exact_probe_persistency": ("qqa.presolve.qubo", "exact_probe_persistency"),
    "extract_scip_state": ("qqa.presolve.scip_bridge", "extract_scip_state"),
    "general_qpbo_persistency": ("qqa.presolve.qubo", "general_qpbo_persistency"),
    "scaled_model": ("qqa.presolve.scaling", "scaled_model"),
    "submodular_roof_duality": ("qqa.presolve.qubo", "submodular_roof_duality"),
    "tighten_singleton_bounds": ("qqa.model.bounds", "tighten_singleton_bounds"),
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
