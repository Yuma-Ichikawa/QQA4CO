"""Presolve state extraction, safe reduction, and numerical scaling helpers."""

from qqa.model.bounds import (
    BoundTighteningInfeasibleError,
    BoundTighteningResult,
    tighten_singleton_bounds,
)
from qqa.presolve.qubo import (
    PersistencyResult,
    detect_qubo_symmetries,
    dominance_fixings,
    exact_probe_persistency,
    general_qpbo_persistency,
    submodular_roof_duality,
)
from qqa.presolve.scaling import ScalingFactors, compute_scaling, scaled_model
from qqa.presolve.scip_bridge import (
    SCIPState,
    SCIPVariableMap,
    build_scip_model,
    extract_scip_state,
)

__all__ = [
    "BoundTighteningInfeasibleError",
    "BoundTighteningResult",
    "PersistencyResult",
    "SCIPState",
    "SCIPVariableMap",
    "ScalingFactors",
    "build_scip_model",
    "compute_scaling",
    "detect_qubo_symmetries",
    "dominance_fixings",
    "exact_probe_persistency",
    "general_qpbo_persistency",
    "extract_scip_state",
    "scaled_model",
    "submodular_roof_duality",
    "tighten_singleton_bounds",
]
