"""Solver-independent structural feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from qqa.algebraic import AlgebraicModel
from qqa.compile import compile_sparse_qubo
from qqa.model import ModelIR, ObjectiveSense, VariableDomain
from qqa.model.adapters import algebraic_to_model_ir, problem_to_model_ir
from qqa.model.capabilities import inspect_capabilities


@dataclass(frozen=True, slots=True)
class ModelInspection:
    name: str
    num_variables: int
    relaxed_variables: int
    num_constraints: int
    domains: dict[str, int]
    factor_counts: dict[str, int]
    nonzeros: int
    density: float
    coefficient_dynamic_range: float
    connected_components: int
    objective_sense: str
    structure: tuple[str, ...]
    qqa_compatible: bool = True
    exact_compatible: bool = True
    missing_bounds: tuple[str, ...] = ()
    unsupported_qqa: tuple[str, ...] = ()
    unsupported_exact: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dynamic_range(tensors: list[torch.Tensor]) -> float:
    nonzero = [tensor.detach().abs().reshape(-1) for tensor in tensors if tensor.numel()]
    if not nonzero:
        return 1.0
    values = torch.cat(nonzero)
    values = values[values > 0]
    if values.numel() == 0:
        return 1.0
    return float((values.max() / values.min()).item())


def inspect_model(model: Any) -> ModelInspection:
    """Return stable planner features without mutating or solving a model."""
    algebraic = model if isinstance(model, AlgebraicModel) else None
    ir = (
        model
        if isinstance(model, ModelIR)
        else algebraic_to_model_ir(model)
        if algebraic is not None
        else problem_to_model_ir(model)
    )
    domains: dict[str, int] = {}
    for block in ir.variables:
        domain = VariableDomain(block.domain).value
        domains[domain] = domains.get(domain, 0) + block.size
    factors: dict[str, int] = {}
    tensors: list[torch.Tensor] = []
    nonzeros = 0
    for factor in ir.objective.factors:
        name = type(factor).__name__
        factors[name] = factors.get(name, 0) + 1
        weights = getattr(factor, "weights", None)
        if torch.is_tensor(weights):
            tensors.append(weights)
            nonzeros += int(torch.count_nonzero(weights).item())
    for row in ir.constraints:
        for factor in row.expression.factors:
            name = type(factor).__name__
            factors[name] = factors.get(name, 0) + 1
            weights = getattr(factor, "weights", None)
            if torch.is_tensor(weights):
                tensors.append(weights)
                nonzeros += int(torch.count_nonzero(weights).item())

    components = 1
    structure = []
    try:
        qubo = compile_sparse_qubo(model)
    except (TypeError, ValueError, AttributeError):
        qubo = None
    if qubo is not None:
        components = len(qubo.connected_components())
        structure.append("sparse-qubo" if qubo.density < 0.2 else "dense-qubo")
        nonzeros = qubo.num_variables + qubo.num_edges
        tensors.extend((qubo.linear, qubo.edge_weight))
        density = qubo.density
    else:
        possible = max(1, ir.num_variables * ir.num_variables)
        density = min(1.0, nonzeros / possible)
    if components > 1:
        structure.append("decomposable")
    if ir.constraints:
        structure.append("constrained")
    if set(domains) == {"binary"}:
        structure.append("pure-binary")
    elif set(domains) <= {"binary", "integer", "real"}:
        structure.append("mixed-domain")
    if any(key in factors for key in ("AssignmentFactor", "AllDifferentFactor")):
        structure.append("assignment")
    if "permutation" in domains:
        structure.append("assignment")
    elif "categorical" in domains:
        structure.append("categorical")
    if any(key == "ClauseFactor" for key in factors):
        structure.append("clause")
    if factors and set(factors) <= {"LinearFactor"}:
        structure.append("linear")

    capability_report = inspect_capabilities(ir)
    return ModelInspection(
        name=ir.metadata.name,
        num_variables=ir.num_variables,
        relaxed_variables=sum(block.size * int(block.categories or 1) for block in ir.variables),
        num_constraints=len(ir.constraints),
        domains=domains,
        factor_counts=factors,
        nonzeros=nonzeros,
        density=density,
        coefficient_dynamic_range=_dynamic_range(tensors),
        connected_components=components,
        objective_sense=ObjectiveSense(ir.sense).value,
        structure=tuple(structure),
        qqa_compatible=capability_report.qqa_compatible,
        exact_compatible=capability_report.exact_compatible,
        missing_bounds=capability_report.missing_bounds,
        unsupported_qqa=capability_report.unsupported_qqa,
        unsupported_exact=capability_report.unsupported_exact,
    )


__all__ = ["ModelInspection", "inspect_model"]
