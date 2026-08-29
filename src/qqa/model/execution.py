"""Immutable factor-split execution plans for QQA-centred model evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from qqa.model.capabilities import (
    FactorCapability,
    factor_backend_registrations,
)
from qqa.model.ir import ModelIR, ObjectiveSense


@dataclass(frozen=True, slots=True)
class FactorExecutionBucket:
    """Factors sharing one concrete backend and execution contract."""

    backend: str
    factor_type: str
    locations: tuple[str, ...]
    capabilities: tuple[str, ...]
    supported_dtypes: tuple[str, ...]
    deterministic: bool


@dataclass(frozen=True, slots=True)
class CompiledExecutionPlan:
    """A device-aware, immutable lowering decision for one :class:`ModelIR`."""

    model: ModelIR
    buckets: tuple[FactorExecutionBucket, ...]
    device_type: str
    dtype: str
    deterministic: bool
    fused_graph: Any = None

    def internal_value(self, values: torch.Tensor) -> torch.Tensor:
        """Evaluate the canonical minimisation objective used by QQA."""
        if self.fused_graph is not None:
            graph = self.fused_graph
            if graph.linear.device != values.device or graph.linear.dtype != values.dtype:
                graph = graph.to(values.device, values.dtype)
            objective = graph.evaluate(values)
            return (
                objective
                if ObjectiveSense(self.model.sense) is ObjectiveSense.MINIMIZE
                else -objective
            )
        return self.model.internal_energy(values)

    def internal_value_and_grad(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return QQA energy and its gradient without mutating caller tensors."""
        differentiable = values.detach().clone().requires_grad_(True)
        energy = self.internal_value(differentiable)
        gradient = torch.autograd.grad(energy.sum(), differentiable)[0]
        return energy.detach(), gradient.detach()

    def constraint_violations(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.fused_graph is not None:
            graph = self.fused_graph
            if graph.linear.device != values.device or graph.linear.dtype != values.dtype:
                graph = graph.to(values.device, values.dtype)
            return graph.constraint_violations(values)
        return self.model.constraint_violations(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "dtype": self.dtype,
            "deterministic": self.deterministic,
            "fused": self.fused_graph is not None,
            "buckets": [asdict(item) for item in self.buckets],
        }


def _select_backend(factor: Any, device_type: str, dtype: str):
    registrations = factor_backend_registrations(factor)
    candidates = [
        item
        for item in registrations
        if dtype in item.supported_dtypes
        and FactorCapability.EVALUATE in item.capabilities
        and bool(
            item.capabilities
            & {
                FactorCapability.DIFFERENTIABLE,
                FactorCapability.SUBGRADIENT,
                FactorCapability.PROX,
            }
        )
    ]
    if device_type == "cuda":
        fused = [item for item in candidates if FactorCapability.GPU_KERNEL in item.capabilities]
        if fused:
            return fused[0]
    eager = [item for item in candidates if item.name == "torch-eager"]
    return eager[0] if eager else candidates[0] if candidates else None


def compile_execution_plan(
    model: ModelIR,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    strict: bool = True,
) -> CompiledExecutionPlan:
    """Select registered factor backends and optionally fuse the whole graph."""
    if not isinstance(model, ModelIR):
        raise TypeError("model must be a ModelIR.")
    resolved = torch.device(device)
    dtype_name = str(dtype).removeprefix("torch.")
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    unsupported = []
    expressions = [("objective", model.objective)] + [
        (f"constraint:{row.name}", row.expression) for row in model.constraints
    ]
    selected = []
    for location, expression in expressions:
        for index, factor in enumerate(expression.factors):
            backend = _select_backend(factor, resolved.type, dtype_name)
            if backend is None:
                unsupported.append(f"{location}[{index}]={type(factor).__name__}")
                continue
            selected.append(backend)
            key = (backend.name, type(factor).__name__)
            bucket = grouped.setdefault(
                key,
                {
                    "backend": backend,
                    "locations": [],
                },
            )
            bucket["locations"].append(f"{location}[{index}]")
    if strict and unsupported:
        raise NotImplementedError(
            "No registered QQA execution backend for: " + ", ".join(unsupported)
        )
    buckets = tuple(
        FactorExecutionBucket(
            backend=item["backend"].name,
            factor_type=factor_type,
            locations=tuple(item["locations"]),
            capabilities=tuple(
                sorted(capability.value for capability in item["backend"].capabilities)
            ),
            supported_dtypes=item["backend"].supported_dtypes,
            deterministic=item["backend"].deterministic,
        )
        for (backend_name, factor_type), item in sorted(grouped.items())
    )
    fused_graph = None
    if selected and all(
        FactorCapability.GPU_KERNEL in item.capabilities for item in selected
    ):
        from qqa.gpu.factors import compile_factor_graph

        fused_graph = compile_factor_graph(model).to(resolved, dtype)
    return CompiledExecutionPlan(
        model,
        buckets,
        resolved.type,
        dtype_name,
        all(item.deterministic for item in selected),
        fused_graph,
    )


__all__ = ["CompiledExecutionPlan", "FactorExecutionBucket", "compile_execution_plan"]
