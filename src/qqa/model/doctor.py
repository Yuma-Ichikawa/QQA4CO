"""Pre-solve mathematical, capability, and resource diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch

from qqa.algebraic import AlgebraicModel
from qqa.model.adapters import algebraic_to_model_ir, problem_to_model_ir
from qqa.model.capabilities import ModelCapabilityReport, inspect_capabilities
from qqa.model.ir import ModelIR, QuadraticEdgeFactor, VariableDomain


class DiagnosticSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ModelDiagnostic:
    severity: DiagnosticSeverity
    code: str
    message: str
    remediation: str | None = None


@dataclass(frozen=True, slots=True)
class SolverCapability:
    route: str
    supported: bool
    proof_available: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ModelDoctorReport:
    model_name: str
    num_variables: int
    num_constraints: int
    diagnostics: tuple[ModelDiagnostic, ...]
    capabilities: ModelCapabilityReport
    solver_routes: tuple[SolverCapability, ...]
    estimated_population_memory_bytes: int
    recommended_budget_seconds: float
    recommended_replicas: int
    expected_kernel: str
    factor_count: int
    estimated_host_device_transfer_bytes: int
    decomposition_method: str

    @property
    def ready(self) -> bool:
        return not any(item.severity is DiagnosticSeverity.ERROR for item in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ready"] = self.ready
        return payload

    def explain(self) -> str:
        lines = [
            f"Model Doctor: {self.model_name}",
            f"Variables/constraints: {self.num_variables}/{self.num_constraints}",
            f"Pure QQA ready: {'yes' if self.capabilities.qqa_compatible else 'no'}",
        ]
        for item in self.diagnostics:
            lines.append(f"{item.severity.value.upper()} [{item.code}] {item.message}")
        return "\n".join(lines)


def _as_ir(model: Any) -> ModelIR:
    if isinstance(model, ModelIR):
        return model
    if isinstance(model, AlgebraicModel):
        return algebraic_to_model_ir(model)
    return problem_to_model_ir(model)


def _coefficient_tensors(model: ModelIR) -> list[torch.Tensor]:
    tensors = []
    expressions = [model.objective, *(row.expression for row in model.constraints)]
    for expression in expressions:
        for factor in expression.factors:
            for name in ("weights", "outputs", "durations", "demands", "supplies"):
                value = getattr(factor, name, None)
                if torch.is_tensor(value) and value.numel():
                    tensors.append(value.detach().to(torch.float64).reshape(-1))
    return tensors


def _quadratic_curvature(model: ModelIR) -> tuple[bool | None, float | None]:
    if model.num_variables > 2048:
        return None, None
    matrix = torch.zeros((model.num_variables, model.num_variables), dtype=torch.float64)
    found = False
    for factor in model.objective.factors:
        if isinstance(factor, QuadraticEdgeFactor):
            found = True
            left, right = factor.edge_index
            weights = factor.weights.to(torch.float64)
            matrix[left, right] += weights / 2
            matrix[right, left] += weights / 2
    if not found:
        return True, 0.0
    minimum = float(torch.linalg.eigvalsh(matrix).amin().item())
    return minimum >= -1e-10, minimum


def diagnose_model(
    model: Any,
    *,
    replicas: int = 128,
    precision_bytes: int = 4,
) -> ModelDoctorReport:
    """Return a deterministic pre-solve report without executing a solver."""
    if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
        raise ValueError("replicas must be a positive integer.")
    if precision_bytes not in {2, 4, 8}:
        raise ValueError("precision_bytes must be 2, 4, or 8.")
    ir = _as_ir(model)
    capabilities = inspect_capabilities(ir)
    diagnostics: list[ModelDiagnostic] = []

    try:
        from qqa.model.presolve import PresolveInfeasibleError, presolve_model

        presolve = presolve_model(ir)
    except PresolveInfeasibleError as exc:
        presolve = None
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.ERROR,
                "CONTRADICTORY_CONSTRAINT",
                str(exc),
                "Inspect the reported constant row or conflicting bound before solving.",
            )
        )
    if presolve is not None and presolve.report.removed_duplicate_constraints:
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.INFO,
                "REDUNDANT_CONSTRAINT",
                f"Presolve can remove {presolve.report.removed_duplicate_constraints} duplicate row(s).",
            )
        )

    if capabilities.missing_bounds:
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.ERROR,
                "BOUND_MISSING",
                "Pure QQA requires finite bounds for: " + ", ".join(capabilities.missing_bounds),
                "Derive valid bounds in presolve, provide them explicitly, or use an exact route that supports infinity.",
            )
        )
    if capabilities.unsupported_qqa:
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.WARNING,
                "QQA_FACTOR_UNSUPPORTED",
                "Some represented factors have no valid pure-QQA derivative: "
                + ", ".join(capabilities.unsupported_qqa),
                "Select CP/SAT/exact propagation for these factors or reformulate them explicitly.",
            )
        )
    if capabilities.unsupported_exact:
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.WARNING,
                "PROOF_UNAVAILABLE",
                "At least one factor has no proof-safe exact encoding.",
                "A result can be a heuristic incumbent but must not be labelled globally optimal.",
            )
        )

    tensors = _coefficient_tensors(ir)
    nonzero = (
        torch.cat([item.abs()[item != 0] for item in tensors if bool((item != 0).any())])
        if any(bool((item != 0).any()) for item in tensors)
        else torch.ones(1)
    )
    dynamic_range = float(nonzero.max().item() / max(nonzero.min().item(), 1e-300))
    if dynamic_range > 1e8:
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.WARNING,
                "POOR_SCALING",
                f"Coefficient dynamic range is {dynamic_range:.3e}.",
                "Apply unit-aware row/objective scaling and verify the result in high precision.",
            )
        )
    maximum = float(nonzero.max().item())
    if maximum > 1e8:
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.WARNING,
                "LARGE_COEFFICIENT",
                f"Largest absolute coefficient is {maximum:.3e}; inspect big-M formulations.",
                "Derive the smallest valid M or replace it with an indicator constraint.",
            )
        )

    convex, minimum_eigenvalue = _quadratic_curvature(ir)
    if convex is False and minimum_eigenvalue is not None:
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.INFO,
                "NONCONVEX_OBJECTIVE",
                f"Quadratic objective has estimated minimum eigenvalue {minimum_eigenvalue:.3e}.",
                "Use curvature-aware c=2 convexification for exploration and retain an exact/nonconvex proof route.",
            )
        )
    if not diagnostics:
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.INFO, "MODEL_READY", "No blocking model issue was detected."
            )
        )

    from qqa.decomposition.planner import detect_decomposition

    decomposition = detect_decomposition(ir)
    if decomposition.method != "monolithic":
        diagnostics.append(
            ModelDiagnostic(
                DiagnosticSeverity.INFO,
                "DECOMPOSABLE",
                f"Detected {len(decomposition.blocks)} block(s) via {decomposition.method}.",
            )
        )

    domains = {VariableDomain(block.domain) for block in ir.variables}
    integral = domains <= {VariableDomain.BINARY, VariableDomain.INTEGER, VariableDomain.SPIN}
    clause = any(record.factor_type == "ClauseFactor" for record in capabilities.factors)
    assignment = any(
        record.factor_type in {"AssignmentFactor", "AllDifferentFactor", "NoOverlapFactor"}
        for record in capabilities.factors
    )
    routes = [
        SolverCapability(
            "qqa",
            capabilities.qqa_compatible,
            False,
            "Differentiable/prox factors and finite domains are required.",
        ),
        SolverCapability(
            "scip",
            capabilities.exact_compatible,
            capabilities.exact_compatible,
            "General exact encoding when every factor is proof-safe.",
        ),
        SolverCapability(
            "cpsat",
            integral and assignment,
            integral and assignment,
            "Bounded integral scheduling/assignment structure.",
        ),
        SolverCapability(
            "sat-maxsat",
            integral and clause,
            integral and clause,
            "Clause/PB propagation and proof logging.",
        ),
    ]
    relaxed = sum(block.size * int(block.categories or 1) for block in ir.variables)
    estimated = replicas * max(1, relaxed) * precision_bytes * 6
    factor_count = len(capabilities.factors)
    gpu_capable = sum("gpu_kernel" in record.capabilities for record in capabilities.factors)
    expected_kernel = (
        "typed-factor-gpu"
        if factor_count and gpu_capable == factor_count
        else "hybrid-factor"
        if gpu_capable
        else "generic-autograd"
    )
    estimated_transfer = max(1, relaxed) * precision_bytes * 2
    complexity = (
        ir.num_variables
        + 10 * len(ir.constraints)
        + sum(len(item.capabilities) for item in capabilities.factors)
    )
    recommended_budget = float(max(1.0, min(3600.0, math.sqrt(max(1, complexity)))))
    return ModelDoctorReport(
        ir.metadata.name,
        ir.num_variables,
        len(ir.constraints),
        tuple(diagnostics),
        capabilities,
        tuple(routes),
        estimated,
        recommended_budget,
        replicas,
        expected_kernel,
        factor_count,
        estimated_transfer,
        decomposition.method,
    )


__all__ = [
    "DiagnosticSeverity",
    "ModelDiagnostic",
    "ModelDoctorReport",
    "SolverCapability",
    "diagnose_model",
]
