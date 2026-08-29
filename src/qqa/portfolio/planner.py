"""Rule-based, explainable QQA-centred portfolio planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qqa.config import SolverConfig
from qqa.portfolio.inspector import ModelInspection, inspect_model


@dataclass(frozen=True, slots=True)
class PlanStage:
    """One node in the explainable solve-plan DAG."""

    name: str
    engine: str
    role: str
    depends_on: tuple[str, ...] = ()
    budget_fraction: float = 0.0
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.engine or not self.role:
            raise ValueError("Plan stage name, engine, and role must be non-empty.")
        if not 0.0 <= self.budget_fraction <= 1.0:
            raise ValueError("Plan stage budget_fraction must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class SolverPlan:
    model: ModelInspection
    profile: str
    primary_engine: str
    refinements: tuple[str, ...]
    exact_backend: str | None
    replicas: int
    estimated_memory_bytes: int
    reasons: tuple[str, ...]
    fallbacks: tuple[str, ...] = ()
    stages: tuple[PlanStage, ...] = ()

    def stage(self, name: str) -> PlanStage | None:
        return next((item for item in self.stages if item.name == name), None)

    def explain(self) -> str:
        """Return a concise, deterministic plan suitable for CLI/UI preview."""
        domain_text = ", ".join(
            f"{count} {domain}" for domain, count in sorted(self.model.domains.items())
        )
        lines = [
            f"Model: {self.model.name} ({domain_text}; {self.model.num_constraints} constraints)",
            f"Detected: {', '.join(self.model.structure) or 'generic factor model'}",
            f"Primary engine: {self.primary_engine}",
            f"Refinement: {', '.join(self.refinements) or 'none'}",
            f"Certification: {self.exact_backend or 'disabled'}",
            f"Chosen replicas: {self.replicas}",
            f"Estimated working memory: {self.estimated_memory_bytes} bytes",
        ]
        lines.extend(f"Reason: {reason}" for reason in self.reasons)
        if self.fallbacks:
            lines.append(f"Fallbacks: {', '.join(self.fallbacks)}")
        if self.stages:
            lines.append("Execution DAG:")
            lines.extend(
                f"  {stage.name}: {stage.engine} ({stage.role}, "
                f"budget={stage.budget_fraction:.0%}, "
                f"after={','.join(stage.depends_on) or 'root'})"
                for stage in self.stages
            )
        return "\n".join(lines)


def _available_device_memory(device: str) -> int | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        free, _ = torch.cuda.mem_get_info(torch.device(device))
    except (RuntimeError, ValueError):
        return None
    return int(free)


def build_plan(model: Any, config: SolverConfig) -> SolverPlan:
    """Select a transparent route and VRAM-safe population size."""
    inspection = inspect_model(model)
    resolved = config.resolved()
    requested = int(resolved.replicas or 1)
    # AdamW keeps latent, gradient, and two moments. Include projected state,
    # factor storage, telemetry, and a conservative 25% workspace margin.
    bytes_per_replica = max(1, inspection.relaxed_variables) * 6 * 4
    factor_bytes = max(1, inspection.nonzeros) * 3 * 8
    available = _available_device_memory(resolved.device)
    if available is not None:
        budget = max(1, int(available * resolved.memory_fraction) - factor_bytes)
        replicas = max(1, min(requested, budget // max(1, bytes_per_replica)))
    else:
        replicas = requested
    estimated = factor_bytes + replicas * bytes_per_replica

    structure = set(inspection.structure)
    reasons: list[str] = []
    fallbacks: list[str] = []
    exact: str | None = None
    primary: str
    refinements: tuple[str, ...]
    if resolved.backend != "qqa":
        primary = resolved.backend
        refinements = ("domain-local-search",)
        reasons.append("A baseline backend was explicitly requested; the default remains QQA.")
    elif "sparse-qubo" in structure:
        primary = "sparse-factor-qqa"
        refinements = ("incremental-one-flip", "diverse-elite-archive")
        reasons.append("Sparse pairwise factors avoid dense N×N objective evaluation.")
    elif "assignment" in structure:
        primary = "permutation-qqa"
        refinements = ("hungarian-projection", "pair-swap-local-search")
        reasons.append("Assignment structure benefits from projection before refinement.")
    elif "mixed-domain" in structure:
        primary = "mixed-factor-qqa"
        refinements = ("scaled-augmented-lagrangian", "constraint-repair")
        reasons.append("Mixed domains require canonical scaling and feasibility-first ranking.")
    else:
        primary = "factor-qqa"
        refinements = ("domain-local-search",)
        reasons.append(
            "No stronger specialised structure was detected; QQA remains the primal engine."
        )
    if inspection.connected_components > 1:
        refinements = ("component-decomposition", *refinements)
        reasons.append(
            f"The factor graph has {inspection.connected_components} independent components."
        )
    if "linear" in structure and resolved.backend == "qqa":
        refinements = ("gpu-pdhg-relaxation", *refinements)
        reasons.append("A linear relaxation can provide an LP warm state and dual bound.")
    if "clause" in structure:
        fallbacks.append("SAT/MaxSAT propagation")
    if inspection.missing_bounds:
        reasons.append(
            "Pure QQA is unavailable until finite bounds are supplied for: "
            + ", ".join(inspection.missing_bounds)
            + "."
        )
    if inspection.unsupported_qqa:
        reasons.append("The canonical model contains factors without a declared QQA derivative.")
    if resolved.require_certificate or resolved.exact_backend not in {"auto", "none"}:
        if resolved.exact_backend == "auto":
            exact = "cpsat" if structure & {"assignment", "clause"} else "scip"
        else:
            exact = resolved.exact_backend
        reasons.append("An exact backend was requested for a bound or certificate.")
    elif resolved.exact_backend == "auto":
        fallbacks.append("optional exact completion")
    if replicas < requested:
        reasons.append(f"Replica count was reduced from {requested} to fit the memory budget.")
    relaxation_fraction = 0.08 if "linear" in structure and resolved.backend == "qqa" else 0.0
    if exact is None:
        qqa_fraction = 0.90 - relaxation_fraction
        refinement_fraction = 0.10
        exact_fraction = 0.0
    else:
        qqa_fraction = 0.25
        if inspection.num_variables >= 1000:
            qqa_fraction += 0.10
        if structure & {"mixed-domain", "constrained", "sparse-qubo"}:
            qqa_fraction += 0.10
        if resolved.profile != "certify" and not resolved.require_certificate:
            qqa_fraction += 0.05
        qqa_fraction = min(0.60, qqa_fraction)
        refinement_fraction = 0.07
        exact_fraction = max(0.10, 1.0 - relaxation_fraction - qqa_fraction - refinement_fraction)
    stages = [PlanStage("compile", "factor-registry", "lowering")]
    previous = "compile"
    if relaxation_fraction:
        stages.append(
            PlanStage(
                "relaxation",
                "gpu-pdhg",
                "bound-and-warm-state",
                ("compile",),
                relaxation_fraction,
                optional=True,
            )
        )
        previous = "relaxation"
    stages.append(
        PlanStage(
            "qqa-primal",
            primary,
            "population-primal-search",
            (previous,),
            qqa_fraction,
        )
    )
    stages.append(
        PlanStage(
            "repair-and-lns",
            "+".join(refinements) or "domain-repair",
            "feasibility-and-incumbent-improvement",
            ("qqa-primal",),
            refinement_fraction,
        )
    )
    if exact is not None:
        stages.append(
            PlanStage(
                "certificate",
                exact,
                "completion-bound-and-proof",
                ("qqa-primal", "repair-and-lns"),
                exact_fraction,
            )
        )
    return SolverPlan(
        inspection,
        resolved.profile,
        primary,
        refinements,
        exact,
        replicas,
        estimated,
        tuple(reasons),
        tuple(fallbacks),
        tuple(stages),
    )


__all__ = ["PlanStage", "SolverPlan", "build_plan"]
