"""Rule-based, explainable QQA-centred portfolio planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from qqa.config import SolverConfig
from qqa.portfolio.inspector import ModelInspection, inspect_model


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
    )


__all__ = ["SolverPlan", "build_plan"]
