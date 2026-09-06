"""Backend-independent solve result contract.

The legacy backend result classes remain available for compatibility.  New
entry points adapt every backend to :class:`SolveResult`, where mathematical
objective, internal search energy, feasibility, repair, timing, and proof
information have distinct meanings.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch


class SolveStatus(str, Enum):
    """Portable termination states shared by heuristic and exact backends."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE_PROVEN = "infeasible_proven"
    INFEASIBLE = "infeasible_proven"  # backwards-compatible alias
    UNBOUNDED_PROVEN = "unbounded_proven"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    NO_SOLUTION_FOUND = "no_solution_found"
    LIMIT_REACHED_WITH_INCUMBENT = "limit_reached_with_incumbent"
    LIMIT_REACHED_NO_INCUMBENT = "limit_reached_no_incumbent"
    MODEL_INVALID = "model_invalid"
    NUMERICAL_FAILURE = "numerical_failure"
    INTERRUPTED = "interrupted"
    LOCALLY_OPTIMAL = "locally_optimal"
    TIME_LIMIT = "time_limit"
    ITERATION_LIMIT = "iteration_limit"
    ERROR = "error"
    FAILED = "error"  # backwards-compatible alias
    UNKNOWN = "unknown"


class GuaranteeLevel(str, Enum):
    """Strength of the claim made by a result, independent of termination."""

    EXACT = "exact"
    CERTIFIED_BOUND = "certified_bound"
    VERIFIED_FEASIBLE = "verified_feasible"
    APPROXIMATE = "approximate"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"


class FeasibilityStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


class CoordinateSpace(str, Enum):
    """Variable order used by a candidate at a solver boundary."""

    ORIGINAL = "original"
    REDUCED = "reduced"


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """Auditable identity and verification summary for one candidate."""

    candidate_id: str
    origin: str
    coordinate_space: CoordinateSpace | str
    objective_value: float | None
    feasibility: FeasibilityStatus | str
    maximum_violation: float | None = None
    selected: bool = False

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.origin:
            raise ValueError("Candidate id and origin must be non-empty.")
        object.__setattr__(self, "coordinate_space", CoordinateSpace(self.coordinate_space))
        object.__setattr__(self, "feasibility", FeasibilityStatus(self.feasibility))
        for name in ("objective_value", "maximum_violation"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or value < 0 and name == "maximum_violation"
            ):
                raise ValueError(
                    f"Candidate {name} must be finite and non-negative where applicable."
                )


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """One canonical constraint residual and its reporting tolerance."""

    name: str
    raw_residual: float
    scaled_residual: float
    tolerance: float
    satisfied: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ConstraintViolation.name must be non-empty.")
        if not all(
            math.isfinite(value)
            for value in (self.raw_residual, self.scaled_residual, self.tolerance)
        ):
            raise ValueError("Constraint residuals and tolerance must be finite.")
        if self.tolerance < 0:
            raise ValueError("Constraint tolerance must be non-negative.")


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    """Aggregate feasibility diagnostics without hiding individual rows."""

    rows: tuple[ConstraintViolation, ...] = ()
    evaluated: bool = False

    @property
    def feasible(self) -> bool:
        return self.evaluated and all(row.satisfied for row in self.rows)

    @property
    def status(self) -> FeasibilityStatus:
        if not self.evaluated:
            return FeasibilityStatus.UNKNOWN
        return FeasibilityStatus.FEASIBLE if self.feasible else FeasibilityStatus.INFEASIBLE

    @property
    def maximum_violation(self) -> float:
        return max((max(0.0, row.raw_residual) for row in self.rows), default=0.0)

    @property
    def l1_violation(self) -> float:
        return sum(max(0.0, row.scaled_residual) for row in self.rows)

    @property
    def l2_violation(self) -> float:
        return math.sqrt(sum(max(0.0, row.scaled_residual) ** 2 for row in self.rows))

    @classmethod
    def unconstrained(cls) -> ConstraintReport:
        return cls((), evaluated=True)

    @classmethod
    def unknown(cls) -> ConstraintReport:
        return cls((), evaluated=False)


@dataclass(frozen=True, slots=True)
class TimingReport:
    """Wall-clock phase breakdown in seconds."""

    total: float
    compile: float = 0.0
    warmup: float = 0.0
    search: float = 0.0
    repair: float = 0.0
    certification: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.total,
            self.compile,
            self.warmup,
            self.search,
            self.repair,
            self.certification,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Timing values must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class ResourceReport:
    """Portable resource metrics; machine names and filesystem paths are excluded."""

    device: str
    precision: str = "fp32"
    peak_device_memory_bytes: int | None = None
    peak_host_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for value in (self.peak_device_memory_bytes, self.peak_host_memory_bytes):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError("Memory measurements must be non-negative integers or None.")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Reproducibility fields that are safe to serialise and publish."""

    backend: str
    seed: int
    profile: str
    config: dict[str, Any] = field(default_factory=dict)
    transformations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class CertificateMetadata:
    """Portable pointer to a proof/certificate without embedding machine paths."""

    proof_system: str
    status: str
    verifier: str | None = None
    sha256: str | None = None
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        if not self.proof_system or not self.status:
            raise ValueError("Certificate proof_system and status must be non-empty.")
        if self.sha256 is not None and (
            len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256)
        ):
            raise ValueError("Certificate sha256 must be a lowercase hexadecimal digest.")
        if self.candidate_id is not None and not self.candidate_id:
            raise ValueError("Certificate candidate_id must be non-empty or None.")


@dataclass(slots=True)
class SolveResult:
    """One unambiguous result contract for all QQA4CO solve routes.

    ``objective_value`` is always the original mathematical objective.
    ``internal_energy`` is the canonical minimisation quantity used by the
    search backend.  ``merit_value`` may additionally contain feasibility or
    augmented-Lagrangian terms.  Repair never overwrites ``raw_solution``.
    """

    status: SolveStatus
    raw_solution: torch.Tensor | None
    objective_value: float | None
    internal_energy: float | None
    merit_value: float | None
    feasible: bool
    violations: ConstraintReport
    timings: TimingReport
    resources: ResourceReport
    provenance: Provenance
    guarantee_level: GuaranteeLevel | None = None
    plan: Any = None
    repaired_solution: torch.Tensor | None = None
    repaired_objective_value: float | None = None
    selected_candidate_id: str | None = None
    candidates: tuple[CandidateRecord, ...] = ()
    best_bound: float | None = None
    relative_gap: float | None = None
    proven_optimal: bool = False
    population: torch.Tensor | None = None
    archive: Any = None
    events: tuple[Any, ...] = ()
    certificate: CertificateMetadata | None = None
    score: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    legacy_result: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.status = SolveStatus(self.status)
        if self.guarantee_level is None:
            if self.status in {
                SolveStatus.OPTIMAL,
                SolveStatus.INFEASIBLE_PROVEN,
                SolveStatus.UNBOUNDED_PROVEN,
            }:
                self.guarantee_level = GuaranteeLevel.EXACT
            elif self.best_bound is not None:
                self.guarantee_level = GuaranteeLevel.CERTIFIED_BOUND
            elif self.violations.status is FeasibilityStatus.FEASIBLE:
                self.guarantee_level = GuaranteeLevel.VERIFIED_FEASIBLE
            elif self.raw_solution is not None:
                self.guarantee_level = GuaranteeLevel.HEURISTIC
            else:
                self.guarantee_level = GuaranteeLevel.UNKNOWN
        else:
            self.guarantee_level = GuaranteeLevel(self.guarantee_level)
        for name in ("objective_value", "internal_energy", "merit_value"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if self.raw_solution is None and any(
            value is not None
            for value in (self.objective_value, self.internal_energy, self.merit_value)
        ):
            raise ValueError("Objective/energy/merit require a raw solution.")
        if self.repaired_solution is None and self.repaired_objective_value is not None:
            raise ValueError("repaired_objective_value requires repaired_solution.")
        if self.repaired_solution is not None and self.repaired_objective_value is None:
            raise ValueError("repaired_solution requires repaired_objective_value.")
        if self.selected_candidate_id is None:
            self.selected_candidate_id = "repaired" if self.repaired_solution is not None else "raw"
        if self.selected_candidate_id not in {"raw", "repaired"}:
            raise ValueError("selected_candidate_id must be 'raw' or 'repaired'.")
        if self.selected_candidate_id == "repaired" and self.repaired_solution is None:
            raise ValueError("A repaired selected candidate requires repaired_solution.")
        if self.relative_gap is not None and (
            not math.isfinite(self.relative_gap) or self.relative_gap < 0
        ):
            raise ValueError("relative_gap must be finite and non-negative or None.")
        if bool(self.feasible) != self.violations.feasible:
            raise ValueError("feasible must agree with violations.feasible.")
        if self.proven_optimal and self.status is not SolveStatus.OPTIMAL:
            raise ValueError("proven_optimal requires status='optimal'.")
        if self.proven_optimal and self.guarantee_level is not GuaranteeLevel.EXACT:
            raise ValueError("proven_optimal requires guarantee_level='exact'.")
        if self.guarantee_level is GuaranteeLevel.EXACT and self.status not in {
            SolveStatus.OPTIMAL,
            SolveStatus.INFEASIBLE_PROVEN,
            SolveStatus.UNBOUNDED_PROVEN,
        }:
            raise ValueError("An exact guarantee requires a proven terminal status.")
        self.events = tuple(self.events)
        self.candidates = tuple(self.candidates)
        if self.candidates:
            selected = [item.candidate_id for item in self.candidates if item.selected]
            if selected != [self.selected_candidate_id]:
                raise ValueError("Candidate records must identify exactly the selected candidate.")

    @property
    def solution(self) -> torch.Tensor | None:
        """Preferred reported solution, using repair when available."""
        if self.selected_candidate_id == "repaired":
            return self.repaired_solution
        return self.raw_solution

    @property
    def best_sol(self) -> torch.Tensor | None:
        """Compatibility alias for :attr:`solution`."""
        return self.solution

    @property
    def best_obj(self) -> float | None:
        """Compatibility alias for the original mathematical objective."""
        if self.selected_candidate_id == "repaired":
            return self.repaired_objective_value
        return self.objective_value

    @property
    def runtime(self) -> float:
        """Compatibility alias for total wall-clock time."""
        return self.timings.total

    @property
    def history(self) -> dict[str, Any]:
        """Compatibility view of backend history for existing visualisations."""
        history = getattr(self.legacy_result, "history", None)
        return history if isinstance(history, dict) else {}

    def to_dict(self, *, include_solutions: bool = False) -> dict[str, Any]:
        """Return a JSON-oriented, environment-neutral representation."""
        guarantee_level = self.guarantee_level
        if guarantee_level is None:  # Defensive guard for bypassed dataclass initialisation.
            raise RuntimeError("SolveResult guarantee level has not been initialised.")
        payload: dict[str, Any] = {
            "status": self.status.value,
            "guarantee_level": guarantee_level.value,
            "objective_value": self.objective_value,
            "repaired_objective_value": self.repaired_objective_value,
            "selected_candidate_id": self.selected_candidate_id,
            "candidates": [
                {
                    **asdict(candidate),
                    "coordinate_space": CoordinateSpace(candidate.coordinate_space).value,
                    "feasibility": FeasibilityStatus(candidate.feasibility).value,
                }
                for candidate in self.candidates
            ],
            "internal_energy": self.internal_energy,
            "merit_value": self.merit_value,
            "feasible": self.feasible,
            "violations": {
                "rows": [asdict(row) for row in self.violations.rows],
                "evaluated": self.violations.evaluated,
                "status": self.violations.status.value,
                "maximum": self.violations.maximum_violation,
                "l1": self.violations.l1_violation,
                "l2": self.violations.l2_violation,
            },
            "best_bound": self.best_bound,
            "relative_gap": self.relative_gap,
            "proven_optimal": self.proven_optimal,
            "certificate": None if self.certificate is None else asdict(self.certificate),
            "timings": asdict(self.timings),
            "resources": asdict(self.resources),
            "provenance": asdict(self.provenance),
            "score": self.score,
            "diagnostics": self.diagnostics,
            "events": [
                event.to_dict() if callable(getattr(event, "to_dict", None)) else event
                for event in self.events
            ],
        }
        if include_solutions:
            payload["raw_solution"] = (
                None if self.raw_solution is None else self.raw_solution.detach().cpu().tolist()
            )
            payload["repaired_solution"] = (
                None
                if self.repaired_solution is None
                else self.repaired_solution.detach().cpu().tolist()
            )
        return payload


__all__ = [
    "ConstraintReport",
    "ConstraintViolation",
    "CertificateMetadata",
    "CandidateRecord",
    "CoordinateSpace",
    "FeasibilityStatus",
    "GuaranteeLevel",
    "Provenance",
    "ResourceReport",
    "SolveResult",
    "SolveStatus",
    "TimingReport",
]
