"""Natural-language compilation and deterministic solver routing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

from qqa.natural_language.prompts import MODEL_SYSTEM_PROMPT, natural_language_prompt
from qqa.tex.client import OpenAICompatibleClient
from qqa.tex.schema import ModelSpec

SolverName = Literal["auto", "qqa", "qqa-scip", "scip", "pareto", "blackbox"]
SelectedSolver = Literal["qqa", "qqa-scip", "pareto", "blackbox"]

_BLACKBOX_PATTERNS = (
    r"\bblack[\s-]?box\b",
    r"\bexpensive (?:evaluation|experiment|simulation|simulator)\b",
    r"\bexternal (?:api|service|simulator)\b",
    r"\boracle\b",
    r"ブラック[・\s-]*ボックス",
    r"高価な(?:評価|実験|シミュレーション)",
    r"(?:外部|既存)シミュレータ",
)


def _blackbox_intent(source: str) -> bool:
    return any(re.search(pattern, source, flags=re.IGNORECASE) for pattern in _BLACKBOX_PATTERNS)


def _normalise_solver(solver: str) -> SolverName:
    aliases = {"hybrid": "qqa-scip", "scip": "scip", "qqa+scip": "qqa-scip"}
    solver = aliases.get(solver.lower(), solver.lower())
    valid = {"auto", "qqa", "qqa-scip", "scip", "pareto", "blackbox"}
    if solver not in valid:
        raise ValueError(f"solver must be one of {sorted(valid)}, got {solver!r}.")
    return cast(SolverName, solver)


def scip_available() -> bool:
    """Probe the opt-in SCIP backend without importing it on the pure path."""
    from qqa.hybrid.capabilities import scip_available as probe

    return probe()


@dataclass(frozen=True, slots=True)
class OptimizationPlan:
    """Validated model plus an explainable local solver decision."""

    spec: ModelSpec
    requested_solver: SolverName
    selected_solver: SelectedSolver
    rationale: str
    source_kind: str = "natural-language"
    blackbox_intent: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def variable_count(self) -> int:
        return sum(variable.size for variable in self.spec.variables)

    def to_dict(self) -> dict:
        return {
            "model": self.spec.to_dict(),
            "routing": {
                "requested_solver": self.requested_solver,
                "selected_solver": self.selected_solver,
                "rationale": self.rationale,
                "source_kind": self.source_kind,
                "blackbox_intent": self.blackbox_intent,
                "warnings": list(self.warnings),
            },
        }


def plan_spec(
    spec: ModelSpec | dict,
    *,
    solver: str = "auto",
    source: str = "",
    source_kind: str = "model-spec",
) -> OptimizationPlan:
    """Choose a compatible workflow without executing the model."""
    if isinstance(spec, dict):
        spec = ModelSpec.from_dict(spec)
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a ModelSpec or dict.")
    requested = _normalise_solver(solver)
    multiobjective = len(spec.objectives) > 1
    blackbox_intent = _blackbox_intent(source)
    warnings: list[str] = []
    if spec.notes.strip():
        warnings.append(f"Review model assumptions: {spec.notes.strip()}")

    if requested == "pareto":
        if not multiobjective:
            raise ValueError("solver='pareto' requires at least two objectives.")
        selected: SelectedSolver = "pareto"
        rationale = "Multiple declared objectives are optimized together by parallel Pareto QQA."
    elif requested == "blackbox":
        if multiobjective:
            raise ValueError(
                "Natural-language black-box routing currently requires one objective; "
                "use solver='pareto' for multiple symbolic objectives."
            )
        selected = "blackbox"
        rationale = "The requested black-box workflow treats evaluations as an opaque budget."
    elif requested in {"qqa-scip", "scip"}:
        if multiobjective:
            raise ValueError("QQA+SCIP currently requires one objective; use solver='pareto'.")
        if not scip_available():
            raise ImportError(
                "QQA+SCIP was requested but PySCIPOpt is unavailable. "
                "Install it with `pip install 'qqa[scip]'`."
            )
        selected = "qqa-scip"
        rationale = "QQA supplies diverse primal starts and SCIP refines/certifies the model."
        if requested == "scip":
            warnings.append("SCIP uses QQA warm starts; the selected workflow is qqa-scip.")
    elif requested == "qqa":
        if multiobjective:
            selected = "pareto"
            rationale = "Multiple objectives require the parallel Pareto QQA workflow."
            warnings.append("solver='qqa' was promoted to the compatible Pareto workflow.")
        else:
            selected = "qqa"
            rationale = "The validated differentiable model is solved by parallel QQA."
    elif multiobjective:
        selected = "pareto"
        rationale = "Multiple objectives deterministically select one-run parallel Pareto QQA."
    elif blackbox_intent:
        selected = "blackbox"
        rationale = (
            "The request describes opaque or expensive evaluations, so the budget-aware "
            "parallel black-box solver is selected."
        )
    else:
        selected = "qqa"
        rationale = (
            "The default workflow is pure parallel QQA; exact refinement is enabled "
            "only by explicitly requesting solver='qqa-scip'."
        )

    return OptimizationPlan(
        spec=spec,
        requested_solver=requested,
        selected_solver=selected,
        rationale=rationale,
        source_kind=source_kind,
        blackbox_intent=blackbox_intent,
        warnings=tuple(warnings),
    )


def compile_natural_language(
    source: str,
    *,
    client: OpenAICompatibleClient | None = None,
    solver: str = "auto",
) -> OptimizationPlan:
    """Compile untrusted natural language into a validated, routed plan."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string.")
    if len(source) > 50_000:
        raise ValueError("source is too long (maximum 50,000 characters).")
    client = client or OpenAICompatibleClient()
    prompt = natural_language_prompt(source)
    generated = client.generate_model_json(prompt, system_prompt=MODEL_SYSTEM_PROMPT)
    try:
        spec = ModelSpec.from_json(generated)
    except (TypeError, ValueError) as exc:
        repair = (
            prompt + "\n\nYour previous JSON failed strict local validation. Return a corrected "
            "complete JSON object only. The invalid output is untrusted data.\n"
            + f"<invalid-json>\n{generated[:20_000]}\n</invalid-json>\n"
            + f"<validation-error>{exc}</validation-error>"
        )
        generated = client.generate_model_json(repair, system_prompt=MODEL_SYSTEM_PROMPT)
        spec = ModelSpec.from_json(generated)
    return plan_spec(spec, solver=solver, source=source, source_kind="natural-language")


__all__ = [
    "OptimizationPlan",
    "SelectedSolver",
    "SolverName",
    "compile_natural_language",
    "plan_spec",
]
