"""Natural-language planning and the unified QQA execution entry point."""

from qqa.natural_language.planner import OptimizationPlan, compile_natural_language, plan_spec
from qqa.natural_language.prompts import MODEL_SYSTEM_PROMPT
from qqa.natural_language.runner import AskResult, ask, blackbox_from_spec, execute_plan

__all__ = [
    "AskResult",
    "MODEL_SYSTEM_PROMPT",
    "OptimizationPlan",
    "ask",
    "blackbox_from_spec",
    "compile_natural_language",
    "execute_plan",
    "plan_spec",
]
