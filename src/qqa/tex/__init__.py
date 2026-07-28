"""Compile TeX optimisation models through an OpenAI-compatible API."""

from qqa.tex.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMAPIError,
    OpenAICompatibleClient,
)
from qqa.tex.compiler import TexSolveResult, compile_tex, problem_from_spec, solve_tex
from qqa.tex.schema import ModelSpec

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "LLMAPIError",
    "ModelSpec",
    "OpenAICompatibleClient",
    "TexSolveResult",
    "compile_tex",
    "problem_from_spec",
    "solve_tex",
]
