"""Compile TeX optimisation models through an OpenAI-compatible API."""

from qqa.tex.client import LLMAPIError, OpenAICompatibleClient
from qqa.tex.compiler import TexSolveResult, compile_tex, problem_from_spec, solve_tex
from qqa.tex.schema import ModelSpec

__all__ = [
    "LLMAPIError",
    "ModelSpec",
    "OpenAICompatibleClient",
    "TexSolveResult",
    "compile_tex",
    "problem_from_spec",
    "solve_tex",
]
