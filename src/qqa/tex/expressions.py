"""A small, non-evaluating expression interpreter for LLM-generated models."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping

import torch

from qqa.mixed.variables import VariableSpec

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": torch.abs,
    "square": torch.square,
    "sqrt": torch.sqrt,
    "exp": torch.exp,
    "log": torch.log,
    "sin": torch.sin,
    "cos": torch.cos,
    "tanh": torch.tanh,
    "minimum": torch.minimum,
    "maximum": torch.maximum,
}


class UnsafeExpressionError(ValueError):
    """Raised when a model expression is outside the safe grammar."""


def _validate_tree(tree: ast.AST, variables: Mapping[str, VariableSpec]) -> None:
    nodes = list(ast.walk(tree))
    if len(nodes) > 256:
        raise UnsafeExpressionError("Expression is too complex (maximum 256 AST nodes).")
    for node in nodes:
        if isinstance(
            node,
            (
                ast.Expression,
                ast.Load,
                ast.Constant,
                ast.Name,
                ast.Subscript,
                ast.BinOp,
                ast.UnaryOp,
                ast.Call,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.Pow,
                ast.UAdd,
                ast.USub,
            ),
        ):
            continue
        raise UnsafeExpressionError(f"Unsupported syntax: {type(node).__name__}.")

    for node in nodes:
        if isinstance(node, ast.Name) and not (
            node.id in variables or node.id in _FUNCTIONS or node.id == "sum"
        ):
            raise UnsafeExpressionError(f"Unknown name {node.id!r}.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsafeExpressionError("Attribute and indirect calls are forbidden.")
            name = node.func.id
            if name not in _FUNCTIONS and name != "sum":
                raise UnsafeExpressionError(f"Function {name!r} is not allowed.")
            if node.keywords:
                raise UnsafeExpressionError("Keyword arguments are not allowed.")
            expected = 1 if name not in {"minimum", "maximum"} else 2
            if len(node.args) != expected:
                raise UnsafeExpressionError(f"{name}() requires {expected} argument(s).")
        if isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name) or node.value.id not in variables:
                raise UnsafeExpressionError("Only direct variable indexing is allowed.")
            if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
                raise UnsafeExpressionError("Variable indices must be integer literals.")
            variable = variables[node.value.id]
            if not 0 <= node.slice.value < variable.size:
                raise UnsafeExpressionError(
                    f"Index {node.slice.value} is outside {node.value.id}[0:{variable.size}]."
                )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(
                node.right.value, (int, float)
            ):
                raise UnsafeExpressionError("Exponents must be numeric literals.")
            if not math.isfinite(float(node.right.value)) or abs(float(node.right.value)) > 8:
                raise UnsafeExpressionError("Exponent magnitude must be <= 8.")
        if isinstance(node, ast.Constant) and (
            not isinstance(node.value, (int, float))
            or isinstance(node.value, bool)
            or not math.isfinite(float(node.value))
        ):
            raise UnsafeExpressionError("Only finite numeric constants are allowed.")


def compile_expression(
    source: str,
    variables: Mapping[str, VariableSpec],
) -> Callable[[Mapping[str, torch.Tensor]], torch.Tensor]:
    """Compile a validated expression without ``eval`` or ``exec``."""
    if not isinstance(source, str) or not source.strip():
        raise UnsafeExpressionError("Expression must be a non-empty string.")
    if len(source) > 2000:
        raise UnsafeExpressionError("Expression is too long (maximum 2000 characters).")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid expression syntax: {exc.msg}.") from exc
    _validate_tree(tree, variables)

    def evaluate(named: Mapping[str, torch.Tensor]) -> torch.Tensor:
        def visit(node: ast.AST):
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id not in named:
                    raise UnsafeExpressionError(f"Variable {node.id!r} is unavailable.")
                return named[node.id]
            if isinstance(node, ast.Subscript):
                value = named[node.value.id]
                return value if value.ndim == 1 else value[..., node.slice.value]
            if isinstance(node, ast.UnaryOp):
                return _UNARY[type(node.op)](visit(node.operand))
            if isinstance(node, ast.BinOp):
                return _BINARY[type(node.op)](visit(node.left), visit(node.right))
            if isinstance(node, ast.Call):
                name = node.func.id
                args = [visit(argument) for argument in node.args]
                if name == "sum":
                    value = args[0]
                    return value.sum(dim=-1) if torch.is_tensor(value) and value.ndim > 1 else value
                tensor_args = []
                like = next((value for value in named.values() if torch.is_tensor(value)), None)
                for value in args:
                    tensor_args.append(
                        torch.as_tensor(value, device=like.device, dtype=like.dtype)
                        if not torch.is_tensor(value)
                        else value
                    )
                return _FUNCTIONS[name](*tensor_args)
            raise UnsafeExpressionError(f"Unsupported expression node {type(node).__name__}.")

        result = visit(tree)
        if torch.is_tensor(result):
            return result
        first = next(iter(named.values()))
        return first.new_full((first.shape[0],), float(result))

    return evaluate


__all__ = ["UnsafeExpressionError", "compile_expression"]
