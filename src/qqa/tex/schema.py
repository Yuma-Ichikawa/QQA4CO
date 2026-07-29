"""Strict JSON model schema shared by the API client and local compiler."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from numbers import Real as RealNumber
from typing import Any

import torch

from qqa.mixed.variables import Binary, Integer, Real, VariableSpace, VariableSpec
from qqa.tex.expressions import compile_expression

MAX_VARIABLE_DECLARATIONS = 512
MAX_TOTAL_DIMENSION = 65_536
MAX_OBJECTIVES = 32
MAX_CONSTRAINTS = 4_096
MAX_NAME_LENGTH = 256
MAX_EXPRESSION_LENGTH = 2_000
MAX_TOTAL_EXPRESSION_CHARACTERS = 1_000_000
MAX_NOTES_LENGTH = 50_000
MAX_MODEL_JSON_LENGTH = 4_000_000
_PREFLIGHT_SAMPLES = 5


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, RealNumber):
        raise ValueError(f"{label} must be a JSON real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _text(
    value: Any,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not allow_empty and not value.strip():
        raise ValueError(f"{label} must not be empty.")
    if len(value) > maximum:
        raise ValueError(f"{label} is too long (maximum {maximum:,} characters).")
    return value


@dataclass(frozen=True, slots=True)
class VariableDeclaration:
    name: str
    kind: str
    lower: float
    upper: float
    size: int

    @classmethod
    def from_dict(cls, value: dict) -> VariableDeclaration:
        _exact_keys(value, {"name", "kind", "lower", "upper", "size"}, "variable")
        declaration = cls(
            name=_text(value["name"], "Variable name", maximum=MAX_NAME_LENGTH),
            kind=value["kind"],
            lower=_finite(value["lower"], "variable lower"),
            upper=_finite(value["upper"], "variable upper"),
            size=value["size"],
        )
        declaration.to_variable()
        return declaration

    def to_variable(self) -> VariableSpec:
        if self.kind == "binary":
            if self.lower != 0 or self.upper != 1:
                raise ValueError("Binary variable bounds must be exactly [0, 1].")
            return Binary(self.name, size=self.size)
        if self.kind == "integer":
            if not self.lower.is_integer() or not self.upper.is_integer():
                raise ValueError("Integer variable bounds must be integral.")
            return Integer(self.name, int(self.lower), int(self.upper), size=self.size)
        if self.kind == "real":
            return Real(self.name, self.lower, self.upper, size=self.size)
        raise ValueError(f"Unknown variable kind {self.kind!r}.")


@dataclass(frozen=True, slots=True)
class ObjectiveDeclaration:
    name: str
    direction: str
    expression: str
    unit: str

    @classmethod
    def from_dict(cls, value: dict) -> ObjectiveDeclaration:
        _exact_keys(value, {"name", "direction", "expression", "unit"}, "objective")
        if value["direction"] not in ("min", "max"):
            raise ValueError("Objective direction must be 'min' or 'max'.")
        return cls(
            name=_text(value["name"], "Objective name", maximum=MAX_NAME_LENGTH),
            direction=value["direction"],
            expression=_text(
                value["expression"],
                "Objective expression",
                maximum=MAX_EXPRESSION_LENGTH,
            ),
            unit=_text(
                value["unit"],
                "Objective unit",
                maximum=MAX_NAME_LENGTH,
                allow_empty=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class ConstraintDeclaration:
    name: str
    expression: str
    sense: str
    rhs: float
    weight: float
    scale: float
    tolerance: float

    @classmethod
    def from_dict(cls, value: dict) -> ConstraintDeclaration:
        _exact_keys(
            value,
            {"name", "expression", "sense", "rhs", "weight", "scale", "tolerance"},
            "constraint",
        )
        if value["sense"] not in ("<=", ">=", "=="):
            raise ValueError("Constraint sense must be '<=', '>=', or '=='.")
        numbers = {
            name: _finite(value[name], f"constraint {name}")
            for name in ("rhs", "weight", "scale", "tolerance")
        }
        if numbers["weight"] <= 0 or numbers["weight"] > 1e12:
            raise ValueError("Constraint weight must be in (0, 1e12].")
        if numbers["scale"] <= 0:
            raise ValueError("Constraint scale must be > 0.")
        if numbers["tolerance"] < 0:
            raise ValueError("Constraint tolerance must be >= 0.")
        return cls(
            name=_text(value["name"], "Constraint name", maximum=MAX_NAME_LENGTH),
            expression=_text(
                value["expression"],
                "Constraint expression",
                maximum=MAX_EXPRESSION_LENGTH,
            ),
            sense=value["sense"],
            **numbers,
        )


def _exact_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise ValueError(f"Invalid {label} keys; missing={sorted(missing)}, extra={sorted(extra)}.")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Validated, JSON-serialisable optimisation model."""

    name: str
    variables: tuple[VariableDeclaration, ...]
    objectives: tuple[ObjectiveDeclaration, ...]
    constraints: tuple[ConstraintDeclaration, ...]
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict) -> ModelSpec:
        _exact_keys(value, {"name", "variables", "objectives", "constraints", "notes"}, "model")
        model_name = _text(value["name"], "Model name", maximum=MAX_NAME_LENGTH)
        notes = _text(
            value["notes"],
            "Model notes",
            maximum=MAX_NOTES_LENGTH,
            allow_empty=True,
        )
        if not isinstance(value["variables"], list) or not value["variables"]:
            raise ValueError("Model variables must be a non-empty list.")
        if not isinstance(value["objectives"], list) or not value["objectives"]:
            raise ValueError("Model objectives must be a non-empty list.")
        if not isinstance(value["constraints"], list):
            raise TypeError("Model constraints must be a list.")
        if len(value["variables"]) > MAX_VARIABLE_DECLARATIONS:
            raise ValueError(
                f"At most {MAX_VARIABLE_DECLARATIONS} variable declarations are allowed."
            )
        if len(value["objectives"]) > MAX_OBJECTIVES:
            raise ValueError(f"At most {MAX_OBJECTIVES} objectives are allowed.")
        if len(value["constraints"]) > MAX_CONSTRAINTS:
            raise ValueError(f"At most {MAX_CONSTRAINTS} constraints are allowed.")
        variables = tuple(VariableDeclaration.from_dict(item) for item in value["variables"])
        total_dimension = sum(item.size for item in variables)
        if total_dimension > MAX_TOTAL_DIMENSION:
            raise ValueError(
                "Model total variable dimension must be at most "
                f"{MAX_TOTAL_DIMENSION:,}, got {total_dimension:,}."
            )
        objectives = tuple(ObjectiveDeclaration.from_dict(item) for item in value["objectives"])
        constraints = tuple(ConstraintDeclaration.from_dict(item) for item in value["constraints"])
        total_expression_characters = sum(len(item.expression) for item in objectives) + sum(
            len(item.expression) for item in constraints
        )
        if total_expression_characters > MAX_TOTAL_EXPRESSION_CHARACTERS:
            raise ValueError(
                "Model expressions contain too much text; the combined limit is "
                f"{MAX_TOTAL_EXPRESSION_CHARACTERS:,} characters."
            )
        variable_names = [item.name for item in variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("Variable names must be unique.")
        objective_names = [item.name for item in objectives]
        if len(objective_names) != len(set(objective_names)):
            raise ValueError("Objective names must be unique.")
        constraint_names = [item.name for item in constraints]
        if len(constraint_names) != len(set(constraint_names)):
            raise ValueError("Constraint names must be unique.")
        model = cls(model_name, variables, objectives, constraints, notes)
        model.validate_semantics()
        return model

    @classmethod
    def from_json(cls, source: str) -> ModelSpec:
        if not isinstance(source, str):
            raise TypeError("Model response must be a JSON string.")
        if len(source) > MAX_MODEL_JSON_LENGTH:
            raise ValueError(
                f"Model response is too large (maximum {MAX_MODEL_JSON_LENGTH:,} characters)."
            )
        try:
            value = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model response is not valid JSON: {exc.msg}.") from exc
        return cls.from_dict(value)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "variables": [asdict(value) for value in self.variables],
            "objectives": [asdict(value) for value in self.objectives],
            "constraints": [asdict(value) for value in self.constraints],
            "notes": self.notes,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @property
    def variable_specs(self) -> tuple[VariableSpec, ...]:
        return tuple(value.to_variable() for value in self.variables)

    def validate_semantics(self) -> None:
        """Preflight every expression on a small deterministic domain sample.

        The safe AST validator prevents code execution, but syntax alone cannot
        establish the numerical contract required by QQA. Objectives and
        constraints must return exactly one finite scalar per candidate. This
        check intentionally includes declared bounds, because the relaxation
        and final projection may evaluate expressions there.
        """

        variable_specs = self.variable_specs
        variable_map = {variable.name: variable for variable in variable_specs}
        space = VariableSpace(variable_specs)
        dimension = space.dimension

        # Three common boundary/interior points plus two deterministic,
        # column-varying interiors catch cross-variable singularities without
        # allocating a solver-sized population or relying on global RNG state.
        fractions = torch.empty((_PREFLIGHT_SAMPLES, dimension), dtype=torch.float64)
        fractions[0].zero_()
        fractions[1].fill_(1.0)
        fractions[2].fill_(0.5)
        columns = torch.arange(dimension, dtype=torch.int64)
        fractions[3] = ((columns * 37 + 11) % 101).to(torch.float64).add_(0.5).div_(101)
        fractions[4] = ((columns * 53 + 17) % 103).to(torch.float64).add_(0.5).div_(103)
        points = space.project(fractions)
        named = space.unpack(points)

        declarations = (
            *(("objective", item.name, item.expression) for item in self.objectives),
            *(("constraint", item.name, item.expression) for item in self.constraints),
        )
        with torch.no_grad():
            for kind, name, expression in declarations:
                function = compile_expression(expression, variable_map)
                try:
                    result = torch.as_tensor(function(named), dtype=points.dtype)
                except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{kind.capitalize()} {name!r} failed numerical preflight: {exc}"
                    ) from exc
                expected = (_PREFLIGHT_SAMPLES,)
                if tuple(result.shape) != expected:
                    raise ValueError(
                        f"{kind.capitalize()} {name!r} must return one scalar per candidate "
                        f"with shape {expected}, got {tuple(result.shape)}."
                    )
                if not torch.isfinite(result).all():
                    bad_samples = (~torch.isfinite(result)).nonzero(as_tuple=False).flatten()
                    first_bad = int(bad_samples[0].item())
                    raise ValueError(
                        f"{kind.capitalize()} {name!r} returned NaN or infinity during "
                        f"numerical preflight at sample {first_bad}. Check expression domains "
                        "over all declared variable bounds."
                    )


MODEL_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "variables", "objectives", "constraints", "notes"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": MAX_NAME_LENGTH},
        "variables": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_VARIABLE_DECLARATIONS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "kind", "lower", "upper", "size"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_NAME_LENGTH,
                    },
                    "kind": {"type": "string", "enum": ["binary", "integer", "real"]},
                    "lower": {"type": "number"},
                    "upper": {"type": "number"},
                    "size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOTAL_DIMENSION,
                    },
                },
            },
        },
        "objectives": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_OBJECTIVES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "direction", "expression", "unit"],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_NAME_LENGTH,
                    },
                    "direction": {"type": "string", "enum": ["min", "max"]},
                    "expression": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_EXPRESSION_LENGTH,
                    },
                    "unit": {"type": "string", "maxLength": MAX_NAME_LENGTH},
                },
            },
        },
        "constraints": {
            "type": "array",
            "maxItems": MAX_CONSTRAINTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name",
                    "expression",
                    "sense",
                    "rhs",
                    "weight",
                    "scale",
                    "tolerance",
                ],
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_NAME_LENGTH,
                    },
                    "expression": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_EXPRESSION_LENGTH,
                    },
                    "sense": {"type": "string", "enum": ["<=", ">=", "=="]},
                    "rhs": {"type": "number"},
                    "weight": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1e12,
                    },
                    "scale": {"type": "number", "exclusiveMinimum": 0},
                    "tolerance": {"type": "number", "minimum": 0},
                },
            },
        },
        "notes": {"type": "string", "maxLength": MAX_NOTES_LENGTH},
    },
}


__all__ = [
    "ConstraintDeclaration",
    "MAX_CONSTRAINTS",
    "MAX_EXPRESSION_LENGTH",
    "MAX_MODEL_JSON_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_NOTES_LENGTH",
    "MAX_OBJECTIVES",
    "MAX_TOTAL_EXPRESSION_CHARACTERS",
    "MAX_TOTAL_DIMENSION",
    "MAX_VARIABLE_DECLARATIONS",
    "MODEL_JSON_SCHEMA",
    "ModelSpec",
    "ObjectiveDeclaration",
    "VariableDeclaration",
]
