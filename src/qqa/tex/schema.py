"""Strict JSON model schema shared by the API client and local compiler."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from qqa.mixed.variables import Binary, Integer, Real, VariableSpec
from qqa.tex.expressions import compile_expression


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a real number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


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
            name=value["name"],
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
        if not isinstance(value["name"], str) or not value["name"]:
            raise ValueError("Objective name must not be empty.")
        if not isinstance(value["unit"], str):
            raise TypeError("Objective unit must be a string.")
        return cls(**value)


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
        if not isinstance(value["name"], str) or not value["name"]:
            raise ValueError("Constraint name must not be empty.")
        return cls(
            name=value["name"],
            expression=value["expression"],
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
        if not isinstance(value["name"], str) or not value["name"]:
            raise ValueError("Model name must not be empty.")
        if not isinstance(value["notes"], str):
            raise TypeError("Model notes must be a string.")
        if not isinstance(value["variables"], list) or not value["variables"]:
            raise ValueError("Model variables must be a non-empty list.")
        if not isinstance(value["objectives"], list) or not value["objectives"]:
            raise ValueError("Model objectives must be a non-empty list.")
        if not isinstance(value["constraints"], list):
            raise TypeError("Model constraints must be a list.")
        if len(value["variables"]) > 512:
            raise ValueError("At most 512 variable declarations are allowed.")
        variables = tuple(VariableDeclaration.from_dict(item) for item in value["variables"])
        objectives = tuple(ObjectiveDeclaration.from_dict(item) for item in value["objectives"])
        constraints = tuple(ConstraintDeclaration.from_dict(item) for item in value["constraints"])
        variable_names = [item.name for item in variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("Variable names must be unique.")
        objective_names = [item.name for item in objectives]
        if len(objective_names) != len(set(objective_names)):
            raise ValueError("Objective names must be unique.")
        constraint_names = [item.name for item in constraints]
        if len(constraint_names) != len(set(constraint_names)):
            raise ValueError("Constraint names must be unique.")
        variable_map = {item.name: item.to_variable() for item in variables}
        for item in (*objectives, *constraints):
            compile_expression(item.expression, variable_map)
        return cls(value["name"], variables, objectives, constraints, value["notes"])

    @classmethod
    def from_json(cls, source: str) -> ModelSpec:
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


MODEL_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "variables", "objectives", "constraints", "notes"],
    "properties": {
        "name": {"type": "string"},
        "variables": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "kind", "lower", "upper", "size"],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["binary", "integer", "real"]},
                    "lower": {"type": "number"},
                    "upper": {"type": "number"},
                    "size": {"type": "integer", "minimum": 1},
                },
            },
        },
        "objectives": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "direction", "expression", "unit"],
                "properties": {
                    "name": {"type": "string"},
                    "direction": {"type": "string", "enum": ["min", "max"]},
                    "expression": {"type": "string"},
                    "unit": {"type": "string"},
                },
            },
        },
        "constraints": {
            "type": "array",
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
                    "name": {"type": "string"},
                    "expression": {"type": "string"},
                    "sense": {"type": "string", "enum": ["<=", ">=", "=="]},
                    "rhs": {"type": "number"},
                    "weight": {"type": "number", "exclusiveMinimum": 0},
                    "scale": {"type": "number", "exclusiveMinimum": 0},
                    "tolerance": {"type": "number", "minimum": 0},
                },
            },
        },
        "notes": {"type": "string"},
    },
}


__all__ = [
    "ConstraintDeclaration",
    "MODEL_JSON_SCHEMA",
    "ModelSpec",
    "ObjectiveDeclaration",
    "VariableDeclaration",
]
