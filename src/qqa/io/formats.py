"""Portable, dependency-light input formats for the stable solve API."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from qqa.compile import SparseQUBO
from qqa.engines.qqa import SparseQUBOProblem
from qqa.model import (
    ClauseFactor,
    ConstraintIR,
    HigherOrderFactor,
    LinearFactor,
    ModelIR,
    ModelMetadata,
    ObjectiveIR,
    ObjectiveSense,
    QuadraticEdgeFactor,
    VariableBlock,
    VariableDomain,
)
from qqa.model.ir import Factor

_MAX_PORTABLE_VARIABLES = 1_000_000


def _validate_variable_count(value: int) -> int:
    if isinstance(value, bool) or not 1 <= value <= _MAX_PORTABLE_VARIABLES:
        raise ValueError(
            f"Portable inputs must declare between 1 and {_MAX_PORTABLE_VARIABLES} variables."
        )
    return value


def _data_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "c", "%", "*"))
    ]


def load_qubo_text(path: str | Path) -> SparseQUBOProblem:
    """Load a simple ``i j coefficient`` QUBO edge list."""
    source = Path(path)
    records = []
    for line in _data_lines(source):
        parts = line.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError("QUBO rows must contain: variable_i variable_j coefficient.")
        left, right, weight = int(parts[0]), int(parts[1]), float(parts[2])
        if left < 0 or right < 0:
            raise ValueError("QUBO variable indices must be non-negative.")
        records.append((left, right, weight))
    if not records:
        raise ValueError("QUBO file contains no coefficients.")
    minimum = min(min(left, right) for left, right, _ in records)
    offset = 1 if minimum == 1 else 0
    maximum = max(max(left, right) for left, right, _ in records) - offset
    _validate_variable_count(maximum + 1)
    linear = torch.zeros(maximum + 1, dtype=torch.float64)
    edges = []
    weights = []
    for left, right, weight in records:
        left -= offset
        right -= offset
        if left == right:
            linear[left] += weight
        else:
            edges.append((left, right))
            weights.append(weight)
    edge_index = (
        torch.as_tensor(edges, dtype=torch.long).T
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    return SparseQUBOProblem(
        SparseQUBO(linear, edge_index, torch.as_tensor(weights, dtype=torch.float64)),
        name=source.stem,
    )


def load_ising_text(path: str | Path) -> ModelIR:
    """Load ``i h_i`` and ``i j J_ij`` Ising edge-list records."""
    source = Path(path)
    linear_records = []
    edge_records = []
    indices = []
    for line in _data_lines(source):
        parts = line.replace(",", " ").split()
        if len(parts) == 2:
            index, weight = int(parts[0]), float(parts[1])
            if index < 0:
                raise ValueError("Ising variable indices must be non-negative.")
            linear_records.append((index, weight))
            indices.append(index)
        elif len(parts) == 3:
            left, right, weight = int(parts[0]), int(parts[1]), float(parts[2])
            if left < 0 or right < 0:
                raise ValueError("Ising variable indices must be non-negative.")
            edge_records.append((left, right, weight))
            indices.extend((left, right))
        else:
            raise ValueError("Ising rows must contain `i h` or `i j J`.")
    if not indices:
        raise ValueError("Ising file contains no coefficients.")
    offset = 1 if min(indices) == 1 else 0
    size = _validate_variable_count(max(indices) - offset + 1)
    linear_index = torch.as_tensor(
        [index - offset for index, _ in linear_records], dtype=torch.long
    )
    linear_weight = torch.as_tensor([weight for _, weight in linear_records], dtype=torch.float64)
    edge_index = (
        torch.as_tensor(
            [(left - offset, right - offset) for left, right, _ in edge_records],
            dtype=torch.long,
        )
        .reshape(-1, 2)
        .T
    )
    edge_weight = torch.as_tensor([weight for _, _, weight in edge_records], dtype=torch.float64)
    factors: list[Factor] = []
    if linear_records:
        factors.append(LinearFactor(linear_index, linear_weight))
    if edge_records:
        factors.append(QuadraticEdgeFactor(edge_index, edge_weight))
    return ModelIR(
        (VariableBlock("spin", VariableDomain.SPIN, (size,), -1.0, 1.0),),
        ObjectiveIR(tuple(factors)),
        metadata=ModelMetadata(source.stem, source_format="ising-edge-list"),
    )


def load_dimacs(path: str | Path) -> ModelIR:
    """Load DIMACS CNF or WCNF as native clause factors."""
    source = Path(path)
    header = None
    clauses: list[tuple[float, list[int]]] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p "):
            if header is not None:
                raise ValueError("DIMACS input must contain exactly one header.")
            header = line.split()
            if len(header) not in {4, 5} or header[1].lower() not in {"cnf", "wcnf"}:
                raise ValueError("Unsupported DIMACS header.")
            if header[1].lower() == "cnf" and len(header) != 4:
                raise ValueError("CNF header must be `p cnf variables clauses`.")
            _validate_variable_count(int(header[2]))
            if int(header[3]) < 0:
                raise ValueError("DIMACS clause count must be non-negative.")
            continue
        if header is None:
            raise ValueError("DIMACS file must contain a `p cnf` or `p wcnf` header.")
        try:
            parts = [int(value) for value in line.split()]
        except ValueError as exc:
            raise ValueError("DIMACS clauses must contain integer literals.") from exc
        if not parts or parts[-1] != 0:
            raise ValueError("Every DIMACS clause must end with 0.")
        if header[1].lower() == "wcnf":
            weight, literals = float(parts[0]), parts[1:-1]
        else:
            weight, literals = 1.0, parts[:-1]
        if not literals:
            raise ValueError("Empty DIMACS clauses are not supported.")
        size = int(header[2])
        if any(value == 0 or abs(value) > size for value in literals):
            raise ValueError("DIMACS literal is outside the declared variable range.")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("DIMACS clause weights must be finite and positive.")
        clauses.append((weight, literals))
    if header is None or header[1].lower() not in {"cnf", "wcnf"}:
        raise ValueError("Unsupported DIMACS header.")
    size = int(header[2])
    if len(clauses) != int(header[3]):
        raise ValueError("DIMACS clause count does not match the header.")
    top = float(header[4]) if header[1].lower() == "wcnf" and len(header) > 4 else None
    objective_factors = []
    constraints: list[ConstraintIR] = []
    for index, (weight, literals) in enumerate(clauses):
        factor = ClauseFactor(
            torch.as_tensor([[abs(value) - 1 for value in literals]], dtype=torch.long),
            torch.as_tensor([[1 if value > 0 else -1 for value in literals]], dtype=torch.int8),
            torch.as_tensor([weight], dtype=torch.float64),
        )
        if top is not None and weight >= top:
            constraints.append(
                ConstraintIR(f"hard_clause_{index}", ObjectiveIR((factor,)), "<=", 0.0)
            )
        else:
            objective_factors.append(factor)
    return ModelIR(
        (VariableBlock("x", VariableDomain.BINARY, (size,), 0.0, 1.0),),
        ObjectiveIR(tuple(objective_factors)),
        tuple(constraints),
        metadata=ModelMetadata(source.stem, source_format=header[1].lower()),
    )


_OPB_TERM = re.compile(r"([+-]?\s*\d+(?:\.\d+)?)\s+([A-Za-z_]\w*)")


def load_opb(path: str | Path) -> ModelIR:
    """Load linear pseudo-Boolean objective and constraints."""
    source = Path(path)
    lines = _data_lines(source)
    variable_names = sorted(
        {match.group(2) for line in lines for match in _OPB_TERM.finditer(line)}
    )
    if not variable_names:
        raise ValueError("OPB file contains no variables.")
    _validate_variable_count(len(variable_names))
    lookup = {name: index for index, name in enumerate(variable_names)}

    def expression(text: str) -> ObjectiveIR:
        matches = list(_OPB_TERM.finditer(text))
        return ObjectiveIR(
            (
                LinearFactor(
                    torch.as_tensor([lookup[item.group(2)] for item in matches], dtype=torch.long),
                    torch.as_tensor(
                        [float(item.group(1).replace(" ", "")) for item in matches],
                        dtype=torch.float64,
                    ),
                ),
            )
        )

    objective = ObjectiveIR(())
    sense = ObjectiveSense.MINIMIZE
    constraints: list[ConstraintIR] = []
    for line in lines:
        line = line.rstrip(";").strip()
        if line.lower().startswith(("min:", "max:")):
            sense = (
                ObjectiveSense.MINIMIZE
                if line.lower().startswith("min:")
                else ObjectiveSense.MAXIMIZE
            )
            objective = expression(line.split(":", 1)[1])
            continue
        match = re.search(r"(<=|>=|=)\s*([+-]?\d+(?:\.\d+)?)$", line)
        if match is None:
            raise ValueError(f"Unsupported OPB row: {line!r}")
        constraints.append(
            ConstraintIR(
                f"constraint_{len(constraints)}",
                expression(line[: match.start()]),
                "==" if match.group(1) == "=" else match.group(1),
                float(match.group(2)),
            )
        )
    return ModelIR(
        (VariableBlock("x", VariableDomain.BINARY, (len(variable_names),), 0.0, 1.0),),
        objective,
        tuple(constraints),
        sense,
        ModelMetadata(
            source.stem, source_format="opb", attributes={"variable_names": variable_names}
        ),
    )


def _factor_from_json(record: dict[str, Any]):
    if not isinstance(record, dict):
        raise ValueError("Every JSON ModelIR factor must be an object.")
    kind = record.get("type")
    required = {
        "linear": {"indices", "weights"},
        "quadratic": {"edge_index", "weights"},
        "higher_order": {"indices", "weights"},
        "clause": {"indices", "signs"},
    }
    if kind in required and not required[kind] <= record.keys():
        raise ValueError(f"JSON ModelIR {kind!r} factor is missing required fields.")
    if kind == "linear":
        return LinearFactor(record["indices"], record["weights"])
    if kind == "quadratic":
        return QuadraticEdgeFactor(record["edge_index"], record["weights"])
    if kind == "higher_order":
        return HigherOrderFactor(record["indices"], record["weights"])
    if kind == "clause":
        return ClauseFactor(record["indices"], record["signs"], record.get("weights"))
    raise ValueError(f"Unsupported JSON ModelIR factor type {kind!r}.")


def model_ir_from_dict(payload: dict[str, Any], *, default_name: str = "model") -> ModelIR:
    """Validate a portable JSON-shaped mapping without executing user code."""
    if not isinstance(payload, dict):
        raise TypeError("JSON ModelIR root must be an object.")
    variable_records = payload.get("variables")
    objective_record = payload.get("objective")
    constraint_records = payload.get("constraints", [])
    if (
        not isinstance(variable_records, list)
        or not variable_records
        or any(
            not isinstance(item, dict) or not {"name", "domain"} <= item.keys()
            for item in variable_records
        )
    ):
        raise ValueError("JSON ModelIR variables must be non-empty schema objects.")
    if not isinstance(objective_record, dict) or not isinstance(
        objective_record.get("factors", []), list
    ):
        raise ValueError("JSON ModelIR objective must be an object with a factor list.")
    if not isinstance(constraint_records, list) or any(
        not isinstance(item, dict)
        or "name" not in item
        or not isinstance(item.get("expression"), dict)
        or not isinstance(item["expression"].get("factors", []), list)
        for item in constraint_records
    ):
        raise ValueError("JSON ModelIR constraints must be schema objects.")
    variables = tuple(
        VariableBlock(
            item["name"],
            item["domain"],
            tuple(item.get("shape", [1])),
            item.get("lower"),
            item.get("upper"),
            item.get("categories"),
        )
        for item in variable_records
    )
    _validate_variable_count(sum(block.size for block in variables))
    objective = ObjectiveIR(
        tuple(_factor_from_json(item) for item in objective_record.get("factors", [])),
        float(objective_record.get("constant", 0.0)),
    )
    constraints = tuple(
        ConstraintIR(
            item["name"],
            ObjectiveIR(
                tuple(
                    _factor_from_json(factor) for factor in item["expression"].get("factors", [])
                ),
                float(item["expression"].get("constant", 0.0)),
            ),
            item.get("sense", "<="),
            float(item.get("rhs", 0.0)),
            float(item.get("scale", 1.0)),
            float(item.get("tolerance", 1e-6)),
        )
        for item in constraint_records
    )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("JSON ModelIR metadata must be an object.")
    return ModelIR(
        variables,
        objective,
        constraints,
        payload.get("sense", "minimize"),
        ModelMetadata(
            metadata.get("name", default_name),
            metadata.get("problem_class"),
            metadata.get("source_format", "json-model-ir"),
            metadata.get("attributes", {}),
        ),
    )


def load_model_ir_json(path: str | Path) -> ModelIR:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return model_ir_from_dict(payload, default_name=source.stem)


def load_portable_model(path: str | Path):
    """Dispatch dependency-light formats by suffix."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        return load_model_ir_json(source)
    if suffix in {".qubo", ".bqm"}:
        return load_qubo_text(source)
    if suffix in {".ising", ".isingtxt"}:
        return load_ising_text(source)
    if suffix in {".cnf", ".wcnf"}:
        return load_dimacs(source)
    if suffix == ".opb":
        return load_opb(source)
    raise ValueError(
        "Unsupported model format. Use MPS, LP, QPLIB, JSON ModelIR, OPB, "
        "DIMACS CNF/WCNF, QUBO text, or Ising edge list."
    )


__all__ = [
    "load_dimacs",
    "load_ising_text",
    "load_model_ir_json",
    "load_opb",
    "load_portable_model",
    "load_qubo_text",
    "model_ir_from_dict",
]
