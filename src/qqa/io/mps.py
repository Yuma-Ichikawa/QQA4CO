"""MPS/MIPLIB importer using SCIP's production file readers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy import sparse

from qqa.algebraic import AlgebraicConstraint, AlgebraicModel, SparseQuadratic, VariableType


def _require_scip():
    try:
        from pyscipopt import Model
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "MPS loading requires SCIP. Install it with `pip install 'qqa[scip]'`."
        ) from exc
    return Model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_stem(path: Path) -> str:
    name = path.name
    for suffix in (".gz", ".bz2", ".zip", ".mps", ".lp"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name or "mip-model"


def load_mps(path: str | Path) -> AlgebraicModel:
    """Read a linear MPS/LP instance and preserve its sparse original model."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"MPS instance does not exist: {source}")
    Model = _require_scip()
    scip = Model(_source_stem(source))
    scip.hideOutput()
    scip.readProblem(str(source))

    variables = list(scip.getVars(transformed=False))
    names = tuple(variable.name for variable in variables)
    index = {variable.name: position for position, variable in enumerate(variables)}
    type_map = {
        "BINARY": VariableType.BINARY,
        "INTEGER": VariableType.INTEGER,
        "IMPLINT": VariableType.IMPLICIT_INTEGER,
        "CONTINUOUS": VariableType.CONTINUOUS,
    }
    variable_types = tuple(type_map[variable.vtype()] for variable in variables)
    lower = np.asarray([variable.getLbGlobal() for variable in variables], dtype=np.float64)
    upper = np.asarray([variable.getUbGlobal() for variable in variables], dtype=np.float64)
    objective_linear = np.asarray([variable.getObj() for variable in variables], dtype=np.float64)
    objective = SparseQuadratic.linear_expression(
        objective_linear,
        constant=float(scip.getObjoffset(original=True)),
    )

    constraints: list[AlgebraicConstraint] = []
    for row_number, constraint in enumerate(scip.getConss(transformed=False), start=1):
        try:
            coefficients = scip.getValsLinear(constraint)
        except Exception as exc:
            raise ValueError(
                "The MPS importer currently supports linear MIPLIB constraints; "
                f"constraint {constraint.name!r} is not linear."
            ) from exc
        columns: list[int] = []
        values: list[float] = []
        for name, value in coefficients.items():
            if name not in index:
                raise ValueError(f"Constraint references unknown variable {name!r}.")
            columns.append(index[name])
            values.append(float(value))
        vector = sparse.coo_matrix(
            (values, ([0] * len(columns), columns)),
            shape=(1, len(variables)),
            dtype=np.float64,
        ).tocsr()
        constraints.append(
            AlgebraicConstraint(
                name=constraint.name or f"c_{row_number}",
                expression=SparseQuadratic.linear_expression(vector),
                lower=float(scip.getLhs(constraint)),
                upper=float(scip.getRhs(constraint)),
            )
        )

    result = AlgebraicModel(
        name=_source_stem(source),
        variable_names=names,
        variable_types=variable_types,
        lower_bounds=lower,
        upper_bounds=upper,
        objective=objective,
        constraints=constraints,
        objective_sense=str(scip.getObjectiveSense()).lower(),
        source_format="mps",
        metadata={
            "source_name": source.name,
            "source_sha256": _sha256(source),
            "reader": "SCIP",
        },
    )
    scip.free()
    return result


__all__ = ["load_mps"]
