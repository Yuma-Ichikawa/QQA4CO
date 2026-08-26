"""QPLIB importer backed by the optional :mod:`pyqplib` parser."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from qqa.algebraic import AlgebraicConstraint, AlgebraicModel, SparseQuadratic, VariableType


def qplib_available() -> bool:
    try:
        import pyqplib  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _require_pyqplib():
    try:
        import pyqplib
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "QPLIB loading requires the optional dependency. "
            "Install it with `pip install 'qqa[qplib]'`."
        ) from exc
    return pyqplib


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hessian_from_lower(lower: Any, dimension: int) -> sparse.csr_matrix:
    """Convert QPLIB's stored lower triangle to a symmetric Hessian.

    A QPLIB expression is ``0.5 * sum(q_ij * x_i * x_j)`` over stored
    lower-triangle entries.  Consequently a non-diagonal stored coefficient
    contributes half that value to each symmetric Hessian entry.  Applying
    ``LowerMatrix.full()`` directly would double those cross terms.
    """
    if lower is None or not bool(lower):
        return sparse.csr_matrix((dimension, dimension), dtype=np.float64)
    diag_rows = np.asarray(lower.diag_rows, dtype=np.int64)
    diag_values = np.asarray(lower.diag_vals, dtype=np.float64)
    rows = np.asarray(lower.subdiag_rows, dtype=np.int64)
    cols = np.asarray(lower.subdiag_cols, dtype=np.int64)
    values = 0.5 * np.asarray(lower.subdiag_vals, dtype=np.float64)
    return sparse.coo_matrix(
        (
            np.concatenate([diag_values, values, values]),
            (
                np.concatenate([diag_rows, rows, cols]),
                np.concatenate([diag_rows, cols, rows]),
            ),
        ),
        shape=(dimension, dimension),
        dtype=np.float64,
    ).tocsr()


def _problem_type(description: Any) -> str:
    maps = (
        {
            "LINEAR": "L",
            "CONVEX_SYMM": "D",
            "CONVEX": "C",
            "GENERAL": "Q",
        },
        {
            "CONTINUOUS": "C",
            "BINARY": "B",
            "MIXED_BINARY": "M",
            "INTEGER": "I",
            "GENERAL": "G",
        },
        {
            "UNCONSTRAINED": "N",
            "BOXED": "B",
            "LINEAR": "L",
            "CONVEX_SYMM": "D",
            "CONVEX": "C",
            "GENERAL": "Q",
        },
    )
    values = (description.obj_type, description.var_type, description.cons_type)
    return "".join(mapping[value.name] for mapping, value in zip(maps, values, strict=True))


def load_qplib(path: str | Path) -> AlgebraicModel:
    """Load one ``.qplib`` file into the sparse algebraic IR.

    Only the source basename and content hash enter metadata.  Absolute local
    paths are intentionally not retained.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"QPLIB instance does not exist: {source}")
    pyqplib = _require_pyqplib()
    parsed = pyqplib.read_problem(source)
    dimension = int(parsed.num_vars)

    if hasattr(parsed.obj, "mat"):
        objective_hessian = _hessian_from_lower(parsed.obj.mat, dimension)
    else:
        objective_hessian = sparse.csr_matrix((dimension, dimension), dtype=np.float64)
    objective = SparseQuadratic(
        objective_hessian,
        np.asarray(parsed.obj.lin, dtype=np.float64),
        float(parsed.obj.offset),
    )

    rows: list[AlgebraicConstraint] = []
    if parsed.constraints is not None:
        linear = sparse.csr_matrix(parsed.constraints.mat, dtype=np.float64)
        hessians = getattr(parsed.constraints, "hess_mats", None)
        for index in range(parsed.num_cons):
            hessian = (
                _hessian_from_lower(hessians[index], dimension)
                if hessians is not None
                else sparse.csr_matrix((dimension, dimension), dtype=np.float64)
            )
            rows.append(
                AlgebraicConstraint(
                    name=f"c_{index + 1}",
                    expression=SparseQuadratic(
                        hessian,
                        linear.getrow(index),
                    ),
                    lower=float(parsed.cons_lb[index]),
                    upper=float(parsed.cons_ub[index]),
                )
            )

    type_map = {
        pyqplib.VarType.CONTINUOUS: VariableType.CONTINUOUS,
        pyqplib.VarType.BINARY: VariableType.BINARY,
        pyqplib.VarType.INTEGER: VariableType.INTEGER,
    }
    variable_types = tuple(type_map[value] for value in parsed.var_types)
    sense = "minimize" if parsed.obj.sense is pyqplib.Sense.MINIMIZE else "maximize"
    model = AlgebraicModel(
        name=str(parsed.name),
        variable_names=tuple(f"x_{index + 1}" for index in range(dimension)),
        variable_types=variable_types,
        lower_bounds=np.asarray(parsed.var_lb, dtype=np.float64),
        upper_bounds=np.asarray(parsed.var_ub, dtype=np.float64),
        objective=objective,
        constraints=rows,
        objective_sense=sense,
        problem_type=_problem_type(parsed.description),
        source_format="qplib",
        metadata={
            "source_name": source.name,
            "source_sha256": _sha256(source),
            "parser": "pyqplib",
            "parser_version": _package_version("pyqplib"),
        },
    )

    # A parser update must never silently change QPLIB semantics. Check both
    # the supplied initial point and a deterministic, bounded nonzero probe;
    # the latter catches lower-triangle cross-term mistakes hidden by x0=0.
    probes: list[np.ndarray] = []
    initial = np.asarray(parsed.x0, dtype=np.float64)
    if initial.shape == (dimension,) and np.isfinite(initial).all():
        probes.append(initial)
    finite_lower = np.where(
        np.isfinite(model.lower_bounds),
        model.lower_bounds,
        np.where(np.isfinite(model.upper_bounds), model.upper_bounds - 2.0, -1.0),
    )
    finite_upper = np.where(
        np.isfinite(model.upper_bounds),
        model.upper_bounds,
        np.where(np.isfinite(model.lower_bounds), model.lower_bounds + 2.0, 1.0),
    )
    probe = finite_lower + 0.37 * (finite_upper - finite_lower)
    for index in model.integer_indices:
        probe[index] = np.rint(probe[index])
    if np.isfinite(probe).all():
        probes.append(probe)
    for probe in probes:
        ours = model.evaluate(probe)
        theirs_objective = float(parsed.obj_val(probe))
        theirs_constraints = np.asarray(parsed.cons_val(probe), dtype=np.float64)
        if not np.isclose(ours.objective, theirs_objective, rtol=1e-9, atol=1e-9):
            raise ValueError("QPLIB objective conversion failed parser cross-check.")
        if not np.allclose(
            ours.constraint_values,
            theirs_constraints,
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError("QPLIB constraint conversion failed parser cross-check.")
    return model


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:  # pragma: no cover - nonstandard environment
        return None


__all__ = ["load_qplib", "qplib_available"]
