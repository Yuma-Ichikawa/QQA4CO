"""Sparse dynamic convexification helpers for non-convex QPLIB blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True, slots=True)
class DCDecomposition:
    """Difference-of-convex representation ``matrix = convex - concave``."""

    convex: sparse.csr_matrix
    concave: sparse.csr_matrix
    minimum_eigenvalue: float
    shift: float


def minimum_eigenvalue(matrix: sparse.spmatrix, *, dense_threshold: int = 256) -> float:
    symmetric = ((sparse.csr_matrix(matrix) + sparse.csr_matrix(matrix).T) * 0.5).tocsr()
    dimension = symmetric.shape[0]
    if dimension == 0 or symmetric.nnz == 0:
        return 0.0
    if dimension <= dense_threshold:
        return float(np.linalg.eigvalsh(symmetric.toarray())[0])
    try:
        value = eigsh(symmetric, k=1, which="SA", return_eigenvectors=False, tol=1e-7)[0]
        return float(value)
    except Exception:
        # Gershgorin is conservative but deterministic and never densifies.
        diagonal = symmetric.diagonal()
        radius = np.asarray(abs(symmetric).sum(axis=1)).reshape(-1) - np.abs(diagonal)
        return float(np.min(diagonal - radius))


def dc_decomposition(
    matrix: sparse.spmatrix,
    *,
    margin: float = 1e-8,
) -> DCDecomposition:
    """Shift an indefinite sparse matrix into two positive-semidefinite parts."""
    if margin < 0 or not np.isfinite(margin):
        raise ValueError("margin must be finite and >= 0.")
    symmetric = ((sparse.csr_matrix(matrix) + sparse.csr_matrix(matrix).T) * 0.5).tocsr()
    eigenvalue = minimum_eigenvalue(symmetric)
    shift = max(0.0, -eigenvalue + margin)
    identity = sparse.eye(symmetric.shape[0], format="csr", dtype=np.float64)
    concave = (shift * identity).tocsr()
    convex = (symmetric + concave).tocsr()
    return DCDecomposition(convex, concave, eigenvalue, shift)


def linearize_concave_part(
    decomposition: DCDecomposition,
    point: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, float]:
    """Return convex Hessian and affine majorisation of ``-0.5*x'Cx``."""
    x = np.asarray(point, dtype=np.float64)
    if x.shape != (decomposition.convex.shape[0],):
        raise ValueError("point dimension does not match decomposition.")
    gradient = -decomposition.concave.dot(x)
    constant = float(0.5 * x @ decomposition.concave.dot(x))
    return decomposition.convex, np.asarray(gradient), constant


__all__ = [
    "DCDecomposition",
    "dc_decomposition",
    "linearize_concave_part",
    "minimum_eigenvalue",
]
