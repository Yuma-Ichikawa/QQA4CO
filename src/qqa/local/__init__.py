"""Structure-aware repair and local refinement plugins."""

from qqa.local.archive import EliteArchive, EliteEntry
from qqa.local.sparse_qubo import LocalSearchResult, sparse_qubo_descent

__all__ = ["EliteArchive", "EliteEntry", "LocalSearchResult", "sparse_qubo_descent"]
