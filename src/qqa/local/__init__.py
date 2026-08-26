"""Structure-aware repair and local refinement plugins."""

from qqa.local.advanced import (
    QUBOLocalSearchResult,
    iterated_local_search,
    k_flip_search,
    path_relink,
    tabu_search,
)
from qqa.local.archive import EliteArchive, EliteEntry
from qqa.local.sparse_qubo import LocalSearchResult, sparse_qubo_descent
from qqa.local.structured import (
    StructuredSearchResult,
    kempe_coloring_search,
    maxcut_fm_search,
    mis_swap_search,
    three_opt_tour,
    two_opt_tour,
    walksat_search,
)

__all__ = [
    "EliteArchive",
    "EliteEntry",
    "LocalSearchResult",
    "QUBOLocalSearchResult",
    "StructuredSearchResult",
    "iterated_local_search",
    "k_flip_search",
    "kempe_coloring_search",
    "maxcut_fm_search",
    "mis_swap_search",
    "path_relink",
    "sparse_qubo_descent",
    "tabu_search",
    "three_opt_tour",
    "two_opt_tour",
    "walksat_search",
]
