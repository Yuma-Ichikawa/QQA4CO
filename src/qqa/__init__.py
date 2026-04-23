"""Quasi-Quantum Annealing (QQA) for combinatorial and spin-glass optimization.

Reference:
    Y. Ichikawa, Y. Arai. "Optimization by Parallel Quasi-Quantum Annealing
    with Gradient-Based Sampling." ICLR 2025.
    https://openreview.net/forum?id=9EfBeXaXf0  (arXiv:2409.02135)

Typical usage::

    import networkx as nx
    import qqa

    qqa.fix_seed(0)
    g = nx.random_regular_graph(d=3, n=50, seed=0)
    problem = qqa.MaximumIndependentSet(g, penalty=2)
    result = qqa.anneal(problem, sol_size=100, num_epochs=1500)
    print(result.best_obj, result.runtime)

Spin-glass example::

    problem = qqa.SherringtonKirkpatrick(N=100, seed=0)
    result = qqa.anneal(problem, sol_size=200, num_epochs=2000)
    print("E_0 per spin:", result.best_obj / 100)
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from qqa import polish, warmstart
from qqa.annealing import AnnealResult, anneal
from qqa.callbacks import (
    AutoDivTuner,
    Callback,
    CallbackState,
    HistoryRecorder,
    PopulationTracker,
    TrajectoryTracker,
)
from qqa.isco import ISCOResult, discrete_langevin, isco_anneal
from qqa.pa import PAResult, population_annealing
from qqa.problems import (
    QAP,
    TSP,
    BalancedGraphPartition,
    BinaryPerceptron,
    Coloring,
    COProblem,
    EdwardsAnderson,
    GraphBisection,
    HopfieldMemory,
    IntegerFactorizationIsing,
    Ising1D,
    Knapsack,
    MaxClique,
    MaxCliqueInstance,
    MaxCut,
    MaxCutInstance,
    MaximumIndependentSet,
    MaximumIndependentSetInstance,
    MaxSAT3,
    MinimumDominatingSet,
    NormalizedCut,
    NormCut,
    NQueens,
    NumberPartitioning,
    PSpinGlass,
    QUBOProblem,
    RandomFieldIsing,
    SherringtonKirkpatrick,
    SpinProblem,
    UserProblem,
    VertexCover,
    load_problem_from_file,
    random_factorization_problems,
    random_prime,
    random_semiprime,
    user_problem_from_source,
)
from qqa.relaxation import (
    BinaryInstanceRelaxation,
    BinaryRelaxation,
    CategoricalRelaxation,
    Relaxation,
    SpinRelaxation,
)
from qqa.sa import SAResult, simulated_annealing
from qqa.schedule import LinearBGSchedule
from qqa.utils import enable_tf32, fix_seed, generate_graph

# Single-source the version from the wheel metadata so ``__version__`` is
# always whatever ``pip install qqa`` actually installed. The fallback covers
# editable installs where the metadata is occasionally absent (e.g. a
# fresh ``git clone`` before any ``uv sync``); we surface the canonical
# value of ``pyproject.toml`` so callers always get a real-looking string.
try:
    __version__ = _pkg_version("qqa")
except PackageNotFoundError:  # pragma: no cover - editable install w/o metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "QAP",
    "TSP",
    "AnnealResult",
    "AutoDivTuner",
    "BalancedGraphPartition",
    "BinaryInstanceRelaxation",
    "BinaryPerceptron",
    "BinaryRelaxation",
    "COProblem",
    "Callback",
    "CallbackState",
    "CategoricalRelaxation",
    "Coloring",
    "EdwardsAnderson",
    "GraphBisection",
    "HistoryRecorder",
    "HopfieldMemory",
    "ISCOResult",
    "IntegerFactorizationIsing",
    "Ising1D",
    "Knapsack",
    "LinearBGSchedule",
    "MaxClique",
    "MaxCliqueInstance",
    "MaxCut",
    "MaxCutInstance",
    "MaxSAT3",
    "MaximumIndependentSet",
    "MaximumIndependentSetInstance",
    "MinimumDominatingSet",
    "NQueens",
    "NormCut",
    "NormalizedCut",
    "NumberPartitioning",
    "PAResult",
    "PSpinGlass",
    "PopulationTracker",
    "QUBOProblem",
    "RandomFieldIsing",
    "Relaxation",
    "SAResult",
    "SherringtonKirkpatrick",
    "SpinProblem",
    "SpinRelaxation",
    "TrajectoryTracker",
    "UserProblem",
    "VertexCover",
    "__version__",
    "anneal",
    "discrete_langevin",
    "enable_tf32",
    "fix_seed",
    "generate_graph",
    "isco_anneal",
    "load_problem_from_file",
    "polish",
    "population_annealing",
    "random_factorization_problems",
    "random_prime",
    "random_semiprime",
    "simulated_annealing",
    "user_problem_from_source",
    "warmstart",
]
