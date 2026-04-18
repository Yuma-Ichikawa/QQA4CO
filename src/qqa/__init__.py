"""Quasi-Quantum Annealing (QQA) for combinatorial and spin-glass optimization.

Reference:
    Y. Ichikawa, Y. Arai. "Continuous Tensor Relaxation for Finding Diverse
    Solutions in Combinatorial Optimization." ICLR 2025.

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

from qqa.annealing import AnnealResult, anneal
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
    Ising1D,
    Knapsack,
    MaxClique,
    MaxCliqueInstance,
    MaxCut,
    MaxCutInstance,
    MaximumIndependentSet,
    MaximumIndependentSetInstance,
    MaxSAT3,
    NQueens,
    NumberPartitioning,
    QUBOProblem,
    SherringtonKirkpatrick,
    SpinProblem,
    UserProblem,
    VertexCover,
    load_problem_from_file,
    user_problem_from_source,
)
from qqa.relaxation import (
    BinaryInstanceRelaxation,
    BinaryRelaxation,
    CategoricalRelaxation,
    SpinRelaxation,
)
from qqa.schedule import LinearBGSchedule
from qqa.utils import fix_seed, generate_graph

__version__ = "0.3.0"

__all__ = [
    "QAP",
    "TSP",
    "AnnealResult",
    "BalancedGraphPartition",
    "BinaryInstanceRelaxation",
    "BinaryPerceptron",
    "BinaryRelaxation",
    "CategoricalRelaxation",
    "COProblem",
    "Coloring",
    "EdwardsAnderson",
    "GraphBisection",
    "HopfieldMemory",
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
    "NQueens",
    "NumberPartitioning",
    "QUBOProblem",
    "SherringtonKirkpatrick",
    "SpinProblem",
    "SpinRelaxation",
    "UserProblem",
    "VertexCover",
    "__version__",
    "anneal",
    "fix_seed",
    "generate_graph",
    "load_problem_from_file",
    "user_problem_from_source",
]
