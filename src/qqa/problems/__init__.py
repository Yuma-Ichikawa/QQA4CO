"""Problem catalog for QQA.

This subpackage groups the concrete combinatorial optimization problems
shipped with QQA. They are organized by the kind of variable they operate on:

* ``qubo.py`` — binary QUBO problems (MIS, MaxClique, MaxCut).
* ``categorical.py`` — one-hot categorical problems (Coloring, BGP).
* ``spin.py`` — ``{-1, +1}`` spin problems (Ising, EA, SK, perceptron,
  Hopfield).

All public names are re-exported from :mod:`qqa` for convenience, so user code
can simply do ``from qqa import MaximumIndependentSet`` regardless of where
the class lives internally.
"""

from __future__ import annotations

from qqa.problems.base import COProblem, QUBOProblem
from qqa.problems.categorical import BalancedGraphPartition, Coloring
from qqa.problems.extras import (
    QAP,
    TSP,
    GraphBisection,
    Knapsack,
    MaxSAT3,
    MinimumDominatingSet,
    NQueens,
    NumberPartitioning,
    VertexCover,
)
from qqa.problems.factorization import (
    IntegerFactorizationIsing,
    random_factorization_problems,
    random_prime,
    random_semiprime,
)
from qqa.problems.normcut import NormalizedCut, NormCut
from qqa.problems.qubo import (
    MaxClique,
    MaxCliqueInstance,
    MaxCut,
    MaxCutInstance,
    MaximumIndependentSet,
    MaximumIndependentSetInstance,
)
from qqa.problems.spin import (
    BinaryPerceptron,
    EdwardsAnderson,
    HopfieldMemory,
    Ising1D,
    PSpinGlass,
    RandomFieldIsing,
    SherringtonKirkpatrick,
    SpinProblem,
)
from qqa.problems.user import (
    UserProblem,
    load_problem_from_file,
    user_problem_from_source,
)

__all__ = [
    "COProblem",
    "QUBOProblem",
    "SpinProblem",
    "MaximumIndependentSet",
    "MaximumIndependentSetInstance",
    "MaxClique",
    "MaxCliqueInstance",
    "MaxCut",
    "MaxCutInstance",
    "BalancedGraphPartition",
    "Coloring",
    "NormalizedCut",
    "NormCut",
    "Ising1D",
    "EdwardsAnderson",
    "SherringtonKirkpatrick",
    "PSpinGlass",
    "RandomFieldIsing",
    "BinaryPerceptron",
    "HopfieldMemory",
    "Knapsack",
    "NumberPartitioning",
    "VertexCover",
    "GraphBisection",
    "MinimumDominatingSet",
    "MaxSAT3",
    "TSP",
    "QAP",
    "NQueens",
    "UserProblem",
    "user_problem_from_source",
    "load_problem_from_file",
    "IntegerFactorizationIsing",
    "random_factorization_problems",
    "random_prime",
    "random_semiprime",
]
