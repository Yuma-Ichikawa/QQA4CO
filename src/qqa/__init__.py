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

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from qqa import polish, warmstart
from qqa.algebraic import (
    AlgebraicConstraint,
    AlgebraicEvaluation,
    AlgebraicModel,
    SparseQuadratic,
    VariableType,
)
from qqa.annealing import AnnealResult, anneal
from qqa.api import inspect, plan, solve
from qqa.applications import (
    APPLICATIONS,
    build_application,
    build_microgrid_dispatch,
    build_microgrid_pareto,
    build_portfolio_pareto,
    build_process_blackbox,
)
from qqa.blackbox import (
    BlackBoxConstraint,
    BlackBoxProblem,
    BlackBoxResult,
    blackbox_optimize,
    plot_blackbox,
)
from qqa.callbacks import (
    AutoDivTuner,
    Callback,
    CallbackState,
    HistoryRecorder,
    PopulationTracker,
    TrajectoryTracker,
)
from qqa.config import SolverConfig
from qqa.isco import ISCOResult, discrete_langevin, isco_anneal
from qqa.mixed import (
    Binary,
    BinaryVariable,
    Constraint,
    Integer,
    IntegerVariable,
    MixedProblem,
    MixedRelaxation,
    Real,
    RealVariable,
    RepairResult,
    VariableSpace,
    repair_mixed_solution,
    solve_mixed,
)
from qqa.model import ModelIR
from qqa.multiobjective import (
    MultiObjectiveProblem,
    Objective,
    ParetoResult,
    pareto_anneal,
    plot_pareto,
    plot_pareto_diagnostics,
)
from qqa.natural_language import (
    MODEL_SYSTEM_PROMPT,
    AskResult,
    OptimizationPlan,
    ask,
    blackbox_from_spec,
    compile_natural_language,
    execute_plan,
    plan_spec,
)
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
    EntropicCategoricalRelaxation,
    MirrorDescentCategoricalRelaxation,
    Relaxation,
    SinkhornRelaxation,
    SoftmaxCategoricalRelaxation,
    SparseCategoricalRelaxation,
    SpinRelaxation,
    StochasticBinaryRelaxation,
    StraightThroughBinaryRelaxation,
)
from qqa.reporting import save_html_report
from qqa.result import SolveResult
from qqa.sa import SAResult, simulated_annealing
from qqa.schedule import LinearBGSchedule
from qqa.tex import (
    TEX_SYSTEM_PROMPT,
    LLMAPIError,
    ModelSpec,
    OpenAICompatibleClient,
    TexSolveResult,
    compile_tex,
    problem_from_spec,
    solve_tex,
)
from qqa.utils import enable_tf32, fix_seed, generate_graph, resolve_device

# Exact-solver and public-benchmark integrations are deliberately absent from
# the eager package graph.  Explicit legacy access such as
# ``qqa.solve_qqa_scip`` remains compatible, but only that access imports the
# opt-in module.  New code should import from ``qqa.hybrid``,
# ``qqa.benchmarking``, or ``qqa.io`` to make the boundary visible.
_OPTIONAL_EXPORTS: dict[str, tuple[str, str]] = {
    "BenchmarkComparisonResult": ("qqa.benchmarking", "BenchmarkComparisonResult"),
    "BenchmarkFailure": ("qqa.benchmarking", "BenchmarkFailure"),
    "BenchmarkResult": ("qqa.benchmarking", "BenchmarkResult"),
    "BenchmarkSuiteResult": ("qqa.benchmarking", "BenchmarkSuiteResult"),
    "QQAHeuristic": ("qqa.hybrid", "QQAHeuristic"),
    "QQAHeuristicConfig": ("qqa.hybrid", "QQAHeuristicConfig"),
    "QQAHeuristicStats": ("qqa.hybrid", "QQAHeuristicStats"),
    "SCIPExpressionError": ("qqa.hybrid", "SCIPExpressionError"),
    "SCIPHybridResult": ("qqa.hybrid", "SCIPHybridResult"),
    "SCIPModelResult": ("qqa.hybrid", "SCIPModelResult"),
    "compare_benchmark_solvers": ("qqa.benchmarking", "compare_benchmark_solvers"),
    "fetch_benchmark": ("qqa.benchmarking", "fetch_benchmark"),
    "fetch_instance": ("qqa.benchmarking", "fetch_instance"),
    "load_mps": ("qqa.io", "load_mps"),
    "load_qplib": ("qqa.io", "load_qplib"),
    "publish_benchmark_campaigns": ("qqa.benchmarking", "publish_benchmark_campaigns"),
    "qplib_available": ("qqa.io", "qplib_available"),
    "run_benchmark_instance": ("qqa.benchmarking", "run_benchmark_instance"),
    "run_benchmark_suite": ("qqa.benchmarking", "run_benchmark_suite"),
    "run_miplib": ("qqa.benchmarking", "run_miplib"),
    "run_qplib": ("qqa.benchmarking", "run_qplib"),
    "scip_available": ("qqa.hybrid", "scip_available"),
    "solve_qqa_scip": ("qqa.hybrid", "solve_qqa_scip"),
    "solve_spec_scip": ("qqa.hybrid", "solve_spec_scip"),
}


def __getattr__(name: str):
    """Resolve backwards-compatible optional exports on explicit access."""
    target = _OPTIONAL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_OPTIONAL_EXPORTS})


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
    "AlgebraicConstraint",
    "AlgebraicEvaluation",
    "AlgebraicModel",
    "QAP",
    "TSP",
    "AnnealResult",
    "APPLICATIONS",
    "AskResult",
    "AutoDivTuner",
    "BalancedGraphPartition",
    "BlackBoxConstraint",
    "BlackBoxProblem",
    "BlackBoxResult",
    "Binary",
    "BinaryInstanceRelaxation",
    "BinaryPerceptron",
    "BinaryRelaxation",
    "BinaryVariable",
    "COProblem",
    "Callback",
    "CallbackState",
    "CategoricalRelaxation",
    "Coloring",
    "Constraint",
    "EdwardsAnderson",
    "EntropicCategoricalRelaxation",
    "GraphBisection",
    "HistoryRecorder",
    "HopfieldMemory",
    "ISCOResult",
    "Integer",
    "IntegerFactorizationIsing",
    "IntegerVariable",
    "Ising1D",
    "Knapsack",
    "LinearBGSchedule",
    "LLMAPIError",
    "MaxClique",
    "MaxCliqueInstance",
    "MaxCut",
    "MaxCutInstance",
    "MaxSAT3",
    "MaximumIndependentSet",
    "MaximumIndependentSetInstance",
    "MinimumDominatingSet",
    "MirrorDescentCategoricalRelaxation",
    "MixedProblem",
    "ModelIR",
    "MixedRelaxation",
    "ModelSpec",
    "MODEL_SYSTEM_PROMPT",
    "MultiObjectiveProblem",
    "NQueens",
    "NormCut",
    "NormalizedCut",
    "NumberPartitioning",
    "Objective",
    "OpenAICompatibleClient",
    "OptimizationPlan",
    "PAResult",
    "ParetoResult",
    "PSpinGlass",
    "PopulationTracker",
    "QUBOProblem",
    "RandomFieldIsing",
    "Real",
    "RealVariable",
    "RepairResult",
    "Relaxation",
    "SAResult",
    "SolveResult",
    "SolverConfig",
    "SherringtonKirkpatrick",
    "SinkhornRelaxation",
    "SoftmaxCategoricalRelaxation",
    "SparseCategoricalRelaxation",
    "SpinProblem",
    "SpinRelaxation",
    "StochasticBinaryRelaxation",
    "StraightThroughBinaryRelaxation",
    "SparseQuadratic",
    "TEX_SYSTEM_PROMPT",
    "TexSolveResult",
    "TrajectoryTracker",
    "UserProblem",
    "VariableSpace",
    "VariableType",
    "VertexCover",
    "__version__",
    "anneal",
    "ask",
    "blackbox_from_spec",
    "blackbox_optimize",
    "build_application",
    "build_microgrid_dispatch",
    "build_microgrid_pareto",
    "build_portfolio_pareto",
    "build_process_blackbox",
    "compile_tex",
    "compile_natural_language",
    "discrete_langevin",
    "enable_tf32",
    "execute_plan",
    "fix_seed",
    "generate_graph",
    "isco_anneal",
    "inspect",
    "load_problem_from_file",
    "polish",
    "population_annealing",
    "pareto_anneal",
    "plan",
    "plan_spec",
    "plot_blackbox",
    "plot_pareto",
    "plot_pareto_diagnostics",
    "problem_from_spec",
    "random_factorization_problems",
    "random_prime",
    "random_semiprime",
    "resolve_device",
    "repair_mixed_solution",
    "simulated_annealing",
    "save_html_report",
    "solve",
    "solve_tex",
    "solve_mixed",
    "user_problem_from_source",
    "warmstart",
]
