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
from qqa.algebraic import (
    AlgebraicConstraint,
    AlgebraicEvaluation,
    AlgebraicModel,
    SparseQuadratic,
    VariableType,
)
from qqa.annealing import AnnealResult, anneal
from qqa.applications import (
    APPLICATIONS,
    build_application,
    build_microgrid_dispatch,
    build_microgrid_pareto,
    build_portfolio_pareto,
    build_process_blackbox,
)
from qqa.benchmarking import (
    BenchmarkComparisonResult,
    BenchmarkFailure,
    BenchmarkResult,
    BenchmarkSuiteResult,
    compare_benchmark_solvers,
    fetch_benchmark,
    fetch_instance,
    publish_benchmark_campaigns,
    run_benchmark_instance,
    run_benchmark_suite,
    run_miplib,
    run_qplib,
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
from qqa.hybrid import (
    QQAHeuristic,
    QQAHeuristicConfig,
    QQAHeuristicStats,
    SCIPExpressionError,
    SCIPHybridResult,
    SCIPModelResult,
    scip_available,
    solve_qqa_scip,
    solve_spec_scip,
)
from qqa.io import load_mps, load_qplib, qplib_available
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
    Relaxation,
    SpinRelaxation,
)
from qqa.reporting import save_html_report
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
    "BenchmarkResult",
    "BenchmarkComparisonResult",
    "BenchmarkFailure",
    "BenchmarkSuiteResult",
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
    "MixedProblem",
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
    "QQAHeuristicConfig",
    "QQAHeuristic",
    "QQAHeuristicStats",
    "RandomFieldIsing",
    "Real",
    "RealVariable",
    "RepairResult",
    "Relaxation",
    "SAResult",
    "SCIPHybridResult",
    "SCIPExpressionError",
    "SCIPModelResult",
    "scip_available",
    "SherringtonKirkpatrick",
    "SpinProblem",
    "SpinRelaxation",
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
    "fetch_benchmark",
    "fetch_instance",
    "generate_graph",
    "isco_anneal",
    "load_problem_from_file",
    "load_mps",
    "load_qplib",
    "polish",
    "population_annealing",
    "pareto_anneal",
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
    "compare_benchmark_solvers",
    "publish_benchmark_campaigns",
    "run_benchmark_instance",
    "run_benchmark_suite",
    "run_miplib",
    "run_qplib",
    "simulated_annealing",
    "save_html_report",
    "solve_qqa_scip",
    "solve_spec_scip",
    "solve_tex",
    "solve_mixed",
    "qplib_available",
    "user_problem_from_source",
    "warmstart",
]
