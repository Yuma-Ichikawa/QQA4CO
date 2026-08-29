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

# Keep ``import qqa`` and metadata-only CLI commands lightweight. Public
# objects retain their historical top-level names and are loaded on first
# explicit access through PEP 562's module ``__getattr__`` hook.
_EXPORTS: dict[str, tuple[str, str | None]] = {
    "AlgebraicConstraint": ("qqa.algebraic", "AlgebraicConstraint"),
    "AlgebraicEvaluation": ("qqa.algebraic", "AlgebraicEvaluation"),
    "AlgebraicModel": ("qqa.algebraic", "AlgebraicModel"),
    "SparseQuadratic": ("qqa.algebraic", "SparseQuadratic"),
    "VariableType": ("qqa.algebraic", "VariableType"),
    "AnnealResult": ("qqa.annealing", "AnnealResult"),
    "anneal": ("qqa.annealing", "anneal"),
    "doctor": ("qqa.api", "doctor"),
    "inspect": ("qqa.api", "inspect"),
    "plan": ("qqa.api", "plan"),
    "solve": ("qqa.api", "solve"),
    "APPLICATIONS": ("qqa.applications", "APPLICATIONS"),
    "build_application": ("qqa.applications", "build_application"),
    "build_microgrid_dispatch": ("qqa.applications", "build_microgrid_dispatch"),
    "build_microgrid_pareto": ("qqa.applications", "build_microgrid_pareto"),
    "build_portfolio_pareto": ("qqa.applications", "build_portfolio_pareto"),
    "build_process_blackbox": ("qqa.applications", "build_process_blackbox"),
    "BlackBoxConstraint": ("qqa.blackbox", "BlackBoxConstraint"),
    "BlackBoxProblem": ("qqa.blackbox", "BlackBoxProblem"),
    "BlackBoxResult": ("qqa.blackbox", "BlackBoxResult"),
    "Study": ("qqa.blackbox", "Study"),
    "Trial": ("qqa.blackbox", "Trial"),
    "TrialState": ("qqa.blackbox", "TrialState"),
    "blackbox_optimize": ("qqa.blackbox", "blackbox_optimize"),
    "create_study": ("qqa.blackbox", "create_study"),
    "plot_blackbox": ("qqa.blackbox", "plot_blackbox"),
    "AutoDivTuner": ("qqa.callbacks", "AutoDivTuner"),
    "Callback": ("qqa.callbacks", "Callback"),
    "CallbackState": ("qqa.callbacks", "CallbackState"),
    "HistoryRecorder": ("qqa.callbacks", "HistoryRecorder"),
    "PopulationTracker": ("qqa.callbacks", "PopulationTracker"),
    "TrajectoryTracker": ("qqa.callbacks", "TrajectoryTracker"),
    "SolverConfig": ("qqa.config", "SolverConfig"),
    "ISCOResult": ("qqa.isco", "ISCOResult"),
    "discrete_langevin": ("qqa.isco", "discrete_langevin"),
    "isco_anneal": ("qqa.isco", "isco_anneal"),
    "Binary": ("qqa.mixed", "Binary"),
    "BinaryVariable": ("qqa.mixed", "BinaryVariable"),
    "Constraint": ("qqa.mixed", "Constraint"),
    "Integer": ("qqa.mixed", "Integer"),
    "IntegerVariable": ("qqa.mixed", "IntegerVariable"),
    "MixedProblem": ("qqa.mixed", "MixedProblem"),
    "MixedRelaxation": ("qqa.mixed", "MixedRelaxation"),
    "Real": ("qqa.mixed", "Real"),
    "RealVariable": ("qqa.mixed", "RealVariable"),
    "RepairResult": ("qqa.mixed", "RepairResult"),
    "VariableSpace": ("qqa.mixed", "VariableSpace"),
    "repair_mixed_solution": ("qqa.mixed", "repair_mixed_solution"),
    "solve_mixed": ("qqa.mixed", "solve_mixed"),
    "ModelIR": ("qqa.model", "ModelIR"),
    "MultiObjectiveProblem": ("qqa.multiobjective", "MultiObjectiveProblem"),
    "Objective": ("qqa.multiobjective", "Objective"),
    "ParetoResult": ("qqa.multiobjective", "ParetoResult"),
    "pareto_anneal": ("qqa.multiobjective", "pareto_anneal"),
    "plot_pareto": ("qqa.multiobjective", "plot_pareto"),
    "plot_pareto_diagnostics": ("qqa.multiobjective", "plot_pareto_diagnostics"),
    "MODEL_SYSTEM_PROMPT": ("qqa.natural_language", "MODEL_SYSTEM_PROMPT"),
    "AskResult": ("qqa.natural_language", "AskResult"),
    "OptimizationPlan": ("qqa.natural_language", "OptimizationPlan"),
    "ask": ("qqa.natural_language", "ask"),
    "blackbox_from_spec": ("qqa.natural_language", "blackbox_from_spec"),
    "compile_natural_language": ("qqa.natural_language", "compile_natural_language"),
    "execute_plan": ("qqa.natural_language", "execute_plan"),
    "plan_spec": ("qqa.natural_language", "plan_spec"),
    "PAResult": ("qqa.pa", "PAResult"),
    "population_annealing": ("qqa.pa", "population_annealing"),
    "QAP": ("qqa.problems", "QAP"),
    "TSP": ("qqa.problems", "TSP"),
    "BalancedGraphPartition": ("qqa.problems", "BalancedGraphPartition"),
    "BinaryPerceptron": ("qqa.problems", "BinaryPerceptron"),
    "Coloring": ("qqa.problems", "Coloring"),
    "COProblem": ("qqa.problems", "COProblem"),
    "EdwardsAnderson": ("qqa.problems", "EdwardsAnderson"),
    "GraphBisection": ("qqa.problems", "GraphBisection"),
    "HopfieldMemory": ("qqa.problems", "HopfieldMemory"),
    "IntegerFactorizationIsing": ("qqa.problems", "IntegerFactorizationIsing"),
    "Ising1D": ("qqa.problems", "Ising1D"),
    "Knapsack": ("qqa.problems", "Knapsack"),
    "MaxClique": ("qqa.problems", "MaxClique"),
    "MaxCliqueInstance": ("qqa.problems", "MaxCliqueInstance"),
    "MaxCut": ("qqa.problems", "MaxCut"),
    "MaxCutInstance": ("qqa.problems", "MaxCutInstance"),
    "MaximumIndependentSet": ("qqa.problems", "MaximumIndependentSet"),
    "MaximumIndependentSetInstance": ("qqa.problems", "MaximumIndependentSetInstance"),
    "MaxSAT3": ("qqa.problems", "MaxSAT3"),
    "MinimumDominatingSet": ("qqa.problems", "MinimumDominatingSet"),
    "NormalizedCut": ("qqa.problems", "NormalizedCut"),
    "NormCut": ("qqa.problems", "NormCut"),
    "NQueens": ("qqa.problems", "NQueens"),
    "NumberPartitioning": ("qqa.problems", "NumberPartitioning"),
    "PSpinGlass": ("qqa.problems", "PSpinGlass"),
    "QUBOProblem": ("qqa.problems", "QUBOProblem"),
    "RandomFieldIsing": ("qqa.problems", "RandomFieldIsing"),
    "SherringtonKirkpatrick": ("qqa.problems", "SherringtonKirkpatrick"),
    "SpinProblem": ("qqa.problems", "SpinProblem"),
    "UserProblem": ("qqa.problems", "UserProblem"),
    "VertexCover": ("qqa.problems", "VertexCover"),
    "load_problem_from_file": ("qqa.problems", "load_problem_from_file"),
    "random_factorization_problems": ("qqa.problems", "random_factorization_problems"),
    "random_prime": ("qqa.problems", "random_prime"),
    "random_semiprime": ("qqa.problems", "random_semiprime"),
    "user_problem_from_source": ("qqa.problems", "user_problem_from_source"),
    "BinaryInstanceRelaxation": ("qqa.relaxation", "BinaryInstanceRelaxation"),
    "BinaryRelaxation": ("qqa.relaxation", "BinaryRelaxation"),
    "CategoricalRelaxation": ("qqa.relaxation", "CategoricalRelaxation"),
    "EntropicCategoricalRelaxation": ("qqa.relaxation", "EntropicCategoricalRelaxation"),
    "MirrorDescentCategoricalRelaxation": (
        "qqa.relaxation",
        "MirrorDescentCategoricalRelaxation",
    ),
    "Relaxation": ("qqa.relaxation", "Relaxation"),
    "SinkhornRelaxation": ("qqa.relaxation", "SinkhornRelaxation"),
    "SoftmaxCategoricalRelaxation": ("qqa.relaxation", "SoftmaxCategoricalRelaxation"),
    "SparseCategoricalRelaxation": ("qqa.relaxation", "SparseCategoricalRelaxation"),
    "SpinRelaxation": ("qqa.relaxation", "SpinRelaxation"),
    "StochasticBinaryRelaxation": ("qqa.relaxation", "StochasticBinaryRelaxation"),
    "StraightThroughBinaryRelaxation": (
        "qqa.relaxation",
        "StraightThroughBinaryRelaxation",
    ),
    "save_html_report": ("qqa.reporting", "save_html_report"),
    "FeasibilityStatus": ("qqa.result", "FeasibilityStatus"),
    "GuaranteeLevel": ("qqa.result", "GuaranteeLevel"),
    "SolveResult": ("qqa.result", "SolveResult"),
    "SolveStatus": ("qqa.result", "SolveStatus"),
    "SAResult": ("qqa.sa", "SAResult"),
    "simulated_annealing": ("qqa.sa", "simulated_annealing"),
    "LinearBGSchedule": ("qqa.schedule", "LinearBGSchedule"),
    "SessionState": ("qqa.session", "SessionState"),
    "SolveSession": ("qqa.session", "SolveSession"),
    "available_templates": ("qqa.templates", "available_templates"),
    "build_template": ("qqa.templates", "build_template"),
    "TEX_SYSTEM_PROMPT": ("qqa.tex", "TEX_SYSTEM_PROMPT"),
    "LLMAPIError": ("qqa.tex", "LLMAPIError"),
    "ModelSpec": ("qqa.tex", "ModelSpec"),
    "OpenAICompatibleClient": ("qqa.tex", "OpenAICompatibleClient"),
    "TexSolveResult": ("qqa.tex", "TexSolveResult"),
    "compile_tex": ("qqa.tex", "compile_tex"),
    "problem_from_spec": ("qqa.tex", "problem_from_spec"),
    "solve_tex": ("qqa.tex", "solve_tex"),
    "tex": ("qqa.tex", None),
    "enable_tf32": ("qqa.utils", "enable_tf32"),
    "fix_seed": ("qqa.utils", "fix_seed"),
    "generate_graph": ("qqa.utils", "generate_graph"),
    "resolve_device": ("qqa.utils", "resolve_device"),
    "polish": ("qqa.polish", None),
    "warmstart": ("qqa.warmstart", None),
}

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
    """Resolve public and backwards-compatible exports on explicit access."""
    target: tuple[str, str | None] | None = _EXPORTS.get(name)
    if target is None:
        target = _OPTIONAL_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_OPTIONAL_EXPORTS})


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
    "Study",
    "Trial",
    "TrialState",
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
    "SolveSession",
    "SessionState",
    "SolveStatus",
    "GuaranteeLevel",
    "FeasibilityStatus",
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
    "available_templates",
    "blackbox_from_spec",
    "blackbox_optimize",
    "create_study",
    "build_application",
    "build_microgrid_dispatch",
    "build_microgrid_pareto",
    "build_portfolio_pareto",
    "build_process_blackbox",
    "build_template",
    "compile_tex",
    "compile_natural_language",
    "discrete_langevin",
    "doctor",
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
    "tex",
    "solve_mixed",
    "user_problem_from_source",
    "warmstart",
]
