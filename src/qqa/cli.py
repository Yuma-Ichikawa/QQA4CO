"""Command-line interface for QQA.

Exposed as the ``qqa`` console script via ``[project.scripts]`` in
``pyproject.toml``. Subcommands:

* ``qqa version`` — print the installed version.
* ``qqa solve`` — solve a single problem from the CLI.
* ``qqa ask`` — describe an optimisation problem in natural language.
* ``qqa bench`` — run a quick benchmark on a bundled dataset.
* ``qqa gui`` — launch the Streamlit GUI in a subprocess.

The CLI uses only :mod:`argparse` and standard library modules so it has no
extra runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

from qqa.commands.runtime import (
    command_version as _cmd_version,
)
from qqa.commands.runtime import (
    print_score as _print_score,
)
from qqa.commands.runtime import (
    resolve_device as _resolve_device,
)
from qqa.commands.system import command_doctor as _cmd_doctor
from qqa.commands.system import command_gui as _cmd_gui

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qqa",
        description="Quasi-Quantum Annealing command-line interface.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("version", help="Show the installed qqa version.")

    inspect_command = sub.add_parser("inspect", help="Inspect a model without solving it.")
    inspect_command.add_argument("model", help="Path to a supported model file.")

    plan_command = sub.add_parser("plan", help="Preview the QQA-centred solver plan.")
    plan_command.add_argument("model", help="Path to a supported model file.")
    plan_command.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality", "certify", "diverse", "pareto", "reproducible"],
        default="balanced",
    )
    plan_command.add_argument("--budget", type=float, default=None)
    plan_command.add_argument("--device", default="auto")

    solve = sub.add_parser("solve", help="Solve a single problem.")
    solve.add_argument(
        "model",
        nargs="?",
        help=(
            "Portable model file (MPS, LP, QPLIB, JSON ModelIR, OPB, CNF/WCNF, "
            "QUBO, or Ising). Omit to use the legacy --problem catalog."
        ),
    )
    solve.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality", "certify", "diverse", "pareto", "reproducible"],
        default="balanced",
    )
    solve.add_argument(
        "--budget", type=float, default=None, help="Total wall-clock budget in seconds."
    )
    solve.add_argument(
        "--problem",
        required=False,
        default=None,
        choices=[
            "mis",
            "maxcut",
            "maxclique",
            "coloring",
            "ising1d",
            "ea",
            "sk",
            "perceptron",
            "hopfield",
            # Phase-A problems (added in v0.3).
            "knapsack",
            "number_partition",
            "vertex_cover",
            "graph_bisection",
            "maxsat3",
            "tsp",
            "qap",
            "nqueens",
            # Phase-B catalog growth (v0.5+).
            "bgp",
            "min_dominating_set",
            # Physics catalog growth (v0.5+).
            "pspin",
            "rfim",
        ],
        help="Problem family. Mutually exclusive with --problem-file.",
    )
    solve.add_argument(
        "--problem-file",
        type=str,
        default=None,
        help=(
            "Path to a Python file that defines `problem` (a qqa.COProblem) "
            "or a `make_problem()` factory. Lets you plug in arbitrary "
            "user-defined problems."
        ),
    )
    solve.add_argument("--graph-file", type=str, default=None, help="Pickled NetworkX graph path.")
    solve.add_argument(
        "--size", type=int, default=50, help="Problem size (for synthetic problems)."
    )
    solve.add_argument("--dim", type=int, default=3, help="Lattice dimension (EA / RFIM).")
    solve.add_argument(
        "--p-order",
        type=int,
        default=3,
        help="Interaction order p for the dense p-spin glass (>= 2; default 3).",
    )
    solve.add_argument(
        "--h-std",
        type=float,
        default=1.0,
        help="Random-field standard deviation σ_h for RFIM.",
    )
    solve.add_argument(
        "--coupling-J",
        type=float,
        default=1.0,
        help="Uniform ferromagnetic coupling J for RFIM.",
    )
    solve.add_argument(
        "--alpha", type=float, default=0.5, help="Loading ratio (perceptron/Hopfield)."
    )
    solve.add_argument(
        "--patterns", type=int, default=1, help="Number of stored patterns (Hopfield)."
    )
    solve.add_argument("--num-category", type=int, default=3, help="Number of colours (coloring).")
    solve.add_argument(
        "--sol-size",
        type=int,
        default=None,
        help=(
            "Parallel population size. Positional model files use the selected "
            "profile when omitted; the legacy problem catalogue defaults to 100."
        ),
    )
    solve.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Number of solver steps. Positional model files use the selected "
            "profile when omitted; the legacy problem catalogue defaults to 1000."
        ),
    )
    solve.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Learning rate. When omitted, defaults to 1.0 for the qqa "
            "annealer (large LR works because BG normalises gradients) and "
            "to 1e-4 for the pignn / cpra backends (matching the CRA paper)."
        ),
    )
    solve.add_argument("--temp", type=float, default=None)
    solve.add_argument("--min-bg", type=float, default=None)
    solve.add_argument("--max-bg", type=float, default=None)
    solve.add_argument("--curve-rate", type=int, default=None)
    solve.add_argument("--div-param", type=float, default=None)
    solve.add_argument(
        "--schedule",
        choices=[
            "linear",
            "cosine",
            "exponential",
            "sigmoid",
            "polynomial",
            "cyclic",
            "reheat",
            "adaptive",
        ],
        default=None,
        help=(
            "QQA discretisation schedule. Positional models inherit the profile; "
            "the legacy problem catalogue keeps its linear default."
        ),
    )
    solve.add_argument(
        "--restart-patience",
        type=int,
        default=None,
        help=(
            "Restart weak QQA replicas after this many stagnant epochs; "
            "0 disables adaptive basin recovery."
        ),
    )
    solve.add_argument(
        "--restart-fraction",
        type=float,
        default=None,
        help="Fraction of weak replicas restarted after stagnation.",
    )
    solve.add_argument(
        "--restart-jitter",
        type=float,
        default=None,
        help="Local latent-space jitter around the incumbent during restarts.",
    )
    solve.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help="QQA latent-gradient norm cap; pass 0 to disable.",
    )
    solve.add_argument(
        "--optimizer",
        choices=["adamw", "lightweight-adamw", "mirror-descent"],
        default="adamw",
        help=(
            "QQA latent optimizer. mirror-descent requires an explicit "
            "MirrorDescentCategoricalRelaxation."
        ),
    )
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device; auto selects CUDA, then MPS, then CPU.",
    )
    solve.add_argument("--quiet", action="store_true", help="Suppress per-epoch logs.")
    solve.add_argument(
        "--no-polish",
        action="store_true",
        help=(
            "Disable the default greedy 1-flip polish applied to QUBO winners "
            "(only affects compatible backends). Use this for a strictly raw "
            "solver comparison."
        ),
    )
    solve.add_argument(
        "--output",
        type=str,
        default=None,
        help="If given, save the AnnealResult (pickle) to this path.",
    )
    solve.add_argument(
        "--report",
        type=str,
        default=None,
        help="If given, write a self-contained interactive HTML result report.",
    )
    # ------------------------------------------------------------------
    # Optional CRA-PI-GNN backend (PyTorch Geometric). Requires installing
    # the ``pignn`` extra: ``pip install qqa[pignn]``. Defaults to ``qqa``,
    # so users without the extra are completely unaffected.
    # ------------------------------------------------------------------
    solve.add_argument(
        "--backend",
        choices=["qqa", "scip", "pignn", "cpra", "sa", "pa", "isco"],
        default="qqa",
        help=(
            "Solver backend. 'qqa' (default) uses the parallel-replica "
            "annealing loop; 'scip' refines QQA QUBO solutions and can prove "
            "optimality (requires qqa[scip]); 'pignn' uses the optional CRA-PI-GNN "
            "(PyTorch Geometric) trainer — graph problems only "
            "(mis/maxcut/maxclique/vertex_cover/graph_bisection); "
            "'cpra' uses the multi-head CPRA extension that produces "
            "diverse solutions in one run (same graph problems); "
            "'sa' is the GPU-parallel Simulated Annealing baseline used in "
            "the QQA papers — no learning, pure Metropolis on the same problem; "
            "'pa' and 'isco' expose the other sampling baselines."
        ),
    )
    solve.add_argument(
        "--exact-backend",
        choices=["auto", "none", "scip", "highs", "cpsat", "cuopt"],
        default="auto",
        help=(
            "Optional certification/completion backend for positional model files. "
            "The default remains pure QQA unless profile=certify."
        ),
    )
    solve.add_argument(
        "--scip-time-limit",
        type=float,
        default=60.0,
        help="Total QQA+SCIP wall-clock limit in seconds (only --backend scip).",
    )
    solve.add_argument(
        "--scip-gap",
        type=float,
        default=0.0,
        help="Target relative optimality gap (only --backend scip).",
    )
    solve.add_argument(
        "--scip-warm-starts",
        type=int,
        default=32,
        help="Maximum diverse QQA primal starts passed to SCIP.",
    )
    solve.add_argument(
        "--scip-threads",
        type=int,
        default=1,
        help="Maximum SCIP threads (QQA GPU exploration is unaffected).",
    )
    solve.add_argument(
        "--sa-num-sweeps",
        type=int,
        default=None,
        help=(
            "Number of SA sweeps (only used when --backend sa). Defaults to "
            "--epochs so SA and QQA can be compared on equal compute budgets."
        ),
    )
    solve.add_argument(
        "--sa-beta-start",
        type=float,
        default=0.1,
        help="SA initial inverse temperature (only used when --backend sa).",
    )
    solve.add_argument(
        "--sa-beta-end",
        type=float,
        default=10.0,
        help="SA final inverse temperature (only used when --backend sa).",
    )
    solve.add_argument(
        "--sa-schedule",
        choices=["geometric", "linear"],
        default="geometric",
        help="SA beta schedule shape (only used when --backend sa).",
    )
    solve.add_argument(
        "--pignn-init-reg-param",
        type=float,
        default=-20.0,
        help="CRA initial gamma (only used when --backend pignn).",
    )
    solve.add_argument(
        "--pignn-annealing-rate",
        type=float,
        default=1e-3,
        help="CRA gamma increment per epoch (only used when --backend pignn).",
    )
    solve.add_argument(
        "--pignn-tol",
        type=float,
        default=1e-4,
        help="Early-stopping tolerance (only used when --backend pignn).",
    )
    solve.add_argument(
        "--pignn-patience",
        type=int,
        default=1000,
        help="Early-stopping patience in epochs (only used when --backend pignn).",
    )
    solve.add_argument(
        "--pignn-hidden",
        type=int,
        default=None,
        help=(
            "Hidden width of the GCN (only used when --backend pignn). Defaults to floor(sqrt(N))."
        ),
    )
    solve.add_argument(
        "--pignn-no-annealing",
        action="store_true",
        help=(
            "When set with --backend pignn or --backend cpra, runs vanilla "
            "PI-GNN (reg_param fixed at 0) instead of CRA-style annealing."
        ),
    )
    solve.add_argument(
        "--cpra-num-replicas",
        type=int,
        default=4,
        help=("Number of parallel CPRA replicas R (only used when --backend cpra)."),
    )
    solve.add_argument(
        "--cpra-vari-param",
        type=float,
        default=0.0,
        help=(
            "CPRA diversity-term coefficient (only used when --backend cpra). "
            "Positive values reward inter-replica spread for variation "
            "diversification on a fixed problem."
        ),
    )
    solve.add_argument(
        "--cpra-penalty-levels",
        type=str,
        default=None,
        help=(
            "Comma-separated list of penalty weights — one per replica — for "
            "penalty diversification (only used when --backend cpra and "
            "--problem in {mis, vertex_cover}). When set, length must equal "
            "--cpra-num-replicas. Example: '1.0,1.5,2.0,2.5'."
        ),
    )

    bench = sub.add_parser("bench", help="Run a small benchmark on bundled data.")
    bench.add_argument("--preset", choices=["er-small", "sk-small", "ea-small"], default="er-small")
    bench.add_argument("--sol-size", type=int, default=64)
    bench.add_argument("--epochs", type=int, default=500)
    bench.add_argument("--device", type=str, default="cpu")
    bench.add_argument("--seed", type=int, default=0)

    # Suite-level benchmark commands (delegate to scripts/bench_discs.py
    # and scripts/plot_benchmarks.py via ``qqa.bench``). These expose the
    # same one-command setup + run + visualise flow recommended for
    # third-party comparisons.
    bench_list = sub.add_parser(
        "bench-list",
        help="List every benchmark suite reachable from the current ./data tree.",
    )
    bench_list.add_argument(
        "--as-suites",
        action="store_true",
        help="Print resolved suite ids (e.g. 'mis-satlib-uf') instead of a nested tree.",
    )

    bench_run = sub.add_parser(
        "bench-run",
        help="Run a benchmark suite and write JSON results.",
        description=(
            "Wrapper around scripts/bench_discs.py. Unknown options (e.g. "
            "--learning-rate / --temp / --curve-rate ...) are forwarded "
            "verbatim so every PQQA hyperparameter remains tunable."
        ),
    )
    bench_run.add_argument("--suite", default="all", help="suite identifier")
    bench_run.add_argument("--backend", default="qqa", choices=("qqa", "sa", "pa"))
    bench_run.add_argument("--instances", type=int, default=None)
    bench_run.add_argument("--sol-size", type=int, default=20)
    bench_run.add_argument("--num-epochs", type=int, default=500)
    bench_run.add_argument("--device", default="auto")
    bench_run.add_argument("--seed", type=int, default=0)
    bench_run.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="JSON output. Relative paths are resolved under ./bench_results/.",
    )
    bench_run.add_argument("--parallel", action="store_true")
    bench_run.add_argument("--penalty", type=float, default=None)

    bench_plot = sub.add_parser(
        "bench-plot",
        help="Render a polished benchmark-report image from results JSON.",
    )
    bench_plot.add_argument("results", nargs="+")
    bench_plot.add_argument("--labels", nargs="+", default=None)
    bench_plot.add_argument(
        "--output",
        type=str,
        default="report.png",
        help="output image path. Relative paths go under ./bench_results/.",
    )
    bench_plot.add_argument("--title", default=None)
    bench_plot.add_argument("--theme", default="light", choices=("light", "dark"))
    bench_plot.add_argument("--dpi", type=int, default=160)
    bench_plot.add_argument("--format", default=None, dest="fmt")

    from qqa.benchmarking.cli import add_benchmark_parser

    add_benchmark_parser(sub)

    tex = sub.add_parser(
        "tex",
        help="Translate a TeX optimisation model and solve it automatically.",
    )
    tex.add_argument(
        "tex",
        nargs="?",
        help="TeX string, or '-' to read TeX from stdin. Quote backslashes in your shell.",
    )
    tex.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Solve an already-audited QQA model JSON file without calling an API.",
    )
    tex.add_argument(
        "--file",
        type=str,
        default=None,
        help="Read a TeX model from a UTF-8 .tex file and translate it through the API.",
    )
    tex.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (or QQA_LLM_BASE_URL).",
    )
    tex.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id used for TeX translation (or QQA_LLM_MODEL).",
    )
    tex.add_argument(
        "--api-style",
        choices=("responses", "messages"),
        default="responses",
        help="Compatible endpoint style (default: responses).",
    )
    tex.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for a private development gateway.",
    )
    tex.add_argument("--timeout", type=float, default=120.0)
    tex.add_argument("--sol-size", type=int, default=256)
    tex.add_argument("--epochs", type=int, default=1500)
    tex.add_argument("--device", type=str, default="cpu")
    tex.add_argument("--seed", type=int, default=0)
    tex.add_argument(
        "--solver",
        choices=("auto", "qqa", "scip"),
        default="auto",
        help=(
            "Numerical solver. auto is pure QQA; SCIP refinement is opt-in via "
            "--solver scip and requires qqa[scip]."
        ),
    )
    tex.add_argument("--scip-time-limit", type=float, default=60.0)
    tex.add_argument("--scip-gap", type=float, default=0.0)
    tex.add_argument("--scip-threads", type=int, default=1)
    tex.add_argument("--scip-warm-starts", type=int, default=32)
    tex.add_argument(
        "--dry-run",
        action="store_true",
        help="Translate and validate the model but do not solve it.",
    )
    tex.add_argument(
        "--output-model",
        type=str,
        default=None,
        help="Write the audited declarative model JSON (never includes credentials).",
    )
    tex.add_argument(
        "--output-result",
        type=str,
        default=None,
        help="Write complete solution/front JSON instead of flooding stdout.",
    )
    tex.add_argument("--report", type=str, default=None, help="Write an interactive HTML report.")
    tex.add_argument(
        "--show-model",
        action="store_true",
        help="Print the validated declarative model before solving.",
    )
    tex.add_argument("--quiet", action="store_true")

    ask = sub.add_parser(
        "ask",
        help="Describe, route, and solve an optimisation problem in natural language.",
        description=(
            "Compile natural language into an audited model, select QQA, QQA+SCIP, "
            "parallel Pareto, or black-box optimisation locally, and solve it."
        ),
    )
    ask.add_argument(
        "prompt",
        nargs="?",
        help="Natural-language optimisation request, or '-' to read it from stdin.",
    )
    ask.add_argument("--file", help="Read a UTF-8 natural-language request from a file.")
    ask.add_argument(
        "--spec",
        help="Use an already-reviewed model JSON without an LLM request or API key.",
    )
    ask.add_argument(
        "--solver",
        choices=("auto", "qqa", "hybrid", "qqa-scip", "scip", "pareto", "blackbox"),
        default="auto",
        help="Workflow override. auto routes from the validated model and request intent.",
    )
    ask.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible API base URL (or QQA_LLM_BASE_URL).",
    )
    ask.add_argument(
        "--model",
        default=None,
        help="Model id used for request compilation (or QQA_LLM_MODEL).",
    )
    ask.add_argument(
        "--api-style",
        choices=("responses", "messages"),
        default="responses",
    )
    ask.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification only for a trusted private development gateway.",
    )
    ask.add_argument("--timeout", type=float, default=120.0)
    ask.add_argument("--device", default="auto")
    ask.add_argument("--seed", type=int, default=0)
    ask.add_argument("--sol-size", type=int, default=256)
    ask.add_argument("--epochs", type=int, default=1500)
    ask.add_argument("--budget", type=int, default=96)
    ask.add_argument("--batch-size", type=int, default=8)
    ask.add_argument("--workers", type=int, default=4)
    ask.add_argument("--scip-time-limit", type=float, default=60.0)
    ask.add_argument("--scip-gap", type=float, default=0.0)
    ask.add_argument("--scip-threads", type=int, default=1)
    ask.add_argument("--scip-warm-starts", type=int, default=32)
    ask.add_argument(
        "--plan-only",
        action="store_true",
        help="Compile, validate, and explain the route without running a solver.",
    )
    ask.add_argument("--show-model", action="store_true")
    ask.add_argument("--json", action="store_true", help="Print the result as JSON.")
    ask.add_argument("--output-plan", help="Write the audited model and routing decision as JSON.")
    ask.add_argument("--output-result", help="Write the numerical result as JSON.")
    ask.add_argument("--report", help="Write an interactive HTML result report.")
    ask.add_argument("--quiet", action="store_true")

    example = sub.add_parser(
        "example",
        help="List or run realistic mixed, Pareto, and black-box applications.",
    )
    example.add_argument("action", choices=("list", "run"), nargs="?", default="list")
    example.add_argument(
        "name",
        nargs="?",
        choices=(
            "microgrid-dispatch",
            "microgrid-pareto",
            "portfolio-pareto",
            "process-blackbox",
        ),
    )
    example.add_argument("--device", default="auto")
    example.add_argument("--sol-size", type=int, default=512)
    example.add_argument("--epochs", type=int, default=1200)
    example.add_argument("--budget", type=int, default=96)
    example.add_argument("--batch-size", type=int, default=8)
    example.add_argument("--workers", type=int, default=4)
    example.add_argument("--seed", type=int, default=0)
    example.add_argument(
        "--output-dir",
        default=None,
        help="Write JSON, CSV, and an interactive HTML report into this directory.",
    )
    example.add_argument("--quiet", action="store_true")

    doctor = sub.add_parser("doctor", help="Check devices and optional solver/UI capabilities.")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    gui = sub.add_parser("gui", help="Launch the Streamlit GUI.")
    gui.add_argument("--port", type=int, default=8501)
    gui.add_argument("--host", type=str, default="localhost")
    gui.add_argument("--headless", action="store_true")

    return parser


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _build_problem(args: argparse.Namespace):
    import networkx as nx

    import qqa

    qqa.fix_seed(args.seed)

    if getattr(args, "problem_file", None):
        if args.problem:
            raise SystemExit("[qqa solve] pass either --problem or --problem-file, not both.")
        return qqa.load_problem_from_file(args.problem_file)

    kind = args.problem
    if kind is None:
        raise SystemExit("[qqa solve] specify either --problem <name> or --problem-file <path>.")
    device = args.device

    if kind in {"mis", "maxcut", "maxclique", "coloring"}:
        if args.graph_file is not None:
            g_path = Path(args.graph_file).expanduser().resolve()
            suffix = g_path.suffix.lower()
            if suffix in {".gpickle", ".pkl", ".pickle"}:
                with open(g_path, "rb") as fh:
                    g = pickle.load(fh)
            elif suffix in {".graphml", ".xml"}:
                g = nx.read_graphml(g_path)
            elif suffix in {".edgelist", ".txt"}:
                g = nx.read_edgelist(g_path, nodetype=int)
            else:
                raise ValueError(f"Unsupported graph extension {suffix!r}.")
        else:
            g = nx.random_regular_graph(d=3, n=args.size, seed=args.seed)
        if kind == "mis":
            return qqa.MaximumIndependentSet(g, device=device)
        if kind == "maxcut":
            return qqa.MaxCut(g, device=device)
        if kind == "maxclique":
            return qqa.MaxClique(g, device=device)
        if kind == "coloring":
            return qqa.Coloring(g, num_category=args.num_category, device=device)

    if kind == "ising1d":
        return qqa.Ising1D(N=args.size, device=device)
    if kind == "ea":
        return qqa.EdwardsAnderson(L=args.size, dim=args.dim, seed=args.seed, device=device)
    if kind == "sk":
        return qqa.SherringtonKirkpatrick(N=args.size, seed=args.seed, device=device)
    if kind == "pspin":
        return qqa.PSpinGlass(N=args.size, p=args.p_order, seed=args.seed, device=device)
    if kind == "rfim":
        return qqa.RandomFieldIsing(
            L=args.size,
            dim=args.dim,
            J=args.coupling_J,
            h_std=args.h_std,
            seed=args.seed,
            device=device,
        )
    if kind == "perceptron":
        return qqa.BinaryPerceptron(N=args.size, alpha=args.alpha, seed=args.seed, device=device)
    if kind == "hopfield":
        return qqa.HopfieldMemory(
            N=args.size, patterns=args.patterns, seed=args.seed, device=device
        )

    # Phase-A problems (added in v0.3).
    if kind == "knapsack":
        return qqa.Knapsack(N=args.size, seed=args.seed, device=device)
    if kind == "number_partition":
        return qqa.NumberPartitioning(N=args.size, seed=args.seed, device=device)
    if kind == "vertex_cover":
        g = nx.random_regular_graph(d=3, n=args.size, seed=args.seed)
        return qqa.VertexCover(g, device=device)
    if kind == "graph_bisection":
        g = nx.random_regular_graph(d=3, n=args.size, seed=args.seed)
        return qqa.GraphBisection(g, device=device)
    if kind == "maxsat3":
        return qqa.MaxSAT3(N=args.size, seed=args.seed, device=device)
    if kind == "tsp":
        return qqa.TSP(N=args.size, seed=args.seed, device=device)
    if kind == "qap":
        return qqa.QAP(N=args.size, seed=args.seed, device=device)
    if kind == "nqueens":
        return qqa.NQueens(N=args.size, device=device)
    if kind == "bgp":
        g = nx.random_regular_graph(d=3, n=args.size, seed=args.seed)
        return qqa.BalancedGraphPartition(g, num_category=args.num_category, device=device)
    if kind == "min_dominating_set":
        g = nx.random_regular_graph(d=3, n=args.size, seed=args.seed)
        return qqa.MinimumDominatingSet(g, device=device)

    raise ValueError(f"Unknown problem kind {kind!r}.")


_PIGNN_SUPPORTED_KINDS = {
    "mis",
    "maxcut",
    "maxclique",
    "vertex_cover",
    "graph_bisection",
}

# Subset of pignn-supported kinds whose constructor accepts a ``penalty``
# kwarg, so we can build per-replica problems for CPRA penalty diversification.
_CPRA_PENALTY_KINDS = {"mis", "vertex_cover"}


def _build_replica_problems(args: argparse.Namespace, base_problem) -> list | None:
    """Return per-replica problems for CPRA penalty diversification, or None."""
    raw = getattr(args, "cpra_penalty_levels", None)
    if raw is None:
        return None
    kind = args.problem
    if kind not in _CPRA_PENALTY_KINDS:
        raise SystemExit(
            f"[qqa solve] --cpra-penalty-levels currently supports "
            f"{sorted(_CPRA_PENALTY_KINDS)}; got --problem {kind!r}."
        )
    try:
        levels = [float(x) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(
            f"[qqa solve] --cpra-penalty-levels must be a comma-separated list "
            f"of floats, got {raw!r}: {exc}"
        ) from exc
    if len(levels) != int(args.cpra_num_replicas):
        raise SystemExit(
            f"[qqa solve] --cpra-penalty-levels has {len(levels)} values but "
            f"--cpra-num-replicas is {args.cpra_num_replicas}; lengths must match."
        )
    import qqa
    from qqa.pignn.graph import extract_nx_graph

    try:
        g = extract_nx_graph(base_problem)
    except TypeError as exc:
        raise SystemExit(f"[qqa solve] --cpra-penalty-levels: {exc}") from exc
    replicas: list = []
    for p in levels:
        if kind == "mis":
            replicas.append(qqa.MaximumIndependentSet(g, penalty=p, device=args.device))
        elif kind == "vertex_cover":
            replicas.append(qqa.VertexCover(g, penalty=p, device=args.device))
    return replicas


def _cmd_solve(args: argparse.Namespace) -> int:
    import qqa

    if args.model is not None:
        backend = getattr(args, "backend", "qqa")
        if backend in {"pignn", "cpra"}:
            raise SystemExit(f"[qqa solve] --backend {backend} does not accept model files.")
        exact_backend = args.exact_backend
        if backend == "scip":
            if exact_backend not in {"auto", "scip"}:
                raise SystemExit("[qqa solve] --backend scip conflicts with --exact-backend.")
            exact_backend = "scip"
            backend = "qqa"
        option_map = {
            "sol_size": "replicas",
            "epochs": "epochs",
            "learning_rate": "learning_rate",
            "temp": "temperature",
            "schedule": "schedule",
            "min_bg": "min_bg",
            "max_bg": "max_bg",
            "curve_rate": "curve_rate",
            "div_param": "diversity",
            "restart_patience": "restart_patience",
            "restart_fraction": "restart_fraction",
            "restart_jitter": "restart_jitter",
            "optimizer": "optimizer",
        }
        overrides = {
            config_name: value
            for argument_name, config_name in option_map.items()
            if (value := getattr(args, argument_name, None)) is not None
        }
        if args.gradient_clip is not None:
            overrides["gradient_clip_norm"] = args.gradient_clip or None
        if args.no_polish:
            overrides["polish"] = False
        result = qqa.solve(
            args.model,
            profile=args.profile,
            budget=args.budget,
            device=args.device,
            seed=args.seed,
            backend=backend,
            exact_backend=exact_backend,
            **overrides,
        )
        print(result.plan.explain())
        print(f"status: {result.status.value}")
        print(f"objective: {result.best_obj}")
        print(f"feasible: {result.feasible}")
        print(f"runtime: {result.runtime:.6g} s")
        if result.best_bound is not None:
            print(f"best bound: {result.best_bound}")
        if result.relative_gap is not None:
            print(f"relative gap: {result.relative_gap}")
        return 0

    # The built-in catalogue predates profiles. Preserve its public defaults
    # while allowing positional model files to inherit profile settings.
    legacy_defaults = {
        "sol_size": 100,
        "epochs": 1000,
        "temp": 0.0,
        "min_bg": -2.0,
        "max_bg": 0.1,
        "curve_rate": 2,
        "div_param": 0.0,
        "restart_patience": 250,
        "restart_fraction": 0.15,
        "restart_jitter": 0.10,
        "gradient_clip": 100.0,
    }
    for name, default in legacy_defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, default)

    args.device = _resolve_device(args.device)
    problem = _build_problem(args)
    qqa_schedule = None
    if args.schedule is not None:
        from qqa.schedule import make_schedule

        qqa_schedule = make_schedule(args.schedule, minimum=args.min_bg, maximum=args.max_bg)

    backend = getattr(args, "backend", "qqa")
    if args.exact_backend not in {"auto", "none"} and backend != "scip":
        raise SystemExit("[qqa solve] --exact-backend is supported for positional model files.")
    if backend == "scip":
        from qqa.hybrid import solve_qqa_scip

        qqa_lr = args.learning_rate if args.learning_rate is not None else 1.0
        try:
            result = solve_qqa_scip(
                problem,
                qqa_kwargs={
                    "sol_size": args.sol_size,
                    "learning_rate": qqa_lr,
                    "temp": args.temp,
                    "min_bg": args.min_bg,
                    "max_bg": args.max_bg,
                    "curve_rate": args.curve_rate,
                    "div_param": args.div_param,
                    "num_epochs": args.epochs,
                    "device": args.device,
                    "polish": not args.no_polish,
                    "restart_patience": args.restart_patience or None,
                    "restart_fraction": args.restart_fraction,
                    "restart_jitter": args.restart_jitter,
                    "gradient_clip_norm": args.gradient_clip or None,
                    "optimizer": args.optimizer,
                    "verbose": not args.quiet,
                    **({"schedule": qqa_schedule} if qqa_schedule is not None else {}),
                },
                time_limit=args.scip_time_limit,
                relative_gap=args.scip_gap,
                max_warm_starts=args.scip_warm_starts,
                threads=args.scip_threads,
                verbose=not args.quiet,
            )
        except TypeError as exc:
            raise SystemExit(f"[qqa solve] --backend scip requires a QUBO problem: {exc}") from exc
    elif backend in {"pignn", "cpra"}:
        kind = args.problem
        if kind is None:
            raise SystemExit(
                f"[qqa solve] --backend {backend} requires a built-in --problem "
                f"(one of {sorted(_PIGNN_SUPPORTED_KINDS)}); --problem-file "
                "is not yet supported by the PyG-based backends."
            )
        if kind not in _PIGNN_SUPPORTED_KINDS:
            raise SystemExit(
                f"[qqa solve] --backend {backend} only supports graph-based "
                f"problems {sorted(_PIGNN_SUPPORTED_KINDS)}; got {kind!r}. "
                "Use the default --backend qqa for the rest."
            )

        # Resolve learning rate with backend-aware default. ``None`` (no
        # --learning-rate given) falls back to 1e-4 for the PyG trainers,
        # matching the CRA / CPRA paper defaults.
        pignn_lr = args.learning_rate if args.learning_rate is not None else 1e-4

        if backend == "pignn":
            from qqa.pignn import train_cra_pi_gnn  # lazy: surfaces clear msg if PyG missing

            result = train_cra_pi_gnn(
                problem,
                hidden_dim=args.pignn_hidden,
                learning_rate=pignn_lr,
                annealing=not args.pignn_no_annealing,
                init_reg_param=args.pignn_init_reg_param,
                annealing_rate=args.pignn_annealing_rate,
                curve_rate=args.curve_rate,
                num_epochs=args.epochs,
                tol=args.pignn_tol,
                patience=args.pignn_patience,
                device=args.device,
                seed=args.seed,
                verbose=not args.quiet,
            )
        else:  # backend == "cpra"
            from qqa.pignn import train_cpra_pi_gnn

            replica_problems = _build_replica_problems(args, problem)
            result = train_cpra_pi_gnn(
                problem,
                num_replicas=args.cpra_num_replicas,
                replica_problems=replica_problems,
                vari_param=args.cpra_vari_param,
                hidden_dim=args.pignn_hidden,
                learning_rate=pignn_lr,
                annealing=not args.pignn_no_annealing,
                init_reg_param=args.pignn_init_reg_param,
                annealing_rate=args.pignn_annealing_rate,
                curve_rate=args.curve_rate,
                num_epochs=args.epochs,
                tol=args.pignn_tol,
                patience=args.pignn_patience,
                device=args.device,
                seed=args.seed,
                verbose=not args.quiet,
            )
    elif backend == "sa":
        # SA shares no hyper-parameters with QQA — keep its own knobs.
        sa_sweeps = args.sa_num_sweeps if args.sa_num_sweeps is not None else args.epochs
        result = qqa.simulated_annealing(
            problem,
            sol_size=args.sol_size,
            num_sweeps=sa_sweeps,
            beta_schedule=args.sa_schedule,
            beta_start=args.sa_beta_start,
            beta_end=args.sa_beta_end,
            seed=args.seed,
            device=args.device,
            verbose=not args.quiet,
        )
    elif backend == "pa":
        temperatures = max(2, int(args.epochs**0.5))
        sweeps = max(1, (args.epochs + temperatures - 1) // temperatures)
        result = qqa.population_annealing(
            problem,
            sol_size=args.sol_size,
            num_temps=temperatures,
            sweeps_per_temp=sweeps,
            seed=args.seed,
            device=args.device,
            polish=not args.no_polish,
            verbose=not args.quiet,
        )
    elif backend == "isco":
        result = qqa.discrete_langevin(
            problem,
            sol_size=args.sol_size,
            num_steps=args.epochs,
            seed=args.seed,
            device=args.device,
            polish=not args.no_polish,
            verbose=not args.quiet,
        )
    else:
        default_lr = 0.05 if isinstance(problem, qqa.MixedProblem) else 1.0
        qqa_lr = args.learning_rate if args.learning_rate is not None else default_lr
        solver_kwargs = {
            "sol_size": args.sol_size,
            "learning_rate": qqa_lr,
            "temp": args.temp,
            "min_bg": args.min_bg,
            "max_bg": args.max_bg,
            "curve_rate": args.curve_rate,
            "div_param": args.div_param,
            "num_epochs": args.epochs,
            "device": args.device,
            "polish": not args.no_polish,
            "restart_patience": args.restart_patience or None,
            "restart_fraction": args.restart_fraction,
            "restart_jitter": args.restart_jitter,
            "gradient_clip_norm": args.gradient_clip or None,
            "optimizer": args.optimizer,
            "verbose": not args.quiet,
        }
        if qqa_schedule is not None:
            solver_kwargs["schedule"] = qqa_schedule
        if isinstance(problem, qqa.MixedProblem):
            result = problem.solve(**solver_kwargs)
        else:
            result = qqa.anneal(problem, **solver_kwargs)
    print("")
    label = args.problem or f"file:{args.problem_file}"
    size = (
        getattr(problem, "num_nodes", None)
        or getattr(problem, "num_vars", None)
        or getattr(problem, "num_spins", None)
        or args.size
    )
    print(f"problem    : {label}")
    print(f"backend    : {backend}")
    print(f"size       : {size}")
    print(f"best_obj   : {result.best_obj}")
    if backend == "scip":
        print(f"scip_status: {result.scip_status}")
        print(f"gap        : {result.gap}")
        print(f"dual_bound : {result.dual_bound}")
    if result.score:
        score = result.score
        feas = "feasible" if score.get("feasible", True) else "INFEASIBLE"
        unit = score.get("unit", "")
        val = score.get("value")
        print(f"{score.get('label', 'score'):<11}: {val} {unit} [{feas}]")
    print(f"runtime    : {result.runtime:.2f} s")
    diagnostics = getattr(result, "diagnostics", {})
    if diagnostics.get("restart_events"):
        print(
            f"restarts   : {diagnostics['restart_events']} events / "
            f"{diagnostics['restart_count']} replicas"
        )
    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            pickle.dump(
                {
                    "best_obj": result.best_obj,
                    "best_sol": result.best_sol.detach().cpu().numpy(),
                    "runtime": result.runtime,
                    "history": result.history,
                    "diagnostics": diagnostics,
                },
                fh,
            )
        print(f"saved      : {out}")
    if args.report:
        report = qqa.save_html_report(result, problem, args.report)
        print(f"report     : {report}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    import qqa

    args.device = _resolve_device(args.device)
    qqa.fix_seed(args.seed)
    if args.preset == "er-small":
        from qqa import datasets

        try:
            problems = datasets.mis_er_small(device=args.device)[:5]
        except FileNotFoundError as e:
            print(f"[qqa bench] dataset not found: {e}", file=sys.stderr)
            return 2
        sizes = []
        for p in problems:
            r = qqa.anneal(
                p,
                sol_size=args.sol_size,
                num_epochs=args.epochs,
                device=args.device,
                verbose=False,
            )
            sizes.append(-int(r.best_obj))
        print(f"preset     : {args.preset}")
        print(f"instances  : {len(problems)}")
        print(f"mean size  : {sum(sizes) / len(sizes):.2f}")
        print(f"sizes      : {sizes}")
        return 0

    if args.preset == "sk-small":
        problem = qqa.SherringtonKirkpatrick(N=100, seed=args.seed, device=args.device)
        r = qqa.anneal(
            problem,
            sol_size=args.sol_size,
            num_epochs=args.epochs,
            device=args.device,
            verbose=False,
        )
        print(f"preset     : {args.preset}")
        print("N          : 100")
        print(f"E_0/N      : {r.best_obj / 100:.4f}")
        print(f"runtime    : {r.runtime:.2f} s")
        return 0

    if args.preset == "ea-small":
        problem = qqa.EdwardsAnderson(L=4, dim=3, seed=args.seed, device=args.device)
        r = qqa.anneal(
            problem,
            sol_size=args.sol_size,
            num_epochs=args.epochs,
            device=args.device,
            verbose=False,
        )
        N = problem.num_spins
        print(f"preset     : {args.preset}")
        print(f"N          : {N}")
        print(f"E_0/N      : {r.best_obj / N:.4f}")
        print(f"runtime    : {r.runtime:.2f} s")
        return 0

    return 1


# --------------------------------------------------------------------------- #
# Suite-level bench commands (delegate to qqa.bench)                          #
# --------------------------------------------------------------------------- #


def _cmd_bench_list(args: argparse.Namespace) -> int:
    from qqa import bench as _b

    try:
        catalog = _b.list_suites()
    except SystemExit as exc:
        print(f"[qqa bench-list] {exc}", file=sys.stderr)
        return 2

    if not catalog:
        print("[qqa bench-list] no benchmark data on disk. Run:")
        print("    ./scripts/setup_benchmarks.sh")
        return 2

    if args.as_suites:
        for fam in sorted(catalog):
            for gt, subs in catalog[fam].items():
                for sub in subs:
                    parts = [fam]
                    if gt:
                        parts.append(gt)
                    if sub:
                        parts.append(sub)
                    print("-".join(parts))
        return 0

    for fam in sorted(catalog):
        types = catalog[fam]
        n = sum(len(s) for s in types.values())
        print(f"{fam}  ({n} subsets)")
        for gt in sorted(types):
            subs = types[gt]
            head = f"  {gt}/" if gt else "  "
            print(f"{head}  " + ", ".join(sorted(subs)) if subs else head)
    return 0


def _cmd_bench_run(args: argparse.Namespace) -> int:
    from qqa import bench as _b

    output = Path(args.output)
    if not output.is_absolute() and not str(output).startswith((".", "~")):
        output = _b.DEFAULT_RESULTS_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        "--suite",
        args.suite,
        "--backend",
        args.backend,
        "--sol-size",
        str(args.sol_size),
        "--num-epochs",
        str(args.num_epochs),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--output",
        str(output),
    ]
    if args.instances is not None:
        argv += ["--instances", str(args.instances)]
    if args.parallel:
        argv += ["--parallel"]
    if args.penalty is not None:
        argv += ["--penalty", str(args.penalty)]

    rc = _b.bench_discs_main(argv)
    if rc == 0:
        print(f"[qqa bench-run] wrote {output}")
    return rc


def _cmd_bench_plot(args: argparse.Namespace) -> int:
    from qqa import bench as _b

    out = Path(args.output)
    # Only auto-prepend DEFAULT_RESULTS_DIR when the user asked for a bare
    # filename ("report.png"); if they gave a directory ("data/fig/x.png")
    # we must not mangle that into "bench_results/data/fig/x.png".
    if not out.is_absolute() and not str(out).startswith((".", "~")) and out.parent == Path(""):
        out = _b.DEFAULT_RESULTS_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)

    argv: list[str] = list(args.results)
    if args.labels:
        argv += ["--labels", *args.labels]
    argv += ["--output", str(out), "--theme", args.theme, "--dpi", str(args.dpi)]
    if args.title:
        argv += ["--title", args.title]
    if args.fmt:
        argv += ["--format", args.fmt]

    rc = _b.plot_benchmarks_main(argv)
    if rc == 0:
        print(f"[qqa bench-plot] wrote {out}")
    return rc


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from qqa.benchmarking.cli import run_benchmark_command

    return run_benchmark_command(args)


def _cmd_example(args: argparse.Namespace) -> int:
    import qqa

    if args.action == "list":
        descriptions = {
            "microgrid-dispatch": "mixed unit commitment, storage, reserve, and dispatch",
            "microgrid-pareto": "cost/emissions/resilience Pareto planning",
            "portfolio-pareto": "risk/return/turnover allocation with cardinality",
            "process-blackbox": "constrained simulator-style process tuning",
        }
        for name in qqa.APPLICATIONS:
            print(f"{name:<22} {descriptions[name]}")
        return 0
    if args.name is None:
        raise SystemExit("[qqa example] `run` requires an application name.")

    device = _resolve_device(args.device)
    qqa.fix_seed(args.seed)
    problem = qqa.build_application(args.name)
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(problem, qqa.MultiObjectiveProblem):
        result = problem.solve_pareto(
            sol_size=args.sol_size,
            num_epochs=args.epochs,
            device=device,
            seed=args.seed,
            verbose=not args.quiet,
        )
        knee = result.select()
        payload = {
            "application": args.name,
            "device": device,
            "pareto_size": len(result.solutions),
            "objective_names": list(result.objective_names),
            "knee_index": knee,
            "knee_objectives": result.objectives[knee].detach().cpu().tolist(),
            "knee_solution": result.solutions[knee].detach().cpu().tolist(),
            "runtime": result.runtime,
        }
        print(f"pareto_size: {payload['pareto_size']}")
        print(f"knee       : {json.dumps(payload['knee_objectives'])}")
        if output_dir is not None:
            result.to_frame(problem).to_csv(output_dir / "pareto.csv", index=False)
            figure = qqa.plot_pareto(
                result,
                show=False,
                title=f"{problem.name} Pareto front",
            )
            figure.write_html(output_dir / "pareto.html", include_plotlyjs=True, full_html=True)
    elif isinstance(problem, qqa.BlackBoxProblem):
        result = problem.solve(
            budget=args.budget,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            seed=args.seed,
            verbose=not args.quiet,
        )
        payload = {
            "application": args.name,
            "device": device,
            "best_point": result.best_point,
            "best_value": result.best_value,
            "feasible": result.feasible,
            "total_violation": result.total_violation,
            "evaluations": result.evaluations,
            "runtime": result.runtime,
            "metadata": result.metadata,
        }
        print(f"best_value : {result.best_value:.8g}")
        print(f"feasible   : {result.feasible}")
        print(f"best_point : {json.dumps(result.best_point)}")
        if output_dir is not None:
            result.to_frame(problem).to_csv(output_dir / "evaluations.csv", index=False)
            figure = qqa.plot_blackbox(
                result,
                show=False,
                title="Process black-box optimisation",
            )
            figure.write_html(output_dir / "blackbox.html", include_plotlyjs=True, full_html=True)
    else:
        result = problem.solve(
            sol_size=args.sol_size,
            num_epochs=args.epochs,
            device=device,
            verbose=not args.quiet,
        )
        payload = {
            "application": args.name,
            "device": device,
            "best_loss": result.best_obj,
            "score": result.score,
            "solution": result.best_sol.detach().cpu().tolist(),
            "runtime": result.runtime,
        }
        print(f"best_loss  : {result.best_obj:.8g}")
        _print_score(result.score)
        if output_dir is not None:
            qqa.save_html_report(result, problem, output_dir / "dispatch.html")

    print(f"runtime    : {result.runtime:.4f} s")
    if output_dir is not None:
        result_path = output_dir / "result.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"artifacts  : {output_dir}")
    return 0


def _cmd_tex(args: argparse.Namespace) -> int:
    import qqa

    sources = sum(value is not None for value in (args.tex, args.spec, args.file))
    if sources != 1:
        raise SystemExit(
            "[qqa tex] provide exactly one TeX string (or '-'), --file MODEL.tex, "
            "or --spec MODEL.json."
        )
    args.device = _resolve_device(args.device)
    qqa.fix_seed(args.seed)
    if args.spec:
        spec_path = Path(args.spec).expanduser().resolve()
        spec = qqa.ModelSpec.from_json(spec_path.read_text(encoding="utf-8"))
    else:
        if args.file:
            tex_path = Path(args.file).expanduser().resolve()
            source = tex_path.read_text(encoding="utf-8")
        else:
            source = sys.stdin.read() if args.tex == "-" else args.tex
        client = qqa.OpenAICompatibleClient(
            base_url=args.api_base,
            model=args.model,
            api_style=args.api_style,
            verify_ssl=not args.insecure,
            timeout=args.timeout,
        )
        spec = qqa.compile_tex(source, client=client)
    if args.show_model:
        print(spec.to_json())

    if args.output_model:
        output = Path(args.output_model).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(spec.to_json() + "\n", encoding="utf-8")
        print(f"model_json : {output}")

    problem = qqa.problem_from_spec(spec)
    print(f"model      : {spec.name}")
    print(f"variables  : {sum(variable.size for variable in spec.variables)}")
    print(f"objectives : {len(spec.objectives)}")
    print(f"constraints: {len(spec.constraints)}")
    if args.dry_run:
        print("status     : validated (dry run)")
        return 0

    if isinstance(problem, qqa.MultiObjectiveProblem):
        if args.solver == "scip":
            raise SystemExit(
                "[qqa tex] --solver scip currently accepts one objective; "
                "use --solver qqa for a one-run parallel Pareto front."
            )
        result = problem.solve_pareto(
            sol_size=args.sol_size,
            num_epochs=args.epochs,
            device=args.device,
            seed=args.seed,
            verbose=not args.quiet,
        )
        print(f"pareto_size: {len(result.solutions)}")
        print(f"runtime    : {result.runtime:.4f} s")
        result_payload = {
            "objectives": list(result.objective_names),
            "front": result.objectives.detach().cpu().tolist(),
            "solutions": result.solutions.detach().cpu().tolist(),
        }
        print(
            "preview    : " + json.dumps(result_payload["front"][: min(10, len(result.solutions))])
        )
        if args.report:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            figure = qqa.plot_pareto(result, backend="plotly", show=False)
            figure.write_html(report_path, include_plotlyjs=True, full_html=True)
            print(f"report     : {report_path}")
    else:
        use_scip = args.solver == "scip"
        if use_scip:
            from qqa.hybrid import solve_spec_scip

            result = solve_spec_scip(
                spec,
                qqa_kwargs={
                    "sol_size": args.sol_size,
                    "num_epochs": args.epochs,
                    "device": args.device,
                    "verbose": not args.quiet,
                },
                time_limit=args.scip_time_limit,
                relative_gap=args.scip_gap,
                max_warm_starts=args.scip_warm_starts,
                threads=args.scip_threads,
                verbose=not args.quiet,
            )
            print("solver     : qqa+scip")
            print(f"scip_status: {result.scip_status}")
            print(f"gap        : {result.gap}")
            print(f"dual_bound : {result.dual_bound}")
            print(f"best_value : {result.objective_value}")
            _print_score(result.score)
            print(f"runtime    : {result.runtime:.4f} s")
            result_payload = {
                "solver": "qqa+scip",
                "best_value": result.objective_value,
                "solver_loss": result.solver_loss,
                "score": result.score,
                "solution": result.best_sol.detach().cpu().tolist(),
                "scip_status": result.scip_status,
                "gap": result.gap,
                "dual_bound": result.dual_bound,
                "proven_optimal": result.proven_optimal,
            }
        else:
            print("solver     : qqa")
            result = problem.solve(
                sol_size=args.sol_size,
                num_epochs=args.epochs,
                device=args.device,
                verbose=not args.quiet,
            )
            print(f"best_loss  : {result.best_obj}")
            _print_score(result.score)
            print(f"runtime    : {result.runtime:.4f} s")
            result_payload = {
                "solver": "qqa",
                "best_loss": result.best_obj,
                "score": result.score,
                "solution": result.best_sol.detach().cpu().tolist(),
            }
        if args.report:
            report = qqa.save_html_report(result, problem, args.report)
            print(f"report     : {report}")
    if args.output_result:
        result_path = Path(args.output_result).expanduser().resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"result_json: {result_path}")
    return 0


def _ask_result_payload(answer) -> dict:
    """Return a credential-free JSON representation of a unified result."""
    import qqa

    result = answer.result
    payload: dict = {
        "model": answer.plan.spec.name,
        "solver": answer.solver,
        "routing": answer.plan.to_dict()["routing"],
        "runtime": float(result.runtime),
    }
    if isinstance(result, qqa.ParetoResult):
        knee = result.select()
        payload.update(
            {
                "pareto_size": len(result.solutions),
                "objective_names": list(result.objective_names),
                "front": result.objectives.detach().cpu().tolist(),
                "solutions": result.solutions.detach().cpu().tolist(),
                "recommended_index": knee,
            }
        )
    elif isinstance(result, qqa.BlackBoxResult):
        payload.update(
            {
                "best_value": result.best_value,
                "best_point": result.best_point,
                "feasible": result.feasible,
                "total_violation": result.total_violation,
                "evaluations": result.evaluations,
            }
        )
    else:
        payload.update(
            {
                "best_value": float(getattr(result, "objective_value", result.best_obj)),
                "solution": result.best_sol.detach().cpu().tolist(),
                "score": result.score,
            }
        )
        if hasattr(result, "scip_status"):
            payload.update(
                {
                    "scip_status": result.scip_status,
                    "gap": result.gap,
                    "dual_bound": result.dual_bound,
                    "proven_optimal": result.proven_optimal,
                }
            )
    return payload


def _cmd_ask(args: argparse.Namespace) -> int:
    import qqa

    sources = sum(value is not None for value in (args.prompt, args.file, args.spec))
    if sources != 1:
        raise SystemExit(
            "[qqa ask] provide exactly one prompt (or '-'), --file REQUEST.txt, "
            "or --spec MODEL.json."
        )
    qqa.fix_seed(args.seed)
    if args.spec:
        path = Path(args.spec).expanduser().resolve()
        spec = qqa.ModelSpec.from_json(path.read_text(encoding="utf-8"))
        plan = qqa.plan_spec(spec, solver=args.solver)
    else:
        if args.file:
            path = Path(args.file).expanduser().resolve()
            source = path.read_text(encoding="utf-8")
        else:
            source = sys.stdin.read() if args.prompt == "-" else args.prompt
        client = qqa.OpenAICompatibleClient(
            base_url=args.api_base,
            model=args.model,
            api_style=args.api_style,
            verify_ssl=not args.insecure,
            timeout=args.timeout,
        )
        plan = qqa.compile_natural_language(source, client=client, solver=args.solver)

    if args.output_plan:
        path = Path(args.output_plan).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"plan_json  : {path}")
    print(f"model      : {plan.spec.name}")
    print(f"variables  : {plan.variable_count}")
    print(f"objectives : {len(plan.spec.objectives)}")
    print(f"constraints: {len(plan.spec.constraints)}")
    print(f"solver     : {plan.selected_solver}")
    print(f"why        : {plan.rationale}")
    for warning in plan.warnings:
        print(f"warning    : {warning}")
    if args.show_model:
        print(plan.spec.to_json())
    if args.plan_only:
        print("status     : validated (plan only)")
        return 0

    answer = qqa.execute_plan(
        plan,
        device=args.device,
        seed=args.seed,
        sol_size=args.sol_size,
        num_epochs=args.epochs,
        budget=args.budget,
        batch_size=args.batch_size,
        workers=args.workers,
        scip_time_limit=args.scip_time_limit,
        scip_gap=args.scip_gap,
        scip_threads=args.scip_threads,
        scip_warm_starts=args.scip_warm_starts,
        verbose=not args.quiet,
    )
    payload = _ask_result_payload(answer)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif isinstance(answer.result, qqa.ParetoResult):
        index = answer.result.select()
        print(f"pareto_size: {len(answer.result.solutions)}")
        print("recommended: " + json.dumps(answer.result.objectives[index].detach().cpu().tolist()))
        print(f"runtime    : {answer.result.runtime:.4f} s")
    elif isinstance(answer.result, qqa.BlackBoxResult):
        print(f"best_value : {answer.result.best_value:.8g}")
        print(f"feasible   : {answer.result.feasible}")
        print(f"evaluations: {answer.result.evaluations}")
        print(f"solution   : {json.dumps(answer.result.best_point, ensure_ascii=False)}")
        print(f"runtime    : {answer.result.runtime:.4f} s")
    else:
        _print_score(answer.result.score)
        if hasattr(answer.result, "scip_status"):
            print(f"scip_status: {answer.result.scip_status}")
            print(f"gap        : {answer.result.gap}")
        print(f"runtime    : {answer.result.runtime:.4f} s")

    if args.output_result:
        path = Path(args.output_result).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"result_json: {path}")
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(answer.result, qqa.ParetoResult):
            figure = qqa.plot_pareto(answer.result, show=False, title=plan.spec.name)
            figure.write_html(report_path, include_plotlyjs=True, full_html=True)
        elif isinstance(answer.result, qqa.BlackBoxResult):
            figure = qqa.plot_blackbox(answer.result, show=False, title=plan.spec.name)
            figure.write_html(report_path, include_plotlyjs=True, full_html=True)
        else:
            qqa.save_html_report(answer.result, answer.problem, report_path)
        print(f"report     : {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "version":
        return _cmd_version()
    if args.command == "inspect":
        import qqa

        print(json.dumps(qqa.inspect(args.model).to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "plan":
        import qqa

        print(
            qqa.plan(
                args.model,
                profile=args.profile,
                budget=args.budget,
                device=args.device,
            ).explain()
        )
        return 0
    if args.command == "solve":
        return _cmd_solve(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "bench-list":
        return _cmd_bench_list(args)
    if args.command == "bench-run":
        return _cmd_bench_run(args)
    if args.command == "bench-plot":
        return _cmd_bench_plot(args)
    if args.command == "benchmark":
        return _cmd_benchmark(args)
    if args.command == "tex":
        return _cmd_tex(args)
    if args.command == "ask":
        return _cmd_ask(args)
    if args.command == "example":
        return _cmd_example(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "gui":
        return _cmd_gui(args)
    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
