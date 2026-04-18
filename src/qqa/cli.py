"""Command-line interface for QQA.

Exposed as the ``qqa`` console script via ``[project.scripts]`` in
``pyproject.toml``. Subcommands:

* ``qqa version`` — print the installed version.
* ``qqa solve`` — solve a single problem from the CLI.
* ``qqa bench`` — run a quick benchmark on a bundled dataset.
* ``qqa gui`` — launch the Streamlit GUI in a subprocess.

The CLI uses only :mod:`argparse` and standard library modules so it has no
extra runtime dependency.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qqa",
        description="Quasi-Quantum Annealing command-line interface.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    sub.add_parser("version", help="Show the installed qqa version.")

    solve = sub.add_parser("solve", help="Solve a single problem.")
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
    solve.add_argument("--dim", type=int, default=3, help="Lattice dimension (EA).")
    solve.add_argument(
        "--alpha", type=float, default=0.5, help="Loading ratio (perceptron/Hopfield)."
    )
    solve.add_argument(
        "--patterns", type=int, default=1, help="Number of stored patterns (Hopfield)."
    )
    solve.add_argument("--num-category", type=int, default=3, help="Number of colours (coloring).")
    solve.add_argument("--sol-size", type=int, default=100)
    solve.add_argument("--epochs", type=int, default=1000)
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
    solve.add_argument("--temp", type=float, default=0.0)
    solve.add_argument("--min-bg", type=float, default=-2.0)
    solve.add_argument("--max-bg", type=float, default=0.1)
    solve.add_argument("--curve-rate", type=int, default=2)
    solve.add_argument("--div-param", type=float, default=0.0)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--device", type=str, default="cpu")
    solve.add_argument("--quiet", action="store_true", help="Suppress per-epoch logs.")
    solve.add_argument(
        "--output",
        type=str,
        default=None,
        help="If given, save the AnnealResult (pickle) to this path.",
    )
    # ------------------------------------------------------------------
    # Optional CRA-PI-GNN backend (PyTorch Geometric). Requires installing
    # the ``pignn`` extra: ``pip install qqa[pignn]``. Defaults to ``qqa``,
    # so users without the extra are completely unaffected.
    # ------------------------------------------------------------------
    solve.add_argument(
        "--backend",
        choices=["qqa", "pignn", "cpra"],
        default="qqa",
        help=(
            "Solver backend. 'qqa' (default) uses the parallel-replica "
            "annealing loop; 'pignn' uses the optional CRA-PI-GNN "
            "(PyTorch Geometric) trainer — graph problems only "
            "(mis/maxcut/maxclique/vertex_cover/graph_bisection); "
            "'cpra' uses the multi-head CPRA extension that produces "
            "diverse solutions in one run (same graph problems)."
        ),
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

    gui = sub.add_parser("gui", help="Launch the Streamlit GUI.")
    gui.add_argument("--port", type=int, default=8501)
    gui.add_argument("--host", type=str, default="localhost")
    gui.add_argument("--headless", action="store_true")

    return parser


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _cmd_version() -> int:
    import qqa

    print(qqa.__version__)
    return 0


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

    problem = _build_problem(args)

    backend = getattr(args, "backend", "qqa")
    if backend in {"pignn", "cpra"}:
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
    else:
        qqa_lr = args.learning_rate if args.learning_rate is not None else 1.0
        result = qqa.anneal(
            problem,
            sol_size=args.sol_size,
            learning_rate=qqa_lr,
            temp=args.temp,
            min_bg=args.min_bg,
            max_bg=args.max_bg,
            curve_rate=args.curve_rate,
            div_param=args.div_param,
            num_epochs=args.epochs,
            device=args.device,
            verbose=not args.quiet,
        )
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
    if result.score:
        score = result.score
        feas = "feasible" if score.get("feasible", True) else "INFEASIBLE"
        unit = score.get("unit", "")
        val = score.get("value")
        print(f"{score.get('label', 'score'):<11}: {val} {unit} [{feas}]")
    print(f"runtime    : {result.runtime:.2f} s")
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
                },
                fh,
            )
        print(f"saved      : {out}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    import qqa

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


def _resolve_streamlit_app() -> Path | None:
    """Locate ``app/streamlit_app.py`` for both wheel and source installs.

    When installed from a wheel, ``hatch`` places the Streamlit app at
    ``qqa/_app/streamlit_app.py`` (see ``pyproject.toml`` ``force-include``).
    When run from a source checkout (``pip install -e .`` or ``python -m``),
    that directory does not exist, so we fall back to ``repo_root/app/``
    resolved relative to this file.
    """
    try:
        from importlib.resources import files

        candidate = Path(str(files("qqa").joinpath("_app", "streamlit_app.py")))
        if candidate.exists():
            return candidate
    except (ModuleNotFoundError, FileNotFoundError, OSError):
        pass

    here = Path(__file__).resolve()
    # src/qqa/cli.py -> src/qqa -> repo_root
    repo_root = here.parents[2]
    candidate = repo_root / "app" / "streamlit_app.py"
    if candidate.exists():
        return candidate
    return None


def _cmd_gui(args: argparse.Namespace) -> int:
    if shutil.which("streamlit") is None:
        print(
            "[qqa gui] 'streamlit' is not on PATH. Install the GUI extras with "
            "`pip install qqa[gui]`.",
            file=sys.stderr,
        )
        return 2

    app = _resolve_streamlit_app()
    if app is None:
        print(
            "[qqa gui] Streamlit app not found in the installed package or "
            "the source tree. Re-install qqa (``pip install qqa[gui]``) or "
            "clone https://github.com/Yuma-Ichikawa/QQA4CO and run from the "
            "repository root.",
            file=sys.stderr,
        )
        return 2

    cmd = [
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])
    env = os.environ.copy()
    env.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    return subprocess.call(cmd, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "version":
        return _cmd_version()
    if args.command == "solve":
        return _cmd_solve(args)
    if args.command == "bench":
        return _cmd_bench(args)
    if args.command == "gui":
        return _cmd_gui(args)
    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
