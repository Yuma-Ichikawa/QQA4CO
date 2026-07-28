# Internals — what lives where

A guided tour of the source tree for new contributors. If you only need
to *use* QQA4CO see [Quickstart](../quickstart.md); if you want to add
a new problem, relaxation, or backend see [Extending
QQA4CO](extending.md). This page is for everyone who wants to read the
code.

## Top-level layout

```
QQA4CO/
├── src/qqa/             # The Python package (everything ships in the wheel)
├── app/                 # Streamlit dashboard (also shipped in the wheel)
├── tests/               # Unit, integration, CLI, security, and UI tests
├── notebooks/           # Curated walkthroughs (CRA-PI-GNN, CPRA)
├── examples/            # Auto-generated per-problem notebooks
├── scripts/             # Demos, benchmarks, gallery / verification regen
├── docs/                # MkDocs-Material source (this site)
├── data/                # Bundled MIS-ER datasets, gallery assets
├── deploy/              # Streamlit Cloud config
├── tasks/               # Maintainer-only journal: todo.md + lessons.md
├── pyproject.toml       # PEP 621, hatchling, ruff, pytest, extras
├── mkdocs.yml           # Site nav (this site is built from docs/)
├── CHANGELOG.md         # Keep-a-Changelog
├── CITATION.cff         # GitHub citation widget + Zenodo
├── CONTRIBUTING.md      # Setup, tests, style, PR process
├── CODE_OF_CONDUCT.md   # Contributor Covenant
├── SECURITY.md          # Vulnerability reporting
├── Makefile             # `make test / lint / format / docs / serve`
└── README.md            # The 30-second pitch and the long sales page
```

`tasks/` is a maintainer journal that we keep in git so future-us can
re-derive design decisions; it is not part of the public surface.

## `src/qqa/` — module map

```
src/qqa/
├── __init__.py          # Public API surface, __version__ via importlib.metadata
├── annealing.py         # The single anneal() loop and AnnealResult
├── relaxation.py        # Binary / Spin / Categorical / Instance relaxations + Protocol
├── schedule.py          # LinearBGSchedule (callable contract)
├── callbacks.py         # Callback ABC, CallbackState, History/AutoDiv/Population/Trajectory
├── utils.py             # fix_seed, generate_graph, MIS / MaxCut analytics
├── datasets.py          # Bundled MIS-ER loaders + benchmark constants
├── visualization.py     # Plot helpers (matplotlib + optional plotly)
├── reporting.py         # Self-contained HTML result export
├── cli.py               # The qqa command (argparse, no third-party deps)
├── legacy.py            # Deprecated 0.2.x batch_annealing_* shims
├── problems/
│   ├── __init__.py
│   ├── base.py          # COProblem / QUBOProblem ABCs + normalize_graph
│   ├── qubo.py          # MIS / MaxClique / MaxCut (+ batched instance variants)
│   ├── categorical.py   # Coloring, BalancedGraphPartition
│   ├── spin.py          # SpinProblem, Ising1D, EA, SK, Perceptron, Hopfield
│   ├── extras.py        # Knapsack, NumberPartitioning, VertexCover, GraphBisection,
│   │                    # MaxSAT3, TSP, QAP, NQueens
│   └── user.py          # UserProblem + load_problem_from_file (--problem-file CLI hook)
├── mixed/               # Typed binary/integer/real modelling
│   ├── variables.py     # Declarations, bounds, pack/unpack layout
│   ├── relaxation.py    # Normalised mixed-domain relaxation
│   ├── problem.py       # Objective + constraint model
│   └── solve.py         # Mixed-friendly high-level entry point
├── multiobjective/      # Reference directions, Pareto archive, front plots
│   ├── problem.py       # Objective directions + mixed-variable model
│   ├── solver.py        # One-run augmented-Tchebycheff optimiser
│   └── visualization.py # 2-D, 3-D, and parallel-coordinate views
├── blackbox/            # Gradient-free expensive-function optimisation
│   ├── problem.py       # Plain-Python objective/constraint contract
│   ├── solver.py        # RBF surrogate + adaptive batch trust region
│   └── visualization.py # Convergence/feasibility/trust diagnostics
├── hybrid/
│   └── scip.py          # Optional QQA population → SCIP exact refinement
├── tex/                 # TeX → audited declarative model → QQA
│   ├── client.py        # Credential-safe Responses/Messages transport
│   ├── schema.py        # Exact JSON model schema and validation
│   ├── expressions.py   # Restricted AST interpreter (no eval/exec)
│   └── compiler.py      # Translation, repair, compilation, solve API
├── visuals/             # Advanced visualisation internals
│   ├── _data.py         # Backend-neutral diagnostics extraction
│   └── dashboard.py     # Plotly / Matplotlib result dashboards
└── pignn/               # Optional CRA-PI-GNN / CPRA backends (PyG)
    ├── __init__.py      # Re-exports train_cra_pi_gnn / train_cpra_pi_gnn
    ├── _import.py       # require_pyg() with an actionable ImportError
    ├── graph.py         # extract_nx_graph + nx_to_edge_index helpers
    ├── model.py         # GCNNet (2-layer GCNConv, multi-head for CPRA)
    └── trainer.py       # The two trainer functions (returning AnnealResult)
```

### Hard rules for changes inside `src/qqa/`

1. **Never import `torch_geometric` from `qqa.__init__` or any module
   that does not live under `qqa/pignn/`.** It pulls in heavy transitive
   deps that core users do not need; a stray `import` here is enough to
   break `pip install qqa` for everyone without `qqa[pignn]`.
2. **Public API additions go in `src/qqa/__init__.py` *and* its
   `__all__`** so they show up in `dir(qqa)` and the auto-generated
   API reference.
3. **Bug fixes that touch `anneal()` need a regression test.** That
   loop is the central nervous system of the package; every previous
   bug there shipped because the diff "looked obvious".
4. **Public function signatures are append-only.** Add a new keyword
   with a sensible default — never reorder, rename, or remove an
   existing argument without a deprecation cycle.
5. **Feature-specific code stays in a feature package.** Transport, schema,
   numerical algorithm, and visualisation are separate modules; importing the
   top-level package must never require optional SCIP or an API credential.

## `app/` — Streamlit dashboard

```
app/
├── streamlit_app.py     # Landing page (cache config + intro)
├── _common.py           # Cached resources shared across pages
├── _solution_viz.py     # Per-problem solution renderers
└── pages/
    ├── 1_Solve.py       # Build a problem + run anneal + show progress
    ├── 2_Visualize.py   # Re-render an AnnealResult pickle
    └── 3_Compare.py     # Hyperparameter sweeps
```

The wheel ships this directory under `qqa/_app/` (see
`pyproject.toml` `[tool.hatch.build.targets.wheel.force-include]`) so
`qqa gui` works after `pip install qqa` without a repo checkout.

## `tests/` — what each file covers

| File | Covers |
|---|---|
| `test_smoke.py` | End-to-end `anneal()` on each variable kind |
| `test_qqa_correctness.py` | Numerical guarantees against ground truth on small instances |
| `test_problems.py` | Constructor / shape contracts of binary/categorical problems |
| `test_extra_problems.py` | The `extras.py` catalogue |
| `test_spin_problems.py` | Spin-glass problems (Ising/EA/SK/Hopfield/Perceptron) |
| `test_pignn.py` | Optional PyG backends — CRA-PI-GNN + CPRA |
| `test_cli.py` | `qqa solve / bench / version` subcommands |
| `test_gui_smoke.py` | Streamlit imports without a browser |
| `test_gui_apptest.py` | `streamlit.testing` harness on the multi-page app |
| `test_visualization.py` | Plot helpers do not crash with a minimal `AnnealResult` |
| `test_legacy.py` | Deprecated `batch_annealing_*` aliases still emit a `DeprecationWarning` |
| `test_mixed.py` | Binary/integer/real modelling and constraint diagnostics |
| `test_advanced_optimization.py` | Pareto, black-box, SCIP, and population hand-off |
| `test_tex.py` | TeX schema safety, API fallback/redaction, and offline CLI |

Every numerical test uses a deliberately small problem. Optional backends are
skipped only when their declared extra is unavailable.

## `scripts/` — runnable demos and reproducibility

Listed in `docs/reference/backends.md` and partially on the README.
The categories:

* `demo_*.py` — single-problem walkthroughs you can run as
  `uv run python scripts/demo_mis.py`.
* `bench_*.py` — small benchmarks; `bench_qqa_vs_pignn.py` regenerates
  the headline comparison table.
* `make_gallery.py` — regenerates the README/docs gallery PNGs.
* `verify_all_problems.py` — the catalogue-wide correctness sweep that
  produces `docs/verification.md`.
* `_generate_notebooks.py` — deterministically rebuilds the
  `examples/` notebooks (the leading underscore signals "internal
  tooling").
* `check_streamlit_deploy.py` — health probe for the hosted demo at
  <https://parallelquasiquantum4co.streamlit.app/>.

## How tests, docs, and CI relate

```
.github/workflows/ci.yml
        │
        ├── ruff check + ruff format --check
        ├── pytest -q                  ← uses tests/
        └── mkdocs build --strict      ← uses docs/ and mkdocs.yml
```

A PR that breaks any of those three steps is automatically blocked.
There is also a Pages workflow that publishes the same `mkdocs build`
output to <https://yuma-ichikawa.github.io/QQA4CO/> on every push to
`main`.

## Where to start a typical PR

1. Open `tasks/todo.md`, scroll to the bottom, and skim the most
   recent dated section — that is where active work is journaled.
2. Run the *whole* test suite once locally before you start; that
   becomes your baseline.
3. After every change run `make lint test docs` (or the equivalent
   `uv run ...` commands listed in `CONTRIBUTING.md`).
4. If your change adds a public-API symbol, also add it to
   `docs/api.md` so mkdocstrings picks it up.
5. Add an entry to `CHANGELOG.md` under the *Unreleased* heading.

That is the whole loop. See [Releasing](releasing.md) for what happens
once changes accumulate to a tag.
