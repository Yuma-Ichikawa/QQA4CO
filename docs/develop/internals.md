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
├── examples/            # Generated and curated executable notebooks
├── scripts/             # Demos, benchmarks, gallery / verification regen
├── docs/                # Guides, deployment runbooks, and documentation assets
├── data/                # Tiny smoke data and public dataset manifests
├── pyproject.toml       # PEP 621, hatchling, ruff, pytest, extras
├── mkdocs.yml           # Site nav (this site is built from docs/)
├── CHANGELOG.md         # Keep-a-Changelog
├── CITATION.cff         # GitHub citation widget + Zenodo
├── CONTRIBUTING.md      # Setup, tests, style, PR process
├── CODE_OF_CONDUCT.md   # Contributor Covenant
├── SECURITY.md          # Vulnerability reporting
├── Makefile             # `make test / lint / format / docs / serve`
└── README.md            # Short landing page; details live in this site
```

Large public datasets and generated campaign trajectories are fetched on
demand and are not part of the source distribution.

## `src/qqa/` — module map

```
src/qqa/
├── __init__.py          # Small stable public surface and legacy exports
├── api.py              # Stable solve / plan / inspect entry points
├── config.py           # Strict SolverConfig and named profiles
├── result.py           # Backend-neutral SolveResult contract
├── annealing.py        # Pure QQA loop; extensions are explicit options
├── relaxation.py       # Binary, spin, simplex, Gumbel and Sinkhorn maps
├── schedule.py         # Fixed, cyclic, reheat and adaptive schedules
├── model/              # Immutable ModelIR, factors, constraints and adapters
├── compile/            # Sparse QUBO compilation
├── engines/            # Sparse/component QQA and independent islands
├── portfolio/          # Inspector, deterministic planner and mini-probes
├── local/              # Sparse delta search and diverse elite archive
├── repair/             # Structure-aware repair registry
├── problems/           # Compatibility problem catalogue
├── mixed/              # Binary/integer/real augmented-Lagrangian path
├── algebraic/          # Compatibility sparse linear/quadratic IR
├── io/                 # MPS/QPLIB/JSON/OPB/DIMACS/QUBO/Ising adapters
├── presolve/           # Reversible scaling and transformed state
├── decomposition/      # Continuous completion
├── benchmarking/       # Public fetch, paired metrics and campaign runner
├── multiobjective/     # Scalarisations, archive and indicators
├── blackbox/           # Persistent asynchronous trust-region search
├── hybrid/             # QQA LNS plus isolated optional exact adapters
├── uncertainty.py      # Scenario, robust and CVaR evaluation
├── rolling.py          # Rolling-horizon warm-state hand-off
├── tex/                # Audited TeX compiler (no eval/exec)
├── visuals/            # Advanced diagnostics
└── pignn/              # Optional experimental PyG backends
```

### Hard rules for changes inside `src/qqa/`

1. **Never import `torch_geometric` from `qqa.__init__` or any module
   that does not live under `qqa/pignn/`.** It pulls in heavy transitive
   deps that core users do not need; a stray `import` here is enough to
   break `pip install qqa` for everyone without `qqa[pignn]`.
2. **Keep the top-level API small.** Stable capabilities converge on
   `solve`, `plan`, `inspect`, `ModelIR`, `SolverConfig`, and `SolveResult`;
   specialised building blocks stay in their feature packages.
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
    ├── 3_Compare.py     # Matched-budget, multi-seed comparisons
    └── 4_Universal.py   # Model-file inspect / plan / solve workflow
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
* `make_gallery.py` — regenerates the documentation gallery PNGs.
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
        ├── mypy stable API
        ├── pytest -q                  ← uses tests/
        ├── deterministic performance guard
        ├── mkdocs build --strict      ← uses docs/ and mkdocs.yml
        └── wheel + source build
```

Nightly CUDA checks cover device parity, mixed precision, compilation and
memory behaviour. A scheduled public-benchmark workflow exercises a pinned,
small MIPLIB/QPLIB subset without committing machine-specific trajectories.

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
