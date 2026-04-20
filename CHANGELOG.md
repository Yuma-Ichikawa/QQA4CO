# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-04-20

### Added

- **`qqa.polish.apply_polish_if_improves`**: single entry point for the
  greedy 1-flip QUBO polish post-processing. `qqa.anneal`,
  `qqa.simulated_annealing`, `qqa.population_annealing` and both
  PI-GNN trainers now route through this helper so every backend has
  the same "monotone free improvement" contract without five copies of
  the same `if polish and Q_mat is not None: …` block.
- **Shared test fixtures** at `tests/conftest.py`: `APP`, `PAGE_DIR`
  path constants, a `make_problem_config(kind, size, **extra)` factory
  and a `set_slider` helper. Test modules now import these directly,
  eliminating twelve copies of the same ``problem_config`` literal in
  `test_gui_apptest.py`.
- **`app/_common.retheme_plotly(fig)`**: replaces the ``_retheme`` clone
  previously defined once per Streamlit page. Import it alongside
  `plotly_layout` so every chart stays in step with the active theme.
- **`app/_common.as_numpy(x)`** (public alias of the former `_as_np`):
  imported by `_solution_viz.py` so the two modules share a single
  tensor-to-numpy conversion path.

### Changed

- **Benchmark suite refreshed**: the project version now tracks the
  "qqa4co-bench" HF dataset (coloring / mis-rrg / ea3d /
  balanced-partition / MaxCut G-set families), wired through the
  `qqa.bench` public API and `qqa bench run|plot|list|setup` CLI.
- `SpinRelaxation.perturb_` now inherits from `BinaryRelaxation` —
  both relaxations share the same latent cube `[0, 1]` and therefore
  the same noise + ``clamp_`` schedule. Removes a silent copy-paste
  drift risk.
- `qqa.bench` collapsed `_load_bench_discs` and `_load_plot_benchmarks`
  onto a shared `_load_scripts_module(name)` helper so the two
  ``sys.path`` / ``importlib`` call sites no longer drift.
- ``tests/`` directory is now on the pytest ``pythonpath`` so test
  modules can ``from conftest import …`` the shared helpers.

### Removed

- **`qqa.sa._qubo_glauber_sweep` deprecated alias** dropped — it
  forwarded to `_qubo_seq_glauber_sweep` and was only referenced by an
  in-tree diagnostic script (updated). The buggy parallel-update
  semantics it warned about have been gone since 0.4.0.

## [0.5.3] - 2026-04-20

### Added

- **Backend-aware Visualize layout**: the Streamlit Visualize page now
  shows PQQA-only tabs for PQQA runs (family tree, PCA embedding,
  diversity, parallel coordinates) and PA-only tabs for PA runs
  (ESS, free-energy trajectory, equilibration diagnostic,
  Thermodynamics, Lineage vs energy, Ancestry Sankey). Empty
  "No snapshots recorded" placeholders are gone.
- **Up-front PA capability probe** in the Solve page: problems that
  PA cannot sample (categorical / structured binary, e.g. TSP, QAP,
  Coloring, NQueens) now trigger a clear warning banner and disable
  the Run button, instead of surfacing a cryptic ``einsum`` /
  ``NotImplementedError`` mid-run.
- **Three PA-specific visualisation tabs**: Thermodynamics (Q vs β,
  internal energy, specific heat), Lineage vs energy, Ancestry
  Sankey.

### Changed

- `qqa.simulated_annealing` / `qqa.population_annealing` now accept
  `polish=True/False` and expose a `polished_sol` field, matching the
  contract `qqa.anneal` has always had. The 1-flip polish is default-on
  across all backends so the "best_obj" score card reflects the same
  post-processing everywhere.
- `_validate_chain_problem` (used by both SA and PA) now rejects
  structured `BinaryRelaxation` (non-flat `shape_fn`, e.g. TSP)
  with an actionable error steering users to `qqa.anneal`.

## [0.5.2] - 2026-04-20

### Added

- **`qqa.bench` public Python API** (`run`, `plot`, `list_suites`,
  `resolve_suite`) mirroring the `qqa bench` CLI so notebooks can
  dispatch a benchmark without subprocess boilerplate.
- **Polished benchmark report figure** (`scripts/plot_benchmarks.py`)
  and the corresponding `qqa bench plot` CLI flow.

### Changed

- HF Hub dataset renamed to `qqa4co-bench` (was `discs-benchmarks`);
  `scripts/setup_discs_data.sh` and all docs follow suit.

## [0.5.1] - 2026-04-19

### Added

- **`qqa.population_annealing`**: Population Annealing backend with
  parallel chain sampling, importance resampling between inverse
  temperatures, full free-energy / log-Z estimates and an optional
  genealogy / ancestry record. `PAResult` dataclass and
  `qqa solve --backend pa` CLI expose the new path.
- **MaxCut G-set benchmark family** via
  `scripts/fetch_gset_data.py` + `scripts/maxcut_gset_g70.py`.

## [0.5.0] - 2026-04-19

### Added

- Streamlit Compare page now offers a **PQQA vs SA shootout** mode that
  runs both backends on the same problem instance and reports the
  per-backend best objective, runtime and a "SA time to PQQA best"
  speed-up factor side-by-side, including a convergence plot.

### Changed

- Internal refactor: `qqa.utils` now exposes
  `require_cuda_if_requested(device)` and
  `safe_score_summary(problem, sol, fallback_obj)` helpers. The QQA,
  SA and PI-GNN/CPRA trainers now route their CUDA-availability check
  and `problem.score_summary` fallback through these shared helpers,
  removing duplicated inline `try/except` blocks while preserving the
  exact user-facing error messages and result dictionaries.
- Marked the legacy graph-evaluation helpers in `qqa.utils`
  (`approximate_mis`, `mis_stats`, `max_cut_stats`, `_gen_combinations`)
  as superseded by `problem.score_summary`. They are kept for backward
  compatibility but are no longer used internally.

### Documentation

- Repo-wide audit of the QQA / CPRA paper citations. Three places had
  silently swapped the QQA paper (Ichikawa & Arai, ICLR 2025) with the
  CPRA paper (Ichikawa & Iwashita, TMLR 2025) — fixed in
  `src/qqa/__init__.py` docstring, `notebooks/cra_pignn_example.ipynb`
  and `notebooks/cpra_pignn_example.ipynb`. Adopted the TMLR-published
  title for CPRA ("Continuous Parallel Relaxation for Finding Diverse
  Solutions in Combinatorial Optimization Problems"); the older
  arXiv-preprint title ("Continuous Tensor Relaxation …") is no longer
  used.
- Added a Codecov coverage badge to `README.md` and a placeholder for
  the Zenodo DOI badge (uncommented and DOI-substituted as soon as the
  first release is minted).
- Fixed `CITATION.cff` `preferred-citation` block: title now correctly
  matches the URL (both point at the QQA ICLR 2025 paper); arXiv:2409.02135
  added as an explicit identifier so citation tooling (Zenodo, ORCID,
  OpenAlex) resolves to the same artefact.

### Infrastructure

- `publish.yml` Trusted Publishing wired up end-to-end on PyPI:
  GitHub Actions environment `pypi` is now connected to the registered
  Trusted Publisher, so future tagged releases upload automatically
  without manual `twine` invocations.
- Broadened PyPI classifiers in `pyproject.toml`
  (`Environment :: Console`, `Environment :: GPU :: NVIDIA CUDA`,
  `Intended Audience :: Education / Developers`, OS-specific tags,
  `Topic :: Mathematics / Physics`, `Typing :: Typed`) for better PyPI
  discoverability.

## [0.4.0] - 2026-04-19

### Added

- **`qqa.simulated_annealing`**: GPU-parallel Simulated Annealing
  baseline with two execution paths:
  - QUBO fast path (Glauber-like parallel update, single matmul per
    sweep) for any problem exposing `Q_mat`.
  - Generic single-spin sequential Metropolis fallback for non-QUBO
    problems.
  - New `SAResult` dataclass mirroring `AnnealResult` for
    interchangeable downstream tooling.
- **CLI**: `qqa solve --backend sa` with `--sa-num-sweeps`,
  `--sa-beta-start`, `--sa-beta-end`, `--sa-schedule`.
- **`qqa.utils.enable_tf32`** helper to opt into TF32 matmul / cuDNN
  on Ampere+ GPUs.
- **`anneal(..., mixed_precision="bf16")`** opt-in for bfloat16
  autocast on the QQA forward pass (CUDA only; falls back to fp32
  silently elsewhere).
- **`train_cra_pi_gnn` / `train_cpra_pi_gnn`**: new
  `early_stop_disc_patience` argument that terminates training when
  the best discrete objective stops improving.
- **CPRA `multi_problem` batching**: when every replica problem has a
  same-shape `Q_mat`, the trainer stacks them into one tensor and
  computes all replica costs in a single batched `einsum`, replacing
  the previous Python-level per-replica loop.
- **`docs/explanation/algorithm.md`**: SA section documenting the
  parallel-Glauber fast path and when to reach for SA vs QQA / CRA /
  CPRA.
- **`notebooks/benchmark_sa_vs_qqa_vs_pignn.ipynb`**: head-to-head
  benchmark notebook comparing all four solver families on a common
  MIS instance with controlled compute budget.

### Changed

- `HistoryRecorder` now buffers per-epoch metrics as GPU scalars and
  performs a single bulk `cpu()` transfer in `on_train_end`,
  eliminating per-epoch host-device synchronisation. Public
  `result.history` shape is unchanged.
- `qqa.anneal` and the PI-GNN trainers now use
  `optimizer.zero_grad(set_to_none=True)` (PyTorch 2.x best practice).
- `SpinRelaxation.project` no longer allocates two `ones_like(x)`
  intermediates per call; uses scalar-broadcast `torch.where`.
- `CategoricalRelaxation.penalty` no longer triggers a redundant
  `forward`: the relaxation now exposes `penalty_from_forward` so
  `anneal` reuses the already-normalised tensor.

### Performance

- ~15 % wall-clock reduction on CPU for `qqa.anneal`-driven workloads
  (HistoryRecorder + `set_to_none` + `SpinRelaxation` together).
- CPRA `multi_problem` runs are 2–4× faster on GPU at `R = 16` thanks
  to the batched `einsum` path.

### Notes

- No public API removed. `qqa.anneal`, `qqa.pignn.train_*` and the
  `AnnealResult` dataclass are unchanged. New keyword arguments
  (`mixed_precision`, `early_stop_disc_patience`) are opt-in and
  default to the prior behaviour.

## [0.3.0] - 2026-04-18

### Added

- **Spin problem family** in `qqa.problems`:
  - `Ising1D`, `EdwardsAnderson`, `SherringtonKirkpatrick`
  - `BinaryPerceptron` (teacher-student), `HopfieldMemory`
  - New `SpinRelaxation` that maps `[0,1]` → `±1` with differentiable forward.
- **Visualization** (`qqa.visualization`):
  - Dual backend (`"matplotlib"` default, `"plotly"` optional).
  - `plot_best_trajectory`, `plot_schedule`, `plot_run_comparison`,
    `plot_parallel_coordinates`, `plot_solution_heatmap`.
- **CLI** (`qqa` entry point): `qqa version`, `qqa solve`, `qqa bench`,
  `qqa gui`.
- **Streamlit GUI** (`qqa gui` / `uv run streamlit run app/streamlit_app.py`):
  problem definition → live annealing → visualization → comparison.
- **Example notebooks**: MIS, coloring, MaxCut, 3D Edwards–Anderson, SK,
  binary perceptron, Hopfield memory, parallel benchmark.
- **Docs site** via MkDocs + Material with auto API reference.
- **Tooling**: GitHub Actions CI, `pre-commit`, `CONTRIBUTING.md`,
  `CITATION.cff`.

### Changed

- `qqa.problems` is now a subpackage (`qubo.py`, `categorical.py`, `spin.py`).
  Public symbols (`MaximumIndependentSet`, `Coloring`, ...) are preserved via
  re-export, so existing code keeps working.

### Deprecated

- `qqa.legacy.*` wrappers still work and emit `DeprecationWarning`; use
  `qqa.anneal` instead.

## [0.2.0]

- Initial unified `qqa.anneal` API, package reorganization under `src/qqa`,
  `uv`/`pyproject.toml` based install, smoke tests and demo scripts.

## [0.1.0]

- Original research release accompanying the ICLR 2025 paper.
