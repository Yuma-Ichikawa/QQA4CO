# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **First-class iSCO sampler support** via `qqa.discrete_langevin`
  (paper-faithful alias `qqa.isco_anneal`). Faithful, GPU-parallel
  implementation of **Algorithm 1 + Appendix C (PAS-MH-Step)** of
  Sun, Goshvadi, Nova, Schuurmans, Dai, *Revisiting Sampling for
  Combinatorial Optimization*, ICML 2023 (pmlr-v202-sun23c). Every
  MH step samples a Poisson-length path `L ~ Poisson(μ)` truncated
  at `L ≥ 1`, picks `L` sites without replacement via Gumbel-top-`L`
  with logits `−Δ_j/(2τ)`, applies the **path-auxiliary MH
  correction** over the ordered permutation σ (Eq. 30), and adapts μ
  toward the paper's 0.574 acceptance target (Eq. 31). Works on
  single-instance (`Q_mat`) and batched-instance (`Q_tensor`) QUBOs;
  spin / categorical / structured-shape relaxations are rejected at
  the API boundary with an actionable `NotImplementedError`. Returns
  an `ISCOResult` that mirrors `SAResult` / `PAResult`
  (`best_sol` / `best_obj` / `runtime` / `history` / `score` /
  `polished_sol`) plus iSCO-specific diagnostics
  (`accept_rate`, `mu_final`, `mean_path_length`, `t_max_used`).
  Cross-checked against the DISCS reference implementation
  (`samplers/path_auxiliary.py`) and the Zhang et al.
  `discrete-langevin` reference. See the new
  *iSCO baseline (Sun et al., ICML 2023)* section in the README and
  citations `sun2023revisiting` + `goshvadi2023discs`.
- **Empirical detailed-balance test for iSCO**
  (`tests/test_isco.py::test_isco_detailed_balance_on_tiny_qubo`)
  enumerates a 2^4-state QUBO, runs the full PAS-MH kernel for 4000
  inner steps × 200 chains at fixed temperature, and asserts
  TV(empirical, exact Boltzmann) < 0.02. Ships as a permanent
  guard against silent MH-correction regressions; offline sweep
  across `N ∈ {3, 4, 5} × seed ∈ {0, 7, 42} × μ ∈ {1, 2, 3} ×
  {float32, float64}` shows the post-fix sampler converges to
  TV ≤ 0.0064 in every cell.

### Fixed

- **iSCO `_plackett_luce_logprob` NaN bug (silent detailed-balance
  violation).** The Plackett-Luce log-prob recursion used
  `diff.clamp(max=-1e-12)` to keep `log1p(-exp(diff))` finite, but
  `-1e-12` round-trips to `0.0` in float32 (machine ε ≈ 1.19e-7),
  sending the recursion into `log(0) = -inf` whenever `sigma`
  contained the repeated indices that `_reverse_path` writes into
  the masked tail (i.e. **every chain with `L_per_chain < L_max`,
  which is every short chain in any batch with variable Poisson
  path length**). Subsequent summation via `* mask.to(dtype)` then
  produced `(-inf) * 0 = NaN`, making `log(u) < log_alpha` evaluate
  to `False` everywhere and silently rejecting every multi-flip
  proposal in the affected chain. Empirical TV(empirical, Boltzmann)
  on a 4-bit enumerable QUBO was ~0.51; after the fix it is
  ~0.001-0.002. Two surgical changes: (a) dtype-aware clamp
  (`eps_clamp = -1e-6` for float32, `-1e-12` for float64); (b)
  `torch.where(mask, value, 0)` instead of `* mask` so masked
  positions can never contaminate the sum via `inf * 0`. Regression
  tests `test_plackett_luce_logprob_handles_repeated_indices_in_float32`
  and `test_isco_detailed_balance_on_tiny_qubo` ensure this stays
  fixed. Lessons L48-L50 in `tasks/lessons.md`.

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
