# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Infrastructure

- `publish.yml` Trusted Publishing wired up end-to-end on PyPI:
  GitHub Actions environment `pypi` is now connected to the registered
  Trusted Publisher, so future tagged releases upload automatically
  without manual `twine` invocations.

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
