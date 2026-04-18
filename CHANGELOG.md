# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

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
