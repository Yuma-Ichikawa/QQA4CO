# Contributing to QQA

Thanks for your interest in improving QQA. This project aims to be a friendly,
reproducible research tool for gradient-based annealing on discrete problems.

## Setup

We recommend [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/Yuma-Ichikawa/QQA4CO.git
cd QQA4CO
uv sync --extra plotly --extra gui --extra docs --extra dev
uv run pre-commit install
```

This installs:

- Core runtime (PyTorch, NumPy, NetworkX, SciPy, Matplotlib, tqdm)
- Optional: Plotly (interactive viz), Streamlit (GUI), MkDocs (docs)
- Dev tools: pytest, ruff, pre-commit, nbval

If you intend to work on the optional CRA-PI-GNN backend (`qqa.pignn`),
also include the `pignn` extra:

```bash
uv sync --extra plotly --extra gui --extra docs --extra dev --extra pignn
```

The `pignn` extra pulls in `torch-geometric`. CI does **not** install it
by default — `tests/test_pignn.py` skips cleanly when PyG is missing.

## Running tests

```bash
uv run pytest -q
```

Tests are designed to finish in under a minute on CPU.

## Code style

- Ruff handles linting and formatting.
- Target Python 3.10+.
- Public functions should have a docstring (one-line summary + args/returns).

```bash
uv run ruff check src tests scripts app
uv run ruff format src tests scripts app
```

## Adding a new problem

1. Add a class to the appropriate file under `src/qqa/problems/`.
2. Attach a `.relaxation` (Binary / Categorical / Spin) so `qqa.anneal()` can
   dispatch.
3. Export it from `src/qqa/problems/__init__.py` and `src/qqa/__init__.py`.
4. Add a smoke test in `tests/` that solves a small instance and checks the
   objective.

## Submitting a PR

- Ensure `ruff check`, `ruff format --check`, and `pytest` all pass locally.
- Describe what you changed and why in the PR body.
- Link to any relevant issues or papers.

## Reporting issues

Please include:

- `qqa.__version__` and PyTorch version
- A minimal reproducing snippet
- Expected vs. observed behavior
