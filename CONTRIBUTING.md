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

## Secrets

- **Never commit API tokens, private keys, or `.env` files.** The
  pre-commit config runs [`gitleaks`](https://github.com/gitleaks/gitleaks)
  and [`nbstripout`](https://github.com/kynan/nbstripout) on every
  commit, and the `Secret scan` GitHub Actions workflow re-runs
  `gitleaks` on every PR plus weekly against the full history.
- Hugging Face tokens belong in the `HUGGINGFACE_HUB_TOKEN` (or
  `HF_TOKEN`) environment variable, not in source files or notebook
  cells. `scripts/setup_benchmarks.sh` already expects this.

## Quick commands (Makefile)

The `Makefile` wraps the exact commands CI runs:

```bash
make install   # uv sync + pre-commit hooks
make lint      # ruff check + ruff format --check
make format    # rewrite with ruff format
make test      # pytest -q
make docs      # mkdocs build --strict
make ci        # lint + test + docs (everything CI runs)
make serve     # live mkdocs preview at http://localhost:8000
```

## Extending QQA4CO

The full extension guide lives in
[`docs/develop/extending.md`](docs/develop/extending.md). Short version:

* **New problem** → subclass `qqa.COProblem`, attach a `relaxation`,
  implement `loss_fn` (and ideally `score_summary`). Register in
  `src/qqa/problems/__init__.py` and `src/qqa/__init__.py`. Add a smoke
  test in `tests/`. Add a row to `docs/problems.md`.
* **New relaxation** → implement the `qqa.Relaxation` `Protocol`.
* **New schedule** → any `(epoch, T) -> float` callable; pass to
  `qqa.anneal(schedule=...)`.
* **New callback** → subclass `qqa.Callback`.
* **New solver backend** → return `qqa.AnnealResult` from your trainer
  function. See `qqa.pignn` for the canonical example.

[`docs/develop/internals.md`](docs/develop/internals.md) is the source-tree
map for new contributors.

## Submitting a PR

- Ensure `make ci` passes locally.
- Describe what you changed and why in the PR body.
- Add an entry to `CHANGELOG.md` under *Unreleased*.
- Link to any relevant issues or papers.

## Releasing

See [`docs/develop/releasing.md`](docs/develop/releasing.md) for the
full release checklist (versioning, CHANGELOG, tagging, PyPI upload).

## Reporting issues

Please include:

- `qqa.__version__` and PyTorch version
- A minimal reproducing snippet
- Expected vs. observed behaviour

For security issues, follow [`SECURITY.md`](SECURITY.md) instead of
filing a public issue.

## Code of Conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.
