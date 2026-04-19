# QQA4CO development tasks. Wraps the `uv run ...` commands that CI
# uses, so contributors get the same checks locally with one keystroke.
#
#   make help    # list all targets
#   make test    # run pytest
#   make lint    # ruff check + ruff format --check (CI's exact commands)
#   make format  # ruff format (rewrite, not just check)
#   make docs    # mkdocs build --strict
#   make serve   # mkdocs serve at http://localhost:8000
#   make ci      # everything CI runs (lint + test + docs)
#   make clean   # remove build artefacts

.DEFAULT_GOAL := help
.PHONY: help test lint format docs serve ci clean install build

UV ?= uv
PY_TARGETS := src tests scripts app

help:  ## list targets
	@awk 'BEGIN {FS=":.*?## "} /^[a-zA-Z_-]+:.*## / {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## sync the dev environment with all extras
	$(UV) sync --extra plotly --extra gui --extra docs --extra dev
	$(UV) run pre-commit install

test:  ## run the pytest suite
	$(UV) run pytest -q

lint:  ## ruff check + ruff format --check (the exact CI commands)
	$(UV) run ruff check $(PY_TARGETS)
	$(UV) run ruff format --check $(PY_TARGETS)

format:  ## rewrite files with ruff format
	$(UV) run ruff format $(PY_TARGETS)

docs:  ## strict mkdocs build (also what CI runs)
	$(UV) run mkdocs build --clean --strict

serve:  ## live preview the docs at http://localhost:8000
	$(UV) run mkdocs serve

ci:  ## everything CI runs, in CI order
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) docs

build:  ## build wheel + sdist into dist/
	rm -rf dist/
	$(UV) build

clean:  ## remove build artefacts and caches
	rm -rf dist/ site/ .pytest_cache/ .ruff_cache/ build/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
