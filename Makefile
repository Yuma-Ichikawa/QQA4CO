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
.PHONY: help test lint format docs serve ci clean install build \
        bench-discs bench-discs-setup bench-discs-smoke

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

bench-discs-setup:  ## download + convert the DISCS CO benchmark suite (~6.7 GB)
	$(UV) run --extra discs scripts/setup_discs_data.sh

bench-discs-smoke:  ## 3 instances per problem family on real DISCS data (CPU)
	$(UV) run --extra discs python scripts/bench_discs.py \
	    --suite all --backend qqa --instances 3 --device cpu \
	    --output bench_discs_smoke.json

bench-discs:  ## full DISCS suite with qqa.anneal (use SUITE=... to scope; PARALLEL=1 to batch)
	$(UV) run --extra discs python scripts/bench_discs.py \
	    --suite $(or $(SUITE),all) \
	    --backend $(or $(BACKEND),qqa) \
	    --device $(or $(DEVICE),auto) \
	    $(if $(filter 1,$(PARALLEL)),--parallel,) \
	    --output $(or $(OUTPUT),bench_discs_$(or $(BACKEND),qqa).json)

bench-discs-paper:  ## reproduce PQQA paper (Ichikawa NeurIPS 2024) settings
	## Defaults match Table 1 (SATLIB MIS, S=100, fewer steps): expect
	## mean_ratio ~0.993 vs KaMIS. Override SOL_SIZE / NUM_EPOCHS for
	## "more steps" (3000 -> 30000) or S=1000 row.
	$(UV) run --extra discs python scripts/bench_discs.py \
	    --suite $(or $(SUITE),mis-satlib-uf) \
	    --backend qqa \
	    --device $(or $(DEVICE),auto) \
	    --sol-size $(or $(SOL_SIZE),100) \
	    --num-epochs $(or $(NUM_EPOCHS),3000) \
	    --learning-rate $(or $(LEARNING_RATE),0.1) \
	    --temp 1e-3 \
	    --curve-rate 4 \
	    --gamma-min -2 \
	    --gamma-max 0.1 \
	    --div-param $(or $(DIV_PARAM),0.2) \
	    --penalty $(or $(PENALTY),2.0) \
	    $(if $(filter 1,$(PARALLEL)),--parallel,) \
	    --output $(or $(OUTPUT),bench_discs_paper.json)

clean:  ## remove build artefacts and caches
	rm -rf dist/ site/ .pytest_cache/ .ruff_cache/ build/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
