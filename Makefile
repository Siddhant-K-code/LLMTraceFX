# LLMTraceFX Makefile

.PHONY: help install install-dev sync test lint lint-changed test-ratchet format format-check clean run-sample run-server deploy-modal install-modal glm-recipe glm-budget glm-plan test-deploy metal-evidence

help:  ## Show this help message
	@echo "LLMTraceFX - evidence-first inference toolkit"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install:  ## Install package with uv
	uv sync

install-dev:  ## Install package with development dependencies
	uv sync --extra dev --extra test --extra docs

sync:  ## Sync dependencies
	uv sync

# Development
test:  ## Run tests
	uv run pytest

test-cov:  ## Run tests with coverage
	uv run pytest --cov=llmtracefx --cov-report=html --cov-report=term

lint:  ## Run linting
	uv run ruff check llmtracefx/
	uv run mypy llmtracefx/

lint-changed:  ## Run the CI quality ratchet over files changed vs origin/main
	./scripts/lint-changed.sh origin/main

test-ratchet:  ## Test the ratchet's own file selection and failure handling
	./scripts/test-lint-changed.sh

# Formatting scope note: the CI ratchet checks every changed *.py file wherever
# it lives, not just llmtracefx/. These targets match that scope so `make format`
# and CI cannot disagree. Scoping them to llmtracefx/ used to hide two files at
# the repo root, launch_dashboard.py and generate_trace.py, which fail both black
# and isort: editing either one meant CI rejected formatting that `make format`
# had refused to fix. black and isort apply their own default exclusions, so `.`
# walks exactly the 84 tracked Python files and no virtualenv or build output.
#
# `lint` below stays on llmtracefx/ deliberately. mypy is configured strictly
# enough that ordinary untyped test functions fail it, so the ratchet scopes mypy
# the same way.
format:  ## Format code (repository wide, matching the CI ratchet scope)
	uv run black .
	uv run isort .

format-check:  ## Check formatting (repository wide, matching the CI ratchet scope)
	uv run black --check .
	uv run isort --check-only .

# Running
run-sample:  ## Run the legacy analyzer on a sample trace
	uv run llmtracefx --trace sample

run-server:  ## Run the legacy analyzer API
	uv run llmtracefx-serve

run-dashboard:  ## Run the legacy Streamlit dashboard
	uv run streamlit run llmtracefx/realtime_dashboard.py --server.port=8501 --server.address=0.0.0.0

create-sample:  ## Create sample trace file
	uv run llmtracefx --create-sample

generate-traces:  ## Generate various example trace files
	uv run python generate_trace.py --profile optimized --tokens "Fast" "optimized" "inference" --output llmtracefx/test_traces/fast_trace.json
	uv run python generate_trace.py --profile memory_bound --tokens "Slow" "memory" "bound" "workload" --output llmtracefx/test_traces/slow_trace.json
	uv run python generate_trace.py --profile balanced --tokens "Normal" "balanced" "performance" "example" --output llmtracefx/test_traces/balanced_trace.json
	@echo "✅ Generated example trace files in llmtracefx/test_traces/"

# Modal deployment
install-modal:  ## Install the optional Modal SDK extra
	uv sync --extra modal

deploy-modal:  ## Deploy the legacy analyzer to Modal (paid)
	uv run --extra modal modal deploy llmtracefx/modal_app.py

serve-modal:  ## Serve the legacy analyzer on Modal (can incur charges)
	uv run --extra modal modal serve llmtracefx/modal_app.py::run_server

test-modal:  ## Run legacy Modal functions (can incur charges)
	uv run --extra modal modal run llmtracefx/modal_app.py

# GLM-5.3-Flash self-hosting harness. `glm-recipe`, `glm-budget` and
# `glm-plan` are offline: they need no Modal account and cannot spend.
# See SELF_HOST_GLM_RUNBOOK.md for the full sequence.
glm-recipe:  ## Print the pinned GLM-5.3-Flash facts and their sources
	uv run llmtracefx-deploy recipe

CREDIT ?= 30
glm-budget:  ## Recommend a session spending cap (make glm-budget CREDIT=30)
	uv run llmtracefx-deploy budget --credit-usd $(CREDIT)

glm-plan:  ## Adjudicate a deployment plan (make glm-plan ARGS="--max-usd 10 ...")
	uv run llmtracefx-deploy plan $(ARGS)

test-deploy:  ## Run the offline deployment harness tests
	uv run pytest tests/deploy

metal-evidence:  ## Capture privacy-safe Metal evidence (requires OUTPUT_DIR)
	@test -n "$(OUTPUT_DIR)" || { echo "OUTPUT_DIR is required" >&2; exit 2; }
	uv run python examples/metal_evidence/evidence_demo.py capture --output-dir "$(OUTPUT_DIR)"

# Documentation
docs:  ## Build documentation
	uv run mkdocs build

docs-serve:  ## Serve documentation locally
	uv run mkdocs serve

# Cleanup
clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf output/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# CI/CD
ci:  ## Run CI checks
	make format-check
	make lint
	make test

# Build
build:  ## Build package
	uv build

# Pre-commit hooks
install-hooks:  ## Install pre-commit hooks
	uv run pre-commit install

run-hooks:  ## Run pre-commit hooks
	uv run pre-commit run --all-files

# All-in-one commands
setup:  ## Setup development environment
	uv sync --extra dev --extra test
	make install-hooks
	make create-sample

check-all:  ## Run all checks
	make format-check
	make lint
	make test
	@echo "✅ All checks passed!"

# Help target (default)
.DEFAULT_GOAL := help
