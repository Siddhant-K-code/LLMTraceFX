# LLMTraceFX Makefile

.PHONY: help install install-dev sync test lint format clean run-sample run-server deploy-modal

help:  ## Show this help message
	@echo "LLMTraceFX - GPU-level LLM inference profiler"
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

format:  ## Format code
	uv run black llmtracefx/
	uv run isort llmtracefx/
	uv run ruff format llmtracefx/

format-check:  ## Check formatting
	uv run black --check llmtracefx/
	uv run isort --check-only llmtracefx/
	uv run ruff format --check llmtracefx/

# Running
run-sample:  ## Run analysis on sample trace
	uv run llmtracefx --trace sample

run-server:  ## Run FastAPI server
	uv run llmtracefx-serve

create-sample:  ## Create sample trace file
	uv run llmtracefx --create-sample

# Modal deployment
deploy-modal:  ## Deploy to Modal
	uv run modal deploy llmtracefx/modal_app.py

serve-modal:  ## Serve on Modal
	uv run modal serve llmtracefx/modal_app.py::run_server

test-modal:  ## Test Modal functions
	uv run modal run llmtracefx/modal_app.py

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
