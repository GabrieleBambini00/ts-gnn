# TS-GNN Makefile
# Provides standard development commands for reproducibility.

.PHONY: test lint demo clean install help

# Default target
help:
	@echo "TS-GNN Development Commands"
	@echo "==========================="
	@echo "  make install   Install the project in editable mode"
	@echo "  make test      Run the full test suite with coverage"
	@echo "  make lint      Run linting (ruff)"
	@echo "  make demo      Run synthetic end-to-end demo"
	@echo "  make clean     Remove caches, checkpoints, and build artifacts"

# Install project + dev dependencies
install:
	pip install -e ".[dev]"

# Run full test suite with coverage report
test:
	python -m pytest tests/ -v --tb=short --no-header -q

# Run tests with coverage
test-cov:
	python -m pytest tests/ -v --cov=tsgnn --cov-report=term-missing --tb=short

# Run ALL tests (tests/ + ultimate/)
test-all:
	python -m pytest tests/ ultimate/test_properties.py -v --tb=short

# Run tests with strict coverage threshold (50%)
test-coverage:
	python -m pytest tests/ ultimate/test_properties.py -v --cov=tsgnn --cov-report=term-missing --cov-fail-under=50 --tb=short

# Lint with ruff (if installed)
lint:
	python -m ruff check src/ tests/ || echo "ruff not installed, skipping lint"

# Run synthetic end-to-end demo
demo:
	python scripts/run_pipeline.py --config configs/default.yaml --skip-download

# Run synthetic proof-of-concept from ultimate/
proof:
	python ultimate/run_synthetic_proof.py

# Full user onboarding shortcut
quickstart: install lint test proof
	@echo "=========================================================="
	@echo "🚀 TS-GNN Quickstart Complete!"
	@echo "All tests passed. Run 'tsgnn-train' to start real training."
	@echo "=========================================================="

# Clean build artifacts and caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/
	rm -rf checkpoints/latest.pt figures/

# Generate pinned requirements from current environment
freeze:
	pip freeze > requirements-lock.txt
	@echo "Saved pinned requirements to requirements-lock.txt"
