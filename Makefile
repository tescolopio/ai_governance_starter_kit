# Makefile for AI Governance Starter Kit
#
# Common commands for development, testing, and validation

.PHONY: help install install-dev test validate clean lint format setup

# Default target
help:
	@echo "AI Governance Starter Kit - Available Commands"
	@echo "==============================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install         Install runtime dependencies"
	@echo "  make install-dev     Install development dependencies"
	@echo "  make setup           Complete setup (install + pre-commit hooks)"
	@echo ""
	@echo "Validation & Testing:"
	@echo "  make validate        Validate model registry"
	@echo "  make test            Run all tests"
	@echo "  make test-coverage   Run tests with coverage report"
	@echo "  make test-registry   Run registry validation tests only"
	@echo "  make test-models     Run model performance tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            Run linting checks"
	@echo "  make format          Format code with black and isort"
	@echo "  make check-docs      Check model documentation completeness"
	@echo ""
	@echo "Model Operations:"
	@echo "  make train-examples  Train example models"
	@echo "  make list-models     List all registered models"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove build artifacts and cache files"
	@echo "  make clean-all       Deep clean (including model artifacts)"
	@echo ""

# Installation
install:
	@echo "Installing runtime dependencies..."
	pip install -r requirements.txt
	@echo "✓ Runtime dependencies installed"

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements-dev.txt
	@echo "✓ Development dependencies installed"

setup: install install-dev
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	@echo "✓ Pre-commit hooks installed"
	@echo ""
	@echo "✓ Setup complete! You're ready to go."
	@echo ""
	@echo "Next steps:"
	@echo "  1. Review the README.md for overview"
	@echo "  2. Check docs/templates/ for documentation templates"
	@echo "  3. Run 'make validate' to check the model registry"
	@echo "  4. Run 'make test' to run the test suite"

# Validation
validate:
	@echo "Validating model registry..."
	@python scripts/validate_registry.py
	@echo ""

check-docs:
	@echo "Checking model documentation..."
	@python scripts/check_documentation.py
	@echo ""

# Testing
test:
	@echo "Running all tests..."
	pytest tests/ -v
	@echo ""

test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

test-registry:
	@echo "Running registry validation tests..."
	pytest tests/test_registry_validation.py -v
	@echo ""

test-models:
	@echo "Running model performance tests..."
	pytest tests/test_model_performance.py -v
	@echo ""

# Code Quality
lint:
	@echo "Running linting checks..."
	@echo "Checking Python files with flake8..."
	flake8 models/ scripts/ tests/ --max-line-length=100 --extend-ignore=E203,W503 || true
	@echo ""
	@echo "Checking for security issues with bandit..."
	bandit -r models/ scripts/ || true
	@echo ""

format:
	@echo "Formatting Python code..."
	@echo "Running black..."
	black models/ scripts/ tests/
	@echo "Running isort..."
	isort models/ scripts/ tests/ --profile black
	@echo "✓ Code formatted"

# Model Operations
train-examples:
	@echo "Training example models..."
	@echo ""
	@echo "Training fraud detection model..."
	cd models/examples/fraud_detection && python fraud_detection_model.py
	@echo ""
	@echo "Training credit risk model..."
	cd models/examples/credit_risk && python credit_risk_model.py
	@echo ""
	@echo "✓ Example models trained"

list-models:
	@echo "Registered Models:"
	@echo "=================="
	@python -c "import yaml; registry = yaml.safe_load(open('inventory/model_registry.yaml')); \
		[print(f\"  {m['model_id']:<30} Risk: {m['risk_tier']:<10} Status: {m.get('deployment', {}).get('environment', 'N/A')}\") \
		for m in registry.get('models', [])]"
	@echo ""

# Cleanup
clean:
	@echo "Cleaning build artifacts and cache files..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '.coverage' -delete
	rm -rf htmlcov/
	rm -rf .coverage
	@echo "✓ Cleaned"

clean-all: clean
	@echo "Deep cleaning (including model artifacts)..."
	find models/ -type f -name '*.pkl' -delete
	find models/ -type f -name '*.joblib' -delete
	find models/ -type f -name '*_metadata.json' -delete
	@echo "✓ Deep clean complete"

# Pre-commit hooks
pre-commit:
	@echo "Running pre-commit hooks on all files..."
	pre-commit run --all-files
	@echo ""

# Quick validation before commit
check: validate test-registry lint
	@echo ""
	@echo "✓ All checks passed! Ready to commit."
	@echo ""

# CI/CD simulation (what runs in GitHub Actions)
ci: validate test lint
	@echo ""
	@echo "✓ CI checks complete"
	@echo ""
