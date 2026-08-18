.PHONY: help install dev test lint format clean docs build ci

help:
	@echo "Available commands:"
	@echo "  install    Install package in production mode"
	@echo "  dev        Install package in development mode with all dependencies"
	@echo "  test       Run tests with coverage"
	@echo "  lint       Run linting checks"
	@echo "  format     Format code with ruff"
	@echo "  clean      Remove build artifacts and cache files"
	@echo "  docs       Build documentation"
	@echo "  build      Build distribution packages"
	@echo "  ci         Run the local release checks"

install:
	uv pip install .

dev:
	uv sync --all-groups
	pre-commit install

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

docs:
	cd docs && make clean && make html
	@echo "Documentation built at docs/build/html/index.html"

build: clean
	uv build

ci: lint test docs build
