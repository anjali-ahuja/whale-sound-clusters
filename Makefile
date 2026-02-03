.PHONY: help setup install lint format test clean run-ui run-pipeline download segment featurize cluster

help:
	@echo "WhaleSound Clusters - Makefile Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Install poetry and dependencies"
	@echo "  make install        - Install dependencies only"
	@echo ""
	@echo "Development:"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Format code with ruff"
	@echo "  make test           - Run tests with coverage"
	@echo ""
	@echo "Pipeline:"
	@echo "  make download       - Download audio files from GCS"
	@echo "  make segment        - Segment audio files"
	@echo "  make featurize      - Extract features from segments"
	@echo "  make cluster        - Run clustering pipeline"
	@echo "  make run-pipeline   - Run full pipeline (download -> cluster)"
	@echo ""
	@echo "UI:"
	@echo "  make run-ui         - Launch Streamlit UI"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove generated files and caches"

setup:
	@echo "Setting up project..."
	@if ! command -v poetry > /dev/null; then \
		echo "Installing poetry..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
	fi
	poetry install
	@echo "Setup complete!"

install:
	poetry install

lint:
	poetry run ruff check src/ tests/

format:
	poetry run ruff format src/ tests/

test:
	poetry run pytest

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info

download:
	poetry run whale-download

segment:
	poetry run whale-segment

featurize:
	poetry run whale-featurize

cluster:
	poetry run whale-cluster

run-pipeline:
	poetry run whale-pipeline

run-ui:
	poetry run whale-ui




