.PHONY: install ab churn test lint fmt

install:
	uv sync

ab:
	uv run suban ab

churn:
	uv run suban churn

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

fmt:
	uv run ruff format .
	uv run ruff check --fix .
