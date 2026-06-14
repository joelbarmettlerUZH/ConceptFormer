.PHONY: check lint typecheck test test-all

# One-command quality gate (run before committing).
check: lint typecheck test

lint:
	uv run ruff check .

typecheck:
	uv run ty check

test:
	uv run pytest -q -m "not integration"

test-all:  # includes live-API integration tests (can flake on Wikidata maxlag)
	uv run pytest -q
