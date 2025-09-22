# Repository Guidelines

## Project Structure & Module Organization
The trading engine lives in `src/`, with trading logic under `src/trading`, shared models in `src/models`, and helpers in `src/utils`. Configuration files sit in `config/`, automation and scripts in `scripts/`, while datasets, experiment artifacts, and logs belong in `data/`, `models/`, `mlruns/`, `reports/`, and `logs/`. Tests mirror the codebase: fast checks in `tests/unit`, broader coverage in `tests/integration`, and end-to-end flows in `tests/system`.

## Build, Test, and Development Commands
Run `make setup` once to create the virtualenv and install editable deps. Use `make test` for the default pytest suite and `make test-cov` when you need coverage numbers. Developers often target `pytest -m unit` or `pytest -m "integration and not slow"` during debugging. Keep quality gates green with `make lint`, `make format`, and reinstall Git hooks through `make pre-commit-install` followed by `make pre-commit-run`.

## Coding Style & Naming Conventions
Code targets Python 3 with four-space indentation and a 100-character guideline. Black enforces formatting, while isort (profile `black`) keeps imports sorted, and flake8 runs with ignores E203, W503, E501. Prefer explicit type hints; lint and mypy must stay clean. Stick to snake_case for modules, functions, and variables, CamelCase for classes, and SCREAMING_SNAKE_CASE for constants.

## Testing Guidelines
Pytest discovers files named `test_*.py` or `*_test.py` and functions beginning `test_`. Aim for ≥80% coverage; confirm via `make test-cov` and review the generated report. Use shared fixtures for setup, mock external services unless a test is tagged `external`, and apply markers like `unit`, `integration`, `trading`, or `slow` so CI selectors remain reliable.

## Commit & Pull Request Guidelines
Follow Conventional Commits such as `feat: add hedging agent` or `fix: correct order sizing`, referencing issue IDs when useful. Each PR should explain motivation, summarise changes, list verification commands, and attach logs or screenshots for trading simulations. Wait for green CI before requesting review and document any schema or API shifts in the PR body.

## Security & Configuration Tips
Copy `.env.template` to `.env`, keep secrets out of Git, and run `python validate_environment.py` to confirm local setups. Review `SECURITY_IMPLEMENTATION_SUMMARY.md` before touching integrations, and tag long-running or network-dependent tests with `slow` or `external` to protect the pipeline.
