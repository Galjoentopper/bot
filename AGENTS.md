# Repository Guidelines

## Project Structure & Module Organization
- Core trading engine in `src/`; trading logic in `src/trading`, shared models in `src/models`, helpers in `src/utils`.
- Configurations live in `config/`, while automation and utilities sit in `scripts/`.
- Datasets, checkpoints, and reports belong in `data/`, `models/`, `mlruns/`, `reports/`, and `logs/`.
- Tests mirror the code: unit specs in `tests/unit`, integration flows in `tests/integration`, and end-to-end coverage in `tests/system`.

## Build, Test, and Development Commands
- `make setup`: create the virtualenv and install editable dependencies.
- `make test`: run the default pytest suite with repository markers from `pytest.ini`.
- `make test-cov`: execute pytest with coverage reporting stored under `reports/`.
- `make lint`: run flake8, mypy, and ancillary linters; resolve all warnings before committing.
- `make format`: apply black and isort (profile `black`) to keep imports and formatting aligned.
- Targeted runs: `pytest -m unit` or `pytest -m "integration and not slow"` for focused feedback loops.

## Coding Style & Naming Conventions
- Python 3 codebase with four-space indentation and a 100-character soft limit.
- black and isort enforce formatting; run them via `make format` before opening a PR.
- flake8 operates with ignores E203, W503, E501. Keep mypy clean and include explicit type hints.
- Use snake_case for modules, functions, and variables; CamelCase for classes; SCREAMING_SNAKE_CASE for constants.

## Testing Guidelines
- Pytest discovers files named `test_*.py` or `*_test.py` and functions prefixed with `test_`.
- Maintain at least 80% coverage; validate with `make test-cov` and review the generated HTML report.
- Share fixtures from `tests/conftest.py`, mock external services unless a test is marked `external`.
- Apply markers like `unit`, `integration`, `trading`, and `slow` so CI selectors remain reliable.

## Commit & Pull Request Guidelines
- Follow Conventional Commits, e.g., `feat: add hedging agent` or `fix: correct order sizing`, referencing issue IDs when helpful.
- PRs must outline motivation, summarize changes, list verification commands, and attach logs or screenshots for trading simulations.
- Wait for green CI before requesting review and document any schema or API shifts directly in the PR body.

## Security & Configuration Tips
- Copy `.env.template` to `.env`, keep secrets out of Git, and run `python validate_environment.py` after environment tweaks.
- Review `SECURITY_IMPLEMENTATION_SUMMARY.md` before modifying integrations or external touchpoints.
- Tag long-running or network-bound tests with `slow` or `external` to protect the pipeline.
