# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`: trading logic in `src/trading`, shared models in `src/models`, utilities in `src/utils`. Tests are grouped by scope (`tests/unit`, `tests/integration`, `tests/system`) so markers stay predictable. Configuration belongs in `config/`, automation in `scripts/`, while datasets and experiment outputs sit under `data/`, `models/`, `mlruns/`, `reports/`, and `logs/`.

## Build, Test, and Development Commands
Run `make setup` to bootstrap the virtualenv and install editable dependencies. Use `make test` for the fast suite, `make test-cov` for coverage, and targeted selectors like `pytest -m unit` or `pytest -m "integration and not slow"` during debugging. Enforce quality with `make lint` (flake8), `make format` (Black + top-level Python files), and keep Git hooks active via `make pre-commit-install` and `make pre-commit-run`.

## Coding Style & Naming Conventions
We target Python 3, four-space indentation, and a 100-character line guide. Black handles formatting, isort (profile `black`) sorts imports, and flake8 runs with ignores E203, W503, E501. Keep modules, functions, and variables in snake_case, classes in CamelCase, and constants in SCREAMING_SNAKE_CASE. Touching code should pass mypy; prefer adding precise type hints over muting errors.

## Testing Guidelines
Pytest discovers files named `test_*.py` or `*_test.py` with test functions `def test_*`. Hold coverage at or above 80%; confirm with `make test-cov` and review reports before merging. Apply markers (`unit`, `integration`, `performance`, `model`, `trading`, `config`, `slow`, `external`) consistently so CI selectors remain stable. Use fixtures or factories to share setup; mock network calls unless the test is tagged `external`.

## Commit & Pull Request Guidelines
Adopt Conventional Commits (`feat: add hedging agent`, `fix: correct order sizing`, `chore: refresh docs`) and reference issue IDs when helpful. Keep commits focused, run lint + tests before pushing, and document schema or API shifts in the message body. Pull requests must explain motivation, summarize changes, and list the exact verification commands; attach logs or screenshots for trading simulations and wait for green CI before review.

## Security & Configuration Tips
Copy `.env.template` to `.env`, keep credentials outside version control, and never commit secrets. Validate local setups with `python validate_environment.py` ahead of training or deployments. Consult `SECURITY_IMPLEMENTATION_SUMMARY.md` when touching integrations, and tag long or network-heavy tests with `slow` or `external` to protect pipelines.

## Architecture & Planning Standards
Outline significant design changes in `docs/` or the linked issue, capturing dependencies and failure modes. Break complex efforts into scoped TODOs and rely on logs in `logs/` for debugging.
