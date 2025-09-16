# Repository Guidelines

## Project Structure & Module Organization
Keep production code inside `src/`, grouping features by domain (for example `src/trading`, `src/models`, `src/utils`). Mirror each module with focused tests under `tests/` using `unit/`, `integration/`, or `system/` so behaviors stay localized. Store configuration in `config/` with `config.yaml` pointing to the active training profile, keep generated data in `data/`, and place model artifacts or reports in `models/` and `reports/`. Automation scripts live in `scripts/` or top-level `*.sh` helpers.

## Build, Test, and Development Commands
Run `make setup` once to create the virtualenv and install editable dev dependencies. During iteration use `make test` for the fast pytest suite and `make test-cov` when validating coverage across `src/`. Enforce style with `make lint` and `make format`, and install or run the pre-commit hooks via `make pre-commit-install` and `make pre-commit-run`.

## Coding Style & Naming Conventions
Write Python 3 with 4-space indents, ≤100-character lines, and snake_case modules, functions, and variables. Classes use CamelCase. Format imports with isort (`profile=black`) and rely on black for layout; flake8 (ignoring E203, W503, E501) guards lint quality. Favor small, composable units and add precise docstrings where logic is non-obvious.

## Testing Guidelines
Use pytest with markers like `unit`, `integration`, `performance`, `trading`, and `slow`. Name tests `test_*` and keep them under the matching layer (`tests/unit`, etc.). Target at least 80% coverage, matching `pytest.ini`. Run subsets via `pytest tests/unit` or `pytest -m "integration and not slow"` during focused debugging.

## Commit & Pull Request Guidelines
Adopt Conventional Commit prefixes (`feat:`, `fix:`, `chore:`) written in the imperative and scoped to a single concern. Each PR should summarize motivation, note architecture or data implications, list verification commands (e.g., `make test`, `make lint`), and link issues when relevant. Attach logs or artifacts that prove expected behavior and confirm CI status before requesting review.

## Security & Configuration Tips
Never commit real credentials; populate `.env` from `.env.template` and validate with `python validate_environment.py`. Review `SECURITY_IMPLEMENTATION_SUMMARY.md` when boundaries change, tag risky scenarios with `external` or `slow`, and keep training outputs inside tracked artifact directories.
