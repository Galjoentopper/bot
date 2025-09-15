# Repository Guidelines

## Project Structure & Module Organization
- Source in `src/` (e.g., `src/trading`, `src/models`, `src/utils`).
- Tests in `tests/` with `unit/`, `integration/`, and `system/` subfolders.
- Config in `config/` (top‑level `config.yaml` is a symlink to a training config).
- Automation in `scripts/` and top‑level `*.sh` helpers.
- Data/artifacts: `data/`, `models/`, `mlruns/`, `reports/`, `logs/`.

## Build, Test, and Development Commands
- `make setup` — Create venv and install dev deps (`pip install -e .[dev]`).
- `make test` — Fast pytest run for local iteration.
- `make test-cov` — Pytest with coverage for `src/`.
- `make lint` — Run flake8 on `src/`.
- `make format` — Run black on `src` and top‑level `*.py`.
- `make pre-commit-install` / `make pre-commit-run` — Install and run hooks.
- Examples: `pytest -m unit`, `pytest -m "integration and not slow"`.

## Coding Style & Naming Conventions
- Python 3; 4‑space indentation; max line length 100.
- Format with Black; import order via isort (`profile=black`).
- Lint with flake8 (ignores: E203, W503, E501); type‑check with mypy.
- Naming: modules/files `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`.

## Testing Guidelines
- Framework: pytest; markers: `unit`, `integration`, `performance`, `model`, `trading`, `config`, `slow`.
- Coverage threshold: 80% (see `pytest.ini`).
- Naming: files `test_*.py` or `*_test.py`; tests `def test_*`.
- Run subsets: `pytest tests/unit`, `pytest -m integration`.

## Commit & Pull Request Guidelines
- Prefer Conventional Commits: `feat: ...`, `fix: ...`, `chore: ...` (imperative, focused; reference issues).
- PRs must include description, motivation/context, test plan (commands + results), and relevant screenshots/logs.
- Ensure CI passes (lint, type‑checks, tests); run `make pre-commit-run` before pushing.

## Security & Configuration Tips
- Never commit secrets. Use `.env.template`; keep real values in `.env`.
- Validate environment with `validate_environment.py`; review `SECURITY_IMPLEMENTATION_SUMMARY.md`.
- Be mindful of data paths and external services; mark such tests `external` or `slow`.

## Architecture & Planning Standards
- Clarify requirements and constraints before coding; outline design and dependencies.
- Break work into TODOs with clear scope and owners.
- Document risks, edge cases, and mitigations.
- Aim for SOLID design, clear errors/logging, and mockable interfaces.
