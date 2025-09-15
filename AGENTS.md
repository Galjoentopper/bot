# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/` (e.g., `src/trading`, `src/models`, `src/utils`).
- Tests live in `tests/` with `unit/`, `integration/`, and `system/` subfolders.
- Config in `config/` (top-level `config.yaml` symlinks to a training config).
- Scripts and automation under `scripts/` and top‑level `*.sh` helpers.
- Data/artifacts: `data/`, `models/`, `mlruns/`, `reports/`, `logs/`.

## Build, Test, and Development Commands
- `make setup` — Create venv and install dev deps (`pip install -e .[dev]`).
- `make test` — Run pytest quickly for local iteration.
- `make test-cov` — Run tests with coverage for `src/`.
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
- Prefer Conventional Commits: `feat: ...`, `fix: ...`, `chore: ...`.
- Keep messages imperative and focused; reference issues when applicable.
- PRs must include: clear description, motivation/context, test plan (commands + results), and relevant screenshots/logs.
- Ensure CI passes (lint, type‑checks, tests); pre‑commit hooks clean.

## Security & Configuration Tips
- Never commit secrets. Use `.env.template` as a guide; keep real values in `.env`.
- Validate environment with `validate_environment.py`; see `SECURITY_IMPLEMENTATION_SUMMARY.md`.
- Be mindful of data paths and external services; mark such tests with `external` or `slow`.

## Architecture & Planning Standards
- Scope: applies repo‑wide; follow structure and `make`/pytest tooling above.
- Deep analysis: clarify requirements, constraints, and context before coding.
- Architecture plan: outline design, integration points, and dependencies.
- TODO breakdown: numbered steps with clear scope and ownership.
- Risk assessment: list edge cases, failure modes, and mitigations.
- Code quality: SOLID, separation of concerns, clear naming, meaningful logs/errors.
- Testing mindset: unit + integration, positive/negative paths, mockable interfaces.
- Operations: plan configuration, security, performance, and observability from the start.
