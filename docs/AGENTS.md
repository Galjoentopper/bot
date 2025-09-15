# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core package (trading, backtesting, utils, notifier, config, rl_env, adapters).
- Tests: root-level `test_*.py` near features for fast iteration.
- Ops & deployment: `scripts/`, `server/`, `systemd/`, `cron/`.
- Assets & artifacts: `data/`, `logs/`, `reports/`, `models/`, `model_packages/`.
- Config: `.env` (see `.env.example`) and `training_config.yaml`.

## Build, Test, and Development Commands
- Create venv & install: `python -m venv venv && source venv/bin/activate && pip install -U pip && pip install -e .[dev]` (or `pip install -r requirements.txt`).
- Run tests: `pytest -q` or `pytest --cov=src -q` for coverage.
- Lint/format: `flake8 src` and `black src *.py`.
- Run locally: `python telegram_bot_listener_systemd.py` or `./start_system.sh` / `./stop_system.sh`.
- Discovery: `python scripts/enhanced_trader.py --config training_config.yaml --show-available` (prints JSON of symbols/models).

## Coding Style & Naming Conventions
- Python ≥ 3.8, 4-space indent, type hints, short docstrings for public APIs.
- Formatting: Black; Linting: Flake8 (fix or justify warnings).
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, `pytest-cov`.
- Names: files `test_<unit>.py`; tests `test_<behavior>()`.
- Mock external I/O (Telegram, exchanges, HTTP). Avoid long trainings; seed ML/RL.
- Quick checks: `python quick_test_system.py` (fast) and `pytest -q`.
- Run one test: `pytest -q test_trading_system.py::test_basic_flow`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise (e.g., `Fix telegram startup error`), one logical change.
- Include context for behavior/config changes; reference issues.
- PRs: motivation, linked issues, test evidence (`pytest --cov`), relevant logs/screenshots.

## Security & Configuration Tips
- Never commit secrets; use `.env`. Validate via `src/config/`; fail fast on missing keys.
- Scrub logs; they may include identifiers or balances.

## Architecture Overview
- Data: `src/data_pipeline/` (features, preprocessing; multi-source failover).
- Models: `src/models/` with artifacts under `models/{type}/{SYMBOL}/` + metadata.
- Trading engine: `scripts/trader.py` (paper trading, risk mgmt, ensemble signals).
- Notifications: `src/notifier/enhanced_telegram.py`.

## Agent-Specific Instructions & Deploy
- Production servers: never train; import via `./import_models.sh` and verify under `models/`.
- Telegram work in `src/notifier/`; validate with `python test_telegram_commands.py`.
- Ops via `./scripts/tmux_manager.sh {start|status|logs|attach}`; deploy with `./deploy_full_system.sh` after tests pass.
- Deployment checklist: ensure `.env`/`training_config.yaml` correct, run `python quick_test_system.py` and `pytest -q`, then tail logs with `./scripts/tmux_manager.sh logs`.
