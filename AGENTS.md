# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core package (trading, backtesting, utils, notifier, config, rl_env, adapters).
- Tests: root-level `test_*.py` near features for fast iteration.
- Ops & deployment: `scripts/`, `server/`, `systemd/`, `cron/`.
- Assets & artifacts: `data/`, `logs/`, `reports/`, `models/`, `model_packages/`.
- Config: `.env` (see `.env.example`) and `training_config.yaml`.

## Build, Test, and Development Commands
- Create venv & install: `python -m venv venv && source venv/bin/activate && pip install -U pip && pip install -e .[dev]` (or `-r requirements.txt`).
- Run tests: `pytest -q` or `pytest --cov=src -q` for coverage.
- Lint/format: `flake8 src` and `black src *.py`.
- Run locally: `python telegram_bot_listener_systemd.py` or `./start_system.sh` / `./stop_system.sh`.
- Discovery: `python scripts/enhanced_trader.py --config training_config.yaml --show-available` (prints JSON of available symbols/models and exits).

Note: On servers, prefer `pip install -r requirements.txt` for full runtime deps. Use `pip install -e .[dev]` for editable local development.

## Coding Style & Naming Conventions
- Python 3.8+, 4-space indent, type hints + short docstrings for public APIs.
- Black for formatting; Flake8 for linting (fix or justify warnings).
- Naming: files/modules and functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.

## Testing Guidelines
- Use `pytest` (+ `pytest-asyncio`, `pytest-cov`).
- Names: `test_<unit>.py` and `test_<behavior>()`.
- Mock network/Exchanges (Telegram, Binance, CCXT, HTTP). Avoid long trainings; seed ML/RL when possible.

### Testing Scripts
- Fast smoke test: `python quick_test_system.py` (under a minute; no service changes).
- Integration checks: `python comprehensive_test_system.py` (broader end-to-end validations).
- Final validation: `python final_test_system.py` (production readiness; longest runtime).

Run a single test or function:
- One file: `pytest -q test_trading_system.py`
- One test: `pytest -q test_trading_system.py::test_basic_flow`

## Commit & Pull Request Guidelines
- Commits: imperative, concise (e.g., `Fix telegram startup error`), one logical change.
- Add context in body for behavior/config changes.
- PRs: motivation, linked issues, test evidence (e.g., `pytest --cov`), relevant logs/screenshots.

## Security & Configuration Tips
- No secrets in VCS; use `.env`. Validate via `src/config/`; fail fast.
- Scrub logs; they may include identifiers or balances.

## Architecture Overview
- Data pipeline: `src/data_pipeline/` (features, preprocessing; multi-source with failover).
- Models: `src/models/` (GRU, LightGBM, PPO) with per‑symbol artifacts in `models/{type}/{SYMBOL}/` and metadata for feature alignment.
- Trading engine: `scripts/trader.py` (paper trading; risk mgmt; ensemble signals with weights from `training_config.yaml`).
- Notifications: `src/notifier/enhanced_telegram.py` for alerts and status.

## Configuration Notes
- `portfolio_optimization`: tune buy scaling
  - `correlation_threshold` (default 0.8), `correlation_min_scale` (default 0.5)
  - `cash_min_pct` (default 0.1), `cash_min_scale` (default 0.5)
- `notifications.telegram.enabled`: set `false` to disable Telegram cleanly (no warnings).

## Agent-Specific Instructions
- Production server only: never train here; import via `./import_models.sh` and verify under `models/`.
- Prefer quick checks: `python quick_test_system.py` and `pytest -q` before PRs.
- Telegram work in `src/notifier/`; validate with `python test_telegram_commands.py` or `bash debug_telegram.sh`.
- New exchange/data code in `src/adapters/`; mock in tests; no hard-coded paths/secrets.
- Ops via `./scripts/tmux_manager.sh {start|status|logs|attach}`; deploy with `./deploy_full_system.sh` after tests pass.

## Deploy Safely (Checklist)
- Check sessions: `./scripts/tmux_manager.sh status` and systemd `sudo systemctl status trading-bot`.
- Validate configs: `.env` present and up to date; `training_config.yaml` matches intended symbols.
- Dry-run tests: `python quick_test_system.py` and `pytest -q`.
- Tail logs on deploy: `./scripts/tmux_manager.sh logs`.
- Rollback plan: be ready to `git revert <sha>` or `./stop_system.sh`.
