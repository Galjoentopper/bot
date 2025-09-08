# Deploy & Rollback Playbook

This server runs the paper-trading system only. Use this checklist for safe deploys and fast rollbacks.

## Before Deploy
- Status: `./scripts/tmux_manager.sh status` and `sudo systemctl status trading-bot`
- Configs: `.env` present and valid; `training_config.yaml` matches intended symbols.
- Quick checks: `python quick_test_system.py` and `pytest -q`
- Logs ready: `./scripts/tmux_manager.sh logs`

## Deploy
- Start/Restart via scripts:
  - Start: `./start_system.sh`
  - Stop: `./stop_system.sh`
- Verify: watch logs for errors and successful start.

## Rollback (Git)
1) Identify last good commit: `git log --oneline -n 10`
2) Revert current commit: `git revert <bad_sha>` (or `git checkout <good_sha>` for quick local test)
3) Install if needed: `pip install -e .` (if deps changed)
4) Restart: `./stop_system.sh && ./start_system.sh`
5) Verify: `./scripts/tmux_manager.sh status` and logs

## Rollback (Config/Models)
- Config: restore known-good `.env` and `training_config.yaml`
- Models: re-import packaged models: `./import_models.sh` and verify under `models/`

## Post-Rollback Validation
- `python quick_test_system.py`
- Targeted tests (if relevant): `pytest -q test_<area>.py`
- Confirm notifier health (Telegram): `bash debug_telegram.sh`

## Notes
- Never commit secrets; use `.env`
- Keep one logical change per commit for easier reverts
