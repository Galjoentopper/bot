## Summary
Briefly describe the change and its scope.

## Risk & Service Impact
- Risk level: low / medium / high
- Affects running services? systemd or tmux sessions: yes / no
- Requires config change (`.env`, `training_config.yaml`): yes / no

## Testing Evidence
- Unit/integration tests run: `pytest -q` / `pytest --cov=src`
- System scripts: `python quick_test_system.py` / `python comprehensive_test_system.py` / `python final_test_system.py`
- Logs/screenshots (if applicable):

## Rollback Plan
How to revert safely (e.g., `git revert <sha>`, config toggle, or stop script).

## Checklist
- [ ] No secrets committed; uses `.env`
- [ ] Docs updated (AGENTS.md/README)
- [ ] Lint/format clean (`flake8`, `black`)
- [ ] Verified tmux/service status unaffected or planned
