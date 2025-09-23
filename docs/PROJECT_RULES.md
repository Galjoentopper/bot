# Project Rules

These rules clarify how development, training, and trading activities are
separated across the available environments.

1. **Repository Source of Truth**: All code changes originate from this
   repository. Keep `main` deployable; use feature branches for work in progress.
2. **Environment Separation**:
   - **Paperspace Training Machine**: Clone the repository and run only the
     `train.ipynb` notebook (and related training utilities) to produce or update
     models. Do not run trading services or background schedulers here.
   - **Trading Server (current machine)**: Hosts the live trading stack,
     monitoring, and any automation that interacts with exchanges. Do not run
     long-running model-training notebooks on this server.
3. **Model Artifacts**: Export trained models from Paperspace to the shared
   storage (for example, S3) before deploying them on the trading server.
   Document the Paperspace → shared storage → trading server hand-off so the
   same bundle is promoted end-to-end. Include symbol/model identifiers and
   timestamps in filenames to avoid overwriting older bundles, and record the
   exact path used during deployment.
4. **Data Refresh**: Rebuild the SQLite caches (or otherwise refresh raw data)
   before every training run so the notebook never consumes stale candles.
5. **Configuration Hygiene**: Never commit secrets. Store environment-specific
   values in `.env` files kept outside version control.
6. **Validation Expectations**: Before shipping changes, run `make lint` and the
   appropriate `pytest` target (unit for quick checks, integration/system for
   release readiness).
7. **Incident Response**: If an unexpected change appears on either environment,
   pause, alert the team, and decide next steps before continuing work.

## Agent Expectations

- Surface residual risks whenever tests cannot be executed or assumptions were
  required, so reviewers understand what still needs verification.
- Flag suspicious metrics (for example, perfect validation/test scores) instead
  of treating them as automatic success; highlight potential overfitting or data
  leakage for follow-up.

These rules complement the guidelines in `README.md` and `AGENTS.md`; update this
document when the workflow or infrastructure changes.
