#!/usr/bin/env python3
"""User-friendly wrapper to launch the superior ensemble training pipeline.

This script keeps the notebook "one click" workflow: invoke it from a cell with
```
!python scripts/run_full_training.py --full
```
or run it directly from the terminal. Because the training runs in a standalone
Python process, the trainer can leverage parallel workers even when triggered
from a notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the superior ensemble training pipeline with simple flags",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run the full ensemble plan (default)",
    )
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Run the abbreviated quick test",
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Optional list of trading symbols to override the config",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["ppo", "gru", "lightgbm"],
        help="Optional list of models to train",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the final training summary as JSON (in addition to logs)",
    )

    return parser.parse_args()


def sanitize_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    return [value.upper() if value.isalpha() else value for value in values]


def run_training(args: argparse.Namespace) -> dict:
    os.chdir(PROJECT_ROOT)

    from paperspace_mlops.paperspace_superior_training import (  # noqa: WPS433
        PaperspaceTrainingRunner,
    )

    quick_test = args.quick
    if not args.quick and not args.full:
        quick_test = False

    runner = PaperspaceTrainingRunner()

    symbols = sanitize_list(args.symbols)
    models = sanitize_list(args.models)

    print("Launching Paperspace superior ensemble training")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Mode: {'quick test' if quick_test else 'full plan'}")
    if symbols:
        print(f"Symbols override: {', '.join(symbols)}")
    if models:
        print(f"Models override: {', '.join(models)}")

    try:
        result = runner.run_training(
            symbols=symbols,
            models=models,
            quick_test=quick_test,
        )
    except Exception as exc:  # pragma: no cover - pass through runner errors
        print(f"❌ Training run failed: {exc}")
        raise

    status = result.get("status", "unknown")
    print("\nTraining run finished")
    print(f"Status: {status}")
    print(f"Symbols trained: {result.get('symbols_trained', [])}")
    print(f"Models trained: {result.get('models_trained', [])}")
    export_state = result.get("export_status", {})
    export_flag = "enabled" if export_state.get("export_enabled") else "disabled"
    print(f"S3 export: {export_flag}")
    if export_state.get("models_exported"):
        print(f"Models exported: {export_state['models_exported']}")

    errors = result.get("errors", [])
    if errors:
        print("Reported errors:")
        for entry in errors:
            print(f"   - {entry}")

    if args.json:
        print("\nJSON summary:")
        print(json.dumps(result, indent=2))

    return result


def main() -> int:
    args = parse_args()
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
