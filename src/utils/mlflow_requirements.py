"""Utility helpers for consistent MLflow dependency pinning."""

from __future__ import annotations

import logging
import subprocess
import sys
from importlib import metadata
from typing import Iterable, List

_DEFAULT_EXTRA_REQUIREMENTS = ["rfc3987-syntax==1.1.0"]


def get_mlflow_extra_requirements() -> List[str]:
    """Return a copy of extra pip requirements needed for MLflow artifacts."""

    return list(_DEFAULT_EXTRA_REQUIREMENTS)


def ensure_mlflow_extra_requirements(logger: logging.Logger | None = None) -> List[str]:
    """Ensure MLflow-specific extras are installed.

    Args:
        logger: Optional logger for status output.

    Returns:
        List of requirement strings that were attempted (empty if nothing missing).
    """

    missing: List[str] = []
    for requirement in _DEFAULT_EXTRA_REQUIREMENTS:
        package = requirement.split("==")[0]
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(requirement)

    if not missing:
        return []

    if logger is not None:
        logger.info("Installing MLflow extras: %s", ", ".join(missing))

    try:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=False)
    except Exception as exc:  # pragma: no cover - pip failures should not crash training
        if logger is not None:
            logger.warning("Failed to install MLflow extras: %s", exc)
    return missing
