#!/usr/bin/env bash
# Run full Paperspace training workflow with environment file configuration.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<USAGE
Usage: $0 <env-file> [--skip-s3] [--skip-setup]

Arguments:
  <env-file>   Path to the environment file to source before running.

Options:
  --skip-s3    Skip S3 storage provisioning step.
  --skip-setup Skip environment bootstrap step (paperspace_setup.py).
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ENV_FILE="$1"
shift

SKIP_S3=0
SKIP_SETUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-s3)
      SKIP_S3=1
      shift
      ;;
    --skip-setup)
      SKIP_SETUP=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Environment file not found: $ENV_FILE" >&2
  exit 1
fi

echo "Loading environment from $ENV_FILE"
set -o allexport
# shellcheck source=/dev/null
source "$ENV_FILE"
set +o allexport

echo "Using repository root: $REPO_ROOT"

GIT_REMOTE_URL="${BOT_REPO_URL:-}"
if [[ -n "$GIT_REMOTE_URL" ]]; then
  echo "Pulling latest changes from $GIT_REMOTE_URL"
  git -C "$REPO_ROOT" pull "$GIT_REMOTE_URL"
else
  echo "Pulling latest changes from default remote"
  git -C "$REPO_ROOT" pull --ff-only || true
fi

cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -f "requirements.txt" ]]; then
  echo "Installing base requirements"
  "$PYTHON_BIN" -m pip install --ignore-installed blinker -r requirements.txt
fi

if [[ -f "paperspace_mlops/requirements_paperspace.txt" ]]; then
  echo "Installing Paperspace-specific requirements"
  "$PYTHON_BIN" -m pip install -r paperspace_mlops/requirements_paperspace.txt
fi

if [[ $SKIP_SETUP -eq 0 ]]; then
  echo "Running Paperspace environment setup"
  "$PYTHON_BIN" paperspace_mlops/paperspace_setup.py
else
  echo "Skipping environment setup as requested"
fi

if [[ $SKIP_S3 -eq 0 ]]; then
  echo "Configuring S3 storage"
  if [[ -f "paperspace_mlops/setup_s3_storage.py" ]]; then
    "$PYTHON_BIN" paperspace_mlops/setup_s3_storage.py
  elif [[ -f "setup_s3_storage.py" ]]; then
    "$PYTHON_BIN" setup_s3_storage.py
  else
    echo "setup_s3_storage.py not found; skipping S3 provisioning" >&2
  fi
else
  echo "Skipping S3 storage setup as requested"
fi

echo "Starting training pipeline"
"$PYTHON_BIN" paperspace_mlops/paperspace_training.py
