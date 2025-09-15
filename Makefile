SHELL := /bin/bash

.PHONY: help setup test test-cov lint format start stop status logs validate import-models telegram-debug pre-commit-install pre-commit-run

help:
	@echo "Available targets:"
	@echo "  make setup              - Create venv and install dev deps"
	@echo "  make test               - Run pytest"
	@echo "  make test-cov           - Run pytest with coverage"
	@echo "  make lint               - Run flake8 on src"
	@echo "  make format             - Run black on src and top-level *.py"
	@echo "  make start/stop         - Start or stop the system"
	@echo "  make status/logs        - Tmux manager status or logs"
	@echo "  make validate           - Quick system validation"
	@echo "  make import-models      - Import packaged models"
	@echo "  make telegram-debug     - Debug Telegram integration"
	@echo "  make pre-commit-install - Install pre-commit hooks"
	@echo "  make pre-commit-run     - Run pre-commit on all files"

setup:
	python -m venv venv && source venv/bin/activate && pip install -U pip && pip install -e .[dev]

test:
	pytest -q

test-cov:
	pytest --cov=src -q

lint:
	flake8 --jobs=1 src

format:
	black src *.py

start:
	./start_system.sh

stop:
	./stop_system.sh

status:
	./scripts/tmux_manager.sh status

logs:
	./scripts/tmux_manager.sh logs

validate:
	python quick_test_system.py

import-models:
	./import_models.sh

telegram-debug:
	bash debug_telegram.sh

pre-commit-install:
	python -m pip install pre-commit && pre-commit install

pre-commit-run:
	pre-commit run --all-files
