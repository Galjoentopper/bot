SHELL := /bin/bash

.PHONY: setup test test-cov lint format start stop status logs validate import-models telegram-debug pre-commit-install pre-commit-run

setup:
	python -m venv venv && source venv/bin/activate && pip install -U pip && pip install -e .[dev]

test:
	pytest -q

test-cov:
	pytest --cov=src -q

lint:
	flake8 src

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
