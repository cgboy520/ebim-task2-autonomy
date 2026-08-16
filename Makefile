# EBiM Task 2 autonomy sidecar. Uses uv when available, falls back to pip.

PYTHON ?= python3
POLICY ?= waypoint
CONFIG ?= config/task2.example.yaml

.PHONY: setup check test build run dry-run clean

setup:
	@command -v uv >/dev/null 2>&1 && { \
		uv venv --python 3.12 .venv && \
		uv pip install -r requirements.txt; } || { \
		$(PYTHON) -m venv .venv && \
		. .venv/bin/activate && pip install -r requirements.txt; }

check:
	PYTHONPATH=src $(PYTHON) -m compileall -q src tests

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q tests

build:
	docker compose build policy

run:
	docker compose run --rm policy python3 -m ebim_task2.runner \
		--config /opt/ebim-task2/config/task2.local.yaml --policy $(POLICY)

dry-run:
	docker compose run --rm policy python3 -m ebim_task2.runner \
		--config /opt/ebim-task2/config/task2.example.yaml --dry-run

clean:
	rm -rf .venv __pycache__ src/ebim_task2/__pycache__ tests/__pycache__ .pytest_cache
