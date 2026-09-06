# Every number in the README is regenerable by one of these targets.
# Windows: use `pwsh -File make.ps1 <target>` or call the python directly.

PY ?= python

.PHONY: help up down smoke probe ingest corpus corpus-offline test lint fmt clean

help:
	@echo "up     - start phoenix + qdrant"
	@echo "down   - stop them"
	@echo "smoke  - M0 gate: chat + embed + rerank + cache + trace"
	@echo "probe  - check every model id in the registry still resolves"
	@echo "ingest - fetch the corpus declared in universe.yaml (resumes)"
	@echo "corpus - M1 gate: re-runs ingest and proves it downloads nothing"
	@echo "test   - unit tests (no network)"

up:
	docker compose up -d

down:
	docker compose down

smoke:
	$(PY) -m filing.cli smoke

probe:
	$(PY) -m filing.cli probe

ingest:
	$(PY) -m filing.cli ingest

# Hits EDGAR: the only honest check of "a second run downloads zero files"
# is to do the second run. `corpus-offline` reports from the manifest alone.
corpus:
	$(PY) -m filing.cli corpus

corpus-offline:
	$(PY) -m filing.cli corpus --offline

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff format src tests

clean:
	$(PY) -m filing.cli cache --clear
