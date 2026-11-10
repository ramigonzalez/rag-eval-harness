PYTHON ?= python

.PHONY: install test smoke bench

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m raglab run --config configs/smoke.yaml --check configs/smoke_thresholds.json

bench:
	$(PYTHON) -m raglab run --config configs/scifact.yaml

generation:
	$(PYTHON) -m raglab run --config configs/generation.yaml
