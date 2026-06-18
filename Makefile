.PHONY: status lint prepare-subset augment run-bfcl analyze

PYTHON ?= python
RUN_STAGE = $(PYTHON) scripts/run_stage.py

status:
	$(RUN_STAGE) --list

lint:
	$(PYTHON) -m ruff check .

prepare-subset:
	$(RUN_STAGE) prepare-subset

augment:
	$(RUN_STAGE) augment

run-bfcl:
	$(RUN_STAGE) run-bfcl

analyze:
	$(RUN_STAGE) analyze
