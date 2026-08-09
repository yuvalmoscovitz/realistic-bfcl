.PHONY: status lint typecheck prepare-subset augment run-bfcl analyze

PYTHON ?= python
RUN_STAGE = $(PYTHON) scripts/run_stage.py

status:
	$(RUN_STAGE) --list

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy --follow-imports=skip src/realistic_bfcl/pipeline.py scripts/run_stage.py

prepare-subset:
	$(RUN_STAGE) prepare-subset

augment:
	$(RUN_STAGE) augment

run-bfcl:
	$(RUN_STAGE) run-bfcl $(if $(MODELS),--models $(MODELS),)

analyze:
	$(RUN_STAGE) analyze
