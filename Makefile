.PHONY: status lint typecheck prepare-subset augment run-bfcl analyze plots

PYTHON ?= python
RUN_STAGE = $(PYTHON) scripts/run_stage.py

status:
	$(RUN_STAGE) --list

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy --follow-imports=skip \
		src/realistic_bfcl/common.py \
		src/realistic_bfcl/pipeline.py \
		src/realistic_bfcl/run_manifest.py \
		src/realistic_bfcl/stats.py \
		scripts/run_stage.py \
		scripts/check_staged_env_files.py \
		plots.py

prepare-subset:
	$(RUN_STAGE) prepare-subset

augment:
	$(RUN_STAGE) augment

run-bfcl:
	$(RUN_STAGE) run-bfcl $(if $(MODELS),--models $(MODELS),)

analyze:
	$(RUN_STAGE) analyze $(if $(MANIFEST),--run-manifest $(MANIFEST),)

plots:
	$(PYTHON) plots.py
