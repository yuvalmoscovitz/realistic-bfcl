.PHONY: status freeze-bfcl clean-baseline augment-overhang augment-incremental verify-noisy paired-eval analyze defenses

PYTHON ?= python
RUN_STAGE = $(PYTHON) scripts/run_stage.py

status:
	$(RUN_STAGE) --list

freeze-bfcl:
	$(RUN_STAGE) freeze-bfcl

clean-baseline:
	$(RUN_STAGE) clean-baseline

augment-overhang:
	$(RUN_STAGE) augment-overhang

augment-incremental:
	$(RUN_STAGE) augment-incremental

verify-noisy:
	$(RUN_STAGE) verify-noisy

paired-eval:
	$(RUN_STAGE) paired-eval

analyze:
	$(RUN_STAGE) analyze

defenses:
	$(RUN_STAGE) defenses
