.PHONY: status lint freeze-bfcl clean-baseline augment-typos augment-cursing augment-irrelevant-context augment-removed-spaces augment-argumentative review-augmentations verify-noisy paired-eval analyze defenses

PYTHON ?= python
RUN_STAGE = $(PYTHON) scripts/run_stage.py

status:
	$(RUN_STAGE) --list

lint:
	$(PYTHON) -m ruff check .

freeze-bfcl:
	$(RUN_STAGE) freeze-bfcl

clean-baseline:
	$(RUN_STAGE) clean-baseline

augment-typos:
	$(RUN_STAGE) augment-typos

augment-cursing:
	$(RUN_STAGE) augment-cursing

augment-irrelevant-context:
	$(RUN_STAGE) augment-irrelevant-context

augment-removed-spaces:
	$(RUN_STAGE) augment-removed-spaces

augment-argumentative:
	$(RUN_STAGE) augment-argumentative

review-augmentations:
	$(RUN_STAGE) review-augmentations

verify-noisy:
	$(RUN_STAGE) verify-noisy

paired-eval:
	$(RUN_STAGE) paired-eval

analyze:
	$(RUN_STAGE) analyze

defenses:
	$(RUN_STAGE) defenses
