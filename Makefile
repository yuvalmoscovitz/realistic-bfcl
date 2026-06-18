.PHONY: status lint freeze-bfcl clean-baseline augment-overhang augment-mobile-shorthand augment-impatient-tone augment-messy-punctuation augment-social-filler verify-noisy paired-eval analyze defenses

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

augment-overhang:
	$(RUN_STAGE) augment-overhang

augment-mobile-shorthand:
	$(RUN_STAGE) augment-mobile-shorthand

augment-impatient-tone:
	$(RUN_STAGE) augment-impatient-tone

augment-messy-punctuation:
	$(RUN_STAGE) augment-messy-punctuation

augment-social-filler:
	$(RUN_STAGE) augment-social-filler

verify-noisy:
	$(RUN_STAGE) verify-noisy

paired-eval:
	$(RUN_STAGE) paired-eval

analyze:
	$(RUN_STAGE) analyze

defenses:
	$(RUN_STAGE) defenses
