# Realistic-BFCL

Realistic-BFCL is a realism-controlled metamorphic benchmark layer over the
Berkeley Function Calling Leaderboard (BFCL).

The research question is whether high clean BFCL scores imply robust
real-world tool routing. Real users add irrelevant context, reveal intent over
multiple turns, correct themselves, type casually, sound impatient, mix
languages, and omit words while remaining understandable. This project measures
where tool routers fail under ordinary production-like conversational noise.

## Core Claim

BFCL provides the trusted deterministic evaluation substrate. Realistic-BFCL
adds realistic conversational transformations on top of BFCL while preserving
the gold tool-call oracle. The benchmark is designed for paired evaluation:

```text
same base example
same tool schema
same model
clean prompt vs. realistic noisy prompt
```

The noisy prompt must preserve the original function name and arguments unless
the transformation explicitly models correction or self-repair. In that case,
the final oracle must be well-defined and derived from the clean oracle.

## Repository Map

```text
configs/
  project.yaml                 Reproducibility pins and current assumptions.
  realism_dimensions.yaml      Initial augmentation dimensions and limits.
  subsets/smoke.yaml           First clean subset definition.
docs/
  research_pipeline.md         Staged execution plan.
  realism_contract.md          Validity rules and rejection criteria.
  evaluation_metrics.md        Paired metrics and error taxonomy.
src/realistic_bfcl/
  pipeline.py                  Stage registry used by scripts and Makefile.
  contracts.py                 Lightweight data contracts for examples.
scripts/
  run_stage.py                 Single entry point for staged execution.
Makefile                       Human-facing stage commands.
```

## Current Stage Commands

The stage commands are intentionally lightweight placeholders. They make the
pipeline executable and fail clearly until the corresponding research artifact
is implemented.

```bash
make status
make lint
make freeze-bfcl
make clean-baseline
make augment-overhang
make augment-mobile-shorthand
make augment-impatient-tone
make augment-messy-punctuation
make augment-social-filler
make verify-noisy
make paired-eval
make analyze
make defenses
```

To inspect all registered stages:

```bash
python scripts/run_stage.py --list
```

To dry-run a stage and see its expected inputs and outputs:

```bash
python scripts/run_stage.py clean-baseline --dry-run
```

`freeze-bfcl` expects access to a checkout of the pinned BFCL upstream repository.
Set `REALISTIC_BFCL_BFCL_ROOT=/path/to/gorilla` when the checkout is not in the
default local inspection path.

`clean-baseline` runs `oracle_replay` and `gpt-5.4-nano`. Provide the OpenAI key
through `OPENAI_API_KEY`, `REALISTIC_BFCL_ENV_FILE=/path/to/.env`, or a sibling
`../underlayer/.env` file. Missing model predictions run in parallel with
`REALISTIC_BFCL_CONCURRENCY`, which defaults to `8`.

## Immediate Milestones

1. Pin BFCL dataset commit, evaluator version, model list, and clean subset.
2. Reproduce clean BFCL-style scores on the selected subset.
3. Implement five high-realism transformations:
   - conversational overhang
   - casual mobile shorthand
   - impatient tone
   - messy punctuation and casing
   - casual social filler
4. Add automatic invariant checks for oracle preservation.
5. Run paired clean-vs-noisy evaluation and report degradation.

## Non-Goals For The First Pass

- Large perturbation catalogs.
- Adversarial or unnatural prompts.
- Unverified synthetic examples.
- Model-specific prompt tuning before the clean baseline is reproduced.
