# Realistic-BFCL

Realistic-BFCL is a realism-controlled metamorphic benchmark layer over the
Berkeley Function Calling Leaderboard (BFCL).

The research question is whether high clean BFCL scores imply robust
real-world tool routing. Real users add irrelevant context, reveal intent over
multiple turns, correct themselves, type casually, sound impatient, mix
languages, and omit words while remaining understandable. This project measures
where tool routers fail under ordinary production-like conversational noise.

## Minimal Example

Clean BFCL prompt:

```text
What's cost of 2 and 4 GB RAM machine on AWS EC2 with one CPU?
```

Realistic noisy prompt:

```text
aws pricing calculator makes no sense. i just want an ec2 machine that has 2 gigabytes of ram and 1 cpu. also the 4gb one. how much is it
```

The tool schema and gold answer are unchanged. The model should still call the
same pricing tool twice: once for `2 GB` and once for `4 GB`. If it only calls
the tool once, that is a paired clean-success/noisy-failure regression.

## What This Tests

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

## Current Findings

We ran a 2,351-example BFCL-derived paired evaluation on `gpt-5.4-nano`.

Clean accuracy was `0.761`. All seven realistic noise dimensions produced
reviewed clean-success/noisy-failure regressions after oracle and manual-review
filtering.

We repeated the evaluation three times with fresh clean and noisy model calls.
Every noise type degraded accuracy in every run. The drops are small, but the
direction is consistent for this model. This is a probe, not a leaderboard: the
point is that oracle-preserving realistic rewrites expose failures hidden by
clean prompts.

This repository currently supports a problem-posing article, not a complete
multi-model benchmark paper.

We also ran `claude-haiku-4-5-20251001` on the same 2,351-example pool through
Anthropic Message Batches at temperature `0`. Haiku's clean accuracy was `0.832`;
pooled noisy accuracy across the seven dimensions was `0.826`, a raw drop of
`0.005`. The effect is much smaller than the nano run and is not uniformly
directional across dimensions. The cleanest Haiku signal is
`telegraphic_request`: 51 clean-success/noisy-failure cases versus 17
clean-failure/noisy-success cases, McNemar exact `p = 0.000045`. Several other
dimensions are weak or near-balanced. The full-pool Haiku regressions have not
yet been manually reviewed, so treat these as raw paired metrics, not final
true-failure rates.

We also ran a controlled 250-example comparison on a rewrite-suitable subset.
Deterministic augmentations are the auditable control surface: easy to
reproduce, inspect, and reject when they touch oracle-bearing content. LLM
rewrites are the realism surface: harder to validate, but closer to actual user
traffic. On this controlled subset, deterministic dimensions had a mean raw
drop of `2.12` points, while LLM rewrite dimensions had a mean raw drop of
`2.95` points. The numeric lift is modest; the stronger reason to keep LLM
rewrites is that they make the failure examples easier to recognize as
realistic user behavior.

See [docs/findings.md](docs/findings.md) for the GitHub-facing research note,
examples, tables, and implications.

The small article-facing analysis bundle is checked in under
`artifacts/analysis/article/` for inspection. Larger generated datasets,
predictions, and intermediate analysis files remain ignored.

## Repository Map

```text
configs/
  project.yaml                 Reproducibility pins and current assumptions.
  realism_dimensions.yaml      Current realism dimensions and limits.
  subsets/smoke.yaml           Current stratified clean subset definition.
  subsets/expanded_live.yaml   Larger single-turn pilot with BFCL live categories.
docs/
  findings.md                  GitHub-facing research note and current results.
  annotation_protocol.md       Manual audit labels and rejection rules.
  research_pipeline.md         Research workflow and artifact contract.
  realism_contract.md          Validity rules and rejection criteria.
  evaluation_metrics.md        Paired metrics and error taxonomy.
src/realistic_bfcl/
  augment.py                   Noisy dataset construction and invariant checks.
  evaluate.py                  BFCL evaluation, provider adapters, batch/cache, pairing.
  analyze.py                   Degradation metrics and regression review files.
  common.py                    Shared constants and small file/config helpers.
  pipeline.py                  Thin CLI orchestration for the four research steps.
scripts/
  run_stage.py                 Single entry point for research steps.
Makefile                       Human-facing research commands.
```

## Research Commands

The repository has three core functions plus one reproducibility setup step:

```bash
make status
make lint
make prepare-subset
make augment
make run-bfcl
make analyze
```

`prepare-subset` freezes the BFCL substrate and materializes the configured
clean subset.

By default it uses `configs/subsets/smoke.yaml`. To materialize the larger
single-turn pilot, run:

```bash
REALISTIC_BFCL_SUBSET_CONFIG=configs/subsets/expanded_live.yaml make prepare-subset
```

`augment` creates the frozen noisy dataset once. It can generate ten implemented
oracle-preserving dimensions:

- `typos`
- `cursing`
- `irrelevant_context`
- `removed_spaces`
- `argumentative_challenge`
- `profane_sandwich`
- `argumentative_sandwich`
- `distractor_sandwich`
- `pasted_context_block`
- `telegraphic_request`

The current article-facing analysis uses the seven reviewed dimensions listed in
[docs/findings.md](docs/findings.md).

It also writes `artifacts/generated/augmentation_review.csv` for human
inspection. Deterministic invariant checks reject examples that alter numbers,
quoted strings, or visible gold argument values.

The article-facing run currently uses the seven reviewed dimensions in
`docs/findings.md`. Treat `cursing` as frustrated user register: profanity is a
surface marker for impatience, not the scientific claim by itself.

`augment-llm-pilot` creates saved LLM-generated augmentation candidates for
human review. These candidates are not part of the default article-facing run
until manually reviewed and promoted.

For the LLM rewrite study, first build the representative 500-example
rewrite-suitable subset:

```bash
python scripts/run_stage.py build-rewrite-subset
```

For LLM-generated realistic rewrite candidates, set the provider separately
from the provider-neutral dimension names. The provider is an implementation
detail; the benchmark dimensions are provider-neutral.

```bash
REALISTIC_BFCL_LLM_PROVIDER=openai
REALISTIC_BFCL_ENV_FILE=/path/to/.env
REALISTIC_BFCL_LLM_DIMENSIONS=llm_super_casual_abbreviations,llm_frustrated_swearing,llm_student_broke_context,llm_typos_shorthand,llm_rambling_overexplaining,llm_impatient_direct_attitude,llm_arguing_correcting_ai,llm_confused_overwhelmed,llm_swearing_urgency_work,llm_vague_slightly_aggressive
```

`run-bfcl` evaluates clean and noisy prompts with the same model, schemas, BFCL
AST checker, cache, and parallel model calls. The model list and temperature are
configured in `configs/project.yaml`; temperature is recorded in run metadata and
included in cache fingerprints. Each model writes its own cache files, so one
frozen noisy dataset can be evaluated across small, mid-tier, and frontier
models without regenerating augmentations.

`analyze` writes paired degradation metrics and review files under
`artifacts/analysis/`, including raw and adjusted degradation plus
`strong_failure_examples.csv` for qualitative inspection. It also writes
`model_comparison.csv`, a per-model/per-dimension table with clean accuracy,
noisy accuracy, degradation, clean-success/noisy-failure counts, reviewed
regression counts, and failure-type taxonomy.

To inspect all registered steps:

```bash
python scripts/run_stage.py --list
```

To dry-run a step and see its expected inputs and outputs:

```bash
python scripts/run_stage.py run-bfcl --dry-run
```

`prepare-subset` expects access to a checkout of the pinned BFCL upstream repository.
Set `REALISTIC_BFCL_BFCL_ROOT=/path/to/gorilla` when the checkout is not in the
default local inspection path.

`run-bfcl` runs `oracle_replay` and the models listed in
`configs/project.yaml`. API keys can be provided through the environment,
`REALISTIC_BFCL_ENV_FILE=/path/to/.env`, or a sibling `../underlayer/.env` file.
Missing synchronous predictions run in parallel with `REALISTIC_BFCL_CONCURRENCY`,
which defaults to `8`.

Anthropic models can also run through Message Batches, which is the practical
path for larger model comparisons:

```bash
REALISTIC_BFCL_EXECUTION=batch \
REALISTIC_BFCL_MODELS=anthropic:claude-haiku-4-5-20251001:mid \
python scripts/run_stage.py run-bfcl
```

Batch runs write the same prediction caches and paired summaries as synchronous
runs, plus a small `_batch_state.json` file for resuming submitted batches. The
Anthropic adapter also sanitizes provider-incompatible BFCL schema argument
names, then restores the original argument names before BFCL scoring.
