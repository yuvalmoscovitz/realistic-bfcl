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
  realism_dimensions.yaml      Current realism dimensions and limits.
  subsets/smoke.yaml           Current stratified clean subset definition.
  subsets/expanded_live.yaml   Larger single-turn pilot with BFCL live categories.
docs/
  paper_flow.md                Compact paper/story structure for the benchmark.
  research_pipeline.md         Research workflow and artifact contract.
  realism_contract.md          Validity rules and rejection criteria.
  evaluation_metrics.md        Paired metrics and error taxonomy.
src/realistic_bfcl/
  augment.py                   Noisy dataset construction and invariant checks.
  evaluate.py                  BFCL/OpenAI evaluation, scoring, cache, pairing.
  analyze.py                   Degradation metrics and regression review files.
  common.py                    Shared constants and small file/config helpers.
  pipeline.py                  Thin CLI orchestration for the four research steps.
  contracts.py                 Lightweight data contracts for examples.
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

`augment` creates the frozen noisy dataset once. It currently writes nine
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

It also writes `artifacts/generated/augmentation_review.csv` for human
inspection. Deterministic invariant checks reject examples that alter numbers,
quoted strings, or visible gold argument values.

`augment-llm-pilot` creates saved LLM-generated augmentation candidates for
human review. It currently supports:

- `llm_work_context`: one workplace-style message with irrelevant context.
- `llm_prior_thread`: pasted stale ticket/thread plus the current request.
- `llm_conversation_history`: short history with stale or discarded context.
- `llm_messy_pre_intent_history`: semi-relevant pre-intent chat where the final
  user turn is appended deterministically from the clean request. Earlier turns
  are intentionally messy and distracting: stale alternatives, operational or
  personal context, mild frustration, prior assistant guesses, and overlapping
  details are allowed as long as they do not add active constraints that conflict
  with the final request.
- `llm_profane_frustration`: one user message with realistic frustration or
  profanity around the exact clean request.
- `llm_argumentative_challenge`: one user message with skeptical or challenging
  tone around the exact clean request.
- `llm_frustrated_distractor_context`: one user message that combines strong
  tone and explicitly inactive distracting context around the exact clean
  request.

`run-bfcl` evaluates clean and noisy prompts with the same model, schemas, BFCL
AST checker, cache, and parallel OpenAI calls.

`analyze` writes paired degradation metrics and review files under
`artifacts/analysis/`, including raw and adjusted degradation plus
`strong_failure_examples.csv` for qualitative inspection.

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

`run-bfcl` runs `oracle_replay` and `gpt-5.4-nano`. Provide the OpenAI key
through `OPENAI_API_KEY`, `REALISTIC_BFCL_ENV_FILE=/path/to/.env`, or a sibling
`../underlayer/.env` file. Missing model predictions run in parallel with
`REALISTIC_BFCL_CONCURRENCY`, which defaults to `8`.

## Immediate Milestones

1. Pin BFCL dataset commit, evaluator version, model list, and clean subset.
2. Reproduce clean BFCL-style scores on the selected subset.
3. Implement five high-realism transformations:
   - typos
   - cursing
   - irrelevant context
   - removed spaces
   - argumentative challenge
4. Add automatic invariant checks for oracle preservation.
5. Run paired clean-vs-noisy evaluation and report degradation.
6. Scale from the 400-example pilot to a 1000-example stratified run.

## Non-Goals For The First Pass

- Large perturbation catalogs.
- Adversarial or unnatural prompts.
- Unverified synthetic examples.
- Model-specific prompt tuning before the clean baseline is reproduced.
