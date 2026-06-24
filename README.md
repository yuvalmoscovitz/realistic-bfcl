# Realistic-BFCL

**Your evals are cleaner than your users.** Clean evals tell you whether your
agent *can* do the task. They do not tell you whether it *will* once a real user
phrases the request the way real users actually do.

Realistic-BFCL is a realism-controlled metamorphic benchmark layer over the
Berkeley Function Calling Leaderboard (BFCL) built to measure that gap. It keeps
BFCL's trusted gold oracle, then rephrases each prompt the way production
traffic arrives: terse, casual, impatient, padded with context, occasionally
rude. Then it checks whether the same model still makes the same correct tool
call.

The research question is whether high clean BFCL scores imply robust real-world
tool routing. They mostly do for capable models, but not entirely, and the gap
is invisible to the clean benchmark. This project locates where it opens.

## Minimal Example

Clean BFCL prompt:

```text
What's cost of 2 and 4 GB RAM machine on AWS EC2 with one CPU?
```

Realistic noisy prompt:

```text
aws ec2 2gb 1cpu price and 4gb 1cpu price pls
```

The tool schema and gold answer are unchanged. The model should still call the
same pricing tool twice: once for `2 GB` and once for `4 GB`. If it only calls
the tool once, that is a paired clean-success/noisy-failure regression.

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

## Current Findings

The short version: **clean BFCL accuracy overstates how robust a model is to the
way people actually type, and the clean benchmark cannot see the gap.** It is
small on capable models and larger on cheap ones, but it is real and it
concentrates in ordinary user input.

We ran the same 2,351-example BFCL-derived paired evaluation across three
models: a cheap model, a capable mid-tier model, and a strong open
function-calling model. All runs used temperature `0`, deterministic
oracle-preserving rewrites, and exact McNemar tests on paired flips.
Drop values are absolute percentage-point drops (`pp`), not relative percent
change.

| Model | Pool | Clean acc. | Avg noisy acc. | Avg drop |
|---|---:|---:|---:|---:|
| `gpt-5.4-nano` | 2,351 | 76.1% | 74.9% | 1.2 pp |
| `claude-haiku-4-5-20251001` | 2,351 | 83.2% | 82.7% | 0.5 pp |
| `z-ai/glm-4.6` | 2,351 | 84.5% | 83.6% | 0.9 pp |

The effect is **not** "messy prompts break every model equally," and it is
**not** a clean capability gradient. Aggregate degradation is largest on the
cheap model, but not monotonic: GLM-4.6 has the highest clean accuracy yet shows
a larger aggregate drop than Haiku. The cheap model is fragile to a broad range
of phrasing noise; capable models shed almost all of it. What survives is
specific.

The through-line is **telegraphic shorthand**: terse, grammar-free phrasing, one
of the most common ways real users address an LLM. It degrades every model, and
it is the only dimension significant after multiple-comparison correction on
both capable models. Profanity, captured by `cursing`, is a secondary signal.
`pasted_context_block` is the instructive reversal: it is the largest and most
significant degradation on the cheap model, but a coin flip on both capable
models.

A symmetric first-pass artifact screen for Haiku and GLM reduces both
clean-to-noisy failures and noisy-to-clean fixes. `telegraphic_request` remains
significant for both capable models after that screen; GLM-4.6 `cursing` also
survives, while Haiku `cursing` does not. The remaining failures are mostly
plausible-looking wrong tool calls, dropped calls, or wrong arguments, not
malformed outputs.

That is the can-vs.-will gap made concrete. Every model here *can* price two
machines when asked in clean prose, and the clean benchmark confirms it. But the
moment the request arrives as
`aws ec2 2gb 1cpu price and 4gb 1cpu price pls`, the model that passed the clean
test starts dropping the second machine. Clean evals never show you that,
because clean prompts are never written that way.

See [docs/findings.md](docs/findings.md) for the full research note,
per-dimension tables, McNemar results, interpretation, and limitations.

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

GLM-4.6 was served via OpenRouter pinned to the DeepInfra backend for the
reported full-pool run:

```bash
REALISTIC_BFCL_MODELS=openrouter:z-ai/glm-4.6:strong-open \
REALISTIC_BFCL_OPENROUTER_PROVIDER_ONLY=DeepInfra \
REALISTIC_BFCL_ROUTER_MAX_OUTPUT_TOKENS=1024 \
python scripts/run_stage.py run-bfcl
```

Because GLM-4.6 is open weights and is served at varying precision across
providers, treat its absolute clean accuracy as provider-dependent. The paired
design controls for this: clean and noisy prompts share the same serving route,
so the degradation comparison is internally valid regardless of precision.
