# Realistic-BFCL

[![CI](https://github.com/yuvalmoscovitz/realistic-bfcl/actions/workflows/ci.yml/badge.svg)](https://github.com/yuvalmoscovitz/realistic-bfcl/actions/workflows/ci.yml)

**Your evals are cleaner than your users.** Clean evals tell you whether your
agent *can* do the task, they don't tell you whether it *will*.

Realistic-BFCL is a realism-controlled metamorphic benchmark layer over the
Berkeley Function Calling Leaderboard (BFCL). It keeps BFCL's trusted gold
oracle, then rephrases each prompt the way production traffic arrives: terse,
casual, impatient, padded with context, occasionally rude. Then it checks
whether the same model still makes the same correct tool call.

The research question is whether high clean BFCL scores imply robust real-world
tool routing. They mostly do for capable models, but not entirely, and the gap
is invisible to the clean benchmark. This project locates where it opens.

Start here:

- Full research note: [docs/findings.md](docs/findings.md)
- Article result bundle: [artifacts/analysis/article/](artifacts/analysis/article/)
- Exhaustive failure table: [artifacts/analysis/article/clean_to_noisy_failures.csv](artifacts/analysis/article/clean_to_noisy_failures.csv)

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
adds realistic conversational transformations on top while preserving the gold
tool-call oracle. The benchmark is designed for matched-pair evaluation:

```text
same base example
same tool schema
same model
clean prompt vs. realistic noisy prompt
```

A noisy prompt counts only if the gold tool call is unchanged. Deterministic
checks reject rewrites that alter numbers, quoted strings, or visible gold
argument values.

## Current Findings

The short version: **clean BFCL accuracy overstates robustness to the way people
actually type, and the clean benchmark cannot see the gap.** The aggregate drop
is small, which is exactly why it is easy to miss, but underneath it are
reproducible, inspectable regressions: prompts the model gets right in clean
prose and wrong after a realistic rewrite.

We ran the same 2,351-example BFCL-derived paired evaluation across three
models: a cheap model, a capable mid-tier model, and a strong open
function-calling model. All runs used temperature `0`, deterministic
oracle-preserving rewrites, and exact McNemar tests on paired flips.
Drop values are absolute percentage-point drops (`pp`), not relative percent
change.

| Model | Pool | Clean acc. | Avg noisy acc. | Avg drop |
|---|---:|---:|---:|---:|
| `gpt-5.4-nano` | 2,351 | 76.1% | 74.9% | 1.2 pp |
| `claude-haiku-4-5-20251001` | 2,351 | 83.2% | 82.6% | 0.5 pp |
| `z-ai/glm-4.6` | 2,351 | 84.5% | 83.6% | 0.9 pp |

These are genuinely three separate model runs, not three labels produced by one
judge: the checked-in `model_comparison.csv` and paired artifacts identify the
provider and model on every row. However, none of these three is a frontier-tier
model. The current evidence therefore does **not** establish that degradation
persists at the frontier. `claude-sonnet-4-6` is registered for that required
run, but it must not be described as evaluated until its full-pool artifacts
exist.

The effect is **not** "messy prompts break every model equally," and it is
**not** a clean capability gradient. Aggregate degradation is largest on the
cheap model, but not monotonic: GLM-4.6 has the highest clean accuracy yet shows
a larger aggregate drop than Haiku. The cheap model is fragile to a broad range
of phrasing noise; capable models shed almost all of it. What survives is
specific.

The largest cross-model signal is **telegraphic shorthand**: terse,
grammar-free phrasing, one of the most common ways real users address an LLM.
It degrades every model. Profanity, captured by `cursing`, is a smaller signal;
both dimensions are significant on all three models after Holm correction.
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
per-dimension tables, McNemar results, verified examples, interpretation, and
limitations.

The small article-facing analysis bundle is checked in under
`artifacts/analysis/article/` for inspection. Larger generated datasets,
predictions, and intermediate analysis files remain ignored.

The main failure table is
`artifacts/analysis/article/clean_to_noisy_failures.csv`: row-level traces where
the same model was correct on the clean prompt and wrong on the augmented
prompt, with the clean prompt, noisy prompt, expected BFCL tool calls, both
model calls, evaluator error, provider response ids, and explicit manual review
status and notes.
Use `artifacts/analysis/significance.csv` for bootstrap confidence intervals,
exact McNemar p-values, and Holm-adjusted decisions; the article bundle's
`model_comparison.csv` contains the source paired counts.

Three inspectable examples from that table:

- **Dropped call**: `parallel_multiple_27` (GLM-4.6, `telegraphic_request`).
  The clean prompt asks to transfer `$5000` and calculate interest. The terse
  prompt preserves both tasks, but the model only calls the interest calculator.
- **Wrong tool**: `live_multiple_718-165-5` (GLM-4.6, `telegraphic_request`).
  The clean prompt asks to book a house in Austin. The terse prompt still asks
  to book it, but the model calls the house search tool instead of the booking
  tool.
- **Wrong date argument across models**: `live_multiple_676-163-1`
  (`telegraphic_request`). The prompt asks for New York weather tomorrow and
  states that today is `2023.10.1`; the noisy run fetches `2023-10-01` instead
  of the gold `2023-10-02`.

## Built On BFCL

This project builds directly on the Berkeley Function Calling Leaderboard
(BFCL), part of the [Gorilla project](https://github.com/ShishirPatil/gorilla)
from UC Berkeley. BFCL provides the example pool, tool schemas, and AST-based
correctness checker that Realistic-BFCL rephrases and reruns.

- BFCL / Gorilla is licensed under Apache-2.0. Any BFCL-derived data, schemas,
  or checker code used here retain their original Apache-2.0 license and
  copyright; see the upstream repository for the authoritative terms.
- The Realistic-BFCL harness in this repository - augmentation, pairing,
  provider adapters, statistics, and analysis - is original work released under
  MIT; see [LICENSE](LICENSE).
- `prepare-subset` expects a checkout of the pinned BFCL upstream. BFCL data is
  not redistributed here beyond derived artifacts under `artifacts/`.

If you use this work, please also cite BFCL / Gorilla.

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
artifacts/analysis/article/
  clean_to_noisy_failures.csv  Exhaustive clean-correct/noisy-wrong rows.
  model_comparison.csv         Cross-model clean/noisy comparison.
  paired_stats.csv             Paired McNemar statistics.
artifacts/analysis/
  significance.csv             Bootstrap CIs and Holm-adjusted paired tests.
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
oracle-preserving dimensions. Seven reached the frozen, full-pool,
article-facing analysis: `typos`, `cursing`, `irrelevant_context`,
`removed_spaces`, `argumentative_challenge`, `pasted_context_block`, and
`telegraphic_request`. Three overlapping wrapper pilots did not:
`profane_sandwich`, `argumentative_sandwich`, and `distractor_sandwich`.
`configs/realism_dimensions.yaml` is the source of truth for each dimension's
`status`, `article_facing` flag, and exclusion reason.

It also writes `artifacts/generated/augmentation_review.csv` for human
inspection. Deterministic invariant checks reject examples that alter numbers,
quoted strings, or visible gold argument values.

Treat `cursing` as frustrated user register: profanity is a surface marker for
impatience, not the scientific claim by itself.

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
AST checker, cache, and parallel model calls. Model ids, providers, sampling
parameters, tiers, and cost rates are configured in `configs/models.yaml`.
Each model writes its own collision-checked cache namespace, so one
frozen noisy dataset can be evaluated across small, mid-tier, and frontier
models without regenerating augmentations.

Select models by registry name or exact id:

```bash
python scripts/run_stage.py run-bfcl --models nano,haiku,glm
make run-bfcl MODELS=nano,haiku,frontier
python scripts/run_stage.py run-bfcl --models nano --env-file /path/to/.env
```

After CLI and configuration preflight, each evaluation run writes an atomic
`artifacts/<run_id>/manifest.json`. Completed runs bind clean repository/BFCL
SHAs, configs, inputs, models, results, usage, cost, and timing; failed setup or
evaluation runs retain the metadata available at failure. Analysis requires a
completed manifest and rejects inconsistent provenance:

```bash
python scripts/run_stage.py analyze --run-manifest artifacts/<run_id>/manifest.json
```

`analyze` writes paired degradation metrics and review files under
`artifacts/analysis/`, including raw and adjusted degradation plus
`strong_failure_examples.csv` for qualitative inspection. It also writes
`model_comparison.csv`, a per-model/per-dimension table with clean accuracy,
noisy accuracy, degradation, clean-success/noisy-failure counts, reviewed
regression counts, and failure-type taxonomy.
It also writes `significance.csv`, with paired bootstrap confidence intervals,
exact McNemar p-values, and Holm-adjusted p-values for every model/dimension
cell.

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
`configs/project.yaml`. API-key resolution is explicit: `--env-file` takes
precedence over `REALISTIC_BFCL_ENV_FILE=/path/to/.env`, which takes precedence
over provider keys in the process environment. No repository or sibling `.env`
file is loaded implicitly.
Missing synchronous predictions run in parallel with `REALISTIC_BFCL_CONCURRENCY`,
which defaults to `8`.

Before committing, install the repository hooks with `pre-commit install`.
They reject private `.env` files and run gitleaks on staged changes. See
[SECURITY.md](SECURITY.md) for the full-history audit result and credential
handling policy.

Anthropic models can also run through Message Batches, which is the practical
path for larger model comparisons:

```bash
REALISTIC_BFCL_EXECUTION=batch \
python scripts/run_stage.py run-bfcl --models haiku
```

Batch runs write the same prediction caches and paired summaries as synchronous
runs, plus a small `_batch_state.json` file for resuming submitted batches. The
Anthropic adapter also sanitizes provider-incompatible BFCL schema argument
names, then restores the original argument names before BFCL scoring.

GLM-4.6 was served via OpenRouter pinned to the DeepInfra backend for the
reported full-pool run:

```bash
python scripts/run_stage.py run-bfcl --models glm
```

Because GLM-4.6 is open weights and is served at varying precision across
providers, treat its absolute clean accuracy as provider-dependent. The paired
design controls for this: clean and noisy prompts share the same serving route,
so the degradation comparison is internally valid regardless of precision.

## License

MIT for the Realistic-BFCL harness; see [LICENSE](LICENSE). BFCL-derived
material remains under Apache-2.0; see attribution above.
