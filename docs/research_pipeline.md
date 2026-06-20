# Research Pipeline

Realistic-BFCL is a research benchmark layer, not a product pipeline. The
repository should make the experimental artifact easy to reproduce:

```text
prepare BFCL subset -> augment once -> run BFCL-style evaluation -> analyze
```

The repo has three core functions after the BFCL subset is prepared:

1. `augment`: create the noisy dataset.
2. `run-bfcl`: evaluate clean and noisy prompts.
3. `analyze`: compute degradation and review artifacts.

## 0. Prepare The BFCL Substrate

Command:

```bash
python scripts/run_stage.py prepare-subset
```

The default subset config is `configs/subsets/smoke.yaml`. For the larger
single-turn pilot, use:

```bash
REALISTIC_BFCL_SUBSET_CONFIG=configs/subsets/expanded_live.yaml \
  python scripts/run_stage.py prepare-subset
```

Purpose:
- Pin the BFCL upstream commit.
- Materialize the configured stratified clean subset.
- Save enough metadata for another researcher to reconstruct the same examples.

Primary outputs:
- `artifacts/frozen/bfcl_manifest.json`
- `artifacts/frozen/clean_subset.jsonl`

## 1. Augment

Command:

```bash
python scripts/run_stage.py augment
```

Purpose:
- Generate the noisy benchmark dataset once for a given subset.
- Preserve the BFCL gold tool-call oracle.
- Reject rows that mutate oracle-bearing prompt content.

Current dimensions:
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

Primary outputs:
- `artifacts/generated/*.jsonl`
- `artifacts/generated/augmentation_review.csv`

The JSONL files are evaluator-ready. The CSV is for human inspection only.

LLM-generated pilot dimensions are generated separately and saved for review:

```bash
python scripts/run_stage.py augment-llm-pilot
```

These candidates are not part of the default article-facing run. They are for
manual review before promotion into the main augmented dataset.

## 2. Run BFCL-Style Evaluation

Command:

```bash
python scripts/run_stage.py run-bfcl
```

Purpose:
- Run the clean model baseline.
- Run the same model on every noisy variant.
- Score clean and noisy predictions with the same BFCL-style AST checker.
- Cache predictions and run missing OpenAI calls in parallel.

Primary outputs:
- `artifacts/results/clean/`
- `artifacts/results/noisy/`
- `artifacts/results/paired/`

## 3. Analyze

Command:

```bash
python scripts/run_stage.py analyze
```

Purpose:
- Compare clean and noisy outcomes for the same base example.
- Separate raw degradation from possible oracle/evaluator strictness issues.
- Produce review files for manual audit and article-facing summaries.

Primary outputs:
- `artifacts/analysis/benchmark_summary.csv`
- `artifacts/analysis/benchmark_summary.json`
- `artifacts/analysis/flip_review.csv`
- `artifacts/analysis/regression_review.csv`

GitHub-facing article outputs are written under:

- `artifacts/analysis/article/dimension_results.csv`
- `artifacts/analysis/article/overall_error_type_counts.csv`
- `artifacts/analysis/article/included_failure_examples.csv`
- `artifacts/analysis/article/oracle_issue_examples.csv`

## Scaling Rule

The 400-example stratified run is smoke testing. The current article-facing run
uses `configs/subsets/expanded_live.yaml`, which expands coverage to 2,351
BFCL-derived examples with gold answers.

For this project, widening the data and keeping the realism contract defensible
is more important than adding many models or many more augmentation types.
