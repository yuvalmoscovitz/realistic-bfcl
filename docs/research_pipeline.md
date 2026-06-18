# Research Pipeline

Realistic-BFCL is a research benchmark layer, not a product pipeline. The
repository should make the experimental artifact easy to reproduce:

```text
BFCL clean subset -> frozen augmented dataset -> paired BFCL evaluation -> analysis
```

## 1. Prepare The BFCL Substrate

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

## 2. Construct The Augmented Dataset

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

Primary outputs:
- `artifacts/generated/*.jsonl`
- `artifacts/generated/augmentation_review.csv`

The JSONL files are evaluator-ready. The CSV is for human inspection only.

LLM-generated pilot dimensions are generated separately:

```bash
python scripts/run_stage.py augment-llm-pilot
```

Current LLM dimensions:
- `llm_work_context`
- `llm_prior_thread`
- `llm_conversation_history`
- `llm_messy_pre_intent_history`

`llm_messy_pre_intent_history` models semi-relevant chat before the user has a
specific intent. The final user turn must contain the clean BFCL request
verbatim, so the prior turns can be distracting without becoming the oracle.

## 3. Run BFCL-Style Evaluation

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

## 4. Analyze Paired Degradation

Command:

```bash
python scripts/run_stage.py analyze
```

Purpose:
- Compare clean and noisy outcomes for the same base example.
- Separate raw degradation from possible oracle/evaluator strictness issues.
- Produce review files for manual audit and paper tables.

Primary outputs:
- `artifacts/analysis/benchmark_summary.csv`
- `artifacts/analysis/benchmark_summary.json`
- `artifacts/analysis/flip_review.csv`
- `artifacts/analysis/regression_review.csv`

## Scaling Rule

The 400-example stratified run is the smoke test. The next pilot uses
`configs/subsets/expanded_live.yaml`, which keeps the original four categories
and adds single-turn BFCL live categories that have gold answers. Add new models
only after this expanded pilot is stable.
