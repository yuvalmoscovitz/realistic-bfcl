# Realistic-BFCL Article Data

These files organize the article-facing Realistic-BFCL analyses.

The reviewed article counts currently apply to the full-pool `gpt-5.4-nano`
evaluation. Those counts exclude rows marked as possible oracle, augmentation,
or baseline dataset issues, plus manually reviewed artifact/questionable rows.

The Haiku files are raw model-comparison summaries for
`claude-haiku-4-5-20251001`. They use the same frozen 2,351-example pool and
seven dimensions, but the full-pool Haiku regressions have not yet been manually
reviewed. The Haiku paired stats include exact McNemar p-values so weak or
near-balanced dimensions are visible rather than overclaimed.

The GLM files are raw model-comparison summaries for `z-ai/glm-4.6` through
OpenRouter pinned to DeepInfra. They use the same frozen 2,351-example pool,
seven dimensions, temperature `0`, and a 1,024-token router output cap.

The significant-cell review files screen the cells that drive the article-level
claim. Nano uses the existing reviewed article artifacts for clean-to-noisy
failures. Haiku and GLM use a first-pass artifact screen over every raw
discordant item in both directions for their significant `telegraphic_request`
and `cursing` cells. The summary reports symmetric screened McNemar p-values:
possible artifacts are removed from both failures and fixes before recomputing
the paired test.

## Dimension Results

| Dimension | Clean acc. | Noisy acc. | Drop | Article regressions | Article rate |
|---|---:|---:|---:|---:|---:|
| telegraphic_request | 0.761 | 0.746 | 0.014 | 75 | 0.042 |
| pasted_context_block | 0.761 | 0.742 | 0.019 | 74 | 0.041 |
| cursing | 0.761 | 0.744 | 0.017 | 67 | 0.037 |
| irrelevant_context | 0.761 | 0.749 | 0.012 | 58 | 0.032 |
| argumentative_challenge | 0.761 | 0.751 | 0.010 | 54 | 0.030 |
| typos | 0.761 | 0.758 | 0.003 | 45 | 0.025 |
| removed_spaces | 0.761 | 0.753 | 0.008 | 41 | 0.023 |

## Repeat-Run Stability

The evaluation was repeated three times with fresh clean and noisy model calls. Every listed noise type degraded accuracy in every run.

| Dimension | Runs | Mean drop | Min drop | Max drop | Drop sd |
|---|---:|---:|---:|---:|---:|
| pasted_context_block | 3 | 0.025 | 0.019 | 0.032 | 0.007 |
| cursing | 3 | 0.022 | 0.017 | 0.026 | 0.005 |
| telegraphic_request | 3 | 0.016 | 0.014 | 0.020 | 0.003 |
| argumentative_challenge | 3 | 0.014 | 0.010 | 0.019 | 0.004 |
| irrelevant_context | 3 | 0.011 | 0.009 | 0.012 | 0.002 |
| removed_spaces | 3 | 0.009 | 0.005 | 0.013 | 0.004 |
| typos | 3 | 0.006 | 0.003 | 0.009 | 0.003 |

## Files

- `unified_report.md`: consolidated article-facing report with coverage, model results, error taxonomy, screened cells, and strict all-three-model failures.
- `clean_to_noisy_failures.csv`: exhaustive clean-correct/noisy-wrong trace rows across the three article-facing models, with a stable row id, run metadata, clean prompt, noisy prompt, expected BFCL tool calls, clean model calls, noisy model calls, evaluator error, provider response ids, and explicit manual review status.
- `clean_to_noisy_failures_schema.md`: column definitions and review-label caveats for `clean_to_noisy_failures.csv`.
- `clean_to_noisy_failures_summary.csv`: compact raw and reviewed counts for `clean_to_noisy_failures.csv` by model, evaluation run, and dimension.
- `all_three_wrong_examples.csv`: strict clean-correct/noisy-wrong examples shared by all three models, using nano `gpt54nano_full_pool_repeat2` for full-pool coverage.
- `dimension_results.csv`: article-ready per-dimension metrics.
- `paired_stats.csv`: full paired contingency counts, McNemar p-values, and multiple-comparison corrections.
- `review_filtering.csv`: raw-to-reviewed regression filtering counts.
- `significant_cell_review.csv`: first-pass artifact screen for clean-to-noisy failures in significant Haiku and GLM cells.
- `significant_cell_fix_review.csv`: first-pass artifact screen for noisy-to-clean fixes in significant Haiku and GLM cells.
- `significant_cell_review_summary.csv`: compact raw-vs-screened counts and symmetric screened McNemar p-values for significant cells, including reviewed nano failure counts.
- `cross_model_failure_examples.csv`: curated true-failure and possible-artifact examples from the cross-model run.
- `stability_repeat_summary.csv`: mean/range across repeated fresh model runs.
- `stability_repeat_runs.csv`: per-run paired metrics used by the stability summary.
- `stability_repeat_summary.json`: JSON form of the stability summary.
- `error_type_counts.csv`: article regressions by manual error type.
- `overall_error_type_counts.csv`: aggregate article regressions by error type.
- `category_counts.csv`: article regressions by BFCL category.
- `overall_category_counts.csv`: aggregate article regressions by BFCL category.
- `candidate_failure_examples.csv`: strongest examples queued for human review.
- `included_failure_examples.csv`: reviewed qualitative examples for the article.
- `realism_audit_summary.csv`: first-pass researcher audit of reviewed failure candidates.
- `oracle_issue_examples.csv`: examples to exclude or discuss as evaluator/oracle ambiguity.
- `haiku_full_pool_summary.csv`: raw full-pool Haiku paired metrics.
- `haiku_full_pool_summary.json`: JSON form of the raw full-pool Haiku metrics.
- `haiku_full_pool_paired_stats.csv`: Haiku clean->noisy vs noisy->clean counts and exact McNemar p-values.
- `haiku_full_pool_paired_stats.json`: JSON form of the Haiku paired stats.
- `glm46_full_pool_paired_summary.csv`: full-pool GLM-4.6 paired metrics, OpenRouter pinned to DeepInfra with 1024 output tokens.
- `glm46_full_pool_paired_summary.json`: JSON form of the full-pool GLM-4.6 paired metrics.
- `model_comparison.csv`: full-pool per-model/per-dimension comparison for nano, Haiku, and GLM-4.6.
- `model_comparison.json`: JSON form of the model comparison table.

## Notes On `clean_to_noisy_failures.csv`

This is the row-level audit table, not the aggregate statistics table. Use
`paired_stats.csv` and `model_comparison.csv` for headline counts and McNemar
statistics.

Important columns:

- `row_id`: stable unique key, formed from model, evaluation run, dimension, and
  base id. `noisy_id` is not unique across models.
- `evaluation_run_id`: named source run for the row.
- `evaluation_run_role`: whether this is the primary full-pool row trace or a
  repeat used for row-level traceability. `gpt54nano_full_pool_repeat2` is a
  full-pool nano repeat; the headline nano statistics remain in
  `paired_stats.csv`.
- `pool_size`, `repeat_index`: run metadata for quick inspection. The article-facing
  rows were all evaluated at temperature 0, so temperature is not repeated as a
  CSV column.
- `expected_tool_calls`: BFCL accepted tool-call oracle for the prompt.
- `review_status`: manual review status.
- `review_scope`: review scope for the row.
- `review_label`: `true_model_failure` or `possible_oracle_artifact`.
- `screened_failure_type`: type assigned during row-level artifact screening.
  This is separate from the broader inclusion labels in `docs/annotation_protocol.md`.
- `review_notes`: short reviewer note explaining the label.
- `noisy_completion_tokens_if_available`: blank where the serving path did not
  log completion tokens, notably the GLM-4.6 OpenRouter run.

All rows in the CSV have been reviewed: 614 are labeled `true_model_failure` and
608 are labeled `possible_oracle_artifact`. The CSV is self-contained for public
inspection: prompts, expected calls, model calls, evaluator errors, review notes,
and provider response ids are embedded. It should be treated as the authoritative
row-level trace table even when an intermediate generated JSONL source is not
checked in for every dimension.
