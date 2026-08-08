# Realistic-BFCL Article Data

This directory contains the small checked-in artifact bundle behind the public
findings note. For the narrative result, start with `docs/findings.md`; this
directory is for inspection and reproducibility.

## Core Files

- `model_comparison.csv`: full-pool per-model/per-dimension comparison for
  nano, Haiku, and GLM-4.6.
- `../significance.csv`: the 21-cell inferential artifact, with paired
  bootstrap confidence intervals, exact McNemar p-values, and Holm-adjusted
  decisions.
- `paired_stats.csv`: paired contingency counts, McNemar p-values, and
  multiple-comparison corrections.
- `clean_to_noisy_failures.csv`: exhaustive reviewed clean-correct/noisy-wrong
  trace rows across the three article-facing model runs.
- `clean_to_noisy_failures_summary.csv`: raw and reviewed counts from the trace
  table by model, evaluation run, and dimension.
- `significant_cell_review_summary.csv`: symmetric screened McNemar counts for
  the significant Haiku/GLM cells.
- `significant_cell_review.csv`: clean-to-noisy artifact screen rows for the
  significant Haiku/GLM cells.
- `significant_cell_fix_review.csv`: noisy-to-clean fix-side screen rows for the
  significant Haiku/GLM cells.
- `cross_model_failure_examples.csv`: small curated examples across models,
  including artifact controls.
- `all_three_wrong_examples.csv`: strict clean-correct/noisy-wrong examples
  shared by all three models.

Additional CSV/JSON files are supporting breakdowns used by the article tables:
dimension-level results, repeat-run stability, category/error taxonomies,
candidate examples, included examples, oracle-issue examples, and per-model raw
summaries.

The three rows above are separate evaluation models, even though the original
frozen-subset manifest predates the Haiku and GLM runs. None is labeled as a
frontier model. Historical raw cost and timing metadata were not retained, so
the repository does not invent those values retroactively; new runs write them
to `artifacts/<run_id>/manifest.json`.

## `clean_to_noisy_failures.csv`

This is the row-level audit table, not the aggregate statistics table. Use
`../significance.csv` for published uncertainty and adjusted decisions, and
`model_comparison.csv` for the source paired counts. `paired_stats.csv` retains
the earlier Bonferroni/FDR columns for provenance but is not the current claims
artifact.

Important columns:

- `row_id`: stable unique key, formed from model, evaluation run, dimension, and
  base id. `noisy_id` is not unique across models.
- `evaluation_run_id`: named source run for the row.
- `evaluation_run_role`: whether this is the primary full-pool row trace or a
  repeat used for row-level traceability. `gpt54nano_full_pool_repeat2` is a
  full-pool nano repeat; headline nano statistics remain in `paired_stats.csv`.
- `pool_size`, `repeat_index`: run metadata for quick inspection. All
  article-facing trace rows were evaluated at temperature 0, so temperature is
  not repeated as a CSV column.
- `expected_tool_calls`: BFCL accepted tool-call oracle.
- `review_status`, `review_scope`, `review_label`, `screened_failure_type`,
  `review_notes`: manual row-level artifact-screen fields.
- `clean_model_calls`, `noisy_model_calls`: model tool calls on the clean and
  noisy prompts.
- `noisy_eval_error_type`, `noisy_eval_error`: BFCL evaluator error details for
  the noisy prediction.
- `clean_provider_response_id`, `noisy_provider_response_id`: provider response
  ids for traceability.

All rows in the CSV have been reviewed. A second-pass audit checked every row
initially labeled `possible_oracle_artifact`; 90 were promoted to
`true_model_failure` and 1 remains `uncertain`. The final split is 704
`true_model_failure`, 517 `possible_oracle_artifact`, and 1 `uncertain`.
