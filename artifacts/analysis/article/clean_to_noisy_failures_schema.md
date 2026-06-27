# `clean_to_noisy_failures.csv` Schema

This CSV is the row-level trace table for clean-correct/noisy-wrong failures.
Use `paired_stats.csv` and `model_comparison.csv` for headline aggregate counts
and statistical tests.

Primary key: `row_id`.

| Column | Meaning |
|---|---|
| `row_id` | Stable unique row key: model, evaluation run, dimension, and base id. |
| `model` | Evaluated model id. |
| `provider` | Serving provider used for the row. |
| `evaluation_run_id` | Human-readable run id for provenance. |
| `evaluation_run_role` | Whether the run is a primary full-pool trace or a repeat trace. |
| `pool_size` | Number of base BFCL examples in the run. |
| `temperature` | Model sampling temperature. |
| `repeat_index` | Repeat number when the row comes from a repeat run; blank otherwise. |
| `dimension` | Realism transformation applied to the clean prompt. |
| `base_id` | BFCL-derived base example id. |
| `noisy_id` | Augmented example id. Not unique across models. |
| `category` | BFCL category. |
| `review_status` | `reviewed` or `not_reviewed`. |
| `review_scope` | Why the row has or lacks manual screening. |
| `review_label` | `true_model_failure`, `possible_oracle_artifact`, or `not_reviewed`. |
| `screened_failure_type` | Failure type assigned by the significant-cell screen, or `not_reviewed`. |
| `clean_prompt` | Original clean user prompt. |
| `noisy_prompt` | Oracle-preserving noisy prompt. |
| `expected_tool_calls` | BFCL accepted tool-call oracle. |
| `expected_function_names` | Function names extracted from `expected_tool_calls`. |
| `clean_model_calls` | Model tool calls on the clean prompt. |
| `clean_model_call_names` | Function names extracted from `clean_model_calls`. |
| `noisy_model_calls` | Model tool calls on the noisy prompt. |
| `noisy_model_call_names` | Function names extracted from `noisy_model_calls`. |
| `noisy_eval_error_type` | BFCL evaluator error class for the noisy prediction. |
| `noisy_eval_error` | BFCL evaluator error text. |
| `noisy_completion_tokens_if_available` | Noisy completion-token count when logged; blank otherwise. |
| `clean_provider_response_id` | Provider response id for the clean call. |
| `noisy_provider_response_id` | Provider response id for the noisy call. |

Review labels in this file are the significant-cell artifact-screen labels, not
the broader article-inclusion labels from `docs/annotation_protocol.md`. Rows
outside that screen are explicitly marked `not_reviewed`.
