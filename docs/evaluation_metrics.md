# Evaluation Metrics

Realistic-BFCL uses paired clean-vs-noisy evaluation. The unit of analysis is a
base BFCL example and one realistic noisy variant derived from it.

## Core Metrics

Clean accuracy:

```text
clean_correct / clean_total
```

Noisy accuracy:

```text
noisy_correct / noisy_total
```

Degradation ratio:

```text
1 - (noisy_accuracy / clean_accuracy)
```

Conditional failure rate given clean success:

```text
count(clean_correct and noisy_incorrect) / count(clean_correct)
```

Clean-success/noisy-failure count:

```text
count(clean_correct and noisy_incorrect)
```

Clean-failure/noisy-success count:

```text
count(clean_incorrect and noisy_correct)
```

Exact McNemar p-value:

```text
two-sided exact binomial test over the two discordant counts
```

This tests whether the paired flips are directionally asymmetric, rather than
only reporting one side of the flip table.

## Paired Outcomes

Each pair should be classified as:

- `both_correct`
- `clean_success_noisy_failure`
- `clean_failure_noisy_success`
- `both_wrong`

The main scientific signal is `clean_success_noisy_failure`, because it
isolates failures caused by realistic conversational noise rather than baseline
tool-routing weakness.

## Review Filtering

Raw degradation counts every clean-success/noisy-failure row.

Adjusted degradation excludes rows marked `oracle_issue=possible` in
`artifacts/analysis/regression_review.csv`. This adjustment is meant to separate
model brittleness from possible alias/normalization strictness in the oracle or
evaluator.

The strongest model-failure count is `real_model_regression_count`. It excludes:

- `oracle_issue=possible`
- `augmentation_issue=possible`
- `baseline_dataset_issue=possible`

These exclusions prevent benchmark artifacts from inflating the robustness
claim.

For the GitHub-facing findings note, the stricter reviewed set also excludes
manually questionable examples from the article review artifacts.

## Error Taxonomy

Noisy failures should be labeled with one primary category:

- `wrong_tool_routing`: wrong function name.
- `wrong_argument_value`: required argument has the wrong value.
- `missing_tool_call`: one or more required tool calls are absent.
- `extra_tool_call`: model emits a tool call not required by the oracle.
- `argument_drop`: required argument is missing.
- `argument_hallucination`: unsupported argument or unsupported value appears.
- `malformed_call`: invalid JSON or schema-incompatible call.
- `false_refusal`: model refuses despite a valid tool request.
- `unnecessary_clarification`: model asks for information already present.
- `baseline_dataset_ambiguity`: the original BFCL prompt/schema/oracle requires
  an unstated convention.

Secondary tags can be added later, but the first pass should keep labels small
enough for consistent auditing.

## Main Outputs

The main analysis stage writes:

- `artifacts/analysis/benchmark_summary.csv`
- `artifacts/analysis/benchmark_summary.json`
- `artifacts/analysis/regression_review.csv`
- `artifacts/analysis/flip_review.csv`
- `artifacts/analysis/article_failure_review.csv`
- `artifacts/analysis/article_failure_examples.csv`

The GitHub-facing findings note uses article artifacts under:

- `artifacts/analysis/article/dimension_results.csv`
- `artifacts/analysis/article/overall_error_type_counts.csv`
- `artifacts/analysis/article/included_failure_examples.csv`
- `artifacts/analysis/article/oracle_issue_examples.csv`
- `artifacts/analysis/article/paired_stats.csv`
- `artifacts/analysis/article/review_filtering.csv`
- `artifacts/analysis/article/stability_repeat_summary.csv`
- `artifacts/analysis/article/stability_repeat_runs.csv`
- `artifacts/analysis/article/stability_repeat_summary.json`

These article-facing files are intentionally small and checked into the repo so
readers can inspect the reported results without rerunning the model.
