# Evaluation Metrics

Realistic-BFCL uses paired clean-vs-noisy evaluation. The unit of analysis is a
base BFCL example and one or more realistic noisy variants derived from it.

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

## Paired Outcomes

Each pair should be classified as:

- `both_correct`
- `clean_success_noisy_failure`
- `clean_failure_noisy_success`
- `both_wrong`

The main scientific signal is `clean_success_noisy_failure`, because it
isolates failures caused by realistic conversational noise rather than baseline
tool-routing weakness.

## Raw And Adjusted Degradation

Raw degradation counts every `clean_success_noisy_failure` row.

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

## Error Taxonomy

Noisy failures should be labeled with one primary category:

- Routing error: wrong function name.
- Argument drop: required argument missing.
- Argument hallucination: extra or unsupported argument value.
- Malformed call: invalid JSON or schema-incompatible call.
- False refusal: model refuses despite a valid tool request.
- Unnecessary clarification: model asks for information already present.
- Baseline dataset ambiguity: the original BFCL prompt/schema/oracle requires an
  unstated convention.

Secondary tags can be added later, but the first pass should keep labels small
enough for consistent auditing.

## Evidence Table

`artifacts/analysis/strong_failure_examples.csv` contains a compact qualitative
sample of strong clean-success/noisy-failure regressions after excluding oracle,
augmentation, and baseline dataset issues. This is the source table for manual
inspection and paper examples.
