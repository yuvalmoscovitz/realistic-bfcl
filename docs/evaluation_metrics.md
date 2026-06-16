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

- clean_correct_noisy_correct
- clean_correct_noisy_incorrect
- clean_incorrect_noisy_correct
- clean_incorrect_noisy_incorrect

The main scientific signal is `clean_correct_noisy_incorrect`, because it
isolates failures caused by realistic conversational noise rather than baseline
tool-routing weakness.

## Error Taxonomy

Noisy failures should be labeled with one primary category:

- Routing error: wrong function name.
- Argument drop: required argument missing.
- Argument hallucination: extra or unsupported argument value.
- Malformed call: invalid JSON or schema-incompatible call.
- False refusal: model refuses despite a valid tool request.
- Unnecessary clarification: model asks for information already present.

Secondary tags can be added later, but the first pass should keep labels small
enough for consistent auditing.
