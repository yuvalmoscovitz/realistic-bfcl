# Annotation Protocol

This protocol is for auditing whether a noisy Realistic-BFCL example supports
the benchmark claim. The unit of review is one clean/noisy pair with the tool
schema, gold oracle, clean prediction, and noisy prediction visible.

## Labels

Use `yes`, `unclear`, or `no` for each field:

- `oracle_preserved`: the noisy prompt still asks for the same function call and
  argument values as the clean BFCL oracle.
- `production_like`: the prompt resembles plausible user traffic, not a
  benchmark trick or jailbreak.
- `natural_user_style`: the wording sounds like something a real user might
  type or paste.
- `non_adversarial`: the prompt does not try to attack the model or evaluator.
- `active_constraints_unchanged`: the noisy prompt does not add, remove, or
  modify a tool-relevant requirement.

Then assign one final decision:

- `include`: usable as evidence of a realistic prompt-induced regression.
- `exclude`: likely oracle issue, augmentation issue, or unrealistic prompt.
- `uncertain`: needs another reviewer or should be excluded from headline
  counts.

## Rejection Rules

Reject an example when the noisy prompt:

- changes a tool-relevant entity, number, quoted string, locale, date, or time
- adds a new required constraint
- removes a required constraint
- makes the final intent ambiguous
- uses benchmark/meta wording that real users would not normally write
- includes adversarial instructions such as telling the model to ignore tools
- has a typo or spacing edit that changes an argument value

Correction and self-repair examples may change earlier context only when the
final request clearly defines the intended oracle.

## Current Use

The checked-in `artifacts/analysis/article/realism_audit_summary.csv` is a
first-pass researcher audit of reviewed failure candidates. It is not a blinded
independent human annotation study.

For a larger paper, sample accepted and rejected rows from each dimension, use
at least two annotators, measure agreement, and report disagreements separately
from model failures.
