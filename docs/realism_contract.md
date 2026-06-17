# Realism Contract

Realistic-BFCL is a metamorphic benchmark. A noisy example is valid only if it
preserves the clean BFCL oracle under a realistic conversational transformation.

## Invariants

Every accepted noisy example must preserve:

- Base BFCL example identifier.
- Tool schema.
- Gold function name.
- Gold argument names.
- Gold argument values.
- Any required constraints in the clean prompt.

Correction and self-repair transformations are the exception. They may include
intermediate conflicting values only when the final user intent is explicit and
the final oracle is derived from the clean oracle by a documented repair rule.

## Production-Likeness

Accepted noise should resemble ordinary product traffic:

- Irrelevant but plausible context.
- Casual or impatient phrasing.
- Incomplete early turns followed by sufficient information.
- Natural multilingual phrasing when the request remains understandable.
- Minor messiness that does not become a puzzle.

## Non-Adversarialness

Reject examples that depend on:

- Hidden instructions.
- Prompt injection.
- Word games.
- Obfuscated constraints.
- Random character corruption.
- Deliberately ambiguous final intent.
- Tool names or schema details leaked into the user prompt.

## Oracle Preservation

The clean and noisy examples should be evaluated as a pair. If the clean oracle
is:

```json
{"name": "book_flight", "arguments": {"from": "SFO", "to": "JFK"}}
```

then conversational overhang can add surrounding context, but the final noisy
request must still require:

```json
{"name": "book_flight", "arguments": {"from": "SFO", "to": "JFK"}}
```

## Rejection Rules

Reject a generated example when it:

- Adds a new date, location, quantity, preference, budget, format, or filter.
- Drops a clean constraint or makes it optional.
- Makes two or more tool calls equally plausible.
- Requires outside knowledge to infer a slot value.
- Makes the user intent less realistic than the clean prompt.
- Turns the benchmark into an adversarial robustness task.

## Audit Record

Each accepted noisy example should store:

- Base BFCL id.
- Transformation dimension.
- Clean oracle hash.
- Noisy oracle hash.
- Automatic invariant check result.
- Human or LLM audit status.
- Rejection reason when rejected.
