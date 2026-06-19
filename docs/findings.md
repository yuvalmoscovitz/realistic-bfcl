# Realistic-BFCL Findings

Realistic-BFCL is a small, reproducible concept study: start from BFCL's clean
function-calling examples, preserve the same tool schema and gold oracle, then
ask whether the same model still routes correctly when the user prompt looks
more like real production traffic.

The short version: clean tool-calling accuracy is useful, but it is not enough.
On a 2,351-example BFCL-derived pool, `gpt-5.4-nano` solved about 76.1% of the
clean examples. Realistic oracle-preserving prompt noise produced hundreds of
clean-success/noisy-failure regressions.

## Why This Matters

Real users do not usually write benchmark prompts. They paste context, type
shorthand, complain, include irrelevant details, forget spaces, and ask for
multiple things casually. This is not adversarial behavior. It is normal usage.

For tool-using agents, these small changes matter. A model can still appear to
understand the request while silently choosing the wrong endpoint, dropping a
required parallel call, or changing an argument value.

Realistic-BFCL tests that gap directly:

```text
same base example
same tool schema
same model
same gold tool-call oracle
clean prompt vs. realistic noisy prompt
```

## Evaluation Setup

- Base pool: 2,351 BFCL-derived examples.
- Model: `gpt-5.4-nano`.
- Evaluator: BFCL-style AST/tool-call checker.
- Evaluation style: paired clean-vs-noisy.
- Main regression: clean prompt correct, noisy prompt incorrect.
- Article counts exclude:
  - possible oracle or alias strictness issues
  - possible augmentation issues
  - possible baseline dataset ambiguity
  - manually reviewed `artifact` or `questionable` rows

The current results are intentionally small-scale. This is not a full
leaderboard, and it is not a claim about all models. It is evidence that a
realism-controlled layer over clean function-calling benchmarks can reveal
deployment-relevant failures.

## Noise Dimensions

The current deterministic dimensions are:

- `typos`
- `cursing`
- `irrelevant_context`
- `removed_spaces`
- `argumentative_challenge`
- `pasted_context_block`
- `telegraphic_request`

The most useful dimensions are not exotic attacks. They are ordinary user
patterns: terse requests, pasted context, frustration, casual profanity, and
irrelevant text around the actual task.

## Headline Results

Clean accuracy is constant across rows because every dimension is paired against
the same clean baseline.

| Dimension | Clean Accuracy | Noisy Accuracy | Drop | Article Regressions | Regression Rate |
|---|---:|---:|---:|---:|---:|
| `telegraphic_request` | 0.761 | 0.746 | 0.014 | 75 | 0.042 |
| `pasted_context_block` | 0.761 | 0.742 | 0.019 | 74 | 0.041 |
| `cursing` | 0.761 | 0.744 | 0.017 | 67 | 0.037 |
| `irrelevant_context` | 0.761 | 0.749 | 0.012 | 58 | 0.032 |
| `argumentative_challenge` | 0.761 | 0.751 | 0.010 | 54 | 0.030 |
| `typos` | 0.761 | 0.758 | 0.003 | 45 | 0.025 |
| `removed_spaces` | 0.761 | 0.753 | 0.008 | 41 | 0.023 |

The largest raw accuracy drop came from pasted context blocks. The largest
article-grade regression count came from telegraphic requests.

## Failure Modes

Across article-grade regressions:

| Error Type | Count |
|---|---:|
| Wrong argument value | 229 |
| Missing required tool call | 115 |
| Wrong tool routing | 63 |
| Extra tool call | 5 |
| Argument drop | 1 |
| Malformed call | 1 |

The important pattern is subtle failure, not total collapse. The model often
still emits a plausible tool call, but it drops a second required call, chooses a
related but wrong endpoint, or fills an argument differently under noise.

## Reviewed Examples

These examples were manually reviewed as strong failures. Full prompts, gold
calls, clean predictions, and noisy predictions are stored in
`artifacts/analysis/article/included_failure_examples.csv`.

### Cursing: Missing Parallel Call

Clean request:

> What's cost of 2 and 4 gb ram machine on aws ec2 with one CPU?

Noisy request:

> what's cost of 2 and 4 gb ram machine on aws ec2 with one CPU? fucking please man

Clean prediction called AWS pricing for both 2 GB and 4 GB RAM. The noisy
prediction only called AWS pricing for 2 GB RAM.

### Telegraphic Request: Missing Second Skill

Clean request:

> interviewers list for Python and Java

Noisy request:

> interviewers list Python and Java

Clean prediction requested interviewer lists for both Python and Java. The noisy
prediction only requested Python.

### Argumentative Challenge: Wrong Tool Routing

Clean request:

> Can you help me translate this English text to Danish: 'The quick brown fox jumps over the lazy dog.'

Noisy request:

> last time you messed this up, so answer carefully can you help me translate this English text to Danish: 'The quick brown fox jumps over the lazy dog.'

The clean prediction used the expected translation function. The noisy
prediction routed to a different translation tool.

### Cursing: Specific Lookup Becomes List Endpoint

Clean request:

> The maintenance config is needed for 'sandcastle'.

Noisy request:

> this is bullshit the maintenance config is needed for 'sandcastle'.

The clean prediction called the specific maintenance-config lookup with
`id="sandcastle"`. The noisy prediction called a list endpoint instead.

### Pasted Context: Multiply Routes To Add

The clean prompt contained a tool list and the query:

> 3 of my friends gave me 10 euros each, how much do I have now?

The noisy prompt added a messy preamble:

> based on the earlier draft please but ignore the opener. also ignore the ending, both made it worse

The clean prediction called `multiply(a=3, b=10)`. The noisy prediction called
`add(a=10, b=10)`.

## What This Does Not Claim

This is not a finished benchmark paper.

Current limitations:

- one evaluated model
- deterministic augmentations
- no broad model ranking
- no full interactive user simulation
- some BFCL examples have oracle/alias ambiguity and are excluded where detected

The useful claim is narrower:

> A clean function-calling benchmark can be made more deployment-relevant by
> adding oracle-preserving realistic user noise and measuring paired degradation.

## Reproduce

Run the full analysis for the current evaluated dimensions:

```bash
REALISTIC_BFCL_DIMENSIONS=typos,cursing,irrelevant_context,removed_spaces,argumentative_challenge,pasted_context_block,telegraphic_request \
python scripts/run_stage.py analyze
```

The GitHub-facing artifacts are generated under:

```text
artifacts/analysis/article/
```

Important files:

- `dimension_results.csv`
- `overall_error_type_counts.csv`
- `included_failure_examples.csv`
- `oracle_issue_examples.csv`

## Next Steps

The highest-value next steps are:

1. Add one or two stronger LLM-generated but manually reviewed realistic noise
   dimensions.
2. Evaluate one stronger and one weaker model to check whether degradation
   patterns generalize.
3. Publish the accepted noisy examples as a small dataset artifact.
4. Keep BFCL as the deterministic substrate and keep oracle preservation as the
   central validity rule.
