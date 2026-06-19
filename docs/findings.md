# Realistic-BFCL: What We Found So Far

This is a small experiment, not a benchmark launch.

The question was simple:

> If a model gets a BFCL tool-call example right when the prompt is clean, does
> it still get it right when the user writes like an actual person?

For `gpt-5.4-nano`, the answer is: often yes, but not reliably enough to ignore.

Clean accuracy on the current 2,351-example BFCL-derived pool was `76.1%`.
After adding realistic, oracle-preserving user mess, we found hundreds of cases
where the clean prompt was correct and the noisy prompt failed.

## The Result

Same model. Same tools. Same gold oracle. Only the user prompt changes.

| Noise type | Clean acc. | Noisy acc. | Drop | Reviewed regressions |
|---|---:|---:|---:|---:|
| `telegraphic_request` | 0.761 | 0.746 | 0.014 | 75 |
| `pasted_context_block` | 0.761 | 0.742 | 0.019 | 74 |
| `cursing` | 0.761 | 0.744 | 0.017 | 67 |
| `irrelevant_context` | 0.761 | 0.749 | 0.012 | 58 |
| `argumentative_challenge` | 0.761 | 0.751 | 0.010 | 54 |
| `typos` | 0.761 | 0.758 | 0.003 | 45 |
| `removed_spaces` | 0.761 | 0.753 | 0.008 | 41 |

“Reviewed regressions” means:

- the clean prompt was correct
- the noisy prompt was wrong
- the original BFCL oracle was preserved
- possible oracle/alias issues were excluded
- possible augmentation mistakes were excluded
- manually questionable examples were excluded

So these are not raw “anything changed” counts. They are the conservative set I
would be willing to show someone else.

## What Broke

The model usually did not completely fall apart. That is the point.

Most failures looked plausible at a glance:

| Failure type | Count |
|---|---:|
| Wrong argument value | 229 |
| Missing required tool call | 115 |
| Wrong tool routing | 63 |
| Extra tool call | 5 |
| Argument drop | 1 |
| Malformed call | 1 |

The failure pattern is very production-like:

- it calls the right kind of thing, but with the wrong argument
- it handles the first requested call and drops the second
- it chooses a nearby endpoint instead of the specific endpoint
- it treats messy context as if it mattered

That is more worrying than a loud malformed-output failure, because it looks
reasonable unless you check the actual call.

## Examples

Full rows are in:

```text
artifacts/analysis/article/included_failure_examples.csv
```

Here are a few reviewed examples in plain English.

### 1. Profanity makes the model drop one of two required calls

Clean:

> What's cost of 2 and 4 gb ram machine on aws ec2 with one CPU?

Noisy:

> what's cost of 2 and 4 gb ram machine on aws ec2 with one CPU? fucking please man

Expected: call AWS pricing twice, once for 2 GB and once for 4 GB.

Noisy result: only calls pricing for 2 GB.

This is a good example because the profanity does not change the task. It just
changes the way a real annoyed user might phrase it.

### 2. Telegraphic input drops the second item

Clean:

> interviewers list for Python and Java

Noisy:

> interviewers list Python and Java

Expected: one call for Python, one call for Java.

Noisy result: only calls the Python lookup.

This is not adversarial. People type like this all the time.

### 3. Frustration changes the selected tool

Clean:

> Can you help me translate this English text to Danish: 'The quick brown fox jumps over the lazy dog.'

Noisy:

> last time you messed this up, so answer carefully can you help me translate this English text to Danish: 'The quick brown fox jumps over the lazy dog.'

Expected: the same translation function that worked on the clean prompt.

Noisy result: routes to a different translation tool.

The task did not change. The tone changed.

### 4. A specific lookup becomes a list endpoint

Clean:

> The maintenance config is needed for 'sandcastle'.

Noisy:

> this is bullshit the maintenance config is needed for 'sandcastle'.

Expected: get the maintenance config with `id = sandcastle`.

Noisy result: calls a generic list-maintenance-configs endpoint.

This is the kind of error that can easily look fine in logs unless you compare
against the exact intended tool call.

### 5. Pasted context pushes a simple math tool call off track

The original task asks:

> 3 of my friends gave me 10 euros each, how much do I have now?

The noisy prompt adds:

> based on the earlier draft please but ignore the opener. also ignore the ending, both made it worse

Expected: `multiply(a=3, b=10)`.

Noisy result: `add(a=10, b=10)`.

The extra text says to ignore the earlier draft, but the router still changes
behavior.

## What This Suggests

Finding 1: clean benchmark success does not imply prompt-surface robustness.

The model can route the clean prompt and still fail when the same request is
written tersely, angrily, or with irrelevant context around it.

Finding 2: the dangerous failures are often quiet.

The model usually emits a valid-looking call. The problem is that one argument
changed, one call disappeared, or the wrong endpoint was selected.

Finding 3: pasted context and telegraphic requests are worth testing.

These were not complicated adversarial transformations. They are normal user
behavior. They also produced some of the strongest signals.

Finding 4: oracle preservation is the whole game.

If the augmentation changes the task, the result is not useful. For this reason,
the analysis excludes possible oracle issues, possible augmentation issues, and
manually questionable rows.

## Why This Is Not a Full Paper Yet

This is intentionally small:

- one model
- one BFCL-derived pool
- deterministic augmentations
- no model leaderboard
- no claim that these seven noise types cover real usage

That is fine for the current goal. The useful claim is narrower:

> A clean function-calling benchmark can miss realistic tool-routing failures,
> and an oracle-preserving noisy layer is a simple way to expose them.

## How This Relates To Other Agent Benchmarks

This is closer in spirit to recent realistic-agent evaluation work than to a
classic static benchmark paper.

Related examples:

- [Surge/Corecraft, "The Hierarchy of Agentic Capabilities"](https://arxiv.org/abs/2601.09032)
  asks whether agents can do realistic workplace tasks, then reports a failure
  hierarchy instead of stopping at one headline score.
- [Tau-bench](https://arxiv.org/abs/2406.12045) tests tool agents through
  user-agent interaction rather than single clean prompts.
- [RealUserSim](https://arxiv.org/abs/2605.20204) argues that simulated users are
  often too formal and cooperative, and that more realistic user behavior
  exposes hidden failures.
- [CRAB-Bench](https://arxiv.org/abs/2606.01815) makes a similar case for
  imperfect users, distractors, and realistic service scenarios.

Realistic-BFCL is smaller than those projects. Its advantage is that it keeps
BFCL's deterministic oracle and makes the comparison paired:

```text
clean prompt succeeds
same oracle + realistic noisy prompt fails
```

That makes the failure easy to inspect.

## Reproduce

Generate the current article-facing analysis:

```bash
REALISTIC_BFCL_DIMENSIONS=typos,cursing,irrelevant_context,removed_spaces,argumentative_challenge,pasted_context_block,telegraphic_request \
python scripts/run_stage.py analyze
```

Main outputs:

```text
artifacts/analysis/article/dimension_results.csv
artifacts/analysis/article/overall_error_type_counts.csv
artifacts/analysis/article/included_failure_examples.csv
artifacts/analysis/article/oracle_issue_examples.csv
```

## What I Would Do Next

I would not immediately add ten more deterministic augmentations.

Better next steps:

1. Add one LLM-generated but manually reviewed noise type based on realistic
   pasted context or workplace history.
2. Run the same setup on one stronger model and one weaker model.
3. Publish the accepted noisy examples as a small dataset artifact.
4. Keep the benchmark paired and oracle-preserving.

The main thing to protect is the validity of the oracle. Without that, the
benchmark becomes just another noisy prompt set.
