# Realistic-BFCL: A Small Test For Messy Tool Use

Suppose we have a function-calling benchmark example:

```text
Find the cost of a 2 GB and a 4 GB AWS EC2 machine with one CPU.
```

The model gets it right. It calls the pricing tool twice: once for `2 GB`, once
for `4 GB`.

Now change only the user prompt:

```text
what's cost of 2 and 4 gb ram machine on aws ec2 with one CPU? fucking please man
```

The tool schema is the same. The gold answer is the same. The only difference is
that the user sounds like a user.

In one reviewed Realistic-BFCL run, `gpt-5.4-nano` now called the pricing tool
only once.

That is the whole point of this project.

Clean function-calling benchmarks are useful. But they usually test clean
requests. Real users paste random context, write shorthand, complain, mistype
things, remove spaces, and ask for multiple actions casually. A model can look
good on the clean benchmark and still be brittle in those cases.

Realistic-BFCL asks a narrow question:

> If the clean BFCL prompt succeeds, does the same model still make the same
> tool call when the prompt is written more like real user traffic?

## The Experiment

I started from a 2,351-example BFCL-derived pool.

For each example, I kept:

- the same tool schema
- the same expected function name
- the same expected arguments
- the same evaluator
- the same model

Then I changed only the user prompt.

The current noise types are deliberately simple:

- `typos`
- `cursing`
- `irrelevant_context`
- `removed_spaces`
- `argumentative_challenge`
- `pasted_context_block`
- `telegraphic_request`

In the discussion below, `cursing` is best read as a frustrated-register
condition. Profanity is one concrete marker of user impatience; it is not the
scientific claim by itself.

These are not jailbreaks. They are not meant to be clever attacks. They are
ordinary ways people write when they are rushed, annoyed, casual, or copying
from somewhere else.

## Result

Clean accuracy on this pool was `76.1%`.

Here is what happened after adding each kind of noise:

| Noise type | Clean acc. | Noisy acc. | Drop | Clean ok -> noisy fail | Clean fail -> noisy ok | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| `pasted_context_block` | 0.761 | 0.742 | 0.019 | 96 | 51 | 0.0003 |
| `cursing` | 0.761 | 0.744 | 0.017 | 97 | 58 | 0.0022 |
| `telegraphic_request` | 0.761 | 0.746 | 0.014 | 98 | 64 | 0.0093 |
| `irrelevant_context` | 0.761 | 0.749 | 0.012 | 84 | 56 | 0.0222 |
| `argumentative_challenge` | 0.761 | 0.751 | 0.010 | 76 | 52 | 0.0416 |
| `removed_spaces` | 0.761 | 0.753 | 0.008 | 76 | 57 | 0.1182 |
| `typos` | 0.761 | 0.758 | 0.003 | 67 | 60 | 0.5946 |

The drops are not huge.

But that is not the interesting part.

The interesting part is that these are paired examples. The same base example
can succeed when written cleanly and fail when written more like real user
traffic. That is the failure surface this project is trying to expose.

The paired statistics are a sanity check, not the main claim. The exact
percentage drop is less important than the existence of reviewed
clean-success/noisy-failure examples under oracle-preserving rewrites. The full
paired counts, McNemar p-values, and multiple-comparison corrections are in
`artifacts/analysis/article/paired_stats.csv`.

We also repeated the evaluation three times with fresh clean and noisy model
calls. The direction held in every run: each noise type reduced accuracy each
time.

| Noise type | Runs | Mean drop | Min drop | Max drop | Drop sd |
|---|---:|---:|---:|---:|---:|
| `pasted_context_block` | 3 | 0.025 | 0.019 | 0.032 | 0.007 |
| `cursing` | 3 | 0.022 | 0.017 | 0.026 | 0.005 |
| `telegraphic_request` | 3 | 0.016 | 0.014 | 0.020 | 0.003 |
| `argumentative_challenge` | 3 | 0.014 | 0.010 | 0.019 | 0.004 |
| `irrelevant_context` | 3 | 0.011 | 0.009 | 0.012 | 0.002 |
| `removed_spaces` | 3 | 0.009 | 0.005 | 0.013 | 0.004 |
| `typos` | 3 | 0.006 | 0.003 | 0.009 | 0.003 |

Reviewed regressions exclude rows where the oracle looked ambiguous, the
augmentation may have changed the task, or the example was manually questionable.
The audit rules are in [annotation_protocol.md](annotation_protocol.md), and the
first-pass realism audit summary is checked in as
`artifacts/analysis/article/realism_audit_summary.csv`.

So the table is trying to answer a conservative question:

```text
How often did ordinary prompt messiness break a tool call that already worked?
```

This small study is a probe, not a leaderboard. Its purpose is to expose a
failure surface and motivate more realistic robustness testing for tool routers.

## A Mid-Model Check

The first result above used one cheap model, so I also ran the same frozen
2,351-example pool on `claude-haiku-4-5-20251001` through Anthropic Message
Batches at temperature `0`.

This was not a new benchmark version. The prompts, schemas, gold answers, and
seven dimensions were the same. Only the model changed.

Raw Haiku results:

| Noise type | Clean acc. | Noisy acc. | Drop | Clean ok -> noisy fail | Clean fail -> noisy ok | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| `telegraphic_request` | 0.832 | 0.817 | 0.014 | 51 | 17 | 0.000045 |
| `cursing` | 0.832 | 0.824 | 0.008 | 30 | 12 | 0.007916 |
| `irrelevant_context` | 0.832 | 0.825 | 0.006 | 29 | 14 | 0.031540 |
| `argumentative_challenge` | 0.832 | 0.827 | 0.005 | 23 | 12 | 0.089531 |
| `removed_spaces` | 0.832 | 0.829 | 0.002 | 13 | 8 | 0.383310 |
| `typos` | 0.832 | 0.830 | 0.001 | 12 | 9 | 0.663624 |
| `pasted_context_block` | 0.832 | 0.831 | 0.001 | 26 | 24 | 0.887725 |

Pooled across all seven dimensions:

```text
paired rows:             16,457
clean accuracy:          0.832
noisy accuracy:          0.826
raw drop:                0.005
clean -> noisy failures: 184
noisy fixes clean:       96
```

The Haiku result disciplines the claim. The effect is much smaller than the
nano result and is not uniform across noise types. `telegraphic_request` is the
cleanest raw Haiku signal: 51 clean-success/noisy-failure cases versus 17
clean-failure/noisy-success cases. `cursing` and `irrelevant_context` are also
directional before manual review. But `pasted_context_block`, `typos`, and
`removed_spaces` are near-balanced for Haiku.

That suggests a more careful story: phrasing brittleness is strongest on the
cheap model, shrinks on a stronger mid-tier model, and does not disappear
entirely. Telegraphic shorthand is the clearest remaining failure surface in
this run.

The full-pool Haiku regressions have not yet been manually reviewed. On the
250-example Haiku pilot, many raw flips were baseline/oracle ambiguities such
as `giant panda` vs. `panda`, `Adidas` vs. `阿迪达斯`, or `brownies` vs.
`dessert`. So the Haiku table should be read as a raw paired robustness result,
not as a final true-failure count.

The checked-in raw summary is in:

```text
artifacts/analysis/article/haiku_full_pool_summary.csv
artifacts/analysis/article/haiku_full_pool_summary.json
artifacts/analysis/article/haiku_full_pool_paired_stats.csv
artifacts/analysis/article/haiku_full_pool_paired_stats.json
```

The paired stats files were added specifically to avoid overstating weak
dimensions. They report the two discordant directions separately and use an
exact McNemar test, so dimensions like `pasted_context_block` are visible as
near-balanced rather than being folded into one pooled degradation number.

## What The Failures Look Like

Most failures were not dramatic.

The model usually still emitted a valid-looking tool call. It was just the wrong
one.

| Failure type | Count |
|---|---:|
| Wrong argument value | 229 |
| Missing required tool call | 115 |
| Wrong tool routing | 63 |
| Extra tool call | 5 |
| Argument drop | 1 |
| Malformed call | 1 |

This matters because these are the failures that are easiest to miss in a real
system. The JSON parses. The endpoint exists. The call looks plausible. But it
does the wrong thing.

## A Few Examples

The full reviewed rows are in:

```text
artifacts/analysis/article/included_failure_examples.csv
```

Here are some examples that show the shape of the problem.

### Profanity Drops A Parallel Call

Clean:

```text
What's cost of 2 and 4 gb ram machine on aws ec2 with one CPU?
```

Noisy:

```text
what's cost of 2 and 4 gb ram machine on aws ec2 with one CPU? fucking please man
```

Expected: two pricing calls.

Noisy result: one pricing call.

The profanity does not change the task. But the model now handles only part of
it.

### Shorthand Drops The Second Item

Clean:

```text
interviewers list for Python and Java
```

Noisy:

```text
interviewers list Python and Java
```

Expected: one lookup for Python and one lookup for Java.

Noisy result: only Python.

This is not a weird prompt. It is exactly the kind of thing someone types into a
chat box.

### Frustration Changes The Tool

Clean:

```text
Can you help me translate this English text to Danish: 'The quick brown fox jumps over the lazy dog.'
```

Noisy:

```text
last time you messed this up, so answer carefully can you help me translate this English text to Danish: 'The quick brown fox jumps over the lazy dog.'
```

Expected: the same translation tool.

Noisy result: a different translation tool.

The user added emotion, not a new requirement.

### A Specific Lookup Becomes A List Endpoint

Clean:

```text
The maintenance config is needed for 'sandcastle'.
```

Noisy:

```text
this is bullshit the maintenance config is needed for 'sandcastle'.
```

Expected: fetch the maintenance config with `id = sandcastle`.

Noisy result: call a generic list-maintenance-configs endpoint.

Again, the output looks reasonable unless you check the exact intended call.

### Pasted Text Changes A Math Call

The actual task was:

```text
3 of my friends gave me 10 euros each, how much do I have now?
```

The noisy prompt also included:

```text
based on the earlier draft please but ignore the opener. also ignore the ending, both made it worse
```

Expected: `multiply(a=3, b=10)`.

Noisy result: `add(a=10, b=10)`.

The extra text should not matter. But it did.

## Why This Is Useful

This is not trying to prove that `gpt-5.4-nano` is bad. It is one cheap model on
one BFCL-derived pool.

The useful point is methodological:

```text
clean benchmark example
+ oracle-preserving realistic rewrite
= paired robustness test
```

That paired setup is powerful. It lets us separate two questions:

1. Can the model solve the original benchmark example?
2. Does realistic user messiness break that solution?

Without the pair, those get mixed together.

BFCL is a convenient place to test this because the evaluator is mostly
black-and-white. Either the model called the right tool with the right arguments,
or it did not.

That makes the stress test easy to inspect.

But the more important thesis is broader. In high-risk AI applications, the
evaluation target is often not a neat exact match. It may be a rubric, an expert
judgment, a preference ranking, or a multi-part assessment of whether the answer
was complete, safe, and context-aware. In those settings, realistic user
messiness may cause smaller, more subtle degradations that are harder to see in
one headline score.

That is exactly why this kind of stress testing matters. If a simple deterministic
tool benchmark already shows prompt-surface brittleness, then messier, higher
stakes domains deserve at least as much pressure testing.

## The Main Lesson

The biggest risk in this project is not that the model fails.

The biggest risk is that the augmentation quietly changes the task.

For example, if the original argument is `deer` and the noisy prompt says
`dear`, that is no longer a clean model failure. It might be an augmentation
mistake. If the original prompt asks for a decimal but the schema is ambiguous
about percent formatting, that may be a baseline ambiguity.

So Realistic-BFCL should stay strict:

- preserve the oracle
- review regressions manually
- exclude ambiguous cases
- report raw and filtered metrics separately

The benchmark is only useful if the noisy prompt still means the same thing.

## Relation To Other Work

This is much smaller than recent realistic-agent benchmark projects, but it is
pointing in the same direction.

- [Surge/Corecraft, "The Hierarchy of Agentic Capabilities"](https://arxiv.org/abs/2601.09032)
  evaluates agents on realistic workplace tasks and emphasizes concrete failure
  modes.
- [Tau-bench](https://arxiv.org/abs/2406.12045) evaluates tool agents through
  user-agent interaction instead of only static prompts.
- [RealUserSim](https://arxiv.org/abs/2605.20204) argues that simulated users are
  often too cooperative, which can hide failures.
- [CRAB-Bench](https://arxiv.org/abs/2606.01815) also tests imperfect users,
  distractors, and realistic service settings.
- Recent clinical AI evaluation work on
  [case-specific clinician-authored rubrics](https://arxiv.org/abs/2604.24710)
  is a useful example of the harder evaluation setting: the target is not one
  exact string or one function call, but expert judgment over many case-specific
  criteria.

Realistic-BFCL is not trying to replace those. It is a smaller layer over an
existing deterministic benchmark:

```text
same BFCL example
same oracle
same evaluator
clean prompt vs. messy prompt
```

That makes the failures easy to inspect and easy to reproduce.

## Reproduce

Generate the current article-facing analysis:

```bash
REALISTIC_BFCL_DIMENSIONS=typos,cursing,irrelevant_context,removed_spaces,argumentative_challenge,pasted_context_block,telegraphic_request \
python scripts/run_stage.py analyze
```

Main outputs:

```text
artifacts/analysis/article/dimension_results.csv
artifacts/analysis/article/paired_stats.csv
artifacts/analysis/article/realism_audit_summary.csv
artifacts/analysis/article/stability_repeat_summary.csv
artifacts/analysis/article/stability_repeat_runs.csv
artifacts/analysis/article/review_filtering.csv
artifacts/analysis/article/overall_error_type_counts.csv
artifacts/analysis/article/included_failure_examples.csv
artifacts/analysis/article/oracle_issue_examples.csv
```

## Implications

The narrow result is about BFCL.

The broader point is about evaluation.

Benchmarks often make the user unrealistically clean because clean inputs are
easier to score. That is reasonable, but it leaves out an important failure
surface. Real users bring context, emotion, shorthand, irrelevant text, copied
material, and last-minute corrections. Tool routers and agents need to work under
that distribution too.

Realistic-BFCL is useful because it gives us a clean laboratory for that idea:

```text
known oracle
+ same benchmark example
+ realistic prompt variation
= measurable robustness gap
```

For harder-to-score tasks, the same idea should be even more important. If the
evaluation requires rubrics or expert review, degradation may not appear as a
single wrong function call. It may appear as a missed caveat, a slightly worse
plan, an ignored constraint, or an overconfident answer. Those failures are
harder to count, but in high-risk settings each one can matter.

The takeaway:

> Clean tool-calling accuracy can hide realistic prompt-surface failures. A
> paired, oracle-preserving noisy layer makes those failures visible.
