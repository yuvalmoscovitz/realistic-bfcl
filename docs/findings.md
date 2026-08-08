# Realistic-BFCL - Findings

## Summary

**Clean evals tell you whether your agent *can* do the task. They do not tell you
whether it *will* once a real user phrases the request the way real users
actually do.** This note measures that gap on BFCL tool-calling.

Across three models of increasing clean accuracy, clean BFCL accuracy overstates
robustness to ordinary, production-like phrasing, and the clean benchmark cannot
see the gap. The cheap model is fragile to a broad range of phrasing noise.
Capable models shed most of it, but two effects survive across the stronger
models tested:

> **Telegraphic shorthand - terse, grammar-free phrasing - is the largest
> cross-model effect. Cursing is smaller, but both remain significant on all
> three tested models after Holm correction.**

This is the "can vs. will" gap made concrete. Every model here can price two
machines when asked in clean prose, and clean BFCL confirms it. Phrase the same
request as `aws ec2 2gb 4gb 1cpu price` and the model that passed the clean test
begins dropping the second machine, a failure the clean benchmark never
surfaces, because clean prompts are never written that way.

## Method

For each base BFCL example we hold everything fixed: the tool schema, the gold
function name and arguments, the AST evaluator, and the model. We change only
the surface phrasing of the user request. A noisy variant counts only if its
correct tool call is identical to the clean one. Deterministic invariant checks
reject any rewrite that alters a number, a quoted string, or a visible gold
argument value.

All runs are full-pool, 2,351 examples, at temperature `0`. For each model and
dimension, we report the two paired discordant counts: clean-to-noisy failures
and noisy-to-clean fixes. We also report an exact McNemar test on those counts.
Reporting both directions prevents weak, near-balanced dimensions from being
laundered into a single pooled degradation number. We use exact, two-sided
McNemar tests and control family-wise error across the seven dimensions within
each model using Holm-Bonferroni at `alpha = 0.05`. The 95% percentile bootstrap
interval resamples paired binary example differences 10,000 times with seed
`20260618`. For the checked-in legacy comparison, those binary differences are
reconstructed exactly from the paired contingency counts: `b` values of `+1`,
`c` values of `-1`, and zero for concordant pairs. Example identity does not
change the bootstrap distribution of their mean. The complete generated table
is `artifacts/analysis/significance.csv`.

### Dimension Scope

Ten deterministic dimensions are implemented, but seven reached this
article-facing run: `typos`, `cursing`, `irrelevant_context`, `removed_spaces`,
`argumentative_challenge`, `pasted_context_block`, and `telegraphic_request`.
The three sandwich variants remained pilots and are excluded from all article
statistics: `profane_sandwich` overlaps `cursing`, `argumentative_sandwich`
overlaps `argumentative_challenge`, and `distractor_sandwich` overlaps the two
context dimensions. In addition to that conceptual overlap, none had the same
frozen full-pool, cross-model, reviewed artifact required for inclusion. This is
a methodological scope decision, not evidence that the excluded dimensions
would or would not produce an effect. The machine-readable status and exclusion
reason for every implemented dimension live in
`configs/realism_dimensions.yaml`.

## Headline Results

Drop values are absolute percentage-point drops (`pp`), not relative percent
change.

| Model | Provider | Tier | Clean acc. | Avg noisy acc. | Avg drop |
|---|---|---|---:|---:|---:|
| `gpt-5.4-nano` | OpenAI | cheap | 76.1% | 74.9% | 1.2 pp |
| `claude-haiku-4-5-20251001` | Anthropic | mid | 83.2% | 82.6% | 0.5 pp |
| `z-ai/glm-4.6` | OpenRouter | strong-open | 84.5% | 83.6% | 0.9 pp |

This table is backed by three distinct provider/model runs. It does not include
a frontier-tier model: GLM-4.6 is intentionally labeled `strong-open`, not
frontier. Consequently, the result currently supports cross-model persistence
across the tested capability range but **does not yet show whether degradation
persists at the frontier**. The frontier run is a required follow-up, and this
note will report a null result plainly if the degradation disappears there.

Clean accuracy rises across these three models, from `76.1%` to `83.2%` to
`84.5%`, but aggregate degradation is **not** monotonic: GLM-4.6 has the highest
clean accuracy yet a larger average drop than Haiku, because it takes bigger
hits on the two dimensions that survive, telegraphic phrasing and cursing, even
as it is robust to the rest. We therefore do **not** claim "more capable models
are uniformly more robust." The robust claim is dimension-specific.

## Per-Dimension Paired Results

`CI` is the 95% paired bootstrap interval for the absolute percentage-point
drop. `p adj.` is the exact McNemar p-value after Holm correction within the
model's seven-dimension family.

### `gpt-5.4-nano` (clean 76.1%)

| Dimension | Noisy acc. | Drop (95% CI) | Fail | Fix | p adj. | Sig |
|---|---:|---:|---:|---:|---:|:--:|
| `pasted_context_block` | 74.2% | 1.9 pp [0.9, 2.9] | 96 | 51 | 0.0018 | **yes** |
| `cursing` | 74.4% | 1.7 pp [0.6, 2.7] | 97 | 58 | 0.0130 | **yes** |
| `telegraphic_request` | 74.6% | 1.4 pp [0.4, 2.5] | 98 | 64 | 0.0465 | **yes** |
| `irrelevant_context` | 74.9% | 1.2 pp [0.2, 2.2] | 84 | 56 | 0.0886 | no |
| `argumentative_challenge` | 75.1% | 1.0 pp [0.1, 2.0] | 76 | 52 | 0.1249 | no |
| `removed_spaces` | 75.3% | 0.8 pp [-0.1, 1.8] | 76 | 57 | 0.2365 | no |
| `typos` | 75.8% | 0.3 pp [-0.6, 1.2] | 67 | 60 | 0.5946 | no |

### `claude-haiku-4-5-20251001` (clean 83.2%)

| Dimension | Noisy acc. | Drop (95% CI) | Fail | Fix | p adj. | Sig |
|---|---:|---:|---:|---:|---:|:--:|
| `telegraphic_request` | 81.7% | 1.4 pp [0.8, 2.1] | 51 | 17 | 0.00031 | **yes** |
| `cursing` | 82.4% | 0.8 pp [0.3, 1.3] | 30 | 12 | 0.0475 | **yes** |
| `irrelevant_context` | 82.5% | 0.6 pp [0.1, 1.2] | 29 | 14 | 0.1577 | no |
| `argumentative_challenge` | 82.7% | 0.5 pp [0.0, 1.0] | 23 | 12 | 0.3581 | no |
| `removed_spaces` | 82.9% | 0.2 pp [-0.2, 0.6] | 13 | 8 | 1.0000 | no |
| `typos` | 83.0% | 0.1 pp [-0.3, 0.5] | 12 | 9 | 1.0000 | no |
| `pasted_context_block` | 83.1% | 0.1 pp [-0.5, 0.7] | 26 | 24 | 1.0000 | no |

### `z-ai/glm-4.6` (clean 84.5%)

| Dimension | Noisy acc. | Drop (95% CI) | Fail | Fix | p adj. | Sig |
|---|---:|---:|---:|---:|---:|:--:|
| `telegraphic_request` | 82.5% | 2.0 pp [1.1, 3.0] | 86 | 38 | 0.00014 | **yes** |
| `cursing` | 83.1% | 1.4 pp [0.5, 2.3] | 76 | 43 | 0.0191 | **yes** |
| `argumentative_challenge` | 83.7% | 0.9 pp [0.0, 1.7] | 60 | 40 | 0.2844 | no |
| `irrelevant_context` | 83.8% | 0.7 pp [-0.1, 1.5] | 52 | 36 | 0.4372 | no |
| `removed_spaces` | 83.9% | 0.6 pp [-0.1, 1.4] | 48 | 33 | 0.4372 | no |
| `typos` | 84.0% | 0.6 pp [-0.2, 1.3] | 49 | 36 | 0.4372 | no |
| `pasted_context_block` | 84.3% | 0.2 pp [-0.6, 0.9] | 46 | 42 | 0.7493 | no |

## The Cross-Model Signal

- **`telegraphic_request`** is the largest cross-model effect. Its adjusted
  p-values are `0.0465`, `0.00031`, and `0.00014`, with 95% CIs excluding zero
  on nano, Haiku, and GLM respectively.
- **`cursing`** is the secondary signal. Its adjusted p-values are `0.0130`,
  `0.0475`, and `0.0191`, again with 95% CIs excluding zero on all three models.
  The Haiku decision is close to the threshold and should be interpreted as a
  small effect, not evidence of a broad collapse.
- **`pasted_context_block`** is the instructive reversal. It is the largest and
  most significant degradation on the cheap model (1.9 pp, 95% CI [0.9, 2.9],
  Holm-adjusted `p = 0.0018`),
  but a near-perfect coin flip on both capable models (Haiku 26 vs. 24,
  `p = 0.89`; GLM 46 vs. 42, `p = 0.75`). It is a cheap-model fragility that
  capability eliminates. It also shows why the paired test matters: a
  one-directional regression count would have overstated this dimension.

The shape across capability: the cheap model has three Holm-significant cells;
the capable models retain the two phrasing-register effects, terseness and
profanity. Most other intervals include zero after paired resampling.

## Artifact Screen

Raw clean-to-noisy failures include real model failures and evaluator/oracle
edge cases. We therefore keep two views: the raw paired McNemar counts above,
and a screened count that removes likely artifacts such as accepted-alias gaps
(`panda` vs. `giant panda`) or baseline ambiguity (`8%` represented as `8`
instead of `0.08`).

| Model | Dimension | Raw fail | Raw fix | Screened fail | Screened fix | Symmetric p | Bonf. p |
|---|---|---:|---:|---:|---:|---:|---:|
| `gpt-5.4-nano` | `telegraphic_request` | 98 | 64 | 75 | - | - | - |
| `gpt-5.4-nano` | `pasted_context_block` | 96 | 51 | 74 | - | - | - |
| `gpt-5.4-nano` | `cursing` | 97 | 58 | 67 | - | - | - |
| `claude-haiku-4-5-20251001` | `telegraphic_request` | 51 | 17 | 37 | 7 | 5.3e-6 | 3.7e-5 |
| `claude-haiku-4-5-20251001` | `cursing` | 30 | 12 | 20 | 7 | 0.019 | 0.134 |
| `z-ai/glm-4.6` | `telegraphic_request` | 86 | 38 | 61 | 19 | 2.7e-6 | 1.9e-5 |
| `z-ai/glm-4.6` | `cursing` | 76 | 43 | 50 | 23 | 0.002 | 0.015 |

The screened p-value applies the artifact screen symmetrically: possible
artifacts are removed from both clean-to-noisy failures and noisy-to-clean
fixes, then McNemar is recomputed. That matters. A one-sided screen, removing
only failure artifacts while keeping all fixes, biases the test toward the null.
The symmetric screen restores the main result for `telegraphic_request` on both
capable models and for `cursing` on GLM-4.6. Haiku `cursing` remains weaker.

The nano rows use the existing reviewed article artifact for clean-to-noisy
failures. The matching full first-run noisy-to-clean fix review is not available
in the checked-in article artifact, so no symmetric screened p-value is reported
for nano.

The Haiku and GLM rows classify every raw discordant item in both directions for
the significant cells with a first-pass researcher screen based on function
names, argument diffs, evaluator errors, and spot-checked prompts. This is not a
blinded adjudication study. That distinction matters: the screened counts should
be read as conservative article-level evidence, not as final benchmark labels.

The remaining failures are not mostly crashes. They are plausible-looking wrong
tool calls: missing the second item in a multi-call request, changing a required
argument, or routing to a related but wrong function. The showcase examples
below are verified `true_model_failure` rows from
`artifacts/analysis/article/clean_to_noisy_failures.csv`; the smaller
`cross_model_failure_examples.csv` file also includes artifact-control rows and
is not the showcase failure list.

## Full Failure Table And Examples

The exhaustive file for the article claim is
`artifacts/analysis/article/clean_to_noisy_failures.csv`. It contains every
clean-correct/noisy-wrong row across the three article-facing model runs, with
the clean prompt, noisy prompt, expected BFCL tool calls, clean model calls,
noisy model calls, evaluator error, provider response ids, and explicit manual
review status and notes. Use `paired_stats.csv` and `model_comparison.csv` for
headline aggregate counts.

| Model | Clean-correct/noisy-wrong rows |
|---|---:|
| `gpt-5.4-nano` | 621 |
| `claude-haiku-4-5-20251001` | 184 |
| `z-ai/glm-4.6` | 417 |
| **Total** | **1222** |

Every row in this table has been reviewed:

Rows initially labeled `possible_oracle_artifact` received a second-pass audit;
90 were promoted to `true_model_failure` and 1 remains `uncertain`.

| Model | `true_model_failure` | `possible_oracle_artifact` | `uncertain` |
|---|---:|---:|---:|
| `gpt-5.4-nano` | 387 | 234 | 0 |
| `claude-haiku-4-5-20251001` | 80 | 104 | 0 |
| `z-ai/glm-4.6` | 237 | 179 | 1 |
| **Total** | **704** | **517** | **1** |

Some concrete inspectable failures:

- **Dropped call**: `parallel_multiple_27` (GLM, `telegraphic_request`). Clean
  prompt asks to transfer `$5000` and calculate interest. The telegraphic prompt
  keeps both tasks, but the model only calls the interest calculator.
- **Wrong tool**: `live_multiple_718-165-5` (GLM, `telegraphic_request`). Clean
  prompt asks to book a house in Austin. The telegraphic prompt still asks to
  book it, but the model calls the house search tool instead of the booking
  tool.
- **Wrong date argument across models**: `live_multiple_676-163-1`
  (`telegraphic_request`). Clean prompt asks for New York weather tomorrow and
  states today is `2023.10.1`. The noisy prompt preserves that information, but
  GPT, Haiku, and GLM all produce `2023-10-01` instead of the gold
  `2023-10-02` in at least one article-facing run.
- **Profanity flips routing**: `live_multiple_992-223-0` (GLM, `cursing`). Clean
  prompt asks to delete the Apdex config for `d0404`. The profane rewrite still
  asks for deletion, but the model calls the list/get configuration tool.

## Interpretation

1. Clean BFCL accuracy does not certify robustness to realistic phrasing. Even a
   strong open function-calling model has a statistically significant weakness
   to telegraphic shorthand that the clean benchmark cannot see.
2. Capability buys real robustness at the dimension level: most perturbations
   that hurt the cheap model do nothing detectable to the capable ones. But it
   does not fully buy robustness to terse or profane phrasing, and aggregate drop
   is not monotonic in clean accuracy.
3. The mechanism in the failing examples is frequently **list compression**:
   terse phrasing of a multi-item request (`price a 2 GB and a 4 GB machine` to
   `2gb 4gb price`) leads the model to fire one tool call instead of two, or to
   drop the second item of a parallel request. This is a specific, reproducible
   failure rather than generic "messiness."

## What This Does And Does Not Show

- This is a focused probe on single-turn BFCL tool-calling, not a general
  agentic benchmark.
- Capability does not uniformly reduce degradation; the aggregate effect is not
  a clean gradient.
- Per-dimension discordant counts are reported raw and feed exact McNemar tests;
  published decisions use Holm-adjusted p-values and paired bootstrap CIs. The
  artifact screen
  above is a separate credibility check on the significant cells. It reduces
  both directions substantially. Under the symmetric screened test,
  `telegraphic_request` remains Bonferroni-significant for both capable models;
  `cursing` remains significant for GLM-4.6 but not Haiku.
- The significant-cell screen is a first-pass review, not a full independent
  annotation study. Some borderline cases remain, especially where the BFCL gold
  oracle is stricter than a human might be.
- A context-length dose-response probe, scaling pasted-context size, did **not**
  find that longer inert context amplifies the phrasing penalty. On a small nano
  run the trend was flat to slightly reversed. Inert filler is the weakest form
  of context pressure; competing or stale context was not tested. This remains
  an open question, not a supported claim.

## Reproducibility

- BFCL substrate and dimensions are pinned in `configs/`.
- All runs are full-pool and temperature `0`; temperature is recorded in run
  metadata and cache fingerprints, but not repeated as a column in
  `clean_to_noisy_failures.csv`.
- Per-model paired statistics and McNemar outputs are written under
  `artifacts/analysis/article/`; the current inferential table is
  `artifacts/analysis/significance.csv`.

Key artifacts:

```text
artifacts/analysis/article/paired_stats.csv
artifacts/analysis/article/model_comparison.csv
artifacts/analysis/significance.csv
artifacts/analysis/article/clean_to_noisy_failures.csv
artifacts/analysis/article/clean_to_noisy_failures_summary.csv
artifacts/analysis/article/significant_cell_review_summary.csv
artifacts/analysis/article/significant_cell_review.csv
artifacts/analysis/article/significant_cell_fix_review.csv
artifacts/analysis/article/cross_model_failure_examples.csv
artifacts/analysis/article/haiku_full_pool_summary.csv
artifacts/analysis/article/glm46_full_pool_paired_summary.csv
artifacts/analysis/article/included_failure_examples.csv
artifacts/analysis/article/oracle_issue_examples.csv
```

> **GLM-4.6 serving.** GLM-4.6 was served via OpenRouter pinned to the DeepInfra
> backend at unverified weight precision. Because GLM-4.6 is open weights and is
> served at varying precision across providers, treat its absolute clean accuracy
> as provider-dependent. The paired design controls for this: clean and noisy
> prompts share the same serving route, so the degradation comparison is
> internally valid regardless of precision. The served responses included
> reasoning tokens, but the headline `telegraphic_request` GLM failures do not
> appear to be output-cap artifacts: among the 86 clean-to-noisy failures, max
> completion length was 612 tokens, p95 was 436, and none were near the
> 1,024-token completion cap.

## Next Steps

1. If this grows from article to benchmark paper, repeat the significant-cell
   screen with independent adjudication.
2. Position this probe relative to prior work on paraphrase and format
   sensitivity in BFCL-style evaluation.
