# Realistic-BFCL - Findings

## Summary

**Clean evals tell you whether your agent *can* do the task. They do not tell you
whether it *will* once a real user phrases the request the way real users
actually do.** This note measures that gap on BFCL tool-calling.

Across three models of increasing clean accuracy, clean BFCL accuracy overstates
robustness to ordinary, production-like phrasing, and the clean benchmark cannot
see the gap. The cheap model is fragile to a broad range of phrasing noise.
Capable models shed almost all of it, with one exception that survives across
the stronger models tested:

> **Telegraphic shorthand - terse, grammar-free phrasing - is the one noise
> dimension that degrades every model tested, and the only one that stays
> significant on both capable models after multiple-comparison correction.**

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
laundered into a single pooled degradation number. Significance below is stated
under Bonferroni correction across the seven dimensions per model
(`alpha ~= 0.0071`), with Benjamini-Hochberg FDR noted where it changes the
call.

## Headline Results

| Model | Provider | Tier | Clean acc. | Avg noisy acc. | Avg drop |
|---|---|---|---:|---:|---:|
| `gpt-5.4-nano` | OpenAI | cheap | 0.761 | 0.749 | 0.012 |
| `claude-haiku-4-5-20251001` | Anthropic | mid | 0.832 | 0.827 | 0.005 |
| `z-ai/glm-4.6` | OpenRouter | strong-open | 0.845 | 0.836 | 0.009 |

Clean accuracy rises across these three models, from `0.761` to `0.832` to
`0.845`, but aggregate degradation is **not** monotonic: GLM-4.6 has the highest
clean accuracy yet a larger average drop than Haiku, because it takes bigger
hits on the two dimensions that survive, telegraphic phrasing and cursing, even
as it is robust to the rest. We therefore do **not** claim "more capable models
are uniformly more robust." The robust claim is dimension-specific.

## Per-Dimension Paired Results

Sorted by exact p-value. `Sig` means significant under Bonferroni correction
(`alpha ~= 0.0071`).

### `gpt-5.4-nano` (clean 0.761)

| Dimension | Noisy | Drop | Fail | Fix | McNemar p | Sig |
|---|---:|---:|---:|---:|---:|:--:|
| `pasted_context_block` | 0.742 | 0.019 | 96 | 51 | 0.00026 | **yes** |
| `cursing` | 0.744 | 0.017 | 97 | 58 | 0.0022 | **yes** |
| `telegraphic_request` | 0.746 | 0.014 | 98 | 64 | 0.0093 | no (FDR yes, q=0.022) |
| `irrelevant_context` | 0.749 | 0.012 | 84 | 56 | 0.0222 | no (FDR yes, q=0.039) |
| `argumentative_challenge` | 0.751 | 0.010 | 76 | 52 | 0.0416 | no |
| `removed_spaces` | 0.753 | 0.008 | 76 | 57 | 0.1182 | no |
| `typos` | 0.758 | 0.003 | 67 | 60 | 0.5946 | no |

### `claude-haiku-4-5-20251001` (clean 0.832)

| Dimension | Noisy | Drop | Fail | Fix | McNemar p | Sig |
|---|---:|---:|---:|---:|---:|:--:|
| `telegraphic_request` | 0.817 | 0.014 | 51 | 17 | 0.000045 | **yes** |
| `cursing` | 0.824 | 0.008 | 30 | 12 | 0.0079 | no (borderline) |
| `irrelevant_context` | 0.825 | 0.006 | 29 | 14 | 0.0315 | no |
| `argumentative_challenge` | 0.827 | 0.005 | 23 | 12 | 0.0895 | no |
| `removed_spaces` | 0.829 | 0.002 | 13 | 8 | 0.3833 | no |
| `typos` | 0.830 | 0.001 | 12 | 9 | 0.6636 | no |
| `pasted_context_block` | 0.831 | 0.001 | 26 | 24 | 0.8877 | no |

### `z-ai/glm-4.6` (clean 0.845)

| Dimension | Noisy | Drop | Fail | Fix | McNemar p | Sig |
|---|---:|---:|---:|---:|---:|:--:|
| `telegraphic_request` | 0.825 | 0.020 | 86 | 38 | 0.000019 | **yes** |
| `cursing` | 0.831 | 0.014 | 76 | 43 | 0.0032 | **yes** |
| `argumentative_challenge` | 0.837 | 0.009 | 60 | 40 | 0.0569 | no |
| `irrelevant_context` | 0.838 | 0.007 | 52 | 36 | 0.1093 | no |
| `removed_spaces` | 0.839 | 0.006 | 48 | 33 | 0.1193 | no |
| `typos` | 0.840 | 0.006 | 49 | 36 | 0.1928 | no |
| `pasted_context_block` | 0.843 | 0.002 | 46 | 42 | 0.7493 | no |

## The Cross-Model Signal

- **`telegraphic_request`** is the through-line. It degrades every model:
  significant after Bonferroni on both capable models (Haiku `p = 4.5e-5`,
  GLM-4.6 `p = 1.9e-5`), and on the cheap model it is significant under FDR
  control (`q = 0.022`) and borderline under the stricter Bonferroni threshold.
  It is the only dimension significant on both capable models.
- **`cursing`** is the secondary signal: significant under Bonferroni on nano
  (`p = 0.0022`) and GLM-4.6 (`p = 0.0032`), and borderline on Haiku
  (`p = 0.0079`, just above the corrected threshold). Profanity shifts routing
  even though it carries no task information.
- **`pasted_context_block`** is the instructive reversal. It is the largest and
  most significant degradation on the cheap model (`p = 0.00026`, drop 0.019),
  but a near-perfect coin flip on both capable models (Haiku 26 vs. 24,
  `p = 0.89`; GLM 46 vs. 42, `p = 0.75`). It is a cheap-model fragility that
  capability eliminates. It also shows why the paired test matters: a
  one-directional regression count would have overstated this dimension.

The shape across capability: the cheap model is broadly fragile. Pasted context,
profanity, terseness, and irrelevant context all register. Capable models drop
almost all of that, but not terseness.

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
argument, or routing to a related but wrong function. Concrete examples are in
`artifacts/analysis/article/cross_model_failure_examples.csv`.

## Interpretation

1. Clean BFCL accuracy does not certify robustness to realistic phrasing. Even a
   strong open function-calling model has a statistically significant weakness
   to telegraphic shorthand that the clean benchmark cannot see.
2. Capability buys real robustness at the dimension level: most perturbations
   that hurt the cheap model do nothing detectable to the capable ones. But it
   does not buy robustness to terse phrasing, and aggregate drop is not monotonic
   in clean accuracy.
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
- Per-dimension McNemar counts are still reported raw because they are the
  correct paired statistical test over all paired flips. The artifact screen
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
  metadata and cache fingerprints.
- Per-model paired statistics and McNemar outputs are written under
  `artifacts/analysis/article/`.

Key artifacts:

```text
artifacts/analysis/article/paired_stats.csv
artifacts/analysis/article/model_comparison.csv
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
