# Realistic-BFCL Unified Article Report

This report consolidates the article-facing artifacts in `artifacts/analysis/article/`. It separates prompt coverage, aggregate metrics, reviewed/screened failures, curated examples, and strict cross-model failures.

## Coverage

- Clean BFCL-derived examples: `2351`
- Reviewed deterministic noisy dimensions: `7`
- Generated noisy examples per dimension: `2351`
- Total generated noisy prompts across reviewed dimensions: `16457`

Important caveat: all prompts and raw prediction caches exist for the article-facing runs, but not every raw failure has a human-reviewed label. The screened labels are strongest for nano reviewed article artifacts and for Haiku/GLM significant cells.

## Headline Model Results

| Model | Clean acc. | Avg noisy acc. | Avg drop |
|---|---:|---:|---:|
| `gpt-5.4-nano` | 76.1% | 74.9% | 1.2 pp |
| `claude-haiku-4-5-20251001` | 83.2% | 82.6% | 0.5 pp |
| `z-ai/glm-4.6` | 84.5% | 83.6% | 0.9 pp |

Drop values are absolute percentage-point drops (`pp`), not relative percent change.

## Significant-Cell Artifact Screen

| Model | Dimension | Raw fail | Raw fix | Screened fail | Screened fix | Symmetric p | Bonf. p |
|---|---|---:|---:|---:|---:|---:|---:|
| `gpt-5.4-nano` | `telegraphic_request` | 98 | 64 | 75 | - | - | - |
| `gpt-5.4-nano` | `pasted_context_block` | 96 | 51 | 74 | - | - | - |
| `gpt-5.4-nano` | `cursing` | 97 | 58 | 67 | - | - | - |
| `claude-haiku-4-5-20251001` | `telegraphic_request` | 51 | 17 | 37 | 7 | 5.29958174411e-06 | 3.70970722088e-05 |
| `claude-haiku-4-5-20251001` | `cursing` | 30 | 12 | 20 | 7 | 0.0191572904587 | 0.134101033211 |
| `z-ai/glm-4.6` | `telegraphic_request` | 86 | 38 | 61 | 19 | 2.73186992288e-06 | 1.91230894602e-05 |
| `z-ai/glm-4.6` | `cursing` | 76 | 43 | 50 | 23 | 0.0021181727464 | 0.0148272092248 |

Nano rows use reviewed clean-to-noisy failures only; a matching full first-run fix-side screen is not checked in. Haiku and GLM rows screen both discordant directions.

## Full Clean-Pass/Noisy-Fail Table

The exhaustive examples are in `clean_to_noisy_failures.csv`: one row per model,
dimension, and base example where the clean prompt was correct and the noisy
prompt was wrong. It includes the clean prompt, noisy prompt, expected BFCL tool
calls, clean model calls, noisy model calls, evaluator error, provider response
ids, explicit review status, and review notes.

| Model | Clean-pass/noisy-fail rows |
|---|---:|
| `gpt-5.4-nano` | 621 |
| `claude-haiku-4-5-20251001` | 184 |
| `z-ai/glm-4.6` | 417 |
| **Total** | **1222** |

Every row has been reviewed. The row-level screen splits the raw failures as
follows:

Rows initially labeled `possible_oracle_artifact` received a second-pass audit;
90 were promoted to `true_model_failure` and 1 remains `uncertain`.

| Model | `true_model_failure` | `possible_oracle_artifact` | `uncertain` |
|---|---:|---:|---:|
| `gpt-5.4-nano` | 387 | 234 | 0 |
| `claude-haiku-4-5-20251001` | 80 | 104 | 0 |
| `z-ai/glm-4.6` | 237 | 179 | 1 |
| **Total** | **704** | **517** | **1** |

The compact count table is in `clean_to_noisy_failures_summary.csv`. These are
row-level trace counts by `evaluation_run_id`, including reviewed true-failure
and possible-artifact counts; use `paired_stats.csv` and `model_comparison.csv`
for headline aggregate statistics. All article-facing trace rows were evaluated
at temperature 0, so temperature is not repeated as a column in the trace table.

The primary key is `row_id` (`model | evaluation_run_id | dimension | base_id`).
`gpt54nano_full_pool_repeat2` is a full-pool nano repeat used for row-level
traceability; headline nano aggregate statistics are in `paired_stats.csv`.
Column definitions are in `clean_to_noisy_failures_schema.md`.

## Error Taxonomy

### Nano Reviewed Article Failures

| Error type | Count |
|---|---:|
| `wrong_argument_value` | 229 |
| `missing_tool_call` | 115 |
| `wrong_tool_routing` | 63 |
| `extra_tool_call` | 5 |
| `argument_drop` | 1 |
| `malformed_call` | 1 |

### Haiku/GLM Screened Significant-Cell Failures

| Model | Wrong argument | Missing/extra call | Wrong tool routing |
|---|---:|---:|---:|
| `claude-haiku-4-5-20251001` | 39 | 14 | 4 |
| `z-ai/glm-4.6` | 73 | 32 | 6 |

The error categories present in all three models are: wrong argument value, missing/extra tool calls, and wrong tool routing.

## Strict All-Three-Model Failures

Definition: clean prompt correct for all three models, noisy prompt wrong for all three models, same `base_id + dimension`. For nano, this uses the full `gpt54nano_full_pool_repeat2` cache because the checked-in first-run noisy cache only has 250 rows.

Found `8` strict all-three failures. Full rows are in `all_three_wrong_examples.csv`.

| Dimension | Base ID | Category | Clean prompt | Main failure pattern |
|---|---|---|---|---|
| `cursing` | `live_simple_228-119-0` | `live_simple` | Convert it to audio format: I am a pretty girl | same tool(s), wrong argument/value: `text_to_speech.convert` |
| `cursing` | `live_multiple_1018-247-0` | `live_multiple` | Provide me with all the configured websites. My API is called YOUR_API_KEY | same tool(s), wrong argument/value: `website_configuration_api.get_websites` |
| `pasted_context_block` | `live_simple_32-9-0` | `live_simple` | Sure, here is the answer to the question:\n\n**Logistic regression is not present in the text, therefore I cannot answer this q... | mixed routing/count failures: GPT `answer.string`, Haiku `answer.string`, GLM `<empty>` |
| `telegraphic_request` | `live_simple_36-13-0` | `live_simple` | Sure, here is the answer to the question:\n\nThe text does not define logistic regression, therefore I cannot answer this quest... | same tool(s), wrong argument/value: `parseAnswer` |
| `telegraphic_request` | `live_simple_37-14-0` | `live_simple` | Sure, here is the answer to the question:\n\nThe text does not define logistic regression, therefore I cannot answer this quest... | same tool(s), wrong argument/value: `parseAnswer` |
| `telegraphic_request` | `live_simple_105-62-0` | `live_simple` | classify these sentences\nlink my account\nconnect my accounts\nhello | same tool(s), wrong argument/value: `classify` |
| `telegraphic_request` | `live_simple_163-96-2` | `live_simple` | The image is a poster for The Lego Movie. It features the Lego logo, which is a red and yellow rectangle with the word "LEGO" w... | same tool(s), wrong argument/value: `get_items` |
| `telegraphic_request` | `live_multiple_676-163-1` | `live_multiple` | What's the weather going to be like in New York tomorrow? today is 2023.10.1 | same tool(s), wrong argument/value: `Weather_1_GetWeather` |

## Curated Example Files

| File | Rows | Meaning |
|---|---:|---|
| `cross_model_failure_examples.csv` | 10 | Small curated examples across models plus artifact controls. |
| `clean_to_noisy_failures.csv` | 1222 | Exhaustive reviewed clean-correct/noisy-wrong rows across all three article-facing model runs. |
| `clean_to_noisy_failures_schema.md` | - | Column definitions and review-label caveats for the trace table. |
| `clean_to_noisy_failures_summary.csv` | 21 | Raw and reviewed counts from the exhaustive failure table by model, evaluation run, and dimension. |
| `included_failure_examples.csv` | 20 | Curated nano article examples. |
| `candidate_failure_examples.csv` | 40 | Candidate examples considered for article inclusion. |
| `significant_cell_review.csv` | 243 | Haiku/GLM clean-to-noisy significant-cell screen. |
| `significant_cell_fix_review.csv` | 110 | Haiku/GLM noisy-to-clean significant-cell screen. |
| `all_three_wrong_examples.csv` | 8 | Strict clean-correct/noisy-wrong examples shared by all three models. |

## Interpretation

- The full generated prompt set is available and evaluator-ready.
- The strongest reviewed failure evidence is not one monolithic file; it is split by purpose: aggregate stats, significant-cell screens, curated examples, and strict cross-model intersections.
- The common failure modes are plausible wrong tool calls: wrong arguments, dropped/extra calls, and wrong routing. Malformed calls are rare in the reviewed nano taxonomy.
- Several strict all-three failures are evaluator-strict exact-string cases, so they should be used carefully. The most interpretable shared failures are the cursing examples that include profanity in an argument and the telegraphic weather example that resolves `tomorrow` to the current date.
