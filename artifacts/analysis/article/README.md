# Realistic-BFCL Article Data

These files organize the full-pool gpt-5.4-nano evaluation for article writing.
Article counts exclude rows marked as possible oracle, augmentation, or baseline dataset issues, plus manually reviewed artifact/questionable rows.

The Haiku files are raw model-comparison summaries for
`claude-haiku-4-5-20251001`. They use the same frozen 2,351-example pool and
seven dimensions, but the full-pool Haiku regressions have not yet been manually
reviewed. The Haiku paired stats include exact McNemar p-values so weak or
near-balanced dimensions are visible rather than overclaimed.

## Dimension Results

| Dimension | Clean acc. | Noisy acc. | Drop | Article regressions | Article rate |
|---|---:|---:|---:|---:|---:|
| telegraphic_request | 0.761 | 0.746 | 0.014 | 75 | 0.042 |
| pasted_context_block | 0.761 | 0.742 | 0.019 | 74 | 0.041 |
| cursing | 0.761 | 0.744 | 0.017 | 67 | 0.037 |
| irrelevant_context | 0.761 | 0.749 | 0.012 | 58 | 0.032 |
| argumentative_challenge | 0.761 | 0.751 | 0.010 | 54 | 0.030 |
| typos | 0.761 | 0.758 | 0.003 | 45 | 0.025 |
| removed_spaces | 0.761 | 0.753 | 0.008 | 41 | 0.023 |

## Repeat-Run Stability

The evaluation was repeated three times with fresh clean and noisy model calls. Every listed noise type degraded accuracy in every run.

| Dimension | Runs | Mean drop | Min drop | Max drop | Drop sd |
|---|---:|---:|---:|---:|---:|
| pasted_context_block | 3 | 0.025 | 0.019 | 0.032 | 0.007 |
| cursing | 3 | 0.022 | 0.017 | 0.026 | 0.005 |
| telegraphic_request | 3 | 0.016 | 0.014 | 0.020 | 0.003 |
| argumentative_challenge | 3 | 0.014 | 0.010 | 0.019 | 0.004 |
| irrelevant_context | 3 | 0.011 | 0.009 | 0.012 | 0.002 |
| removed_spaces | 3 | 0.009 | 0.005 | 0.013 | 0.004 |
| typos | 3 | 0.006 | 0.003 | 0.009 | 0.003 |

## Files

- `dimension_results.csv`: article-ready per-dimension metrics.
- `paired_stats.csv`: full paired contingency counts, McNemar p-values, and multiple-comparison corrections.
- `review_filtering.csv`: raw-to-reviewed regression filtering counts.
- `stability_repeat_summary.csv`: mean/range across repeated fresh model runs.
- `stability_repeat_runs.csv`: per-run paired metrics used by the stability summary.
- `stability_repeat_summary.json`: JSON form of the stability summary.
- `error_type_counts.csv`: article regressions by manual error type.
- `overall_error_type_counts.csv`: aggregate article regressions by error type.
- `category_counts.csv`: article regressions by BFCL category.
- `overall_category_counts.csv`: aggregate article regressions by BFCL category.
- `candidate_failure_examples.csv`: strongest examples queued for human review.
- `included_failure_examples.csv`: reviewed qualitative examples for the article.
- `realism_audit_summary.csv`: first-pass researcher audit of reviewed failure candidates.
- `oracle_issue_examples.csv`: examples to exclude or discuss as evaluator/oracle ambiguity.
- `haiku_full_pool_summary.csv`: raw full-pool Haiku paired metrics.
- `haiku_full_pool_summary.json`: JSON form of the raw full-pool Haiku metrics.
- `haiku_full_pool_paired_stats.csv`: Haiku clean->noisy vs noisy->clean counts and exact McNemar p-values.
- `haiku_full_pool_paired_stats.json`: JSON form of the Haiku paired stats.
