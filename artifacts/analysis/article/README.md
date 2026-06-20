# Realistic-BFCL Article Data

These files organize the full-pool gpt-5.4-nano evaluation for article writing.
Article counts exclude rows marked as possible oracle, augmentation, or baseline dataset issues, plus manually reviewed artifact/questionable rows.

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

## Files

- `dimension_results.csv`: article-ready per-dimension metrics.
- `paired_stats.csv`: full paired contingency counts and McNemar p-values.
- `review_filtering.csv`: raw-to-reviewed regression filtering counts.
- `error_type_counts.csv`: article regressions by manual error type.
- `overall_error_type_counts.csv`: aggregate article regressions by error type.
- `category_counts.csv`: article regressions by BFCL category.
- `overall_category_counts.csv`: aggregate article regressions by BFCL category.
- `candidate_failure_examples.csv`: strongest examples queued for human review.
- `included_failure_examples.csv`: reviewed qualitative examples for the article.
- `oracle_issue_examples.csv`: examples to exclude or discuss as evaluator/oracle ambiguity.
