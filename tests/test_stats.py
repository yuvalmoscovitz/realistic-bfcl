from __future__ import annotations

import math

import numpy as np
import pytest

from realistic_bfcl.stats import (
    ARTICLE_SIGNIFICANCE_DIMENSIONS,
    REPO_ROOT,
    exact_mcnemar,
    holm_bonferroni,
    paired_differences_from_counts,
    percentile_bootstrap_ci,
    read_comparison_csv,
    significance_rows_from_comparison,
)


def comparison_row(
    model: str,
    dimension: str,
    *,
    total: int = 100,
    clean_accuracy: float = 0.8,
    b: int = 12,
    c: int = 3,
) -> dict[str, object]:
    return {
        "model": model,
        "dimension": dimension,
        "total": total,
        "clean_accuracy": clean_accuracy,
        "noisy_accuracy": clean_accuracy - (b - c) / total,
        "absolute_degradation": (b - c) / total,
        "clean_success_noisy_failure": b,
        "clean_failure_noisy_success": c,
    }


def test_exact_mcnemar_textbook_example() -> None:
    assert exact_mcnemar(12, 3) == pytest.approx(0.03515625)
    assert exact_mcnemar(3, 12) == pytest.approx(0.03515625)
    assert exact_mcnemar(0, 0) == 1.0


def test_exact_mcnemar_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        exact_mcnemar(-1, 2)


def test_holm_bonferroni_known_values_and_order() -> None:
    adjusted = holm_bonferroni([0.01, 0.04, 0.03, 0.002])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.008])
    assert holm_bonferroni([]) == []
    assert holm_bonferroni([0.8, 0.8]) == [1.0, 1.0]


def test_holm_bonferroni_rejects_invalid_p_value() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        holm_bonferroni([math.nan])


def test_paired_difference_reconstruction_and_bootstrap_are_deterministic() -> None:
    differences = paired_differences_from_counts(10, 3, 2)
    assert differences.tolist() == [1, 1, 1, -1, -1, 0, 0, 0, 0, 0]
    assert differences.mean() == pytest.approx(0.1)

    first = percentile_bootstrap_ci(differences, n_resamples=2_000, seed=7)
    second = percentile_bootstrap_ci(differences, n_resamples=2_000, seed=7)
    assert first == second
    assert first[0] <= differences.mean() <= first[1]
    assert percentile_bootstrap_ci(np.zeros(8), n_resamples=100) == (0.0, 0.0)


def test_significance_rows_adjust_within_each_model_family() -> None:
    rows = significance_rows_from_comparison(
        [
            comparison_row("model-b", "two", b=0, c=0),
            comparison_row("model-a", "two", b=5, c=5),
            comparison_row("model-a", "one", b=12, c=3),
            comparison_row("model-b", "one", b=15, c=1),
        ],
        n_resamples=200,
        seed=9,
    )
    assert [(row["model"], row["dimension"]) for row in rows] == [
        ("model-a", "one"),
        ("model-a", "two"),
        ("model-b", "one"),
        ("model-b", "two"),
    ]
    by_key = {(row["model"], row["dimension"]): row for row in rows}
    assert by_key[("model-a", "one")]["p_adjusted"] == pytest.approx(0.0703125)
    assert by_key[("model-a", "one")]["degradation"] == pytest.approx(0.09)
    assert by_key[("model-a", "one")]["noisy_acc"] == pytest.approx(0.71)
    assert by_key[("model-b", "one")]["significant"] is True


def test_significance_rows_reject_duplicate_cells() -> None:
    row = comparison_row("model", "dimension")
    with pytest.raises(ValueError, match="Duplicate"):
        significance_rows_from_comparison([row, row], n_resamples=10)


def test_published_comparison_is_three_complete_seven_dimension_families() -> None:
    source = read_comparison_csv(
        REPO_ROOT / "artifacts/analysis/article/model_comparison.csv"
    )
    rows = significance_rows_from_comparison(
        source,
        expected_dimensions=ARTICLE_SIGNIFICANCE_DIMENSIONS,
        n_resamples=20,
    )
    assert len(rows) == 21
    assert len({str(row["model"]) for row in rows}) == 3


def test_significance_rows_reject_incomplete_family() -> None:
    with pytest.raises(ValueError, match="Incomplete significance family"):
        significance_rows_from_comparison(
            [comparison_row("model", "only")],
            expected_dimensions=ARTICLE_SIGNIFICANCE_DIMENSIONS,
            n_resamples=10,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("noisy_accuracy", 0.9),
        ("absolute_degradation", 0.5),
        ("clean_success_noisy_failure", 90),
    ],
)
def test_significance_rows_reject_inconsistent_aggregates(
    field: str,
    value: object,
) -> None:
    row = comparison_row("model", "dimension")
    row[field] = value
    with pytest.raises(ValueError, match="disagrees|Infeasible"):
        significance_rows_from_comparison([row], n_resamples=10)
