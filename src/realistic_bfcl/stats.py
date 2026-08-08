from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import REPO_ROOT, write_csv

SIGNIFICANCE_FIELDS = [
    "model",
    "dimension",
    "n_pairs",
    "clean_acc",
    "noisy_acc",
    "degradation",
    "b",
    "c",
    "n_discordant",
    "ci_low",
    "ci_high",
    "p_value",
    "p_adjusted",
    "significant",
]

ARTICLE_SIGNIFICANCE_DIMENSIONS = frozenset(
    {
        "argumentative_challenge",
        "cursing",
        "irrelevant_context",
        "pasted_context_block",
        "removed_spaces",
        "telegraphic_request",
        "typos",
    }
)


def exact_mcnemar(b: int, c: int) -> float:
    if b < 0 or c < 0:
        raise ValueError("McNemar counts must be non-negative.")
    discordant = b + c
    if discordant == 0:
        return 1.0
    smaller = min(b, c)
    log_half = math.log(0.5)
    lower_tail = math.fsum(
        math.exp(
            math.lgamma(discordant + 1)
            - math.lgamma(index + 1)
            - math.lgamma(discordant - index + 1)
            + discordant * log_half
        )
        for index in range(smaller + 1)
    )
    return min(1.0, 2 * lower_tail)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    if any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("p-values must be between 0 and 1.")
    total = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * total
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        running_max = max(running_max, (total - rank) * p_value)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted


def paired_differences_from_counts(n_pairs: int, b: int, c: int) -> np.ndarray:
    if n_pairs < 1:
        raise ValueError("n_pairs must be positive.")
    if b < 0 or c < 0 or b + c > n_pairs:
        raise ValueError("Discordant counts must be non-negative and no larger than n_pairs.")
    return np.concatenate(
        (
            np.ones(b, dtype=np.int8),
            -np.ones(c, dtype=np.int8),
            np.zeros(n_pairs - b - c, dtype=np.int8),
        )
    )


def percentile_bootstrap_ci(
    paired_differences: np.ndarray,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260618,
    chunk_size: int = 256,
) -> tuple[float, float]:
    differences = np.asarray(paired_differences, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("paired_differences must be a non-empty one-dimensional array.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    completed = 0
    while completed < n_resamples:
        current_size = min(chunk_size, n_resamples - completed)
        sample_indices = rng.integers(
            0,
            differences.size,
            size=(current_size, differences.size),
        )
        estimates[completed : completed + current_size] = differences[
            sample_indices
        ].mean(axis=1)
        completed += current_size

    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail], method="linear")
    return float(low), float(high)


def significance_rows_from_comparison(
    comparison_rows: list[dict[str, object]],
    *,
    expected_dimensions: frozenset[str] | None = None,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260618,
) -> list[dict[str, object]]:
    if not comparison_rows:
        raise ValueError("At least one comparison row is required.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in comparison_rows:
        model = str(row["model"])
        dimension = str(row["dimension"])
        key = (model, dimension)
        if key in seen:
            raise ValueError(f"Duplicate model/dimension row: {model}/{dimension}.")
        seen.add(key)
        grouped[model].append(row)

    if expected_dimensions is not None:
        for model, model_rows in grouped.items():
            actual_dimensions = {str(row["dimension"]) for row in model_rows}
            if actual_dimensions != expected_dimensions:
                missing = sorted(expected_dimensions - actual_dimensions)
                unexpected = sorted(actual_dimensions - expected_dimensions)
                details = []
                if missing:
                    details.append(f"missing={','.join(missing)}")
                if unexpected:
                    details.append(f"unexpected={','.join(unexpected)}")
                raise ValueError(
                    f"Incomplete significance family for {model}: {'; '.join(details)}."
                )

    pair_counts = {int(row["total"]) for row in comparison_rows}
    if len(pair_counts) != 1:
        raise ValueError("All significance cells must use the same number of pairs.")

    output = []
    for model in sorted(grouped):
        model_rows = sorted(grouped[model], key=lambda row: str(row["dimension"]))
        raw_rows = []
        for dimension_index, row in enumerate(model_rows):
            n_pairs = int(row["total"])
            b = int(row["clean_success_noisy_failure"])
            c = int(row["clean_failure_noisy_success"])
            differences = paired_differences_from_counts(n_pairs, b, c)
            ci_low, ci_high = percentile_bootstrap_ci(
                differences,
                n_resamples=n_resamples,
                confidence=confidence,
                seed=seed + dimension_index,
            )
            clean_acc = float(row["clean_accuracy"])
            noisy_acc = float(row["noisy_accuracy"])
            degradation = float(row["absolute_degradation"])
            clean_correct = round(clean_acc * n_pairs)
            noisy_correct = clean_correct - b + c
            tolerance = max(1e-6, 0.5 / n_pairs + 1e-12)
            if b > clean_correct or c > n_pairs - clean_correct:
                raise ValueError(
                    f"Infeasible contingency counts for {model}/{row['dimension']}."
                )
            derived_clean_acc = clean_correct / n_pairs
            derived_noisy_acc = noisy_correct / n_pairs
            derived_degradation = (b - c) / n_pairs
            if abs(clean_acc - derived_clean_acc) > tolerance:
                raise ValueError(
                    f"clean_accuracy disagrees with counts for {model}/{row['dimension']}."
                )
            if abs(noisy_acc - derived_noisy_acc) > tolerance:
                raise ValueError(
                    f"noisy_accuracy disagrees with counts for {model}/{row['dimension']}."
                )
            if (
                abs(degradation - derived_degradation) > tolerance
                or abs((clean_acc - noisy_acc) - derived_degradation) > tolerance
            ):
                raise ValueError(
                    f"absolute_degradation disagrees with counts for "
                    f"{model}/{row['dimension']}."
                )
            raw_rows.append(
                {
                    "model": model,
                    "dimension": str(row["dimension"]),
                    "n_pairs": n_pairs,
                    "clean_acc": derived_clean_acc,
                    "noisy_acc": derived_noisy_acc,
                    "degradation": derived_degradation,
                    "b": b,
                    "c": c,
                    "n_discordant": b + c,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": exact_mcnemar(b, c),
                }
            )

        adjusted = holm_bonferroni([float(row["p_value"]) for row in raw_rows])
        for row, p_adjusted in zip(raw_rows, adjusted):
            row["p_adjusted"] = p_adjusted
            row["significant"] = p_adjusted <= alpha
            output.append(row)
    return output


def read_comparison_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Comparison artifact not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "model",
            "dimension",
            "total",
            "clean_accuracy",
            "noisy_accuracy",
            "absolute_degradation",
            "clean_success_noisy_failure",
            "clean_failure_noisy_success",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Comparison artifact is missing columns: {', '.join(sorted(missing))}."
            )
        return list(reader)


def write_significance_csv(
    comparison_rows: list[dict[str, object]],
    output_path: Path,
    *,
    expected_dimensions: frozenset[str] = ARTICLE_SIGNIFICANCE_DIMENSIONS,
    n_resamples: int = 10_000,
    seed: int = 20260618,
) -> list[dict[str, object]]:
    rows = significance_rows_from_comparison(
        comparison_rows,
        expected_dimensions=expected_dimensions,
        n_resamples=n_resamples,
        seed=seed,
    )
    write_csv(output_path, rows, SIGNIFICANCE_FIELDS)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute paired significance statistics.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "artifacts/analysis/article/model_comparison.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/analysis/significance.csv",
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = write_significance_csv(
        read_comparison_csv(args.input),
        args.output,
        n_resamples=args.resamples,
        seed=args.seed,
    )
    print(f"Wrote {args.output} ({len(rows)} model/dimension rows)")


if __name__ == "__main__":
    main()
