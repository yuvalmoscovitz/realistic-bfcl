from __future__ import annotations

import json
import re

from .common import (
    DIMENSION_FILES,
    OPENAI_MODEL,
    REPO_ROOT,
    compact_text,
    conversation_text,
    read_jsonl,
    utc_now,
    write_csv,
    write_json,
    write_jsonl,
)
from .evaluate import generated_dimensions


def call_names(calls: object) -> list[str]:
    if not isinstance(calls, list):
        return []
    return [str(call.get("name", "")) for call in calls if isinstance(call, dict)]


def gold_call_names(gold: object) -> list[str]:
    if not isinstance(gold, list):
        return []
    names = []
    for call in gold:
        if not isinstance(call, dict):
            continue
        names.extend(str(name) for name in call)
    return names


def call_arguments(calls: object) -> list[dict[str, object]]:
    if not isinstance(calls, list):
        return []
    arguments = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        value = call.get("arguments", {})
        arguments.append(value if isinstance(value, dict) else {"__non_dict_arguments__": value})
    return arguments


def gold_arguments(gold: object) -> list[dict[str, object]]:
    if not isinstance(gold, list):
        return []
    arguments = []
    for call in gold:
        if not isinstance(call, dict):
            continue
        for value in call.values():
            arguments.append(value if isinstance(value, dict) else {})
    return arguments


def argument_keys(calls: object) -> set[str]:
    keys: set[str] = set()
    for arguments in call_arguments(calls):
        keys.update(str(key) for key in arguments)
    return keys


def gold_argument_keys(gold: object) -> set[str]:
    keys: set[str] = set()
    for arguments in gold_arguments(gold):
        keys.update(str(key) for key in arguments)
    return keys


def heuristic_error_type(gold: object, _clean_prediction: object, noisy_prediction: object) -> str:
    noisy_names = call_names(noisy_prediction)
    if not noisy_names:
        return "no_call"
    if any(
        "__malformed_arguments__" in arguments for arguments in call_arguments(noisy_prediction)
    ):
        return "malformed_arguments"
    gold_names = gold_call_names(gold)
    if len(gold_names) != len(noisy_names):
        return "call_count_mismatch"
    if gold_names != noisy_names:
        return "routing_error"

    gold_keys = gold_argument_keys(gold)
    noisy_keys = argument_keys(noisy_prediction)
    if gold_keys - noisy_keys:
        return "argument_drop"
    if noisy_keys - gold_keys:
        return "argument_hallucination"
    return "argument_value_error"


def count_by(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def flat_values(value: object) -> list[object]:
    if isinstance(value, dict):
        values = []
        for child in value.values():
            values.extend(flat_values(child))
        return values
    if isinstance(value, list):
        values = []
        for child in value:
            values.extend(flat_values(child))
        return values
    return [value]


def parsed_json_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def gold_values(gold: object) -> list[str]:
    values = []
    for arguments in gold_arguments(gold):
        values.extend(str(value) for value in flat_values(arguments) if value not in ("", None))
    return values


def typo_copied_into_argument(
    clean_prompt: str, noisy_prompt: str, noisy_prediction: object
) -> bool:
    clean_tokens = set(re.findall(r"[A-Za-z][A-Za-z']+", clean_prompt))
    noisy_tokens = set(re.findall(r"[A-Za-z][A-Za-z']+", noisy_prompt))
    introduced_tokens = {token.lower() for token in noisy_tokens - clean_tokens if len(token) >= 4}
    argument_text = json.dumps(call_arguments(noisy_prediction)).lower()
    return any(token in argument_text for token in introduced_tokens)


def likely_alias_or_normalization_issue(gold: object, noisy_prediction: object) -> bool:
    accepted_values = gold_values(gold)
    noisy_values = [
        str(value)
        for arguments in call_arguments(noisy_prediction)
        for value in flat_values(arguments)
    ]
    for noisy_value in noisy_values:
        compact_noisy = compact_text(noisy_value)
        if not compact_noisy:
            continue
        for accepted_value in accepted_values:
            compact_accepted = compact_text(accepted_value)
            if (
                compact_accepted
                and compact_accepted != compact_noisy
                and (compact_accepted in compact_noisy or compact_noisy in compact_accepted)
            ):
                return True
    return False


def regression_label(review: dict[str, object]) -> dict[str, str]:
    gold = parsed_json_value(review["gold"])
    noisy_prediction = parsed_json_value(review["noisy_prediction"])
    heuristic = str(review["heuristic_error_type"])
    noisy_names = call_names(noisy_prediction)
    expected_names = gold_call_names(gold)
    oracle_issue = "no"
    augmentation_issue = "no"

    if heuristic == "call_count_mismatch":
        if len(noisy_names) < len(expected_names):
            manual_error_type = "missing_tool_call"
            notes = "Noisy prediction emits fewer calls than the gold oracle."
        elif len(noisy_names) > len(expected_names):
            manual_error_type = "extra_tool_call"
            notes = "Noisy prediction emits more calls than the gold oracle."
        else:
            manual_error_type = "call_count_mismatch"
            notes = "Noisy prediction call count differs from the gold oracle."
    elif heuristic == "routing_error":
        manual_error_type = "wrong_tool_routing"
        notes = "Noisy prediction calls a different tool than the gold oracle."
    elif heuristic == "argument_value_error":
        if str(review["dimension"]) == "typos" and typo_copied_into_argument(
            str(review["clean_prompt"]),
            str(review["noisy_prompt"]),
            noisy_prediction,
        ):
            manual_error_type = "typo_copied_into_argument_value"
            augmentation_issue = "possible"
            notes = "The typo appears to have been copied into an argument value."
        elif likely_alias_or_normalization_issue(gold, noisy_prediction):
            manual_error_type = "entity_or_alias_normalization_mismatch"
            oracle_issue = "possible"
            notes = (
                "Noisy prediction appears semantically close but uses a different "
                "alias or formatting than accepted gold."
            )
        else:
            manual_error_type = "wrong_argument_value"
            notes = (
                "Noisy prediction uses the right tool and call count, but at least "
                "one argument value differs."
            )
    elif heuristic == "argument_drop":
        manual_error_type = "argument_drop"
        notes = "Noisy prediction omits an argument required by the gold oracle."
    elif heuristic == "argument_hallucination":
        manual_error_type = "argument_hallucination"
        notes = "Noisy prediction adds an argument not present in the gold oracle."
    elif heuristic == "no_call":
        manual_error_type = "no_tool_call"
        notes = "Noisy prediction did not emit a tool call."
    elif heuristic == "malformed_arguments":
        manual_error_type = "malformed_call"
        notes = "Noisy prediction emitted malformed tool-call arguments."
    else:
        manual_error_type = "other"
        notes = "Needs manual inspection."

    if str(review["dimension"]) == "argumentative_challenge":
        notes += " The argumentative wrapper may have distracted call decomposition."
    if str(review["dimension"]) == "removed_spaces":
        notes += " Missing whitespace may have reduced parsing of slots or calls."

    return {
        "review_status": "labeled_v1",
        "manual_error_type": manual_error_type,
        "oracle_issue": oracle_issue,
        "augmentation_issue": augmentation_issue,
        "notes": notes,
    }


def analysis_review_row(
    row: dict[str, object],
    clean_examples: dict[str, dict[str, object]],
    noisy_examples: dict[str, dict[str, object]],
    outcome: str,
    include_noisy_error_type: bool,
) -> dict[str, object]:
    clean_example = clean_examples[str(row["base_id"])]
    noisy_example = noisy_examples[str(row["noisy_id"])]
    gold = noisy_example["ground_truth"]
    return {
        "base_id": row["base_id"],
        "noisy_id": row["noisy_id"],
        "category": row["category"],
        "dimension": row["dimension"],
        "outcome": outcome,
        "heuristic_error_type": (
            heuristic_error_type(gold, row["clean_prediction"], row["noisy_prediction"])
            if include_noisy_error_type
            else ""
        ),
        "clean_prompt": conversation_text(clean_example["question"]),
        "noisy_prompt": conversation_text(noisy_example["question"]),
        "gold": gold,
        "clean_prediction": row["clean_prediction"],
        "noisy_prediction": row["noisy_prediction"],
    }


def analyze_dimension(dimension: str) -> dict[str, object]:
    paired_path = REPO_ROOT / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_paired.jsonl"
    clean_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    noisy_path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"
    summary_path = REPO_ROOT / f"artifacts/analysis/{dimension}/summary.json"
    regressions_jsonl_path = REPO_ROOT / f"artifacts/analysis/{dimension}/regressions.jsonl"
    regressions_csv_path = REPO_ROOT / f"artifacts/analysis/{dimension}/regressions.csv"
    recoveries_jsonl_path = REPO_ROOT / f"artifacts/analysis/{dimension}/recoveries.jsonl"

    for path in (paired_path, clean_path, noisy_path):
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(REPO_ROOT)}.")

    paired_rows = read_jsonl(paired_path)
    clean_examples = {row["id"]: row for row in read_jsonl(clean_path)}
    noisy_examples = {row["id"]: row for row in read_jsonl(noisy_path)}

    regressions = [
        analysis_review_row(
            row,
            clean_examples,
            noisy_examples,
            "clean_success_noisy_failure",
            include_noisy_error_type=True,
        )
        for row in paired_rows
        if row["clean_correct"] and not row["noisy_correct"]
    ]
    recoveries = [
        analysis_review_row(
            row,
            clean_examples,
            noisy_examples,
            "clean_failure_noisy_success",
            include_noisy_error_type=False,
        )
        for row in paired_rows
        if not row["clean_correct"] and row["noisy_correct"]
    ]
    stable_failures = [
        row for row in paired_rows if not row["clean_correct"] and not row["noisy_correct"]
    ]
    stable_successes = [row for row in paired_rows if row["clean_correct"] and row["noisy_correct"]]

    write_jsonl(regressions_jsonl_path, regressions)
    write_jsonl(recoveries_jsonl_path, recoveries)
    write_csv(
        regressions_csv_path,
        [
            {
                **row,
                "gold": json.dumps(row["gold"], sort_keys=True),
                "clean_prediction": json.dumps(row["clean_prediction"], sort_keys=True),
                "noisy_prediction": json.dumps(row["noisy_prediction"], sort_keys=True),
            }
            for row in regressions
        ],
        [
            "base_id",
            "noisy_id",
            "category",
            "dimension",
            "outcome",
            "heuristic_error_type",
            "clean_prompt",
            "noisy_prompt",
            "gold",
            "clean_prediction",
            "noisy_prediction",
        ],
    )
    write_json(
        summary_path,
        {
            "created_at": utc_now(),
            "stage": "analyze",
            "model": OPENAI_MODEL,
            "dimension": dimension,
            "total": len(paired_rows),
            "regressions": {
                "total": len(regressions),
                "by_category": count_by(regressions, "category"),
                "by_heuristic_error_type": count_by(regressions, "heuristic_error_type"),
                "jsonl": regressions_jsonl_path.relative_to(REPO_ROOT).as_posix(),
                "csv": regressions_csv_path.relative_to(REPO_ROOT).as_posix(),
            },
            "recoveries": {
                "total": len(recoveries),
                "by_category": count_by(recoveries, "category"),
                "jsonl": recoveries_jsonl_path.relative_to(REPO_ROOT).as_posix(),
            },
            "stable_successes": len(stable_successes),
            "stable_failures": len(stable_failures),
            "taxonomy_note": (
                "heuristic_error_type is a triage label for manual review, not final taxonomy."
            ),
        },
    )
    print(f"Wrote {summary_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {regressions_jsonl_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {regressions_csv_path.relative_to(REPO_ROOT)}")
    return {
        "dimension": dimension,
        "summary": summary_path.relative_to(REPO_ROOT).as_posix(),
        "regression_total": len(regressions),
        "recovery_total": len(recoveries),
    }


def flip_review_rows(dimension: str) -> list[dict[str, object]]:
    paired_path = REPO_ROOT / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_paired.jsonl"
    clean_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    noisy_path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"

    paired_rows = read_jsonl(paired_path)
    clean_examples = {row["id"]: row for row in read_jsonl(clean_path)}
    noisy_examples = {row["id"]: row for row in read_jsonl(noisy_path)}
    review_rows = []
    for row in paired_rows:
        if row["clean_correct"] and row["noisy_correct"]:
            continue
        if row["clean_correct"] and not row["noisy_correct"]:
            outcome = "clean_success_noisy_failure"
            include_noisy_error_type = True
        elif not row["clean_correct"] and row["noisy_correct"]:
            outcome = "clean_failure_noisy_success"
            include_noisy_error_type = False
        else:
            outcome = "both_wrong"
            include_noisy_error_type = True
        review = analysis_review_row(
            row,
            clean_examples,
            noisy_examples,
            outcome,
            include_noisy_error_type=include_noisy_error_type,
        )
        review_rows.append(
            {
                **review,
                "clean_correct": row["clean_correct"],
                "noisy_correct": row["noisy_correct"],
                "gold": json.dumps(review["gold"], sort_keys=True),
                "clean_prediction": json.dumps(review["clean_prediction"], sort_keys=True),
                "noisy_prediction": json.dumps(review["noisy_prediction"], sort_keys=True),
            }
        )
    return review_rows


def regression_review_rows(flip_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in flip_rows:
        if row["outcome"] != "clean_success_noisy_failure":
            continue
        label = regression_label(row)
        rows.append(
            {
                **label,
                "base_id": row["base_id"],
                "noisy_id": row["noisy_id"],
                "category": row["category"],
                "dimension": row["dimension"],
                "heuristic_error_type": row["heuristic_error_type"],
                "clean_prompt": row["clean_prompt"],
                "noisy_prompt": row["noisy_prompt"],
                "gold": row["gold"],
                "clean_prediction": row["clean_prediction"],
                "noisy_prediction": row["noisy_prediction"],
            }
        )
    return rows


def benchmark_summary_rows(
    dimensions: list[str],
    regression_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for dimension in dimensions:
        summary_path = (
            REPO_ROOT / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_summary.json"
        )
        paired_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = paired_summary["metrics"]
        dimension_regressions = [row for row in regression_rows if row["dimension"] == dimension]
        possible_oracle_issues = [
            row for row in dimension_regressions if row["oracle_issue"] == "possible"
        ]
        possible_augmentation_issues = [
            row for row in dimension_regressions if row["augmentation_issue"] == "possible"
        ]
        adjusted_regression_count = int(metrics["clean_success_noisy_failure"]) - len(
            possible_oracle_issues
        )
        clean_correct = int(metrics["clean_correct"])
        rows.append(
            {
                "model": OPENAI_MODEL,
                "dimension": dimension,
                "total": metrics["total"],
                "clean_accuracy": metrics["clean_accuracy"],
                "noisy_accuracy": metrics["noisy_accuracy"],
                "absolute_degradation": metrics["absolute_degradation"],
                "clean_success_noisy_failure": metrics["clean_success_noisy_failure"],
                "conditional_failure_given_clean_success": metrics[
                    "conditional_failure_given_clean_success"
                ],
                "raw_regression_count": metrics["clean_success_noisy_failure"],
                "possible_oracle_issue_regressions": len(possible_oracle_issues),
                "possible_augmentation_issue_regressions": len(possible_augmentation_issues),
                "adjusted_regression_count": adjusted_regression_count,
                "adjusted_regression_rate_given_clean_success": (
                    adjusted_regression_count / clean_correct if clean_correct else 0.0
                ),
                "regressions_by_category": json.dumps(
                    count_by(dimension_regressions, "category"),
                    sort_keys=True,
                ),
                "regressions_by_manual_error_type": json.dumps(
                    count_by(dimension_regressions, "manual_error_type"),
                    sort_keys=True,
                ),
            }
        )
    return rows


def analyze() -> None:
    dimensions = [
        dimension
        for dimension in generated_dimensions()
        if (
            REPO_ROOT / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_paired.jsonl"
        ).exists()
    ]
    if not dimensions:
        raise SystemExit("No paired results found. Run run-bfcl first.")
    summaries = [analyze_dimension(dimension) for dimension in dimensions]
    flip_rows = []
    for dimension in dimensions:
        flip_rows.extend(flip_review_rows(dimension))
    flip_review_path = REPO_ROOT / "artifacts/analysis/flip_review.csv"
    regression_review_path = REPO_ROOT / "artifacts/analysis/regression_review.csv"
    benchmark_summary_csv_path = REPO_ROOT / "artifacts/analysis/benchmark_summary.csv"
    benchmark_summary_json_path = REPO_ROOT / "artifacts/analysis/benchmark_summary.json"
    write_csv(
        flip_review_path,
        flip_rows,
        [
            "base_id",
            "noisy_id",
            "category",
            "dimension",
            "outcome",
            "heuristic_error_type",
            "clean_correct",
            "noisy_correct",
            "clean_prompt",
            "noisy_prompt",
            "gold",
            "clean_prediction",
            "noisy_prediction",
        ],
    )
    regression_rows = regression_review_rows(flip_rows)
    write_csv(
        regression_review_path,
        regression_rows,
        [
            "review_status",
            "manual_error_type",
            "oracle_issue",
            "augmentation_issue",
            "notes",
            "base_id",
            "noisy_id",
            "category",
            "dimension",
            "heuristic_error_type",
            "clean_prompt",
            "noisy_prompt",
            "gold",
            "clean_prediction",
            "noisy_prediction",
        ],
    )
    benchmark_rows = benchmark_summary_rows(dimensions, regression_rows)
    benchmark_fieldnames = [
        "model",
        "dimension",
        "total",
        "clean_accuracy",
        "noisy_accuracy",
        "absolute_degradation",
        "clean_success_noisy_failure",
        "conditional_failure_given_clean_success",
        "raw_regression_count",
        "possible_oracle_issue_regressions",
        "possible_augmentation_issue_regressions",
        "adjusted_regression_count",
        "adjusted_regression_rate_given_clean_success",
        "regressions_by_category",
        "regressions_by_manual_error_type",
    ]
    write_csv(benchmark_summary_csv_path, benchmark_rows, benchmark_fieldnames)
    write_json(
        benchmark_summary_json_path,
        {
            "created_at": utc_now(),
            "stage": "analyze",
            "model": OPENAI_MODEL,
            "adjusted_metric_rule": (
                "adjusted_regression_count excludes only rows with " "oracle_issue=possible"
            ),
            "dimensions": benchmark_rows,
        },
    )
    write_json(
        REPO_ROOT / "artifacts/analysis/summary.json",
        {
            "created_at": utc_now(),
            "stage": "analyze",
            "model": OPENAI_MODEL,
            "dimensions": summaries,
            "benchmark_summary_csv": benchmark_summary_csv_path.relative_to(REPO_ROOT).as_posix(),
            "benchmark_summary_json": benchmark_summary_json_path.relative_to(REPO_ROOT).as_posix(),
            "flip_review_csv": flip_review_path.relative_to(REPO_ROOT).as_posix(),
            "regression_review_csv": regression_review_path.relative_to(REPO_ROOT).as_posix(),
        },
    )
    print(f"Wrote {benchmark_summary_csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {benchmark_summary_json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {flip_review_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {regression_review_path.relative_to(REPO_ROOT)}")
