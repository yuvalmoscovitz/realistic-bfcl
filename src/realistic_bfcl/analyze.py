from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import defaultdict

from .common import (
    DIMENSION_FILES,
    REPO_ROOT,
    article_primary_model,
    compact_text,
    configured_model_runs,
    conversation_text,
    read_jsonl,
    utc_now,
    write_csv,
    write_json,
    write_jsonl,
)
from .evaluate import generated_dimensions

ARTICLE_PRIMARY_MODEL = article_primary_model()

ARTICLE_DIMENSIONS = {
    "typos",
    "cursing",
    "irrelevant_context",
    "removed_spaces",
    "argumentative_challenge",
    "pasted_context_block",
    "telegraphic_request",
}


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
    return argument_value_issue_kind(gold, noisy_prediction) == "oracle_alias_or_format"


def json_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def accepted_options(value: object) -> list[object]:
    return value if isinstance(value, list) else [value]


def value_matches_option(value: object, option: object) -> bool:
    if value == option:
        return True
    if is_number(value) and is_number(option):
        return float(value) == float(option)
    if isinstance(value, str) and isinstance(option, str):
        return compact_text(value) == compact_text(option)
    if isinstance(value, list) and isinstance(option, list):
        return json_key(value) == json_key(option) or compact_text(value) == compact_text(option)
    return False


def value_matches_any_option(value: object, options: list[object]) -> bool:
    return any(value_matches_option(value, option) for option in options)


def value_has_numeric_mismatch(value: object, options: list[object]) -> bool:
    predicted_numbers = [item for item in flat_values(value) if is_number(item)]
    accepted_numbers = [
        item for option in options for item in flat_values(option) if is_number(item)
    ]
    if not predicted_numbers or not accepted_numbers:
        return False
    return not all(
        any(float(predicted) == float(accepted) for accepted in accepted_numbers)
        for predicted in predicted_numbers
    )


def percent_scale_ambiguity(value: object, options: list[object]) -> bool:
    predicted_numbers = [item for item in flat_values(value) if is_number(item)]
    accepted_numbers = [
        item for option in options for item in flat_values(option) if is_number(item)
    ]
    if not predicted_numbers or not accepted_numbers:
        return False
    return all(
        any(float(predicted) == float(accepted) * 100 for accepted in accepted_numbers)
        for predicted in predicted_numbers
    )


def alias_like_value(value: object, options: list[object]) -> bool:
    compact_value = compact_text(value)
    if not compact_value:
        return False
    for option in options:
        compact_option = compact_text(option)
        if (
            compact_option
            and compact_option != compact_value
            and (compact_option in compact_value or compact_value in compact_option)
        ):
            return True
    return False


def list_items_alias_like(value: object, options: list[object]) -> bool:
    if not isinstance(value, list):
        return False
    accepted_items = [item for option in options for item in flat_values(option)]
    if not accepted_items:
        return False
    return all(
        value_matches_any_option(item, accepted_items) or alias_like_value(item, accepted_items)
        for item in value
    )


def argument_value_issue_kind(gold: object, noisy_prediction: object) -> str:
    """Classify only value-level differences after call names/counts already match."""
    gold_args = gold_arguments(gold)
    predicted_args = call_arguments(noisy_prediction)
    if len(gold_args) != len(predicted_args):
        return "real"

    saw_alias_or_format = False
    for expected, predicted in zip(gold_args, predicted_args):
        for key, accepted in expected.items():
            if key not in predicted:
                return "real"
            predicted_value = predicted[key]
            options = accepted_options(accepted)
            if value_matches_any_option(predicted_value, options):
                continue
            if value_has_numeric_mismatch(predicted_value, options):
                return "real"
            if isinstance(predicted_value, bool) or any(
                isinstance(option, bool) for option in options
            ):
                return "real"
            if alias_like_value(predicted_value, options) or list_items_alias_like(
                predicted_value, options
            ):
                saw_alias_or_format = True
                continue
            return "real"

    return "oracle_alias_or_format" if saw_alias_or_format else "real"


def has_baseline_dataset_ambiguity(
    clean_prompt: str, noisy_prompt: str, gold: object, noisy_prediction: object
) -> bool:
    """Identify BFCL prompt/schema cases where the oracle requires unstated normalization."""
    if "%" not in clean_prompt and "%" not in noisy_prompt:
        return False

    gold_args = gold_arguments(gold)
    predicted_args = call_arguments(noisy_prediction)
    if len(gold_args) != len(predicted_args):
        return False

    saw_percent_scale_issue = False
    for expected, predicted in zip(gold_args, predicted_args):
        for key, accepted in expected.items():
            if key not in predicted:
                return False
            predicted_value = predicted[key]
            options = accepted_options(accepted)
            if value_matches_any_option(predicted_value, options):
                continue
            if percent_scale_ambiguity(predicted_value, options):
                saw_percent_scale_issue = True
                continue
            if alias_like_value(predicted_value, options) or list_items_alias_like(
                predicted_value, options
            ):
                continue
            return False
    return saw_percent_scale_issue


def augmentation_text_copied_into_argument(
    clean_prompt: str, noisy_prompt: str, noisy_prediction: object
) -> bool:
    clean_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", clean_prompt)}
    noisy_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", noisy_prompt)}
    introduced_tokens = noisy_tokens - clean_tokens
    argument_text = json.dumps(call_arguments(noisy_prediction)).lower()
    return any(len(token) >= 5 and token in argument_text for token in introduced_tokens)


def regression_label(review: dict[str, object]) -> dict[str, str]:
    gold = parsed_json_value(review["gold"])
    noisy_prediction = parsed_json_value(review["noisy_prediction"])
    heuristic = str(review["heuristic_error_type"])
    noisy_names = call_names(noisy_prediction)
    expected_names = gold_call_names(gold)
    oracle_issue = "no"
    augmentation_issue = "no"
    baseline_dataset_issue = "no"

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
        if str(review["dimension"]) == "removed_spaces" and augmentation_text_copied_into_argument(
            str(review["clean_prompt"]),
            str(review["noisy_prompt"]),
            noisy_prediction,
        ):
            manual_error_type = "augmentation_text_copied_into_argument_value"
            augmentation_issue = "possible"
            notes = "The removed-space artifact appears to have been copied into an argument value."
        elif str(review["dimension"]) == "typos" and typo_copied_into_argument(
            str(review["clean_prompt"]),
            str(review["noisy_prompt"]),
            noisy_prediction,
        ):
            manual_error_type = "typo_copied_into_argument_value"
            augmentation_issue = "possible"
            notes = "The typo appears to have been copied into an argument value."
        elif has_baseline_dataset_ambiguity(
            str(review["clean_prompt"]),
            str(review["noisy_prompt"]),
            gold,
            noisy_prediction,
        ):
            manual_error_type = "baseline_dataset_ambiguity"
            baseline_dataset_issue = "possible"
            notes = (
                "The BFCL prompt and schema appear ambiguous relative to the gold "
                "argument convention."
            )
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
        "baseline_dataset_issue": baseline_dataset_issue,
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
    paired_path = REPO_ROOT / (
        f"artifacts/results/paired/{dimension}/{ARTICLE_PRIMARY_MODEL.filename}_paired.jsonl"
    )
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
            "model": ARTICLE_PRIMARY_MODEL.id,
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
    paired_path = REPO_ROOT / (
        f"artifacts/results/paired/{dimension}/{ARTICLE_PRIMARY_MODEL.filename}_paired.jsonl"
    )
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


def strong_failure_reason(row: dict[str, object]) -> str:
    error_type = str(row["manual_error_type"])
    if error_type == "missing_tool_call":
        return (
            "Clean succeeds, but the noisy prompt causes one or more required "
            "tool calls to be dropped."
        )
    if error_type == "wrong_tool_routing":
        return "Clean succeeds, but the noisy prompt routes to a different function."
    if error_type == "extra_tool_call":
        return "Clean succeeds, but the noisy prompt causes an unnecessary extra tool call."
    if error_type == "wrong_argument_value":
        return "Clean succeeds, but the noisy prompt changes a required argument value."
    return (
        "Clean succeeds and noisy fails without an identified oracle, "
        "augmentation, or baseline ambiguity issue."
    )


def strong_failure_examples(regression_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    priority = {
        "missing_tool_call": 0,
        "wrong_tool_routing": 1,
        "extra_tool_call": 2,
        "wrong_argument_value": 3,
    }
    candidates = [
        row
        for row in regression_rows
        if row["oracle_issue"] == "no"
        and row["augmentation_issue"] == "no"
        and row["baseline_dataset_issue"] == "no"
        and row["manual_error_type"] in priority
    ]
    candidates.sort(
        key=lambda row: (
            priority[str(row["manual_error_type"])],
            str(row["dimension"]),
            str(row["category"]),
            str(row["base_id"]),
        )
    )

    selected = []
    seen_keys: set[tuple[str, str]] = set()
    for row in candidates:
        key = (str(row["dimension"]), str(row["manual_error_type"]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(row)

    selected_ids = {str(row["noisy_id"]) for row in selected}
    dimensions = sorted({str(row["dimension"]) for row in candidates})
    while len(selected) < 30:
        added = False
        for dimension in dimensions:
            for row in candidates:
                if str(row["dimension"]) != dimension:
                    continue
                if str(row["noisy_id"]) in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(str(row["noisy_id"]))
                added = True
                break
            if len(selected) >= 30:
                break
        if not added:
            break

    return [
        {
            "rank": index,
            "evidence_reason": strong_failure_reason(row),
            "base_id": row["base_id"],
            "noisy_id": row["noisy_id"],
            "category": row["category"],
            "dimension": row["dimension"],
            "manual_error_type": row["manual_error_type"],
            "clean_prompt": row["clean_prompt"],
            "noisy_prompt": row["noisy_prompt"],
            "gold": row["gold"],
            "clean_prediction": row["clean_prediction"],
            "noisy_prediction": row["noisy_prediction"],
        }
        for index, row in enumerate(selected, start=1)
    ]


def strong_failure_example_row(row: dict[str, object], rank: int) -> dict[str, object]:
    return {
        "rank": rank,
        "evidence_reason": strong_failure_reason(row),
        "base_id": row["base_id"],
        "noisy_id": row["noisy_id"],
        "category": row["category"],
        "dimension": row["dimension"],
        "manual_error_type": row["manual_error_type"],
        "clean_prompt": row["clean_prompt"],
        "noisy_prompt": row["noisy_prompt"],
        "gold": row["gold"],
        "clean_prediction": row["clean_prediction"],
        "noisy_prediction": row["noisy_prediction"],
    }


def article_review_labels() -> dict[str, dict[str, str]]:
    path = REPO_ROOT / "configs/article_failure_review_labels.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["noisy_id"]: {
                "article_include": row["article_include"],
                "human_judgment": row["human_judgment"],
                "short_explanation": row["short_explanation"],
            }
            for row in csv.DictReader(handle)
        }


def article_failure_review_rows(
    strong_examples: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    labels = article_review_labels()
    rows_by_id = {str(row["noisy_id"]): row for row in strong_examples}
    next_rank = len(strong_examples) + 1
    for row in regression_rows:
        noisy_id = str(row["noisy_id"])
        if noisy_id in rows_by_id:
            continue
        if labels.get(noisy_id, {}).get("article_include") != "yes":
            continue
        rows_by_id[noisy_id] = strong_failure_example_row(row, next_rank)
        next_rank += 1

    rows = []
    for row in sorted(rows_by_id.values(), key=lambda item: int(item["rank"])):
        label = labels.get(
            str(row["noisy_id"]),
            {
                "article_include": "no",
                "human_judgment": "needs_review",
                "short_explanation": "Not reviewed yet.",
            },
        )
        rows.append({**label, **row})
    return rows


def benchmark_summary_rows(
    dimensions: list[str],
    regression_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for dimension in dimensions:
        summary_path = (
            REPO_ROOT
            / (
                f"artifacts/results/paired/{dimension}/"
                f"{ARTICLE_PRIMARY_MODEL.filename}_summary.json"
            )
        )
        paired_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = paired_summary["metrics"]
        clean_success_noisy_failure = int(metrics["clean_success_noisy_failure"])
        clean_failure_noisy_success = int(metrics["clean_failure_noisy_success"])
        dimension_regressions = [row for row in regression_rows if row["dimension"] == dimension]
        possible_oracle_issues = [
            row for row in dimension_regressions if row["oracle_issue"] == "possible"
        ]
        possible_augmentation_issues = [
            row for row in dimension_regressions if row["augmentation_issue"] == "possible"
        ]
        possible_baseline_dataset_issues = [
            row
            for row in dimension_regressions
            if row["baseline_dataset_issue"] == "possible"
        ]
        adjusted_regression_count = clean_success_noisy_failure - len(
            possible_oracle_issues
        )
        real_model_regression_count = adjusted_regression_count - len(
            possible_augmentation_issues
        ) - len(
            possible_baseline_dataset_issues
        )
        clean_correct = int(metrics["clean_correct"])
        rows.append(
            {
                "model": ARTICLE_PRIMARY_MODEL.id,
                "dimension": dimension,
                "total": metrics["total"],
                "clean_accuracy": metrics["clean_accuracy"],
                "noisy_accuracy": metrics["noisy_accuracy"],
                "absolute_degradation": metrics["absolute_degradation"],
                "both_correct": metrics["both_correct"],
                "both_wrong": metrics["both_wrong"],
                "clean_success_noisy_failure": clean_success_noisy_failure,
                "clean_failure_noisy_success": clean_failure_noisy_success,
                "net_degradation_count": (
                    clean_success_noisy_failure - clean_failure_noisy_success
                ),
                "mcnemar_exact_p_value": mcnemar_exact_p_value(
                    clean_success_noisy_failure,
                    clean_failure_noisy_success,
                ),
                "conditional_failure_given_clean_success": metrics[
                    "conditional_failure_given_clean_success"
                ],
                "raw_regression_count": clean_success_noisy_failure,
                "possible_oracle_issue_regressions": len(possible_oracle_issues),
                "possible_augmentation_issue_regressions": len(possible_augmentation_issues),
                "possible_baseline_dataset_issue_regressions": len(
                    possible_baseline_dataset_issues
                ),
                "adjusted_regression_count": adjusted_regression_count,
                "adjusted_regression_rate_given_clean_success": (
                    adjusted_regression_count / clean_correct if clean_correct else 0.0
                ),
                "real_model_regression_count": real_model_regression_count,
                "real_model_regression_rate_given_clean_success": (
                    real_model_regression_count / clean_correct if clean_correct else 0.0
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


def model_comparison_rows(dimensions: list[str]) -> list[dict[str, object]]:
    labels = article_review_labels()
    clean_examples = {
        row["id"]: row
        for row in read_jsonl(REPO_ROOT / "artifacts/frozen/clean_subset.jsonl")
    }
    rows = []
    for model in configured_model_runs():
        for dimension in dimensions:
            summary_path = (
                REPO_ROOT / f"artifacts/results/paired/{dimension}/{model.filename}_summary.json"
            )
            paired_path = (
                REPO_ROOT / f"artifacts/results/paired/{dimension}/{model.filename}_paired.jsonl"
            )
            if not summary_path.exists() or not paired_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metrics = summary["metrics"]
            regression_rows = [
                row
                for row in read_jsonl(paired_path)
                if row["clean_correct"] and not row["noisy_correct"]
            ]
            taxonomy: dict[str, int] = {}
            reviewed_taxonomy: dict[str, int] = {}
            reviewed_count = 0
            for row in regression_rows:
                clean_example = clean_examples[str(row["base_id"])]
                error_type = heuristic_error_type(
                    clean_example["ground_truth"],
                    row["clean_prediction"],
                    row["noisy_prediction"],
                )
                taxonomy[error_type] = taxonomy.get(error_type, 0) + 1
                if labels.get(str(row["noisy_id"]), {}).get("article_include") == "yes":
                    reviewed_count += 1
                    reviewed_taxonomy[error_type] = reviewed_taxonomy.get(error_type, 0) + 1
            rows.append(
                {
                    "model": model.id,
                    "provider": model.provider,
                    "tier": model.tier,
                    "temperature": model.temperature,
                    "dimension": dimension,
                    "total": metrics["total"],
                    "clean_accuracy": metrics["clean_accuracy"],
                    "noisy_accuracy": metrics["noisy_accuracy"],
                    "absolute_degradation": metrics["absolute_degradation"],
                    "clean_success_noisy_failure": metrics["clean_success_noisy_failure"],
                    "clean_failure_noisy_success": metrics["clean_failure_noisy_success"],
                    "reviewed_regressions": reviewed_count,
                    "unreviewed_regressions": int(metrics["clean_success_noisy_failure"])
                    - reviewed_count,
                    "failure_type_taxonomy": json.dumps(taxonomy, sort_keys=True),
                    "reviewed_failure_type_taxonomy": json.dumps(
                        reviewed_taxonomy, sort_keys=True
                    ),
                }
            )
    return rows


def mcnemar_exact_p_value(
    clean_success_noisy_failure: int,
    clean_failure_noisy_success: int,
) -> float:
    discordant = clean_success_noisy_failure + clean_failure_noisy_success
    if discordant == 0:
        return 1.0
    smaller = min(clean_success_noisy_failure, clean_failure_noisy_success)
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


def paired_stats_rows(dimensions: list[str]) -> list[dict[str, object]]:
    rows = []
    for dimension in dimensions:
        summary_path = (
            REPO_ROOT
            / (
                f"artifacts/results/paired/{dimension}/"
                f"{ARTICLE_PRIMARY_MODEL.filename}_summary.json"
            )
        )
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))["metrics"]
        clean_success_noisy_failure = int(metrics["clean_success_noisy_failure"])
        clean_failure_noisy_success = int(metrics["clean_failure_noisy_success"])
        rows.append(
            {
                "model": ARTICLE_PRIMARY_MODEL.id,
                "dimension": dimension,
                "total": metrics["total"],
                "both_correct": metrics["both_correct"],
                "both_wrong": metrics["both_wrong"],
                "clean_success_noisy_failure": clean_success_noisy_failure,
                "clean_failure_noisy_success": clean_failure_noisy_success,
                "net_degradation_count": (
                    clean_success_noisy_failure - clean_failure_noisy_success
                ),
                "clean_accuracy": metrics["clean_accuracy"],
                "noisy_accuracy": metrics["noisy_accuracy"],
                "absolute_degradation": metrics["absolute_degradation"],
                "mcnemar_exact_p_value": mcnemar_exact_p_value(
                    clean_success_noisy_failure,
                    clean_failure_noisy_success,
                ),
            }
        )
    return add_multiple_comparison_adjustments(rows)


def add_multiple_comparison_adjustments(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not rows:
        return rows

    total = len(rows)
    for row in rows:
        p_value = float(row["mcnemar_exact_p_value"])
        row["mcnemar_bonferroni_p_value"] = min(1.0, p_value * total)

    sorted_rows = sorted(rows, key=lambda row: float(row["mcnemar_exact_p_value"]))
    running_q = 1.0
    for rank, row in reversed(list(enumerate(sorted_rows, start=1))):
        p_value = float(row["mcnemar_exact_p_value"])
        running_q = min(running_q, p_value * total / rank)
        row["benjamini_hochberg_q_value"] = min(1.0, running_q)

    return rows


def paired_summary_path(dimension: str, suffix: str) -> object:
    result_dir = REPO_ROOT / f"artifacts/results/paired/{dimension}"
    if suffix:
        result_dir = result_dir / suffix
    return result_dir / f"{ARTICLE_PRIMARY_MODEL.filename}_summary.json"


def complete_repeat_runs(dimensions: list[str]) -> list[tuple[str, str]]:
    runs = [("run_1", "")]
    clean_results_dir = REPO_ROOT / "artifacts/results/clean"
    if not clean_results_dir.exists():
        return runs

    for path in sorted(clean_results_dir.iterdir()):
        if not path.is_dir():
            continue
        suffix = path.name
        if not (path / f"{ARTICLE_PRIMARY_MODEL.filename}_predictions.jsonl").exists():
            continue
        if all(paired_summary_path(dimension, suffix).exists() for dimension in dimensions):
            runs.append((suffix, suffix))
    return runs


def stability_repeat_run_rows(dimensions: list[str]) -> list[dict[str, object]]:
    rows = []
    for run_name, suffix in complete_repeat_runs(dimensions):
        for dimension in dimensions:
            summary = json.loads(
                paired_summary_path(dimension, suffix).read_text(encoding="utf-8")
            )
            metrics = summary["metrics"]
            rows.append(
                {
                    "run": run_name,
                    "dimension": dimension,
                    "total": metrics["total"],
                    "clean_accuracy": metrics["clean_accuracy"],
                    "noisy_accuracy": metrics["noisy_accuracy"],
                    "absolute_degradation": metrics["absolute_degradation"],
                    "clean_success_noisy_failure": metrics[
                        "clean_success_noisy_failure"
                    ],
                    "clean_failure_noisy_success": metrics[
                        "clean_failure_noisy_success"
                    ],
                    "both_correct": metrics["both_correct"],
                    "both_wrong": metrics["both_wrong"],
                }
            )
    return rows


def stability_repeat_summary_rows(
    run_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summary_rows = []
    dimensions = sorted({str(row["dimension"]) for row in run_rows})
    for dimension in dimensions:
        dimension_rows = [row for row in run_rows if row["dimension"] == dimension]
        if len(dimension_rows) < 2:
            continue
        drops = [float(row["absolute_degradation"]) for row in dimension_rows]
        clean_accuracies = [float(row["clean_accuracy"]) for row in dimension_rows]
        noisy_accuracies = [float(row["noisy_accuracy"]) for row in dimension_rows]
        regressions = [
            int(row["clean_success_noisy_failure"]) for row in dimension_rows
        ]
        recoveries = [
            int(row["clean_failure_noisy_success"]) for row in dimension_rows
        ]
        summary_rows.append(
            {
                "dimension": dimension,
                "runs": len(dimension_rows),
                "mean_clean_accuracy": statistics.mean(clean_accuracies),
                "min_clean_accuracy": min(clean_accuracies),
                "max_clean_accuracy": max(clean_accuracies),
                "mean_noisy_accuracy": statistics.mean(noisy_accuracies),
                "min_noisy_accuracy": min(noisy_accuracies),
                "max_noisy_accuracy": max(noisy_accuracies),
                "mean_absolute_degradation": statistics.mean(drops),
                "min_absolute_degradation": min(drops),
                "max_absolute_degradation": max(drops),
                "drop_range": max(drops) - min(drops),
                "drop_sample_stdev": statistics.stdev(drops),
                "mean_clean_success_noisy_failure": statistics.mean(regressions),
                "min_clean_success_noisy_failure": min(regressions),
                "max_clean_success_noisy_failure": max(regressions),
                "mean_clean_failure_noisy_success": statistics.mean(recoveries),
                "min_clean_failure_noisy_success": min(recoveries),
                "max_clean_failure_noisy_success": max(recoveries),
            }
        )
    return summary_rows


def review_filtering_rows(
    benchmark_rows: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
    labels: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    review_excluded_counts: dict[str, int] = {}
    for row in regression_rows:
        if is_likely_real_regression(row) and manually_excluded_from_article(row, labels):
            dimension = str(row["dimension"])
            review_excluded_counts[dimension] = review_excluded_counts.get(dimension, 0) + 1

    rows = []
    for row in benchmark_rows:
        dimension = str(row["dimension"])
        automatic_count = int(row["real_model_regression_count"])
        review_excluded_count = review_excluded_counts.get(dimension, 0)
        rows.append(
            {
                "dimension": dimension,
                "raw_regression_count": row["raw_regression_count"],
                "possible_oracle_issue_regressions": row[
                    "possible_oracle_issue_regressions"
                ],
                "possible_augmentation_issue_regressions": row[
                    "possible_augmentation_issue_regressions"
                ],
                "possible_baseline_dataset_issue_regressions": row[
                    "possible_baseline_dataset_issue_regressions"
                ],
                "automatic_real_model_regression_count": row[
                    "real_model_regression_count"
                ],
                "review_excluded_regression_count": review_excluded_count,
                "article_regression_count": automatic_count - review_excluded_count,
            }
        )
    return rows


def is_likely_real_regression(row: dict[str, object]) -> bool:
    return (
        row["oracle_issue"] != "possible"
        and row["augmentation_issue"] != "possible"
        and row["baseline_dataset_issue"] != "possible"
    )


def manually_excluded_from_article(
    row: dict[str, object], labels: dict[str, dict[str, str]]
) -> bool:
    judgment = labels.get(str(row["noisy_id"]), {}).get("human_judgment", "")
    return judgment in {"artifact", "questionable"}


def is_article_regression(
    row: dict[str, object], labels: dict[str, dict[str, str]]
) -> bool:
    return is_likely_real_regression(row) and not manually_excluded_from_article(row, labels)


def article_dimension_rows(
    rows: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
    labels: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    review_excluded_counts: dict[str, int] = {}
    for row in regression_rows:
        if is_likely_real_regression(row) and manually_excluded_from_article(row, labels):
            dimension = str(row["dimension"])
            review_excluded_counts[dimension] = review_excluded_counts.get(dimension, 0) + 1

    article_rows = []
    for row in rows:
        automatic_count = int(row["real_model_regression_count"])
        review_excluded_count = review_excluded_counts.get(str(row["dimension"]), 0)
        article_count = automatic_count - review_excluded_count
        automatic_rate = float(row["real_model_regression_rate_given_clean_success"])
        clean_successes = automatic_count / automatic_rate if automatic_rate else 0
        article_rows.append(
            {
                "dimension": row["dimension"],
                "total": row["total"],
                "clean_accuracy": row["clean_accuracy"],
                "noisy_accuracy": row["noisy_accuracy"],
                "absolute_degradation": row["absolute_degradation"],
                "raw_regression_count": row["raw_regression_count"],
                "possible_oracle_issue_regressions": row[
                    "possible_oracle_issue_regressions"
                ],
                "automatic_real_model_regression_count": automatic_count,
                "review_excluded_regression_count": review_excluded_count,
                "article_regression_count": article_count,
                "article_regression_rate_given_clean_success": (
                    article_count / clean_successes if clean_successes else 0.0
                ),
            }
        )
    return sorted(
        article_rows,
        key=lambda row: (
            -float(row["article_regression_rate_given_clean_success"]),
            -float(row["absolute_degradation"]),
        ),
    )


def grouped_article_count_rows(
    regression_rows: list[dict[str, object]],
    group_key: str,
    labels: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], int] = {}
    for row in regression_rows:
        if not is_article_regression(row, labels):
            continue
        key = (str(row["dimension"]), str(row[group_key]))
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "dimension": dimension,
            group_key: value,
            "article_regression_count": count,
        }
        for (dimension, value), count in sorted(
            counts.items(),
            key=lambda item: (item[0][0], -item[1], item[0][1]),
        )
    ]


def overall_article_count_rows(
    regression_rows: list[dict[str, object]],
    group_key: str,
    labels: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for row in regression_rows:
        if not is_article_regression(row, labels):
            continue
        value = str(row[group_key])
        counts[value] = counts.get(value, 0) + 1
    return [
        {
            group_key: value,
            "article_regression_count": count,
        }
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def realism_reviewer_notes(dimension: str) -> str:
    notes = {
        "argumentative_challenge": (
            "Accepted examples model distrust or impatience while keeping the final "
            "request; questionable rows are excluded."
        ),
        "cursing": (
            "Accepted examples model frustrated user register; profanity is a surface "
            "marker, not the claim itself."
        ),
        "irrelevant_context": (
            "Accepted examples add plausible side context without changing the task; "
            "questionable rows are excluded."
        ),
        "pasted_context_block": (
            "Accepted examples model copied surrounding context and instructions; this "
            "dimension needs broader human audit before a larger claim."
        ),
        "removed_spaces": (
            "Weakest realism dimension; spacing edits can create lexical artifacts, so "
            "unclear and artifact rows are excluded."
        ),
        "telegraphic_request": (
            "Accepted examples model terse user shorthand; this should be audited "
            "beyond failure candidates before scaling."
        ),
        "typos": (
            "Mixed: accepted examples preserve intent, but typo edits can collide with "
            "entities or argument strings; questionable rows are excluded."
        ),
    }
    return notes.get(dimension, "")


def realism_audit_summary_rows(
    article_review_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    by_dimension: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in article_review_rows:
        by_dimension[str(row["dimension"])].append(row)

    for dimension in sorted(by_dimension):
        candidates = by_dimension[dimension]
        strong_count = sum(
            1 for row in candidates if row["human_judgment"] == "strong_failure"
        )
        unclear_count = sum(
            1 for row in candidates if row["human_judgment"] == "questionable"
        )
        artifact_count = sum(1 for row in candidates if row["human_judgment"] == "artifact")
        included_count = sum(1 for row in candidates if row["article_include"] == "yes")
        sampled_count = len(candidates)

        rows.append(
            {
                "dimension": dimension,
                "sampled_failure_candidates": sampled_count,
                "oracle_preserved_yes": strong_count,
                "oracle_preserved_unclear": unclear_count,
                "oracle_preserved_no": artifact_count,
                "production_like_yes": strong_count,
                "production_like_unclear": unclear_count,
                "production_like_no": artifact_count,
                "natural_user_style_yes": strong_count,
                "natural_user_style_unclear": unclear_count,
                "natural_user_style_no": artifact_count,
                "active_constraints_unchanged_yes": strong_count,
                "active_constraints_unchanged_unclear": unclear_count,
                "active_constraints_unchanged_no": artifact_count,
                "included_rate": included_count / sampled_count if sampled_count else 0.0,
                "reviewer_notes": realism_reviewer_notes(dimension),
            }
        )
    return rows


def write_article_bundle(
    benchmark_rows: list[dict[str, object]],
    regression_rows: list[dict[str, object]],
    article_review_rows: list[dict[str, object]],
) -> None:
    article_dir = REPO_ROOT / "artifacts/analysis/article"
    article_dir.mkdir(parents=True, exist_ok=True)
    labels = article_review_labels()
    benchmark_rows = [
        row for row in benchmark_rows if str(row["dimension"]) in ARTICLE_DIMENSIONS
    ]
    regression_rows = [
        row for row in regression_rows if str(row["dimension"]) in ARTICLE_DIMENSIONS
    ]
    article_review_rows = [
        row for row in article_review_rows if str(row["dimension"]) in ARTICLE_DIMENSIONS
    ]

    article_dimensions = [str(row["dimension"]) for row in benchmark_rows]
    dimension_rows = article_dimension_rows(benchmark_rows, regression_rows, labels)
    paired_rows = paired_stats_rows(article_dimensions)
    filtering_rows = review_filtering_rows(benchmark_rows, regression_rows, labels)
    stability_run_rows = stability_repeat_run_rows(article_dimensions)
    stability_summary_rows = sorted(
        stability_repeat_summary_rows(stability_run_rows),
        key=lambda row: -float(row["mean_absolute_degradation"]),
    )
    dimension_fieldnames = [
        "dimension",
        "total",
        "clean_accuracy",
        "noisy_accuracy",
        "absolute_degradation",
        "raw_regression_count",
        "possible_oracle_issue_regressions",
        "automatic_real_model_regression_count",
        "review_excluded_regression_count",
        "article_regression_count",
        "article_regression_rate_given_clean_success",
    ]
    write_csv(article_dir / "dimension_results.csv", dimension_rows, dimension_fieldnames)
    write_csv(
        article_dir / "paired_stats.csv",
        paired_rows,
        [
            "model",
            "dimension",
            "total",
            "both_correct",
            "both_wrong",
            "clean_success_noisy_failure",
            "clean_failure_noisy_success",
            "net_degradation_count",
            "clean_accuracy",
            "noisy_accuracy",
            "absolute_degradation",
            "mcnemar_exact_p_value",
            "mcnemar_bonferroni_p_value",
            "benjamini_hochberg_q_value",
        ],
    )
    write_csv(
        article_dir / "review_filtering.csv",
        filtering_rows,
        [
            "dimension",
            "raw_regression_count",
            "possible_oracle_issue_regressions",
            "possible_augmentation_issue_regressions",
            "possible_baseline_dataset_issue_regressions",
            "automatic_real_model_regression_count",
            "review_excluded_regression_count",
            "article_regression_count",
        ],
    )
    stability_run_fieldnames = [
        "run",
        "dimension",
        "total",
        "clean_accuracy",
        "noisy_accuracy",
        "absolute_degradation",
        "clean_success_noisy_failure",
        "clean_failure_noisy_success",
        "both_correct",
        "both_wrong",
    ]
    stability_summary_fieldnames = [
        "dimension",
        "runs",
        "mean_clean_accuracy",
        "min_clean_accuracy",
        "max_clean_accuracy",
        "mean_noisy_accuracy",
        "min_noisy_accuracy",
        "max_noisy_accuracy",
        "mean_absolute_degradation",
        "min_absolute_degradation",
        "max_absolute_degradation",
        "drop_range",
        "drop_sample_stdev",
        "mean_clean_success_noisy_failure",
        "min_clean_success_noisy_failure",
        "max_clean_success_noisy_failure",
        "mean_clean_failure_noisy_success",
        "min_clean_failure_noisy_success",
        "max_clean_failure_noisy_success",
    ]
    if stability_summary_rows:
        write_csv(
            article_dir / "stability_repeat_runs.csv",
            stability_run_rows,
            stability_run_fieldnames,
        )
        write_csv(
            article_dir / "stability_repeat_summary.csv",
            stability_summary_rows,
            stability_summary_fieldnames,
        )
        write_json(
            article_dir / "stability_repeat_summary.json",
            {
                "model": ARTICLE_PRIMARY_MODEL.id,
                "runs": sorted({str(row["run"]) for row in stability_run_rows}),
                "dimensions": stability_summary_rows,
            },
        )

    error_type_rows = grouped_article_count_rows(
        regression_rows, "manual_error_type", labels
    )
    write_csv(
        article_dir / "error_type_counts.csv",
        error_type_rows,
        ["dimension", "manual_error_type", "article_regression_count"],
    )
    write_csv(
        article_dir / "overall_error_type_counts.csv",
        overall_article_count_rows(regression_rows, "manual_error_type", labels),
        ["manual_error_type", "article_regression_count"],
    )

    category_rows = grouped_article_count_rows(regression_rows, "category", labels)
    write_csv(
        article_dir / "category_counts.csv",
        category_rows,
        ["dimension", "category", "article_regression_count"],
    )
    write_csv(
        article_dir / "overall_category_counts.csv",
        overall_article_count_rows(regression_rows, "category", labels),
        ["category", "article_regression_count"],
    )

    oracle_rows = [row for row in regression_rows if row["oracle_issue"] == "possible"]
    write_csv(
        article_dir / "oracle_issue_examples.csv",
        oracle_rows,
        [
            "review_status",
            "manual_error_type",
            "oracle_issue",
            "augmentation_issue",
            "baseline_dataset_issue",
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

    article_failure_fieldnames = [
        "article_include",
        "human_judgment",
        "short_explanation",
        "rank",
        "evidence_reason",
        "base_id",
        "noisy_id",
        "category",
        "dimension",
        "manual_error_type",
        "clean_prompt",
        "noisy_prompt",
        "gold",
        "clean_prediction",
        "noisy_prediction",
    ]
    write_csv(
        article_dir / "included_failure_examples.csv",
        [row for row in article_review_rows if row["article_include"] == "yes"],
        article_failure_fieldnames,
    )
    write_csv(
        article_dir / "candidate_failure_examples.csv",
        article_review_rows,
        article_failure_fieldnames,
    )
    write_csv(
        article_dir / "realism_audit_summary.csv",
        realism_audit_summary_rows(article_review_rows),
        [
            "dimension",
            "sampled_failure_candidates",
            "oracle_preserved_yes",
            "oracle_preserved_unclear",
            "oracle_preserved_no",
            "production_like_yes",
            "production_like_unclear",
            "production_like_no",
            "natural_user_style_yes",
            "natural_user_style_unclear",
            "natural_user_style_no",
            "active_constraints_unchanged_yes",
            "active_constraints_unchanged_unclear",
            "active_constraints_unchanged_no",
            "included_rate",
            "reviewer_notes",
        ],
    )

    summary_lines = [
        "# Realistic-BFCL Article Data",
        "",
        "These files organize the full-pool gpt-5.4-nano evaluation for article writing.",
        (
            "Article counts exclude rows marked as possible oracle, augmentation, "
            "or baseline dataset issues, plus manually reviewed artifact/questionable rows."
        ),
        "",
        "## Dimension Results",
        "",
        "| Dimension | Clean acc. | Noisy acc. | Drop | Article regressions | Article rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in dimension_rows:
        summary_lines.append(
            (
                "| {dimension} | {clean:.3f} | {noisy:.3f} | {drop:.3f} "
                "| {count} | {rate:.3f} |"
            ).format(
                dimension=row["dimension"],
                clean=float(row["clean_accuracy"]),
                noisy=float(row["noisy_accuracy"]),
                drop=float(row["absolute_degradation"]),
                count=row["article_regression_count"],
                rate=float(row["article_regression_rate_given_clean_success"]),
            )
        )
    if stability_summary_rows:
        summary_lines.extend(
            [
                "",
                "## Repeat-Run Stability",
                "",
                (
                    "The evaluation was repeated three times with fresh clean and noisy "
                    "model calls. Every listed noise type degraded accuracy in every run."
                ),
                "",
                "| Dimension | Runs | Mean drop | Min drop | Max drop | Drop sd |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in stability_summary_rows:
            summary_lines.append(
                (
                    "| {dimension} | {runs} | {mean:.3f} | {min_drop:.3f} "
                    "| {max_drop:.3f} | {stdev:.3f} |"
                ).format(
                    dimension=row["dimension"],
                    runs=row["runs"],
                    mean=float(row["mean_absolute_degradation"]),
                    min_drop=float(row["min_absolute_degradation"]),
                    max_drop=float(row["max_absolute_degradation"]),
                    stdev=float(row["drop_sample_stdev"]),
                )
            )
    summary_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `dimension_results.csv`: article-ready per-dimension metrics.",
            (
                "- `paired_stats.csv`: full paired contingency counts, McNemar "
                "p-values, and multiple-comparison corrections."
            ),
            "- `review_filtering.csv`: raw-to-reviewed regression filtering counts.",
            (
                "- `stability_repeat_summary.csv`: mean/range across repeated "
                "fresh model runs."
            ),
            (
                "- `stability_repeat_runs.csv`: per-run paired metrics used by the "
                "stability summary."
            ),
            "- `stability_repeat_summary.json`: JSON form of the stability summary.",
            "- `error_type_counts.csv`: article regressions by manual error type.",
            "- `overall_error_type_counts.csv`: aggregate article regressions by error type.",
            "- `category_counts.csv`: article regressions by BFCL category.",
            "- `overall_category_counts.csv`: aggregate article regressions by BFCL category.",
            "- `candidate_failure_examples.csv`: strongest examples queued for human review.",
            "- `included_failure_examples.csv`: reviewed qualitative examples for the article.",
            (
                "- `realism_audit_summary.csv`: first-pass researcher audit of "
                "reviewed failure candidates."
            ),
            (
                "- `oracle_issue_examples.csv`: examples to exclude or discuss as "
                "evaluator/oracle ambiguity."
            ),
        ]
    )
    (article_dir / "README.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def analyze() -> None:
    dimensions = [
        dimension
        for dimension in generated_dimensions()
        if (
            REPO_ROOT
            / (
                f"artifacts/results/paired/{dimension}/"
                f"{ARTICLE_PRIMARY_MODEL.filename}_paired.jsonl"
            )
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
    strong_failure_examples_path = (
        REPO_ROOT / "artifacts/analysis/strong_failure_examples.csv"
    )
    article_failure_review_path = REPO_ROOT / "artifacts/analysis/article_failure_review.csv"
    article_failure_examples_path = (
        REPO_ROOT / "artifacts/analysis/article_failure_examples.csv"
    )
    benchmark_summary_csv_path = REPO_ROOT / "artifacts/analysis/benchmark_summary.csv"
    benchmark_summary_json_path = REPO_ROOT / "artifacts/analysis/benchmark_summary.json"
    model_comparison_path = REPO_ROOT / "artifacts/analysis/model_comparison.csv"
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
            "baseline_dataset_issue",
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
    strong_examples = strong_failure_examples(regression_rows)
    write_csv(
        strong_failure_examples_path,
        strong_examples,
        [
            "rank",
            "evidence_reason",
            "base_id",
            "noisy_id",
            "category",
            "dimension",
            "manual_error_type",
            "clean_prompt",
            "noisy_prompt",
            "gold",
            "clean_prediction",
            "noisy_prediction",
        ],
    )
    article_review_rows = article_failure_review_rows(strong_examples, regression_rows)
    article_failure_fieldnames = [
        "article_include",
        "human_judgment",
        "short_explanation",
        "rank",
        "evidence_reason",
        "base_id",
        "noisy_id",
        "category",
        "dimension",
        "manual_error_type",
        "clean_prompt",
        "noisy_prompt",
        "gold",
        "clean_prediction",
        "noisy_prediction",
    ]
    write_csv(article_failure_review_path, article_review_rows, article_failure_fieldnames)
    write_csv(
        article_failure_examples_path,
        [row for row in article_review_rows if row["article_include"] == "yes"],
        article_failure_fieldnames,
    )
    benchmark_rows = benchmark_summary_rows(dimensions, regression_rows)
    benchmark_fieldnames = [
        "model",
        "dimension",
        "total",
        "clean_accuracy",
        "noisy_accuracy",
        "absolute_degradation",
        "both_correct",
        "both_wrong",
        "clean_success_noisy_failure",
        "clean_failure_noisy_success",
        "net_degradation_count",
        "mcnemar_exact_p_value",
        "conditional_failure_given_clean_success",
        "raw_regression_count",
        "possible_oracle_issue_regressions",
        "possible_augmentation_issue_regressions",
        "possible_baseline_dataset_issue_regressions",
        "adjusted_regression_count",
        "adjusted_regression_rate_given_clean_success",
        "real_model_regression_count",
        "real_model_regression_rate_given_clean_success",
        "regressions_by_category",
        "regressions_by_manual_error_type",
    ]
    write_csv(benchmark_summary_csv_path, benchmark_rows, benchmark_fieldnames)
    model_comparison = model_comparison_rows(
        [dimension for dimension in dimensions if dimension in ARTICLE_DIMENSIONS]
    )
    write_csv(
        model_comparison_path,
        model_comparison,
        [
            "model",
            "provider",
            "tier",
            "temperature",
            "dimension",
            "total",
            "clean_accuracy",
            "noisy_accuracy",
            "absolute_degradation",
            "clean_success_noisy_failure",
            "clean_failure_noisy_success",
            "reviewed_regressions",
            "unreviewed_regressions",
            "failure_type_taxonomy",
            "reviewed_failure_type_taxonomy",
        ],
    )
    write_json(
        benchmark_summary_json_path,
        {
            "created_at": utc_now(),
            "stage": "analyze",
            "model": ARTICLE_PRIMARY_MODEL.id,
            "adjusted_metric_rule": (
                "adjusted_regression_count excludes only rows with " "oracle_issue=possible"
            ),
            "real_model_metric_rule": (
                "real_model_regression_count excludes rows with oracle_issue=possible "
                "or augmentation_issue=possible or baseline_dataset_issue=possible"
            ),
            "dimensions": benchmark_rows,
        },
    )
    write_json(
        REPO_ROOT / "artifacts/analysis/summary.json",
        {
            "created_at": utc_now(),
            "stage": "analyze",
            "model": ARTICLE_PRIMARY_MODEL.id,
            "dimensions": summaries,
            "benchmark_summary_csv": benchmark_summary_csv_path.relative_to(REPO_ROOT).as_posix(),
            "benchmark_summary_json": benchmark_summary_json_path.relative_to(REPO_ROOT).as_posix(),
            "model_comparison_csv": model_comparison_path.relative_to(REPO_ROOT).as_posix(),
            "flip_review_csv": flip_review_path.relative_to(REPO_ROOT).as_posix(),
            "regression_review_csv": regression_review_path.relative_to(REPO_ROOT).as_posix(),
            "strong_failure_examples_csv": strong_failure_examples_path.relative_to(
                REPO_ROOT
            ).as_posix(),
            "article_failure_review_csv": article_failure_review_path.relative_to(
                REPO_ROOT
            ).as_posix(),
            "article_failure_examples_csv": article_failure_examples_path.relative_to(
                REPO_ROOT
            ).as_posix(),
        },
    )
    write_article_bundle(benchmark_rows, regression_rows, article_review_rows)
    print(f"Wrote {benchmark_summary_csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {benchmark_summary_json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {model_comparison_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {flip_review_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {regression_review_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {strong_failure_examples_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {article_failure_review_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {article_failure_examples_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {(REPO_ROOT / 'artifacts/analysis/article').relative_to(REPO_ROOT)}")
