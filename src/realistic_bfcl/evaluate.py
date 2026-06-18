from __future__ import annotations

import concurrent.futures
import importlib
import json
import os
import re
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

from .common import (
    BFCL_CATEGORY_FILES,
    BFCL_COMMIT,
    BFCL_REPOSITORY,
    DIMENSION_FILES,
    OPENAI_MAX_ATTEMPTS,
    OPENAI_MODEL,
    OPENAI_RESPONSES_URL,
    REPO_ROOT,
    RETRYABLE_HTTP_STATUS,
    ROUTER_MAX_OUTPUT_TOKENS,
    ROUTER_MESSAGE_SERIALIZATION,
    ROUTER_SYSTEM_INSTRUCTION,
    ROUTER_TOOL_CHOICE,
    append_jsonl,
    file_sha256,
    openai_api_key,
    openai_concurrency,
    read_int_setting,
    read_jsonl,
    read_list_setting,
    reject_placeholders,
    stable_hash,
    utc_now,
    write_json,
    write_jsonl,
)


def bfcl_data_root() -> Path:
    root = os.environ.get("REALISTIC_BFCL_BFCL_ROOT")
    if root:
        return Path(root) / "berkeley-function-call-leaderboard/bfcl_eval/data"
    default_root = Path(
        "/tmp/gorilla-bfcl-inspect/berkeley-function-call-leaderboard/bfcl_eval/data"
    )
    if default_root.exists():
        return default_root
    raise SystemExit(
        "Set REALISTIC_BFCL_BFCL_ROOT to a checkout of "
        "https://github.com/ShishirPatil/gorilla at the pinned commit."
    )


def bfcl_eval_root() -> Path:
    root = os.environ.get("REALISTIC_BFCL_BFCL_ROOT")
    if root:
        return Path(root) / "berkeley-function-call-leaderboard"
    default_root = Path("/tmp/gorilla-bfcl-inspect/berkeley-function-call-leaderboard")
    if default_root.exists():
        return default_root
    raise SystemExit(
        "Set REALISTIC_BFCL_BFCL_ROOT to a checkout of "
        "https://github.com/ShishirPatil/gorilla at the pinned commit."
    )


def materialize_smoke_subset(subset_config: Path, manifest_path: Path) -> Path:
    categories = read_list_setting(subset_config, "bfcl_categories")
    max_examples = read_int_setting(subset_config, "max_examples")
    examples_per_category = read_int_setting(subset_config, "examples_per_category")
    data_root = bfcl_data_root()
    rows: list[dict[str, object]] = []

    for category in categories:
        question_file, answer_file = BFCL_CATEGORY_FILES[category]
        questions = read_jsonl(data_root / question_file)
        answers = {row["id"]: row for row in read_jsonl(data_root / answer_file)}
        category_count = 0
        for question in questions:
            answer = answers[question["id"]]
            rows.append(
                {
                    "id": question["id"],
                    "category": category,
                    "question": question["question"],
                    "function": question["function"],
                    "ground_truth": answer["ground_truth"],
                }
            )
            category_count += 1
            if category_count >= examples_per_category:
                break
            if len(rows) >= max_examples:
                break
        if len(rows) >= max_examples:
            break

    subset_path = manifest_path.parent / "clean_subset.jsonl"
    write_jsonl(subset_path, rows)
    return subset_path


def openai_type(type_name: str) -> str:
    return {"any": "string", "dict": "object", "float": "number", "tuple": "array"}.get(
        type_name, type_name
    )


def normalize_json_schema(value: object) -> object:
    if isinstance(value, dict):
        normalized = {}
        for key, child in value.items():
            if key == "type" and isinstance(child, str):
                normalized[key] = openai_type(child)
            else:
                normalized[key] = normalize_json_schema(child)
        return normalized
    if isinstance(value, list):
        return [normalize_json_schema(item) for item in value]
    return value


def safe_tool_name(name: str, index: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return f"{safe}_{index}"


def openai_tool(function_doc: dict[str, object], name: str) -> dict[str, object]:
    parameters = normalize_json_schema(function_doc["parameters"])
    return {
        "type": "function",
        "name": name,
        "description": function_doc.get("description", ""),
        "parameters": parameters,
    }


def bfcl_messages(example: dict[str, object]) -> list[dict[str, str]]:
    messages = example["question"][0]
    return [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        for message in messages
    ]


def openai_retry_json(payload: dict[str, object], api_key: str) -> dict[str, object]:
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in RETRYABLE_HTTP_STATUS and attempt < OPENAI_MAX_ATTEMPTS:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAI API request failed: HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            if attempt < OPENAI_MAX_ATTEMPTS:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"OpenAI API request failed: {error}") from error

    raise RuntimeError("OpenAI API request failed without returning a response.")


def call_openai_tool_router(example: dict[str, object], api_key: str) -> dict[str, object]:
    tools = []
    for index, function_doc in enumerate(example["function"]):
        tools.append(openai_tool(function_doc, safe_tool_name(str(function_doc["name"]), index)))
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_INSTRUCTION,
            },
            *bfcl_messages(example),
        ],
        "tools": tools,
        "tool_choice": ROUTER_TOOL_CHOICE,
        "max_output_tokens": ROUTER_MAX_OUTPUT_TOKENS,
    }
    return openai_retry_json(payload, api_key)


def tool_name_map(example: dict[str, object]) -> dict[str, str]:
    return {
        safe_tool_name(str(function_doc["name"]), index): str(function_doc["name"])
        for index, function_doc in enumerate(example["function"])
    }


def response_function_calls(
    response: dict[str, object], name_map: dict[str, str]
) -> list[dict[str, object]]:
    calls = []
    for item in response.get("output", []):
        if item.get("type") != "function_call":
            continue
        try:
            arguments = json.loads(item.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"__malformed_arguments__": item.get("arguments")}
        name = str(item.get("name"))
        calls.append({"name": name_map.get(name, name), "arguments": arguments})
    return calls


def load_bfcl_ast_checker() -> tuple[object, object]:
    eval_root = str(bfcl_eval_root())
    if eval_root not in sys.path:
        sys.path.insert(0, eval_root)

    # ast_checker only needs this mapping for dot/underscore function-name conversion.
    # Importing the full upstream model registry pulls every provider SDK.
    if "bfcl_eval.constants.model_config" not in sys.modules:
        model_config = types.ModuleType("bfcl_eval.constants.model_config")
        model_config.MODEL_CONFIG_MAPPING = {
            OPENAI_MODEL: types.SimpleNamespace(underscore_to_dot=False)
        }
        sys.modules["bfcl_eval.constants.model_config"] = model_config

    ast_checker = importlib.import_module("bfcl_eval.eval_checker.ast_eval.ast_checker").ast_checker
    language = importlib.import_module("bfcl_eval.constants.enums").Language.PYTHON
    return ast_checker, language


def bfcl_model_output(calls: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{call["name"]: call["arguments"]} for call in calls]


def bfcl_ast_result(
    example: dict[str, object], calls: list[dict[str, object]]
) -> dict[str, object]:
    ast_checker, language = load_bfcl_ast_checker()
    return ast_checker(
        example["function"],
        bfcl_model_output(calls),
        example["ground_truth"],
        language,
        str(example["category"]),
        OPENAI_MODEL,
    )


def aggregate_usage(predictions: list[dict[str, object]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for prediction in predictions:
        usage = prediction.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def input_fingerprint(example: dict[str, object]) -> str:
    return stable_hash(
        {
            "model": OPENAI_MODEL,
            "question": example["question"],
            "function": example["function"],
            "ground_truth": example["ground_truth"],
            "router": {
                "system": ROUTER_SYSTEM_INSTRUCTION,
                "tool_choice": ROUTER_TOOL_CHOICE,
                "max_output_tokens": ROUTER_MAX_OUTPUT_TOKENS,
                "message_serialization": ROUTER_MESSAGE_SERIALIZATION,
            },
        }
    )


def accuracy_metrics(predictions: list[dict[str, object]]) -> dict[str, object]:
    correct = sum(1 for prediction in predictions if prediction["correct"])
    total = len(predictions)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
    }


def category_metrics(
    predictions: list[dict[str, object]], examples_by_id: dict[str, dict[str, object]]
) -> dict[str, dict[str, object]]:
    by_category: dict[str, list[dict[str, object]]] = {}
    for prediction in predictions:
        category = str(examples_by_id[prediction["id"]]["category"])
        by_category.setdefault(category, []).append(prediction)
    return {
        category: accuracy_metrics(category_predictions)
        for category, category_predictions in sorted(by_category.items())
    }


def run_or_load_model_predictions(
    examples: list[dict[str, object]], model_predictions_path: Path
) -> list[dict[str, object]]:
    expected_fingerprints = {example["id"]: input_fingerprint(example) for example in examples}
    if model_predictions_path.exists() and not os.environ.get("REALISTIC_BFCL_FORCE_MODEL_RUN"):
        cached_predictions = {}
        stale_count = 0
        for prediction in read_jsonl(model_predictions_path):
            prediction_id = prediction["id"]
            if prediction.get("input_fingerprint") != expected_fingerprints.get(prediction_id):
                stale_count += 1
                continue
            cached_predictions[prediction_id] = prediction
        print(
            f"Loaded {len(cached_predictions)} cached {OPENAI_MODEL} predictions "
            f"({stale_count} stale ignored)"
        )
    else:
        cached_predictions = {}

    missing_examples = [example for example in examples if example["id"] not in cached_predictions]
    if missing_examples:
        api_key = openai_api_key()
        concurrency = min(openai_concurrency(), len(missing_examples))
        print(
            f"Running {len(missing_examples)} missing {OPENAI_MODEL} calls "
            f"at concurrency {concurrency}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(run_openai_prediction, example, api_key): example
                for example in missing_examples
            }
            for future in concurrent.futures.as_completed(futures):
                example = futures[future]
                prediction = future.result()
                prediction["input_fingerprint"] = expected_fingerprints[example["id"]]
                cached_predictions[example["id"]] = prediction
                append_jsonl(model_predictions_path, prediction)
                print(f"Ran {OPENAI_MODEL} on {example['id']}")
    else:
        print(f"All {OPENAI_MODEL} predictions were cached")

    missing_ids = [example["id"] for example in examples if example["id"] not in cached_predictions]
    if missing_ids:
        raise SystemExit(f"Missing predictions after model run: {missing_ids[:5]}")

    predictions = [cached_predictions[example["id"]] for example in examples]
    examples_by_id = {example["id"]: example for example in examples}
    rescored_predictions = []
    for prediction in predictions:
        calls = prediction["prediction"]
        eval_result = bfcl_ast_result(examples_by_id[prediction["id"]], calls)
        rescored_prediction = dict(prediction)
        rescored_prediction["correct"] = eval_result["valid"]
        rescored_prediction["evaluator"] = "bfcl_ast_checker"
        rescored_prediction["eval_result"] = eval_result
        rescored_prediction["input_fingerprint"] = expected_fingerprints[prediction["id"]]
        rescored_predictions.append(rescored_prediction)
    write_jsonl(model_predictions_path, rescored_predictions)
    return rescored_predictions


def load_current_clean_predictions(
    clean_predictions_path: Path, base_ids: set[str]
) -> dict[str, dict[str, object]]:
    clean_subset_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    if not clean_subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")

    clean_examples = {
        example["id"]: example
        for example in read_jsonl(clean_subset_path)
        if example["id"] in base_ids
    }
    missing_examples = sorted(base_ids - set(clean_examples))
    if missing_examples:
        raise SystemExit(f"Clean subset is missing paired base ids: {missing_examples[:10]}")

    expected_fingerprints = {
        example_id: input_fingerprint(example) for example_id, example in clean_examples.items()
    }
    clean_predictions = {
        prediction["id"]: prediction for prediction in read_jsonl(clean_predictions_path)
    }
    missing_predictions = sorted(base_ids - set(clean_predictions))
    stale_predictions = sorted(
        prediction_id
        for prediction_id in base_ids & set(clean_predictions)
        if clean_predictions[prediction_id].get("input_fingerprint")
        != expected_fingerprints[prediction_id]
    )
    if missing_predictions or stale_predictions:
        details = []
        if missing_predictions:
            details.append(f"{len(missing_predictions)} missing")
        if stale_predictions:
            details.append(f"{len(stale_predictions)} stale")
        raise SystemExit(
            "Clean model predictions are not current for paired eval "
            f"({', '.join(details)}). Run run-bfcl first."
        )
    return {prediction_id: clean_predictions[prediction_id] for prediction_id in base_ids}


def run_openai_prediction(example: dict[str, object], api_key: str) -> dict[str, object]:
    response = call_openai_tool_router(example, api_key)
    calls = response_function_calls(response, tool_name_map(example))
    eval_result = bfcl_ast_result(example, calls)
    return {
        "id": example["id"],
        "model": OPENAI_MODEL,
        "prediction": calls,
        "correct": eval_result["valid"],
        "evaluator": "bfcl_ast_checker",
        "eval_result": eval_result,
        "response_id": response.get("id"),
        "usage": response.get("usage"),
    }


def freeze_bfcl() -> None:
    project_config = REPO_ROOT / "configs/project.yaml"
    subset_config = REPO_ROOT / "configs/subsets/smoke.yaml"
    manifest_path = REPO_ROOT / "artifacts/frozen/bfcl_manifest.json"
    reject_placeholders((project_config, subset_config))
    categories = read_list_setting(subset_config, "bfcl_categories")
    subset_path = materialize_smoke_subset(subset_config, manifest_path)

    write_json(
        manifest_path,
        {
            "created_at": utc_now(),
            "bfcl": {
                "upstream_repository": BFCL_REPOSITORY,
                "dataset_commit": BFCL_COMMIT,
                "evaluator_version": f"gorilla@{BFCL_COMMIT}",
            },
            "clean_subset": {
                "config_path": "configs/subsets/smoke.yaml",
                "config_sha256": file_sha256(subset_config),
                "categories": categories,
                "max_examples": read_int_setting(subset_config, "max_examples"),
                "examples_per_category": read_int_setting(subset_config, "examples_per_category"),
                "materialized_path": "artifacts/frozen/clean_subset.jsonl",
                "materialized_sha256": file_sha256(subset_path),
                "materialized_total": len(read_jsonl(subset_path)),
                "status": "materialized",
            },
            "local_configs": {
                "project_yaml_sha256": file_sha256(project_config),
                "subset_yaml_sha256": file_sha256(subset_config),
            },
            "model_list": {
                "status": "configured",
                "models": [OPENAI_MODEL],
            },
            "status": "source_pinned_subset_materialized",
            "notes": [
                "This pins the BFCL upstream commit and materializes the local smoke subset.",
                "Model API evaluation runs in the run-bfcl step.",
            ],
        },
    )
    print(f"Wrote {manifest_path.relative_to(REPO_ROOT)}")


def clean_baseline() -> None:
    manifest_path = REPO_ROOT / "artifacts/frozen/bfcl_manifest.json"
    subset_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    result_path = REPO_ROOT / "artifacts/results/clean/clean_baseline_summary.json"
    oracle_predictions_path = REPO_ROOT / "artifacts/results/clean/oracle_replay_predictions.jsonl"
    model_predictions_path = REPO_ROOT / f"artifacts/results/clean/{OPENAI_MODEL}_predictions.jsonl"

    if not manifest_path.exists():
        raise SystemExit("Missing artifacts/frozen/bfcl_manifest.json. Run prepare-subset first.")
    if not subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = read_jsonl(subset_path)
    predictions = [
        {
            "id": example["id"],
            "model": "oracle_replay",
            "prediction": example["ground_truth"],
            "correct": True,
        }
        for example in examples
    ]
    write_jsonl(oracle_predictions_path, predictions)

    examples_by_id = {example["id"]: example for example in examples}
    model_predictions = run_or_load_model_predictions(examples, model_predictions_path)
    oracle_metrics = accuracy_metrics(predictions)
    model_metrics = accuracy_metrics(model_predictions)
    oracle_metrics["usage"] = {}
    model_metrics["usage"] = aggregate_usage(model_predictions)

    write_json(
        result_path,
        {
            "created_at": utc_now(),
            "stage": "run-bfcl",
            "status": "ran_model_baseline",
            "reason": "Ran oracle replay and a real OpenAI model baseline.",
            "bfcl_manifest": "artifacts/frozen/bfcl_manifest.json",
            "bfcl_dataset_commit": manifest["bfcl"]["dataset_commit"],
            "predictions": {
                "oracle_replay": "artifacts/results/clean/oracle_replay_predictions.jsonl",
                OPENAI_MODEL: f"artifacts/results/clean/{OPENAI_MODEL}_predictions.jsonl",
            },
            "models": ["oracle_replay", OPENAI_MODEL],
            "metrics": {
                "oracle_replay": oracle_metrics,
                OPENAI_MODEL: model_metrics,
            },
            "category_metrics": {
                "oracle_replay": category_metrics(predictions, examples_by_id),
                OPENAI_MODEL: category_metrics(model_predictions, examples_by_id),
            },
            "next_required_work": [
                "Compare this clean baseline against noisy variants.",
            ],
        },
    )
    print(f"Wrote {result_path.relative_to(REPO_ROOT)}")


def paired_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    clean_correct = sum(1 for row in rows if row["clean_correct"])
    noisy_correct = sum(1 for row in rows if row["noisy_correct"])
    clean_success_noisy_failure = sum(
        1 for row in rows if row["clean_correct"] and not row["noisy_correct"]
    )
    clean_failure_noisy_success = sum(
        1 for row in rows if not row["clean_correct"] and row["noisy_correct"]
    )
    both_correct = sum(1 for row in rows if row["clean_correct"] and row["noisy_correct"])
    both_wrong = sum(1 for row in rows if not row["clean_correct"] and not row["noisy_correct"])
    clean_accuracy = clean_correct / total if total else None
    noisy_accuracy = noisy_correct / total if total else None
    degradation = (
        clean_accuracy - noisy_accuracy
        if clean_accuracy is not None and noisy_accuracy is not None
        else None
    )
    return {
        "total": total,
        "clean_correct": clean_correct,
        "noisy_correct": noisy_correct,
        "clean_accuracy": clean_accuracy,
        "noisy_accuracy": noisy_accuracy,
        "absolute_degradation": degradation,
        "relative_degradation": degradation / clean_accuracy if clean_accuracy else None,
        "conditional_failure_given_clean_success": (
            clean_success_noisy_failure / clean_correct if clean_correct else None
        ),
        "clean_success_noisy_failure": clean_success_noisy_failure,
        "clean_failure_noisy_success": clean_failure_noisy_success,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def generated_dimensions() -> list[str]:
    dimensions = [
        dimension
        for dimension, filename in DIMENSION_FILES.items()
        if (REPO_ROOT / f"artifacts/generated/{filename}").exists()
    ]
    requested = os.environ.get("REALISTIC_BFCL_DIMENSIONS")
    if not requested:
        return dimensions
    requested_dimensions = [dimension.strip() for dimension in requested.split(",")]
    unknown = sorted(set(requested_dimensions) - set(DIMENSION_FILES))
    if unknown:
        known = ", ".join(DIMENSION_FILES)
        raise SystemExit(f"Unknown dimensions {unknown}. Known dimensions: {known}")
    missing_artifacts = sorted(set(requested_dimensions) - set(dimensions))
    if missing_artifacts:
        raise SystemExit(
            f"Requested dimensions are not generated: {missing_artifacts}. " "Run augment first."
        )
    return [dimension for dimension in dimensions if dimension in requested_dimensions]


def paired_eval_dimension(dimension: str) -> dict[str, object]:
    noisy_path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"
    clean_predictions_path = REPO_ROOT / f"artifacts/results/clean/{OPENAI_MODEL}_predictions.jsonl"
    noisy_predictions_path = (
        REPO_ROOT / f"artifacts/results/noisy/{dimension}/{OPENAI_MODEL}_predictions.jsonl"
    )
    paired_path = REPO_ROOT / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_paired.jsonl"
    summary_path = REPO_ROOT / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_summary.json"

    if not clean_predictions_path.exists():
        raise SystemExit("Missing clean model predictions. Run run-bfcl first.")

    noisy_examples = read_jsonl(noisy_path)
    clean_predictions = load_current_clean_predictions(
        clean_predictions_path,
        {str(noisy_example["base_id"]) for noisy_example in noisy_examples},
    )
    noisy_predictions = run_or_load_model_predictions(noisy_examples, noisy_predictions_path)
    noisy_predictions_by_id = {prediction["id"]: prediction for prediction in noisy_predictions}

    paired_rows = []
    for noisy_example in noisy_examples:
        clean_prediction = clean_predictions[noisy_example["base_id"]]
        noisy_prediction = noisy_predictions_by_id[noisy_example["id"]]
        paired_rows.append(
            {
                "base_id": noisy_example["base_id"],
                "noisy_id": noisy_example["id"],
                "dimension": noisy_example["dimension"],
                "category": noisy_example["category"],
                "clean_correct": clean_prediction["correct"],
                "noisy_correct": noisy_prediction["correct"],
                "clean_prediction": clean_prediction["prediction"],
                "noisy_prediction": noisy_prediction["prediction"],
            }
        )

    write_jsonl(paired_path, paired_rows)
    metrics = paired_metrics(paired_rows)
    noisy_accuracy = accuracy_metrics(noisy_predictions)
    noisy_accuracy["usage"] = aggregate_usage(noisy_predictions)
    write_json(
        summary_path,
        {
            "created_at": utc_now(),
            "stage": "run-bfcl",
            "model": OPENAI_MODEL,
            "dimension": dimension,
            "clean_predictions": clean_predictions_path.relative_to(REPO_ROOT).as_posix(),
            "noisy_predictions": noisy_predictions_path.relative_to(REPO_ROOT).as_posix(),
            "paired_results": paired_path.relative_to(REPO_ROOT).as_posix(),
            "metrics": metrics,
            "noisy_metrics": noisy_accuracy,
        },
    )
    print(f"Wrote {paired_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {summary_path.relative_to(REPO_ROOT)}")
    return {
        "dimension": dimension,
        "summary": summary_path.relative_to(REPO_ROOT).as_posix(),
        "metrics": metrics,
    }


def paired_eval() -> None:
    dimensions = generated_dimensions()
    if not dimensions:
        raise SystemExit("No generated noisy dimensions found. Run augment first.")
    summaries = [paired_eval_dimension(dimension) for dimension in dimensions]
    write_json(
        REPO_ROOT / f"artifacts/results/paired/{OPENAI_MODEL}_summary.json",
        {
            "created_at": utc_now(),
            "stage": "run-bfcl",
            "model": OPENAI_MODEL,
            "dimensions": summaries,
        },
    )


def run_bfcl() -> None:
    clean_baseline()
    paired_eval()
