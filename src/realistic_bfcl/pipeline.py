from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib
import json
import os
import re
import sys
import time
import types
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
OPENAI_MODEL = "gpt-5.4-nano"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_CONCURRENCY = 8
OPENAI_MAX_ATTEMPTS = 4
RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}
BFCL_CATEGORY_FILES = {
    "simple_python": (
        "BFCL_v4_simple_python.json",
        "possible_answer/BFCL_v4_simple_python.json",
    ),
    "multiple": (
        "BFCL_v4_multiple.json",
        "possible_answer/BFCL_v4_multiple.json",
    ),
    "parallel": (
        "BFCL_v4_parallel.json",
        "possible_answer/BFCL_v4_parallel.json",
    ),
    "parallel_multiple": (
        "BFCL_v4_parallel_multiple.json",
        "possible_answer/BFCL_v4_parallel_multiple.json",
    ),
}


@dataclass(frozen=True)
class Stage:
    name: str
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    next_action: str


STAGES: tuple[Stage, ...] = (
    Stage(
        name="freeze-bfcl",
        purpose="Pin BFCL source metadata and local subset configuration.",
        inputs=("configs/project.yaml", "configs/subsets/smoke.yaml"),
        outputs=("artifacts/frozen/bfcl_manifest.json",),
        next_action="Materialize the configured subset from the pinned BFCL commit.",
    ),
    Stage(
        name="clean-baseline",
        purpose="Reproduce BFCL-style clean scores before adding noise.",
        inputs=("artifacts/frozen/bfcl_manifest.json",),
        outputs=("artifacts/results/clean/",),
        next_action="Wire the BFCL evaluator adapter and run the selected models on clean prompts.",
    ),
    Stage(
        name="augment-overhang",
        purpose="Generate realistic irrelevant conversational context around clean requests.",
        inputs=("artifacts/frozen/bfcl_manifest.json", "configs/realism_dimensions.yaml"),
        outputs=("artifacts/generated/conversational_overhang.jsonl",),
        next_action=(
            "Implement the conversational overhang generator with oracle-preservation metadata."
        ),
    ),
    Stage(
        name="augment-incremental",
        purpose="Split clean requests across natural multi-turn slot revelation.",
        inputs=("artifacts/frozen/bfcl_manifest.json", "configs/realism_dimensions.yaml"),
        outputs=("artifacts/generated/incremental_slot_revelation.jsonl",),
        next_action=(
            "Implement multi-turn prompt construction without changing the final oracle."
        ),
    ),
    Stage(
        name="verify-noisy",
        purpose="Run invariant checks and realism audit before evaluation.",
        inputs=("artifacts/generated/",),
        outputs=("artifacts/audits/noisy_examples_audit.jsonl", "artifacts/accepted/"),
        next_action=(
            "Add deterministic schema and oracle checks, then attach human or LLM audit labels."
        ),
    ),
    Stage(
        name="paired-eval",
        purpose="Evaluate clean and noisy variants with identical models and schemas.",
        inputs=("artifacts/results/clean/", "artifacts/accepted/"),
        outputs=("artifacts/results/noisy/", "artifacts/results/paired/"),
        next_action=(
            "Run the evaluator for each accepted noisy variant and join results to clean runs."
        ),
    ),
    Stage(
        name="analyze",
        purpose="Compute degradation metrics and error taxonomy.",
        inputs=("artifacts/results/paired/",),
        outputs=("artifacts/analysis/metrics.json", "artifacts/analysis/error_taxonomy.csv"),
        next_action="Implement paired metrics and classify noisy failures.",
    ),
    Stage(
        name="defenses",
        purpose="Ablate simple defenses against realistic conversational noise.",
        inputs=("artifacts/accepted/", "artifacts/results/paired/"),
        outputs=("artifacts/analysis/defense_ablations/"),
        next_action=(
            "Run denoising, stricter tool-use instructions, schema variants, and decoding variants."
        ),
    ),
)


def stage_by_name(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    known = ", ".join(stage.name for stage in STAGES)
    raise SystemExit(f"Unknown stage '{name}'. Known stages: {known}")


def list_stages() -> None:
    for stage in STAGES:
        print(f"{stage.name}: {stage.purpose}")


def describe_stage(stage: Stage) -> None:
    print(f"Stage: {stage.name}")
    print(f"Purpose: {stage.purpose}")
    print("Inputs:")
    for item in stage.inputs:
        print(f"  - {item}")
    print("Outputs:")
    for item in stage.outputs:
        print(f"  - {item}")
    print(f"Next action: {stage.next_action}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_int_setting(path: Path, key: str) -> int:
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(prefix):
            return int(line.split(":", 1)[1].strip())
    raise SystemExit(f"Missing required setting '{key}' in {path.relative_to(REPO_ROOT)}")


def read_list_setting(path: Path, key: str) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values: list[str] = []
    in_block = False
    prefix = f"{key}:"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            after_colon = stripped.split(":", 1)[1].strip()
            if after_colon == "[]":
                return []
            in_block = True
            continue
        if in_block:
            if stripped.startswith("- "):
                values.append(stripped[2:].strip())
                continue
            if stripped and not line.startswith(" "):
                break

    return values


def reject_placeholders(paths: tuple[Path, ...]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "TODO" in text:
            raise SystemExit(
                f"Refusing to freeze with TODO placeholders in {path.relative_to(REPO_ROOT)}"
            )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def conversation_text(question: object) -> str:
    conversations = question if isinstance(question, list) else []
    if not conversations:
        return ""
    messages = conversations[0] if isinstance(conversations[0], list) else []
    return "\n".join(str(message.get("content", "")) for message in messages)


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def openai_api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]

    candidates = [
        Path(os.environ["REALISTIC_BFCL_ENV_FILE"])
        if os.environ.get("REALISTIC_BFCL_ENV_FILE")
        else None,
        REPO_ROOT.parent / "underlayer/.env",
    ]
    for path in candidates:
        if path is None:
            continue
        key = read_env_file(path).get("OPENAI_API_KEY")
        if key:
            return key

    raise SystemExit(
        "Missing OPENAI_API_KEY. Set it in the environment or REALISTIC_BFCL_ENV_FILE."
    )


def openai_concurrency() -> int:
    value = os.environ.get("REALISTIC_BFCL_CONCURRENCY", str(DEFAULT_OPENAI_CONCURRENCY))
    try:
        concurrency = int(value)
    except ValueError as error:
        raise SystemExit("REALISTIC_BFCL_CONCURRENCY must be an integer.") from error
    if concurrency < 1:
        raise SystemExit("REALISTIC_BFCL_CONCURRENCY must be >= 1.")
    return concurrency


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    data_root = bfcl_data_root()
    rows: list[dict[str, object]] = []

    for category in categories:
        question_file, answer_file = BFCL_CATEGORY_FILES[category]
        questions = read_jsonl(data_root / question_file)
        answers = {row["id"]: row for row in read_jsonl(data_root / answer_file)}
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


def call_openai_tool_router(example: dict[str, object], api_key: str) -> dict[str, object]:
    messages = example["question"][0]
    user_content = "\n".join(message["content"] for message in messages)
    tools = []
    for index, function_doc in enumerate(example["function"]):
        tools.append(openai_tool(function_doc, safe_tool_name(str(function_doc["name"]), index)))
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": (
                    "Call the provided tool that best satisfies the user request. "
                    "Do not answer in prose when a tool call is appropriate."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "tools": tools,
        "tool_choice": "required",
        "max_output_tokens": 256,
    }
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

    ast_checker = importlib.import_module(
        "bfcl_eval.eval_checker.ast_eval.ast_checker"
    ).ast_checker
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


OVERHANG_TEMPLATES = (
    "hey quick one - {prompt}",
    "before i forget, {prompt}",
    "can you help me with this? {prompt}",
    "sorry, different topic for a sec: {prompt}",
    "i'm updating some notes and need this too: {prompt}",
    "ok while i'm here, {prompt}",
    "my notes are kind of all over the place. {prompt}",
    "one more thing: {prompt}",
    "btw {prompt}",
    "{prompt} thanks",
    "{prompt} - trying to finish this before my next meeting",
    "{prompt} if that makes sense",
)


def lowercase_first_alpha(text: str) -> str:
    for index, char in enumerate(text):
        if char.isalpha():
            return f"{text[:index]}{char.lower()}{text[index + 1 :]}"
    return text


def overhang_prompt(clean_prompt: str, index: int) -> str:
    template = OVERHANG_TEMPLATES[index % len(OVERHANG_TEMPLATES)]
    return template.format(prompt=lowercase_first_alpha(clean_prompt))


def overhang_messages(question: object, index: int) -> object:
    conversations = json.loads(json.dumps(question))
    first_message = conversations[0][0]
    first_message["content"] = overhang_prompt(str(first_message["content"]), index)
    return conversations


def augment_overhang() -> None:
    subset_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    output_path = REPO_ROOT / "artifacts/generated/conversational_overhang.jsonl"

    if not subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run freeze-bfcl first.")

    rows = []
    for index, example in enumerate(read_jsonl(subset_path)):
        rows.append(
            {
                "id": f"{example['id']}__overhang",
                "base_id": example["id"],
                "category": example["category"],
                "dimension": "conversational_overhang",
                "question": overhang_messages(example["question"], index),
                "function": example["function"],
                "ground_truth": example["ground_truth"],
                "oracle_preservation": {
                    "function_schema_unchanged": True,
                    "ground_truth_unchanged": True,
                },
            }
        )

    write_jsonl(output_path, rows)
    print(f"Wrote {output_path.relative_to(REPO_ROOT)}")


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
    if model_predictions_path.exists() and not os.environ.get("REALISTIC_BFCL_FORCE_MODEL_RUN"):
        cached_predictions = {
            prediction["id"]: prediction for prediction in read_jsonl(model_predictions_path)
        }
        print(f"Loaded {len(cached_predictions)} cached {OPENAI_MODEL} predictions")
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
        rescored_predictions.append(rescored_prediction)
    write_jsonl(model_predictions_path, rescored_predictions)
    return rescored_predictions


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
                "Model API evaluation is still pending; clean-baseline runs oracle replay only.",
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
        raise SystemExit("Missing artifacts/frozen/bfcl_manifest.json. Run freeze-bfcl first.")
    if not subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run freeze-bfcl first.")

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
            "stage": "clean-baseline",
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


def paired_eval() -> None:
    noisy_path = REPO_ROOT / "artifacts/generated/conversational_overhang.jsonl"
    clean_predictions_path = REPO_ROOT / f"artifacts/results/clean/{OPENAI_MODEL}_predictions.jsonl"
    noisy_predictions_path = (
        REPO_ROOT
        / f"artifacts/results/noisy/conversational_overhang/{OPENAI_MODEL}_predictions.jsonl"
    )
    paired_path = (
        REPO_ROOT
        / f"artifacts/results/paired/conversational_overhang/{OPENAI_MODEL}_paired.jsonl"
    )
    summary_path = (
        REPO_ROOT
        / f"artifacts/results/paired/conversational_overhang/{OPENAI_MODEL}_summary.json"
    )

    if not noisy_path.exists():
        raise SystemExit("Missing artifacts/generated/conversational_overhang.jsonl.")
    if not clean_predictions_path.exists():
        raise SystemExit("Missing clean model predictions. Run clean-baseline first.")

    noisy_examples = read_jsonl(noisy_path)
    clean_predictions = {
        prediction["id"]: prediction for prediction in read_jsonl(clean_predictions_path)
    }
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
            "stage": "paired-eval",
            "model": OPENAI_MODEL,
            "dimension": "conversational_overhang",
            "clean_predictions": clean_predictions_path.relative_to(REPO_ROOT).as_posix(),
            "noisy_predictions": noisy_predictions_path.relative_to(REPO_ROOT).as_posix(),
            "paired_results": paired_path.relative_to(REPO_ROOT).as_posix(),
            "metrics": metrics,
            "noisy_metrics": noisy_accuracy,
        },
    )
    print(f"Wrote {paired_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {summary_path.relative_to(REPO_ROOT)}")


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


def heuristic_error_type(
    gold: object, _clean_prediction: object, noisy_prediction: object
) -> str:
    noisy_names = call_names(noisy_prediction)
    if not noisy_names:
        return "no_call"
    if any(
        "__malformed_arguments__" in arguments
        for arguments in call_arguments(noisy_prediction)
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


def analysis_review_row(
    row: dict[str, object],
    clean_examples: dict[str, dict[str, object]],
    noisy_examples: dict[str, dict[str, object]],
    outcome: str,
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
        "heuristic_error_type": heuristic_error_type(
            gold, row["clean_prediction"], row["noisy_prediction"]
        ),
        "clean_prompt": conversation_text(clean_example["question"]),
        "noisy_prompt": conversation_text(noisy_example["question"]),
        "gold": gold,
        "clean_prediction": row["clean_prediction"],
        "noisy_prediction": row["noisy_prediction"],
    }


def analyze() -> None:
    paired_path = (
        REPO_ROOT
        / f"artifacts/results/paired/conversational_overhang/{OPENAI_MODEL}_paired.jsonl"
    )
    clean_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    noisy_path = REPO_ROOT / "artifacts/generated/conversational_overhang.jsonl"
    summary_path = REPO_ROOT / "artifacts/analysis/conversational_overhang/summary.json"
    regressions_jsonl_path = (
        REPO_ROOT / "artifacts/analysis/conversational_overhang/regressions.jsonl"
    )
    regressions_csv_path = REPO_ROOT / "artifacts/analysis/conversational_overhang/regressions.csv"
    recoveries_jsonl_path = (
        REPO_ROOT / "artifacts/analysis/conversational_overhang/recoveries.jsonl"
    )

    for path in (paired_path, clean_path, noisy_path):
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(REPO_ROOT)}.")

    paired_rows = read_jsonl(paired_path)
    clean_examples = {row["id"]: row for row in read_jsonl(clean_path)}
    noisy_examples = {row["id"]: row for row in read_jsonl(noisy_path)}

    regressions = [
        analysis_review_row(row, clean_examples, noisy_examples, "clean_success_noisy_failure")
        for row in paired_rows
        if row["clean_correct"] and not row["noisy_correct"]
    ]
    recoveries = [
        analysis_review_row(row, clean_examples, noisy_examples, "clean_failure_noisy_success")
        for row in paired_rows
        if not row["clean_correct"] and row["noisy_correct"]
    ]
    stable_failures = [
        row for row in paired_rows if not row["clean_correct"] and not row["noisy_correct"]
    ]
    stable_successes = [
        row for row in paired_rows if row["clean_correct"] and row["noisy_correct"]
    ]

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
            "dimension": "conversational_overhang",
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
                "by_heuristic_error_type": count_by(recoveries, "heuristic_error_type"),
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


def run_stage(stage: Stage, dry_run: bool) -> None:
    describe_stage(stage)
    if dry_run:
        return
    if stage.name == "freeze-bfcl":
        freeze_bfcl()
        return
    if stage.name == "clean-baseline":
        clean_baseline()
        return
    if stage.name == "augment-overhang":
        augment_overhang()
        return
    if stage.name == "paired-eval":
        paired_eval()
        return
    if stage.name == "analyze":
        analyze()
        return
    raise SystemExit(
        "Stage implementation pending. Use --dry-run for planning, then replace "
        "this placeholder once the corresponding artifact is implemented."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Realistic-BFCL research stages.")
    parser.add_argument("stage", nargs="?", help="Stage name to inspect or run.")
    parser.add_argument("--list", action="store_true", help="List available stages.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the stage without running it.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_stages()
        return
    if not args.stage:
        parser.error("provide a stage name or --list")

    run_stage(stage_by_name(args.stage), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
