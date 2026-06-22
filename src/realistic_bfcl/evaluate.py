from __future__ import annotations

import concurrent.futures
import importlib
import json
import os
import re
import socket
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

from .common import (
    ANTHROPIC_MESSAGE_BATCHES_URL,
    ANTHROPIC_MESSAGES_URL,
    BFCL_CATEGORY_FILES,
    BFCL_COMMIT,
    BFCL_REPOSITORY,
    DIMENSION_FILES,
    OPENAI_MAX_ATTEMPTS,
    OPENAI_RESPONSES_URL,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    REPO_ROOT,
    RETRYABLE_HTTP_STATUS,
    ROUTER_MAX_OUTPUT_TOKENS,
    ROUTER_MESSAGE_SERIALIZATION,
    ROUTER_SYSTEM_INSTRUCTION,
    ROUTER_TOOL_CHOICE,
    XAI_CHAT_COMPLETIONS_URL,
    ModelRun,
    anthropic_api_key,
    append_jsonl,
    configured_model_runs,
    file_sha256,
    openai_api_key,
    openai_concurrency,
    openrouter_api_key,
    optional_positive_int_env,
    read_int_setting,
    read_jsonl,
    read_list_setting,
    reject_placeholders,
    stable_hash,
    utc_now,
    write_json,
    write_jsonl,
    xai_api_key,
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


def materialize_subset(subset_config: Path, manifest_path: Path) -> Path:
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


def responses_tool(function_doc: dict[str, object], name: str) -> dict[str, object]:
    parameters = normalize_json_schema(function_doc["parameters"])
    return {
        "type": "function",
        "name": name,
        "description": function_doc.get("description", ""),
        "parameters": parameters,
    }


def chat_completion_tool(function_doc: dict[str, object], name: str) -> dict[str, object]:
    parameters = normalize_json_schema(function_doc["parameters"])
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": function_doc.get("description", ""),
            "parameters": parameters,
        },
    }


def anthropic_safe_property_name(name: str, used_names: set[str]) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    safe = safe.strip("._-") or "arg"
    if len(safe) > 64:
        safe = f"{safe[:55]}_{stable_hash(name)[:8]}"
    candidate = safe
    counter = 2
    while candidate in used_names:
        suffix = f"_{counter}"
        candidate = f"{safe[: 64 - len(suffix)]}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def anthropic_safe_schema(value: object) -> tuple[object, dict[str, object]]:
    if isinstance(value, list):
        converted_items = []
        item_maps = []
        for item in value:
            converted_item, item_map = anthropic_safe_schema(item)
            converted_items.append(converted_item)
            item_maps.append(item_map)
        return converted_items, {"items": item_maps}

    if not isinstance(value, dict):
        return value, {}

    converted: dict[str, object] = {}
    property_map: dict[str, str] = {}
    child_maps: dict[str, object] = {}
    current_key_map: dict[str, str] | None = None

    for key, child in value.items():
        if key == "properties" and isinstance(child, dict):
            current_key_map = {}
            converted_properties: dict[str, object] = {}
            used_names: set[str] = set()
            for property_name, property_schema in child.items():
                safe_name = anthropic_safe_property_name(str(property_name), used_names)
                converted_property_schema, child_map = anthropic_safe_schema(property_schema)
                converted_properties[safe_name] = converted_property_schema
                current_key_map[str(property_name)] = safe_name
                property_map[safe_name] = str(property_name)
                if child_map:
                    child_maps[safe_name] = child_map
            converted[key] = converted_properties
            continue
        if key == "required" and isinstance(child, list):
            converted[key] = child
            continue

        converted_child, child_map = anthropic_safe_schema(child)
        converted[key] = converted_child
        if child_map:
            child_maps[key] = child_map

    if current_key_map and isinstance(converted.get("required"), list):
        converted["required"] = [
            current_key_map.get(str(required_key), str(required_key))
            for required_key in converted["required"]
        ]

    schema_map: dict[str, object] = {}
    if property_map:
        schema_map["properties"] = property_map
    if child_maps:
        schema_map["children"] = child_maps
    return converted, schema_map


def restore_anthropic_arguments(value: object, schema_map: object) -> object:
    if isinstance(value, list):
        item_maps = schema_map.get("items", []) if isinstance(schema_map, dict) else []
        restored_items = []
        for index, item in enumerate(value):
            item_map = item_maps[index] if index < len(item_maps) else {}
            restored_items.append(restore_anthropic_arguments(item, item_map))
        return restored_items

    if not isinstance(value, dict) or not isinstance(schema_map, dict):
        return value

    property_map = schema_map.get("properties", {})
    child_maps = schema_map.get("children", {})
    restored = {}
    for key, child in value.items():
        original_key = property_map.get(key, key) if isinstance(property_map, dict) else key
        child_map = child_maps.get(key, {}) if isinstance(child_maps, dict) else {}
        restored[original_key] = restore_anthropic_arguments(child, child_map)
    return restored


def anthropic_argument_schema_maps(example: dict[str, object]) -> dict[str, object]:
    maps = {}
    for index, function_doc in enumerate(example["function"]):
        name = safe_tool_name(str(function_doc["name"]), index)
        _schema, schema_map = anthropic_safe_schema(
            normalize_json_schema(function_doc["parameters"])
        )
        maps[name] = schema_map
    return maps


def anthropic_tool(function_doc: dict[str, object], name: str) -> dict[str, object]:
    parameters, _schema_map = anthropic_safe_schema(
        normalize_json_schema(function_doc["parameters"])
    )
    return {
        "name": name,
        "description": function_doc.get("description", ""),
        "input_schema": parameters,
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


def anthropic_system_and_messages(example: dict[str, object]) -> tuple[str, list[dict[str, str]]]:
    system_parts = [ROUTER_SYSTEM_INSTRUCTION]
    messages = []
    for message in bfcl_messages(example):
        if message["role"] == "system":
            system_parts.append(message["content"])
            continue
        messages.append(message)
    return "\n\n".join(part for part in system_parts if part), messages


def openai_retry_json(payload: dict[str, object], api_key: str) -> dict[str, object]:
    request_data = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=request_data,
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in RETRYABLE_HTTP_STATUS and attempt < OPENAI_MAX_ATTEMPTS:
                time.sleep(openai_retry_delay(error, body, attempt))
                continue
            raise RuntimeError(f"OpenAI API request failed: HTTP {error.code}: {body}") from error
        except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
            if attempt < OPENAI_MAX_ATTEMPTS:
                time.sleep(min(60, 2**attempt))
                continue
            raise RuntimeError(f"OpenAI API request failed: {error}") from error

    raise RuntimeError("OpenAI API request failed without returning a response.")


def xai_retry_json(payload: dict[str, object], api_key: str) -> dict[str, object]:
    return chat_completion_retry_json(
        payload,
        api_key,
        XAI_CHAT_COMPLETIONS_URL,
        "xAI",
    )


def openrouter_retry_json(payload: dict[str, object], api_key: str) -> dict[str, object]:
    return chat_completion_retry_json(
        payload,
        api_key,
        OPENROUTER_CHAT_COMPLETIONS_URL,
        "OpenRouter",
    )


def chat_completion_retry_json(
    payload: dict[str, object],
    api_key: str,
    url: str,
    provider_label: str,
) -> dict[str, object]:
    request_data = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            url,
            data=request_data,
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in RETRYABLE_HTTP_STATUS and attempt < OPENAI_MAX_ATTEMPTS:
                time.sleep(openai_retry_delay(error, body, attempt))
                continue
            raise RuntimeError(
                f"{provider_label} API request failed: HTTP {error.code}: {body}"
            ) from error
        except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
            if attempt < OPENAI_MAX_ATTEMPTS:
                time.sleep(min(60, 2**attempt))
                continue
            raise RuntimeError(f"{provider_label} API request failed: {error}") from error

    raise RuntimeError(f"{provider_label} API request failed without returning a response.")


def anthropic_retry_json(payload: dict[str, object], api_key: str) -> dict[str, object]:
    return anthropic_request_json("POST", ANTHROPIC_MESSAGES_URL, api_key, payload)


def anthropic_request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    response_text = anthropic_request_text(method, url, api_key, payload)
    return json.loads(response_text)


def anthropic_request_text(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, object] | None = None,
) -> str:
    request_data = json.dumps(payload).encode("utf-8")
    if payload is None:
        request_data = None
    request_headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            url,
            data=request_data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in RETRYABLE_HTTP_STATUS and attempt < OPENAI_MAX_ATTEMPTS:
                time.sleep(openai_retry_delay(error, body, attempt))
                continue
            raise RuntimeError(
                f"Anthropic API request failed: HTTP {error.code}: {body}"
            ) from error
        except (TimeoutError, socket.timeout, urllib.error.URLError) as error:
            if attempt < OPENAI_MAX_ATTEMPTS:
                time.sleep(min(60, 2**attempt))
                continue
            raise RuntimeError(f"Anthropic API request failed: {error}") from error

    raise RuntimeError("Anthropic API request failed without returning a response.")


def openai_retry_delay(error: urllib.error.HTTPError, body: str, attempt: int) -> float:
    retry_after = error.headers.get("Retry-After")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    match = re.search(r"try again in ([0-9.]+)(ms|s)", body, flags=re.IGNORECASE)
    if match:
        delay = float(match.group(1))
        if match.group(2).lower() == "ms":
            delay /= 1000
        if error.code == 429:
            return max(5.0, delay)
        return max(1.0, delay)
    if error.code == 429:
        return min(60, 4 * attempt)
    return min(60, 2**attempt)


def call_openai_tool_router(example: dict[str, object], model: ModelRun) -> dict[str, object]:
    tools = []
    for index, function_doc in enumerate(example["function"]):
        tools.append(
            responses_tool(function_doc, safe_tool_name(str(function_doc["name"]), index))
        )
    payload = {
        "model": model.id,
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
        "temperature": model.temperature,
    }
    return openai_retry_json(payload, openai_api_key())


def call_xai_tool_router(example: dict[str, object], model: ModelRun) -> dict[str, object]:
    tools = []
    for index, function_doc in enumerate(example["function"]):
        tools.append(
            chat_completion_tool(
                function_doc,
                safe_tool_name(str(function_doc["name"]), index),
            )
        )
    payload = {
        "model": model.id,
        "messages": [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_INSTRUCTION,
            },
            *bfcl_messages(example),
        ],
        "tools": tools,
        "tool_choice": "required",
        "temperature": model.temperature,
        "max_tokens": ROUTER_MAX_OUTPUT_TOKENS,
    }
    return xai_retry_json(payload, xai_api_key())


def openrouter_provider_routing() -> dict[str, object]:
    routing: dict[str, object] = {
        "require_parameters": True,
        "allow_fallbacks": False,
    }
    order = os.environ.get("REALISTIC_BFCL_OPENROUTER_PROVIDER_ORDER", "").strip()
    if order:
        routing["order"] = [item.strip() for item in order.split(",") if item.strip()]
    only = os.environ.get("REALISTIC_BFCL_OPENROUTER_PROVIDER_ONLY", "").strip()
    if only:
        routing["only"] = [item.strip() for item in only.split(",") if item.strip()]
    allow_fallbacks = os.environ.get("REALISTIC_BFCL_OPENROUTER_ALLOW_FALLBACKS", "")
    if allow_fallbacks:
        routing["allow_fallbacks"] = allow_fallbacks.strip().lower() in {"1", "true", "yes"}
    return routing


def call_openrouter_tool_router(example: dict[str, object], model: ModelRun) -> dict[str, object]:
    tools = []
    for index, function_doc in enumerate(example["function"]):
        tools.append(
            chat_completion_tool(
                function_doc,
                safe_tool_name(str(function_doc["name"]), index),
            )
        )
    payload = {
        "model": model.id,
        "messages": [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_INSTRUCTION,
            },
            *bfcl_messages(example),
        ],
        "tools": tools,
        "tool_choice": "required",
        "temperature": model.temperature,
        "max_tokens": ROUTER_MAX_OUTPUT_TOKENS,
        "provider": openrouter_provider_routing(),
    }
    return openrouter_retry_json(payload, openrouter_api_key())


def anthropic_messages_payload(
    example: dict[str, object], model: ModelRun
) -> dict[str, object]:
    tools = []
    for index, function_doc in enumerate(example["function"]):
        tools.append(
            anthropic_tool(function_doc, safe_tool_name(str(function_doc["name"]), index))
        )
    system, messages = anthropic_system_and_messages(example)
    payload = {
        "model": model.id,
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "any"},
        "temperature": model.temperature,
        "max_tokens": ROUTER_MAX_OUTPUT_TOKENS,
    }
    return payload


def call_anthropic_tool_router(example: dict[str, object], model: ModelRun) -> dict[str, object]:
    payload = anthropic_messages_payload(example, model)
    return anthropic_retry_json(payload, anthropic_api_key())


def call_tool_router(example: dict[str, object], model: ModelRun) -> dict[str, object]:
    if model.provider == "openai":
        return call_openai_tool_router(example, model)
    if model.provider in {"anthropic", "claude"}:
        return call_anthropic_tool_router(example, model)
    if model.provider in {"xai", "grok"}:
        return call_xai_tool_router(example, model)
    if model.provider == "openrouter":
        return call_openrouter_tool_router(example, model)
    raise SystemExit(f"Unsupported evaluation provider for {model.id}: {model.provider}")


def use_batch_backend(model: ModelRun) -> bool:
    backend = os.environ.get("REALISTIC_BFCL_EXECUTION", "").strip().lower()
    if backend not in {"batch", "anthropic_batch"}:
        return False
    if model.provider not in {"anthropic", "claude"}:
        raise SystemExit("REALISTIC_BFCL_EXECUTION=batch currently supports Anthropic only.")
    return True


def anthropic_batch_custom_id(index: int, example_id: object) -> str:
    digest = stable_hash({"index": index, "id": example_id})[:16]
    return f"rbfcl_{index}_{digest}"


def anthropic_batch_request(
    custom_id: str, example: dict[str, object], model: ModelRun
) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "params": anthropic_messages_payload(example, model),
    }


def anthropic_batch_state_path(model_predictions_path: Path) -> Path:
    return model_predictions_path.with_name(f"{model_predictions_path.stem}_batch_state.json")


def create_or_load_anthropic_batch(
    examples: list[dict[str, object]],
    model_predictions_path: Path,
    model: ModelRun,
) -> dict[str, object]:
    state_path = anthropic_batch_state_path(model_predictions_path)
    if state_path.exists() and not os.environ.get("REALISTIC_BFCL_FORCE_MODEL_RUN"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "completed":
            print(f"Resuming Anthropic batch {state['batch_id']} for {model.id}")
            return state

    requests = []
    custom_id_to_example_id = {}
    for index, example in enumerate(examples):
        custom_id = anthropic_batch_custom_id(index, example["id"])
        custom_id_to_example_id[custom_id] = example["id"]
        requests.append(anthropic_batch_request(custom_id, example, model))

    response = anthropic_request_json(
        "POST",
        ANTHROPIC_MESSAGE_BATCHES_URL,
        anthropic_api_key(),
        {"requests": requests},
    )
    state = {
        "batch_id": response["id"],
        "created_at": utc_now(),
        "model": model.id,
        "provider": model.provider,
        "temperature": model.temperature,
        "custom_id_to_example_id": custom_id_to_example_id,
        "request_count": len(requests),
        "status": response.get("processing_status"),
        "response": response,
    }
    write_json(state_path, state)
    print(f"Submitted Anthropic batch {response['id']} with {len(requests)} requests")
    return state


def poll_anthropic_batch(batch_id: str) -> dict[str, object]:
    poll_seconds = int(os.environ.get("REALISTIC_BFCL_BATCH_POLL_SECONDS", "30"))
    max_wait_seconds = int(os.environ.get("REALISTIC_BFCL_BATCH_MAX_WAIT_SECONDS", "3600"))
    deadline = time.time() + max_wait_seconds
    url = f"{ANTHROPIC_MESSAGE_BATCHES_URL}/{batch_id}"

    while True:
        response = anthropic_request_json("GET", url, anthropic_api_key())
        status = response.get("processing_status")
        counts = response.get("request_counts", {})
        print(f"Anthropic batch {batch_id}: {status} {counts}")
        if status == "ended":
            return response
        if time.time() >= deadline:
            raise SystemExit(
                f"Anthropic batch {batch_id} is still {status}; rerun to resume later."
            )
        time.sleep(poll_seconds)


def read_anthropic_batch_results(batch_id: str) -> list[dict[str, object]]:
    url = f"{ANTHROPIC_MESSAGE_BATCHES_URL}/{batch_id}/results"
    text = anthropic_request_text("GET", url, anthropic_api_key())
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def run_or_load_anthropic_batch_predictions(
    examples: list[dict[str, object]],
    model_predictions_path: Path,
    model: ModelRun,
    expected_fingerprints: dict[object, str],
    cached_predictions: dict[object, dict[str, object]],
) -> dict[object, dict[str, object]]:
    state = create_or_load_anthropic_batch(examples, model_predictions_path, model)
    batch_response = poll_anthropic_batch(str(state["batch_id"]))
    results = read_anthropic_batch_results(str(state["batch_id"]))
    custom_id_to_example_id = state["custom_id_to_example_id"]
    examples_by_id = {example["id"]: example for example in examples}
    predictions = []
    failed_example_ids = []

    for result in results:
        custom_id = result["custom_id"]
        example_id = custom_id_to_example_id.get(custom_id)
        if example_id is None:
            raise SystemExit(f"Anthropic batch returned unknown custom_id: {custom_id}")
        result_payload = result.get("result", {})
        if result_payload.get("type") != "succeeded":
            failed_example_ids.append(example_id)
            print(
                "Anthropic batch request failed for "
                f"{example_id}: {json.dumps(result_payload, sort_keys=True)}"
            )
            continue
        response = result_payload["message"]
        calls = function_calls(response, examples_by_id[example_id], model)
        eval_result = bfcl_ast_result(examples_by_id[example_id], calls, model)
        predictions.append(
            {
                "id": example_id,
                "model": model.id,
                "provider": model.provider,
                "temperature": model.temperature,
                "prediction": calls,
                "correct": eval_result["valid"],
                "evaluator": "bfcl_ast_checker",
                "eval_result": eval_result,
                "response_id": response.get("id"),
                "usage": response.get("usage"),
                "input_fingerprint": expected_fingerprints[example_id],
            }
        )

    for prediction in predictions:
        cached_predictions[prediction["id"]] = prediction
    write_jsonl(model_predictions_path, list(cached_predictions.values()))

    for example_id in failed_example_ids:
        prediction = run_model_prediction(examples_by_id[example_id], model)
        prediction["input_fingerprint"] = expected_fingerprints[example_id]
        cached_predictions[example_id] = prediction
        append_jsonl(model_predictions_path, prediction)
        print(f"Retried failed batch request synchronously for {example_id}")

    state_path = anthropic_batch_state_path(model_predictions_path)
    state["status"] = "completed"
    state["completed_at"] = utc_now()
    state["response"] = batch_response
    state["failed_request_count"] = len(failed_example_ids)
    write_json(state_path, state)
    print(
        f"Loaded {len(predictions)} predictions from Anthropic batch {state['batch_id']}"
    )
    return cached_predictions


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


def chat_response_function_calls(
    response: dict[str, object], name_map: dict[str, str]
) -> list[dict[str, object]]:
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return []
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    calls = []
    for item in tool_calls:
        if not isinstance(item, dict):
            continue
        function = item.get("function", {})
        if not isinstance(function, dict):
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"__malformed_arguments__": function.get("arguments")}
        name = str(function.get("name"))
        calls.append({"name": name_map.get(name, name), "arguments": arguments})
    return calls


def anthropic_response_function_calls(
    response: dict[str, object],
    name_map: dict[str, str],
    argument_schema_maps: dict[str, object],
) -> list[dict[str, object]]:
    calls = []
    content = response.get("content", [])
    if not isinstance(content, list):
        return calls
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool_use":
            continue
        arguments = item.get("input", {})
        if not isinstance(arguments, dict):
            arguments = {"__non_dict_arguments__": arguments}
        name = str(item.get("name"))
        restored_arguments = restore_anthropic_arguments(
            arguments,
            argument_schema_maps.get(name, {}),
        )
        calls.append({"name": name_map.get(name, name), "arguments": restored_arguments})
    return calls


def function_calls(
    response: dict[str, object],
    example: dict[str, object],
    model: ModelRun,
) -> list[dict[str, object]]:
    name_map = tool_name_map(example)
    if model.provider == "openai":
        return response_function_calls(response, name_map)
    if model.provider in {"anthropic", "claude"}:
        return anthropic_response_function_calls(
            response,
            name_map,
            anthropic_argument_schema_maps(example),
        )
    if model.provider in {"xai", "grok", "openrouter"}:
        return chat_response_function_calls(response, name_map)
    raise SystemExit(f"Unsupported evaluation provider for {model.id}: {model.provider}")


def load_bfcl_ast_checker() -> tuple[object, object]:
    eval_root = str(bfcl_eval_root())
    if eval_root not in sys.path:
        sys.path.insert(0, eval_root)

    # ast_checker only needs this mapping for dot/underscore function-name conversion.
    # Importing the full upstream model registry pulls every provider SDK.
    if "bfcl_eval.constants.model_config" not in sys.modules:
        model_config = types.ModuleType("bfcl_eval.constants.model_config")
        model_config.MODEL_CONFIG_MAPPING = {}
        sys.modules["bfcl_eval.constants.model_config"] = model_config
    sys.modules["bfcl_eval.constants.model_config"].MODEL_CONFIG_MAPPING.update(
        {
            model.id: types.SimpleNamespace(underscore_to_dot=False)
            for model in configured_model_runs()
        }
    )

    ast_checker = importlib.import_module("bfcl_eval.eval_checker.ast_eval.ast_checker").ast_checker
    language = importlib.import_module("bfcl_eval.constants.enums").Language.PYTHON
    return ast_checker, language


def bfcl_model_output(calls: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{call["name"]: call["arguments"]} for call in calls]


def bfcl_ast_result(
    example: dict[str, object], calls: list[dict[str, object]], model: ModelRun
) -> dict[str, object]:
    ast_checker, language = load_bfcl_ast_checker()
    return ast_checker(
        example["function"],
        bfcl_model_output(calls),
        example["ground_truth"],
        language,
        str(example["category"]),
        model.id,
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


def input_fingerprint(example: dict[str, object], model: ModelRun) -> str:
    return stable_hash(
        {
            "model": model.id,
            "provider": model.provider,
            "temperature": model.temperature,
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
    examples: list[dict[str, object]], model_predictions_path: Path, model: ModelRun
) -> list[dict[str, object]]:
    expected_fingerprints = {
        example["id"]: input_fingerprint(example, model) for example in examples
    }
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
            f"Loaded {len(cached_predictions)} cached {model.id} predictions "
            f"({stale_count} stale ignored)"
        )
    else:
        cached_predictions = {}

    missing_examples = [example for example in examples if example["id"] not in cached_predictions]
    if missing_examples:
        if use_batch_backend(model):
            print(
                f"Running {len(missing_examples)} missing {model.id} calls "
                "through Anthropic batch"
            )
            cached_predictions = run_or_load_anthropic_batch_predictions(
                missing_examples,
                model_predictions_path,
                model,
                expected_fingerprints,
                cached_predictions,
            )
        else:
            concurrency = min(openai_concurrency(), len(missing_examples))
            print(
                f"Running {len(missing_examples)} missing {model.id} calls "
                f"at concurrency {concurrency}"
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(run_model_prediction, example, model): example
                    for example in missing_examples
                }
                for future in concurrent.futures.as_completed(futures):
                    example = futures[future]
                    prediction = future.result()
                    prediction["input_fingerprint"] = expected_fingerprints[example["id"]]
                    cached_predictions[example["id"]] = prediction
                    append_jsonl(model_predictions_path, prediction)
                    print(f"Ran {model.id} on {example['id']}")
    else:
        print(f"All {model.id} predictions were cached")

    missing_ids = [example["id"] for example in examples if example["id"] not in cached_predictions]
    if missing_ids:
        raise SystemExit(f"Missing predictions after model run: {missing_ids[:5]}")

    predictions = [cached_predictions[example["id"]] for example in examples]
    examples_by_id = {example["id"]: example for example in examples}
    rescored_predictions = []
    for prediction in predictions:
        calls = prediction["prediction"]
        eval_result = bfcl_ast_result(examples_by_id[prediction["id"]], calls, model)
        rescored_prediction = dict(prediction)
        rescored_prediction["correct"] = eval_result["valid"]
        rescored_prediction["evaluator"] = "bfcl_ast_checker"
        rescored_prediction["eval_result"] = eval_result
        rescored_prediction["input_fingerprint"] = expected_fingerprints[prediction["id"]]
        rescored_predictions.append(rescored_prediction)
    write_jsonl(model_predictions_path, rescored_predictions)
    return rescored_predictions


def result_suffix() -> str:
    parts = []
    configured_suffix = os.environ.get("REALISTIC_BFCL_RESULT_SUFFIX", "").strip()
    if configured_suffix:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", configured_suffix):
            raise SystemExit(
                "REALISTIC_BFCL_RESULT_SUFFIX may contain only letters, numbers, "
                "underscore, dash, and dot."
            )
        parts.append(configured_suffix)
    limit = optional_positive_int_env("REALISTIC_BFCL_EVAL_LIMIT")
    if limit is not None:
        parts.append(f"limit_{limit}")
    return "_".join(parts)


def load_current_clean_predictions(
    clean_predictions_path: Path, base_ids: set[str], model: ModelRun
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
        example_id: input_fingerprint(example, model)
        for example_id, example in clean_examples.items()
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


def run_model_prediction(example: dict[str, object], model: ModelRun) -> dict[str, object]:
    response = call_tool_router(example, model)
    calls = function_calls(response, example, model)
    eval_result = bfcl_ast_result(example, calls, model)
    return {
        "id": example["id"],
        "model": model.id,
        "provider": model.provider,
        "temperature": model.temperature,
        "prediction": calls,
        "correct": eval_result["valid"],
        "evaluator": "bfcl_ast_checker",
        "eval_result": eval_result,
        "response_id": response.get("id"),
        "usage": response.get("usage"),
    }


def clean_examples_for_current_run() -> list[dict[str, object]]:
    subset_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    examples = read_jsonl(subset_path)
    limit = optional_positive_int_env("REALISTIC_BFCL_EVAL_LIMIT")
    if limit is None:
        return examples

    dimensions = generated_dimensions()
    base_ids: set[str] = set()
    for dimension in dimensions:
        noisy_path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"
        base_ids.update(str(row["base_id"]) for row in read_jsonl(noisy_path)[:limit])
    if not base_ids:
        return examples[:limit]

    filtered_examples = [example for example in examples if str(example["id"]) in base_ids]
    print(
        f"Limiting clean baseline to {len(filtered_examples)} base examples "
        f"needed by paired eval limit {limit}"
    )
    return filtered_examples


def freeze_bfcl() -> None:
    project_config = REPO_ROOT / "configs/project.yaml"
    subset_config = Path(
        os.environ.get("REALISTIC_BFCL_SUBSET_CONFIG", "configs/subsets/smoke.yaml")
    )
    if not subset_config.is_absolute():
        subset_config = REPO_ROOT / subset_config
    manifest_path = REPO_ROOT / "artifacts/frozen/bfcl_manifest.json"
    reject_placeholders((project_config, subset_config))
    categories = read_list_setting(subset_config, "bfcl_categories")
    subset_path = materialize_subset(subset_config, manifest_path)
    config_path = subset_config.relative_to(REPO_ROOT).as_posix()

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
                "config_path": config_path,
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
                "models": [
                    {
                        "id": model.id,
                        "provider": model.provider,
                        "tier": model.tier,
                        "temperature": model.temperature,
                    }
                    for model in configured_model_runs()
                ],
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
    clean_results_dir = REPO_ROOT / "artifacts/results/clean"
    suffix = result_suffix()
    if suffix:
        clean_results_dir = clean_results_dir / suffix
    result_path = clean_results_dir / "clean_baseline_summary.json"
    oracle_predictions_path = clean_results_dir / "oracle_replay_predictions.jsonl"
    models = configured_model_runs()

    if not manifest_path.exists():
        raise SystemExit("Missing artifacts/frozen/bfcl_manifest.json. Run prepare-subset first.")
    if not subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = clean_examples_for_current_run()
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
    oracle_metrics = accuracy_metrics(predictions)
    oracle_metrics["usage"] = {}
    model_metrics = {}
    model_category_metrics = {}
    model_prediction_paths = {}
    for model in models:
        model_predictions_path = clean_results_dir / f"{model.filename}_predictions.jsonl"
        model_predictions = run_or_load_model_predictions(
            examples, model_predictions_path, model
        )
        metrics = accuracy_metrics(model_predictions)
        metrics["usage"] = aggregate_usage(model_predictions)
        model_metrics[model.id] = metrics
        model_category_metrics[model.id] = category_metrics(model_predictions, examples_by_id)
        model_prediction_paths[model.id] = model_predictions_path.relative_to(
            REPO_ROOT
        ).as_posix()

    write_json(
        result_path,
        {
            "created_at": utc_now(),
            "stage": "run-bfcl",
            "status": "ran_model_baseline",
            "reason": "Ran oracle replay and configured model baselines.",
            "bfcl_manifest": "artifacts/frozen/bfcl_manifest.json",
            "bfcl_dataset_commit": manifest["bfcl"]["dataset_commit"],
            "predictions": {
                "oracle_replay": oracle_predictions_path.relative_to(REPO_ROOT).as_posix(),
                **model_prediction_paths,
            },
            "models": ["oracle_replay", *[model.id for model in models]],
            "metrics": {
                "oracle_replay": oracle_metrics,
                **model_metrics,
            },
            "category_metrics": {
                "oracle_replay": category_metrics(predictions, examples_by_id),
                **model_category_metrics,
            },
            "temperature": models[0].temperature if models else None,
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


def paired_eval_dimension(dimension: str, model: ModelRun) -> dict[str, object]:
    noisy_path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"
    clean_results_dir = REPO_ROOT / "artifacts/results/clean"
    suffix = result_suffix()
    if suffix:
        clean_results_dir = clean_results_dir / suffix
    clean_predictions_path = clean_results_dir / f"{model.filename}_predictions.jsonl"
    limit = optional_positive_int_env("REALISTIC_BFCL_EVAL_LIMIT")
    noisy_results_dir = REPO_ROOT / f"artifacts/results/noisy/{dimension}"
    paired_results_dir = REPO_ROOT / f"artifacts/results/paired/{dimension}"
    if suffix:
        noisy_results_dir = noisy_results_dir / suffix
        paired_results_dir = paired_results_dir / suffix
    noisy_predictions_path = noisy_results_dir / f"{model.filename}_predictions.jsonl"
    paired_path = paired_results_dir / f"{model.filename}_paired.jsonl"
    summary_path = paired_results_dir / f"{model.filename}_summary.json"

    if not clean_predictions_path.exists():
        raise SystemExit("Missing clean model predictions. Run run-bfcl first.")

    noisy_examples = read_jsonl(noisy_path)
    if limit is not None:
        noisy_examples = noisy_examples[:limit]
        print(
            f"Limiting paired evaluation for {dimension} to {len(noisy_examples)} "
            f"examples under {suffix}"
        )
    clean_predictions = load_current_clean_predictions(
        clean_predictions_path,
        {str(noisy_example["base_id"]) for noisy_example in noisy_examples},
        model,
    )
    noisy_predictions = run_or_load_model_predictions(
        noisy_examples, noisy_predictions_path, model
    )
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
            "model": model.id,
            "provider": model.provider,
            "tier": model.tier,
            "temperature": model.temperature,
            "dimension": dimension,
            "clean_predictions": clean_predictions_path.relative_to(REPO_ROOT).as_posix(),
            "noisy_predictions": noisy_predictions_path.relative_to(REPO_ROOT).as_posix(),
            "paired_results": paired_path.relative_to(REPO_ROOT).as_posix(),
            "eval_limit": limit,
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
    suffix = result_suffix()
    limit = optional_positive_int_env("REALISTIC_BFCL_EVAL_LIMIT")
    models = configured_model_runs()
    summary_name = f"summary_{suffix}.json" if suffix else "summary.json"
    summaries = [
        paired_eval_dimension(dimension, model)
        for model in models
        for dimension in dimensions
    ]
    write_json(
        REPO_ROOT / f"artifacts/results/paired/{summary_name}",
        {
            "created_at": utc_now(),
            "stage": "run-bfcl",
            "models": [
                {
                    "id": model.id,
                    "provider": model.provider,
                    "tier": model.tier,
                    "temperature": model.temperature,
                }
                for model in models
            ],
            "eval_limit": limit,
            "dimensions": summaries,
        },
    )


def run_bfcl() -> None:
    clean_baseline()
    paired_eval()
