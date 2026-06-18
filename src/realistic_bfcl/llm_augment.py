from __future__ import annotations

import concurrent.futures
import csv
import json
import os
from dataclasses import dataclass

from .augment import (
    literal_visible_in_text,
    numeric_tokens,
    primitive_gold_values,
    quoted_literals,
)
from .common import (
    DIMENSION_FILES,
    OPENAI_MODEL,
    REPO_ROOT,
    conversation_text,
    openai_api_key,
    openai_concurrency,
    optional_positive_int_env,
    read_jsonl,
    write_jsonl,
)
from .evaluate import openai_retry_json


@dataclass(frozen=True)
class LlmDimension:
    name: str
    suffix: str
    instruction: str


LLM_DIMENSIONS = (
    LlmDimension(
        name="llm_work_context",
        suffix="llm_work",
        instruction=(
            "Rewrite the request as one realistic workplace message. Add specific but "
            "irrelevant operational context, product/workflow details, and light urgency. "
            "The final actionable request must still be exactly the original request."
        ),
    ),
    LlmDimension(
        name="llm_prior_thread",
        suffix="llm_thread",
        instruction=(
            "Rewrite the request as a pasted prior ticket/thread followed by the user's "
            "current request. The earlier ticket can contain stale context and irrelevant "
            "entities, but it must explicitly be background. The final current request "
            "must still be exactly the original request."
        ),
    ),
    LlmDimension(
        name="llm_conversation_history",
        suffix="llm_history",
        instruction=(
            "Create a short multi-turn conversation history before the final user turn. "
            "Earlier turns should include realistic stale facts, partial progress, or "
            "discarded side tasks. The last user message must clearly ask for exactly the "
            "original request and must determine the tool call."
        ),
    ),
)

SENSITIVE_SLOT_TERMS = {
    "accepts_insurance": ("insurance", "insured", "self-pay", "self pay"),
    "accessibility": ("accessibility", "accessible", "wheelchair", "mobility"),
    "include_disabled": ("disabled", "enabled", "active", "inactive"),
    "include_hidden": ("hidden", "dotfile", "dot file", "dotfiles"),
    "include_images": ("image", "images", "photo", "photos", "picture", "pictures"),
    "language": ("language", "english", "spanish", "french", "german", "hebrew"),
    "number_of_rooms": ("room", "rooms", "bedroom", "bedrooms"),
    "output_format": ("format", "json", "csv", "markdown", "xml"),
    "price": ("price", "budget", "cost", "cheap", "affordable", "expensive", "free", "$"),
    "private_visibility": ("private", "privacy", "public", "visibility"),
    "rating": ("rating", "rated", "stars", "review score", "reviews"),
    "recursive": ("recursive", "recursively", "subdirectory", "subdirectories"),
    "retry_attempts": ("retry", "retries", "attempts"),
    "review_score": ("rating", "rated", "stars", "review score", "reviews"),
    "smoking_allowed": ("smoking", "smoke", "non-smoking", "nonsmoking"),
    "timeout": ("timeout", "time out", "timed out"),
    "use_ssl": ("ssl", "tls", "encrypted", "encryption"),
}


def llm_augment_model() -> str:
    return os.environ.get("REALISTIC_BFCL_LLM_AUGMENT_MODEL", OPENAI_MODEL)


def output_text(response: dict[str, object]) -> str:
    parts = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "\n".join(parts).strip()


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped.removeprefix("```json").strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```").strip()
    if stripped.endswith("```"):
        stripped = stripped[: -len("```")].strip()
    return stripped


def schema_property_names(example: dict[str, object]) -> set[str]:
    names: set[str] = set()
    functions = example.get("function", [])
    if not isinstance(functions, list):
        return names

    for function in functions:
        if not isinstance(function, dict):
            continue
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            continue
        names.update(str(name) for name in properties)

    return names


def forbidden_active_slot_terms(
    example: dict[str, object],
    clean_prompt: str,
) -> dict[str, tuple[str, ...]]:
    lowered_clean = clean_prompt.lower()
    forbidden = {}
    for property_name in sorted(schema_property_names(example)):
        terms = SENSITIVE_SLOT_TERMS.get(property_name)
        if not terms:
            continue
        absent_terms = tuple(term for term in terms if term.lower() not in lowered_clean)
        if absent_terms:
            forbidden[property_name] = absent_terms
    return forbidden


def augmentation_prompt(example: dict[str, object], dimension: LlmDimension) -> str:
    function_names = [function["name"] for function in example["function"]]
    clean_prompt = conversation_text(example["question"])
    forbidden_terms = forbidden_active_slot_terms(example, clean_prompt)
    return json.dumps(
        {
            "task": "Generate an oracle-preserving Realistic-BFCL augmentation.",
            "dimension": dimension.name,
            "dimension_instruction": dimension.instruction,
            "clean_prompt": clean_prompt,
            "gold_tool_calls": example["ground_truth"],
            "available_function_names": function_names,
            "forbidden_active_slot_terms": forbidden_terms,
            "hard_rules": [
                "Preserve the final BFCL oracle: same function names, same "
                "argument values, same number of calls.",
                "Do not change any active number, date, name, location, ID, "
                "quoted string, unit, or boolean from the clean prompt.",
                "You may add distractor entities or numbers only if they are "
                "explicitly stale, background, cancelled, or irrelevant.",
                "Do not add active constraints such as new filters, output "
                "formats, privacy settings, locations, dates, counts, or preferences.",
                "For multi-turn output, the final user message must be the actionable request.",
                "Keep the text realistic, not adversarial puzzle text.",
                "Do not mention BFCL, benchmark, oracle, gold, original request, "
                "tool call, function call, or this augmentation task in the messages.",
                "Do not tell the assistant which tool/function to call unless the "
                "clean prompt already says that in user-facing language.",
                "The messages must read like normal user/assistant chat, not benchmark "
                "instructions or developer instructions.",
                "Do not mention forbidden_active_slot_terms anywhere in the messages, "
                "even as stale context. Those terms correspond to available schema slots "
                "that are not active in the clean request.",
            ],
            "output_schema": {
                "messages": [
                    {
                        "role": "user | assistant",
                        "content": "message text",
                    }
                ],
                "oracle_preservation_notes": "short explanation",
                "distractors_added": ["short descriptions"],
                "risk_flags": ["possible risks, or empty list"],
            },
        },
        ensure_ascii=False,
    )


def call_llm_augmenter(example: dict[str, object], dimension: LlmDimension) -> dict[str, object]:
    payload = {
        "model": llm_augment_model(),
        "input": [
            {
                "role": "system",
                "content": (
                    "You generate realistic benchmark augmentations. Return only valid "
                    "JSON matching the requested schema."
                ),
            },
            {"role": "user", "content": augmentation_prompt(example, dimension)},
        ],
        "max_output_tokens": 1200,
    }
    response = openai_retry_json(payload, openai_api_key())
    text = strip_json_fence(output_text(response))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"LLM augmentation returned invalid JSON: {text}") from error
    return parsed


def validate_llm_messages(
    example: dict[str, object],
    messages: list[dict[str, str]],
) -> list[str]:
    clean_prompt = conversation_text(example["question"])
    augmented_text = "\n".join(message["content"] for message in messages)
    lowered_text = augmented_text.lower()
    lowered_clean = clean_prompt.lower()
    reasons = []

    for number in numeric_tokens(clean_prompt):
        if number not in numeric_tokens(augmented_text):
            reasons.append(f"clean numeric token missing from augmentation: {number!r}")

    for quoted in quoted_literals(clean_prompt):
        if quoted not in quoted_literals(augmented_text):
            reasons.append(f"clean quoted literal missing from augmentation: {quoted!r}")

    for literal in primitive_gold_values(example["ground_truth"]):
        if literal_visible_in_text(literal, clean_prompt) and not literal_visible_in_text(
            literal,
            augmented_text,
        ):
            reasons.append(f"visible gold literal missing from augmentation: {literal!r}")

    for property_name, terms in forbidden_active_slot_terms(example, clean_prompt).items():
        for term in terms:
            if term.lower() in lowered_text:
                reasons.append(
                    "augmentation introduced inactive schema slot term "
                    f"{term!r} for property {property_name!r}"
                )

    if not messages:
        reasons.append("augmentation returned no messages")
    elif messages[-1]["role"] != "user":
        reasons.append("final message is not a user turn")

    meta_terms = (
        "bfcl",
        "benchmark",
        "oracle",
        "gold",
        "original request",
        "same call",
        "tool call",
        "tool",
        "function call",
        "augmentation",
    )
    for term in meta_terms:
        if term in lowered_text and term not in lowered_clean:
            reasons.append(f"benchmark meta-language leaked into prompt: {term!r}")

    return reasons


def normalized_messages(payload: dict[str, object]) -> list[dict[str, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError("LLM augmentation missing messages list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise RuntimeError("LLM augmentation message is not an object")
        role = str(message.get("role", "user")).lower()
        if role not in {"user", "assistant"}:
            raise RuntimeError(f"Unsupported LLM augmentation role: {role}")
        content = str(message.get("content", "")).strip()
        if not content:
            raise RuntimeError("LLM augmentation message has empty content")
        normalized.append({"role": role, "content": content})
    return normalized


def generate_dimension(dimension: LlmDimension, examples: list[dict[str, object]]) -> None:
    output_path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension.name]}"
    rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []

    def generate_one(index_and_example: tuple[int, dict[str, object]]) -> dict[str, object]:
        index, example = index_and_example
        last_errors = []
        for attempt in range(1, 13):
            try:
                payload = call_llm_augmenter(example, dimension)
            except RuntimeError as error:
                last_errors = [str(error)]
                continue
            try:
                messages = normalized_messages(payload)
            except RuntimeError as error:
                last_errors = [str(error)]
                continue
            validation_errors = validate_llm_messages(example, messages)
            if not validation_errors:
                return {
                    "index": index,
                    "example": example,
                    "messages": messages,
                    "payload": payload,
                    "validation_errors": [],
                    "attempt": attempt,
                }
            last_errors = validation_errors
        return {
            "index": index,
            "example": example,
            "messages": [],
            "payload": {},
            "validation_errors": last_errors,
            "attempt": 12,
        }

    indexed_examples = list(enumerate(examples))
    concurrency = min(openai_concurrency(), len(indexed_examples))
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(generate_one, item) for item in indexed_examples]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            example = result["example"]
            if result["validation_errors"]:
                review_rows.append(
                    {
                        "dimension": dimension.name,
                        "base_id": example["id"],
                        "category": example["category"],
                        "clean_prompt": conversation_text(example["question"]),
                        "augmented_prompt": "",
                        "function_names": ", ".join(
                            str(function["name"]) for function in example["function"]
                        ),
                        "ground_truth": json.dumps(
                            example["ground_truth"],
                            ensure_ascii=False,
                        ),
                        "oracle_preservation_notes": "",
                        "distractors_added": "[]",
                        "risk_flags": json.dumps(
                            result["validation_errors"],
                            ensure_ascii=False,
                        ),
                        "review_status": "rejected_auto",
                    }
                )
                print(
                    f"Skipped {dimension.name} for {example['id']}: "
                    + "; ".join(result["validation_errors"])
                )
                continue
            question = [result["messages"]]
            rows.append(
                {
                    "id": f"{example['id']}__{dimension.suffix}",
                    "base_id": example["id"],
                    "category": example["category"],
                    "dimension": dimension.name,
                    "question": question,
                    "function": example["function"],
                    "ground_truth": example["ground_truth"],
                    "oracle_preservation": {
                        "function_schema_unchanged": True,
                        "ground_truth_unchanged": True,
                        "llm_notes": result["payload"].get("oracle_preservation_notes", ""),
                        "distractors_added": result["payload"].get("distractors_added", []),
                        "risk_flags": result["payload"].get("risk_flags", []),
                    },
                }
            )
            review_rows.append(
                {
                    "dimension": dimension.name,
                    "base_id": example["id"],
                    "category": example["category"],
                    "clean_prompt": conversation_text(example["question"]),
                    "augmented_prompt": conversation_text(question),
                    "function_names": ", ".join(
                        str(function["name"]) for function in example["function"]
                    ),
                    "ground_truth": json.dumps(example["ground_truth"], ensure_ascii=False),
                    "oracle_preservation_notes": result["payload"].get(
                        "oracle_preservation_notes",
                        "",
                    ),
                    "distractors_added": json.dumps(
                        result["payload"].get("distractors_added", []),
                        ensure_ascii=False,
                    ),
                    "risk_flags": json.dumps(
                        result["payload"].get("risk_flags", []),
                        ensure_ascii=False,
                    ),
                    "review_status": "needs_review",
                }
            )
            print(f"Generated {dimension.name} for {example['id']}")

    rows.sort(key=lambda row: str(row["base_id"]))
    write_jsonl(output_path, rows)
    print(f"Wrote {output_path.relative_to(REPO_ROOT)}")

    review_path = REPO_ROOT / f"artifacts/generated/{dimension.name}_review.csv"
    review_rows.sort(key=lambda row: str(row["base_id"]))
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dimension",
                "base_id",
                "category",
                "clean_prompt",
                "augmented_prompt",
                "function_names",
                "ground_truth",
                "oracle_preservation_notes",
                "distractors_added",
                "risk_flags",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerows(review_rows)
    print(f"Wrote {review_path.relative_to(REPO_ROOT)}")


def augment_llm_pilot() -> None:
    subset_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    if not subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")

    examples = select_llm_examples(read_jsonl(subset_path))
    limit = optional_positive_int_env("REALISTIC_BFCL_LLM_AUGMENT_LIMIT") or 50
    examples = examples[:limit]
    print(
        f"Generating LLM pilot augmentations for {len(examples)} examples "
        f"with {llm_augment_model()}"
    )
    requested = os.environ.get("REALISTIC_BFCL_LLM_DIMENSIONS")
    dimensions = LLM_DIMENSIONS
    if requested:
        requested_names = {name.strip() for name in requested.split(",") if name.strip()}
        dimensions = tuple(
            dimension for dimension in LLM_DIMENSIONS if dimension.name in requested_names
        )
        missing = requested_names - {dimension.name for dimension in dimensions}
        if missing:
            raise SystemExit(f"Unknown LLM augmentation dimensions: {sorted(missing)}")
    for dimension in dimensions:
        generate_dimension(dimension, examples)


def select_llm_examples(examples: list[dict[str, object]]) -> list[dict[str, object]]:
    selection = os.environ.get("REALISTIC_BFCL_LLM_SELECTION", "first").strip()
    if selection == "first":
        return examples
    if selection == "hard_many_tools":
        priority = {
            "live_parallel_multiple": 0,
            "live_parallel": 1,
            "live_multiple": 2,
            "parallel_multiple": 3,
            "parallel": 4,
            "multiple": 5,
            "live_simple": 6,
            "simple_python": 7,
        }
        return sorted(
            examples,
            key=lambda example: (
                -len(example["function"]),
                priority.get(str(example["category"]), 99),
                str(example["id"]),
            ),
        )
    raise SystemExit(
        "REALISTIC_BFCL_LLM_SELECTION must be one of: first, hard_many_tools."
    )
