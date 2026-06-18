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
ROUTER_SYSTEM_INSTRUCTION = (
    "Call the provided tool that best satisfies the user request. "
    "Do not answer in prose when a tool call is appropriate."
)
ROUTER_TOOL_CHOICE = "required"
ROUTER_MAX_OUTPUT_TOKENS = 256
ROUTER_MESSAGE_SERIALIZATION = "preserve_bfcl_turns_v1"
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
DIMENSION_FILES = {
    "typos": "typos.jsonl",
    "cursing": "cursing.jsonl",
    "irrelevant_context": "irrelevant_context.jsonl",
    "removed_spaces": "removed_spaces.jsonl",
    "argumentative_challenge": "argumentative_challenge.jsonl",
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
        name="prepare-subset",
        purpose="Freeze the reproducible BFCL clean subset used as the substrate.",
        inputs=("configs/project.yaml", "configs/subsets/smoke.yaml"),
        outputs=(
            "artifacts/frozen/bfcl_manifest.json",
            "artifacts/frozen/clean_subset.jsonl",
        ),
        next_action="Run augment to construct the oracle-preserving noisy dataset.",
    ),
    Stage(
        name="augment",
        purpose="Construct the frozen noisy dataset with oracle-preserving transforms.",
        inputs=("artifacts/frozen/bfcl_manifest.json", "configs/realism_dimensions.yaml"),
        outputs=(
            "artifacts/generated/typos.jsonl",
            "artifacts/generated/cursing.jsonl",
            "artifacts/generated/irrelevant_context.jsonl",
            "artifacts/generated/removed_spaces.jsonl",
            "artifacts/generated/argumentative_challenge.jsonl",
            "artifacts/generated/augmentation_review.csv",
        ),
        next_action="Run BFCL clean/noisy paired evaluation on the frozen dataset.",
    ),
    Stage(
        name="run-bfcl",
        purpose="Run clean and noisy BFCL-style evaluation with identical schemas.",
        inputs=("artifacts/frozen/clean_subset.jsonl", "artifacts/generated/"),
        outputs=(
            "artifacts/results/clean/",
            "artifacts/results/noisy/",
            "artifacts/results/paired/",
        ),
        next_action="Analyze paired clean-vs-noisy degradation and failure types.",
    ),
    Stage(
        name="analyze",
        purpose="Compute degradation metrics and error taxonomy.",
        inputs=("artifacts/results/paired/",),
        outputs=(
            "artifacts/analysis/benchmark_summary.csv",
            "artifacts/analysis/benchmark_summary.json",
            "artifacts/analysis/flip_review.csv",
            "artifacts/analysis/regression_review.csv",
        ),
        next_action=(
            "Use adjusted regression metrics to decide whether the pilot is ready to scale."
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
    print(f"Step: {stage.name}")
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


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def optional_positive_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise SystemExit(f"{name} must be an integer.") from error
    if parsed < 1:
        raise SystemExit(f"{name} must be >= 1.")
    return parsed


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


Cursing = str

CURSING_TEMPLATES: tuple[Cursing, ...] = (
    "for fuck sake {prompt}",
    "holy shit just {prompt}",
    "this is annoying, {prompt}",
    "goddamn it {prompt}",
    "i'm tired of this, {prompt}",
    "ffs {prompt}",
)
IRRELEVANT_CONTEXT_TEMPLATES = (
    "it was horrible today but anyway, {prompt}",
    "my commute was a mess. unrelated, {prompt}",
    "i'm in the middle of three things right now; {prompt}",
    "the meeting earlier was useless, but can you do this: {prompt}",
    "my laptop is being weird today. {prompt}",
    "i spilled coffee and lost my notes, so {prompt}",
)
ARGUMENTATIVE_TEMPLATES = (
    "you are wrong all the time, what do you really think is {prompt}",
    "last time you messed this up, so answer carefully: {prompt}",
    "i don't trust your first answer, but {prompt}",
    "prove you can actually do this: {prompt}",
    "you keep getting these wrong. {prompt}",
    "be honest and don't dodge it: {prompt}",
)
TYPO_REPLACEMENTS = (
    ("what", "wat"),
    ("please", "plese"),
    ("using", "useing"),
    ("given", "givn"),
    ("number", "numbr"),
    ("numbers", "numbrs"),
    ("calculate", "calcuate"),
    ("factorial", "factroial"),
    ("triangle", "traingle"),
    ("area", "aera"),
    ("height", "heigth"),
    ("circle", "circel"),
    ("radius", "raduis"),
    ("equation", "eqaution"),
    ("coefficients", "coeficients"),
    ("function", "funciton"),
    ("temperature", "temprature"),
    ("weather", "weahter"),
    ("distance", "distnace"),
    ("between", "betwen"),
    ("lengths", "lenghts"),
    ("hypotenuse", "hypotnuse"),
)


def lowercase_first_alpha(text: str) -> str:
    protected = quoted_literal_spans(text)
    for index, char in enumerate(text):
        if span_overlaps((index, index + 1), protected):
            continue
        if char.isalpha():
            return f"{text[:index]}{char.lower()}{text[index + 1 :]}"
    return text


def cursing_prompt(clean_prompt: str, index: int) -> str:
    template = CURSING_TEMPLATES[index % len(CURSING_TEMPLATES)]
    return template.format(prompt=lowercase_first_alpha(clean_prompt))


def irrelevant_context_prompt(clean_prompt: str, index: int) -> str:
    template = IRRELEVANT_CONTEXT_TEMPLATES[index % len(IRRELEVANT_CONTEXT_TEMPLATES)]
    return template.format(prompt=lowercase_first_alpha(clean_prompt))


def argumentative_prompt(clean_prompt: str, index: int) -> str:
    template = ARGUMENTATIVE_TEMPLATES[index % len(ARGUMENTATIVE_TEMPLATES)]
    return template.format(prompt=lowercase_first_alpha(clean_prompt))


def quoted_literal_spans(text: str) -> list[tuple[int, int]]:
    return [
        match.span()
        for match in re.finditer(r"(?<!\w)(['\"])(.*?)(?<!\\)\1(?!\w)", text)
    ]


def quoted_literals(text: str) -> list[str]:
    return [
        match.group(0)
        for match in re.finditer(r"(?<!\w)(['\"])(.*?)(?<!\\)\1(?!\w)", text)
    ]


def span_overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(
        start < protected_end and end > protected_start
        for protected_start, protected_end in spans
    )


def literal_spans(text: str, literal: str) -> list[tuple[int, int]]:
    if not literal:
        return []
    return [
        match.span()
        for match in re.finditer(re.escape(literal), text, flags=re.IGNORECASE)
    ]


def visible_gold_literal_spans(
    text: str, example: dict[str, object]
) -> list[tuple[int, int]]:
    spans = []
    for literal in primitive_gold_values(example["ground_truth"]):
        if not isinstance(literal, str):
            continue
        literal_text = literal.strip()
        if not literal_text:
            continue
        candidates = {literal_text, literal_text.replace("_", " ")}
        for candidate in candidates:
            spans.extend(literal_spans(text, candidate))
    return spans


def replace_first_unprotected_word(
    text: str,
    source: str,
    replacement: str,
    extra_protected: list[tuple[int, int]] | None = None,
) -> str:
    pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
    protected = quoted_literal_spans(text) + (extra_protected or [])
    for match in pattern.finditer(text):
        if span_overlaps(match.span(), protected):
            continue
        return f"{text[: match.start()]}{replacement}{text[match.end() :]}"
    return text


def typo_prompt(
    clean_prompt: str,
    index: int,
    protected_spans: list[tuple[int, int]] | None = None,
) -> str:
    text = clean_prompt
    start = index % len(TYPO_REPLACEMENTS)
    replacements_applied = 0
    for offset in range(len(TYPO_REPLACEMENTS)):
        source, replacement = TYPO_REPLACEMENTS[(start + offset) % len(TYPO_REPLACEMENTS)]
        updated = replace_first_unprotected_word(
            text,
            source,
            replacement,
            protected_spans,
        )
        if updated != text:
            text = updated
            replacements_applied += 1
            if replacements_applied == 2:
                return text
    if replacements_applied:
        return text
    return f"plese {lowercase_first_alpha(text)}"


def removed_spaces_prompt(clean_prompt: str, index: int) -> str:
    pairs = list(re.finditer(r"\b[A-Za-z]{2,}\s+[A-Za-z]{2,}\b", clean_prompt))
    protected = quoted_literal_spans(clean_prompt)
    pairs = [match for match in pairs if not span_overlaps(match.span(), protected)]
    if not pairs:
        return clean_prompt
    match = pairs[index % len(pairs)]
    return (
        clean_prompt[: match.start()]
        + match.group(0).replace(" ", "", 1)
        + clean_prompt[match.end() :]
    )


def transform_messages(question: object, index: int, transform: object) -> object:
    conversations = json.loads(json.dumps(question))
    first_message = conversations[0][0]
    first_message["content"] = transform(str(first_message["content"]), index)
    return conversations


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?(?![A-Za-z0-9.])", text)


def primitive_gold_values(value: object) -> list[object]:
    if isinstance(value, dict):
        values: list[object] = []
        for item in value.values():
            values.extend(primitive_gold_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(primitive_gold_values(item))
        return values
    if isinstance(value, (str, int, float, bool)):
        return [value]
    return []


def literal_visible_in_text(literal: object, text: str) -> bool:
    if isinstance(literal, bool):
        return str(literal).lower() in text.lower()
    if isinstance(literal, (int, float)):
        return str(literal) in numeric_tokens(text)
    literal_text = str(literal).strip()
    if not literal_text:
        return False
    return compact_text(literal_text) in compact_text(text)


def validate_augmented_prompt(
    example: dict[str, object],
    clean_prompt: str,
    noisy_prompt: str,
) -> list[str]:
    reasons = []
    clean_numbers = numeric_tokens(clean_prompt)
    noisy_numbers = numeric_tokens(noisy_prompt)
    if clean_numbers != noisy_numbers:
        reasons.append(
            f"numeric tokens changed from {clean_numbers!r} to {noisy_numbers!r}"
        )

    clean_quotes = quoted_literals(clean_prompt)
    noisy_quotes = quoted_literals(noisy_prompt)
    if clean_quotes != noisy_quotes:
        reasons.append(
            f"quoted literals changed from {clean_quotes!r} to {noisy_quotes!r}"
        )

    for literal in primitive_gold_values(example["ground_truth"]):
        if literal_visible_in_text(literal, clean_prompt) and not literal_visible_in_text(
            literal,
            noisy_prompt,
        ):
            reasons.append(f"gold literal no longer visible in noisy prompt: {literal!r}")
    return reasons


def augment_dimension(dimension: str, suffix: str, transform: object) -> None:
    subset_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    output_path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"

    if not subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")

    rows = []
    examples = read_jsonl(subset_path)
    limit = optional_positive_int_env("REALISTIC_BFCL_AUGMENT_LIMIT")
    if limit is not None:
        examples = examples[:limit]
        print(f"Limiting augmentation to first {len(examples)} examples")

    for index, example in enumerate(examples):
        if dimension == "typos":
            clean_prompt = conversation_text(example["question"])
            protected = visible_gold_literal_spans(clean_prompt, example)

            def protected_typo_prompt(
                prompt: str,
                prompt_index: int,
                protected_spans: list[tuple[int, int]] = protected,
            ) -> str:
                return typo_prompt(prompt, prompt_index, protected_spans)

            question = transform_messages(
                example["question"],
                index,
                protected_typo_prompt,
            )
        else:
            question = transform_messages(example["question"], index, transform)
        clean_prompt = conversation_text(example["question"])
        noisy_prompt = conversation_text(question)
        validation_errors = validate_augmented_prompt(example, clean_prompt, noisy_prompt)
        if validation_errors:
            joined_errors = "; ".join(validation_errors)
            raise RuntimeError(
                f"{dimension} augmentation changed oracle-bearing text for "
                f"{example['id']}: {joined_errors}"
            )
        rows.append(
            {
                "id": f"{example['id']}__{suffix}",
                "base_id": example["id"],
                "category": example["category"],
                "dimension": dimension,
                "question": question,
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


def augment_typos() -> None:
    augment_dimension("typos", "typos", typo_prompt)


def augment_cursing() -> None:
    augment_dimension("cursing", "cursing", cursing_prompt)


def augment_irrelevant_context() -> None:
    augment_dimension("irrelevant_context", "context", irrelevant_context_prompt)


def augment_removed_spaces() -> None:
    augment_dimension("removed_spaces", "spaces", removed_spaces_prompt)


def augment_argumentative() -> None:
    augment_dimension("argumentative_challenge", "argue", argumentative_prompt)


def augment() -> None:
    augment_typos()
    augment_cursing()
    augment_irrelevant_context()
    augment_removed_spaces()
    augment_argumentative()
    review_augmentations()


def prompt_text(example: dict[str, object]) -> str:
    return conversation_text(example["question"])


def review_augmentations() -> None:
    clean_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    output_path = REPO_ROOT / "artifacts/generated/augmentation_review.csv"

    if not clean_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")

    dimensions = (
        ("typos", "aug_typo"),
        ("cursing", "aug_cursing"),
        ("irrelevant_context", "aug_irrelevant_context"),
        ("removed_spaces", "aug_removed_spaces"),
        ("argumentative_challenge", "aug_argumentative"),
    )
    generated_by_dimension = {}
    for dimension, _column in dimensions:
        path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(REPO_ROOT)}. Run augment first.")
        generated_by_dimension[dimension] = {
            row["base_id"]: row for row in read_jsonl(path)
        }

    examples = read_jsonl(clean_path)
    limit = optional_positive_int_env("REALISTIC_BFCL_AUGMENT_LIMIT")
    if limit is not None:
        examples = examples[:limit]
        print(f"Limiting review CSV to first {len(examples)} examples")

    fieldnames = [
        "base_id",
        "category",
        "clean_prompt",
        "aug_typo",
        "aug_cursing",
        "aug_irrelevant_context",
        "aug_removed_spaces",
        "aug_argumentative",
        "function_names",
        "ground_truth",
    ]
    rows = []
    for example in examples:
        row = {
            "base_id": example["id"],
            "category": example["category"],
            "clean_prompt": prompt_text(example),
            "function_names": ", ".join(function["name"] for function in example["function"]),
            "ground_truth": json.dumps(example["ground_truth"], ensure_ascii=False),
        }
        for dimension, column in dimensions:
            augmented = generated_by_dimension[dimension][example["id"]]
            row[column] = prompt_text(augmented)
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
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
    expected_fingerprints = {
        example["id"]: input_fingerprint(example) for example in examples
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
        raise SystemExit(
            "Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first."
        )

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
                "examples_per_category": read_int_setting(
                    subset_config, "examples_per_category"
                ),
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
        raise SystemExit(
            "Missing artifacts/frozen/bfcl_manifest.json. Run prepare-subset first."
        )
    if not subset_path.exists():
        raise SystemExit(
            "Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first."
        )

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
            f"Requested dimensions are not generated: {missing_artifacts}. "
            "Run augment first."
        )
    return [dimension for dimension in dimensions if dimension in requested_dimensions]


def paired_eval_dimension(dimension: str) -> dict[str, object]:
    noisy_path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"
    clean_predictions_path = REPO_ROOT / f"artifacts/results/clean/{OPENAI_MODEL}_predictions.jsonl"
    noisy_predictions_path = (
        REPO_ROOT
        / f"artifacts/results/noisy/{dimension}/{OPENAI_MODEL}_predictions.jsonl"
    )
    paired_path = (
        REPO_ROOT
        / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_paired.jsonl"
    )
    summary_path = (
        REPO_ROOT
        / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_summary.json"
    )

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


def compact_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


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
        values.extend(
            str(value) for value in flat_values(arguments) if value not in ("", None)
        )
    return values


def typo_copied_into_argument(
    clean_prompt: str, noisy_prompt: str, noisy_prediction: object
) -> bool:
    clean_tokens = set(re.findall(r"[A-Za-z][A-Za-z']+", clean_prompt))
    noisy_tokens = set(re.findall(r"[A-Za-z][A-Za-z']+", noisy_prompt))
    introduced_tokens = {
        token.lower() for token in noisy_tokens - clean_tokens if len(token) >= 4
    }
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
                and (
                    compact_accepted in compact_noisy
                    or compact_noisy in compact_accepted
                )
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
    paired_path = (
        REPO_ROOT
        / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_paired.jsonl"
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
    paired_path = (
        REPO_ROOT
        / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_paired.jsonl"
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
                "clean_prediction": json.dumps(
                    review["clean_prediction"], sort_keys=True
                ),
                "noisy_prediction": json.dumps(
                    review["noisy_prediction"], sort_keys=True
                ),
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
            REPO_ROOT
            / f"artifacts/results/paired/{dimension}/{OPENAI_MODEL}_summary.json"
        )
        paired_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = paired_summary["metrics"]
        dimension_regressions = [
            row for row in regression_rows if row["dimension"] == dimension
        ]
        possible_oracle_issues = [
            row for row in dimension_regressions if row["oracle_issue"] == "possible"
        ]
        possible_augmentation_issues = [
            row
            for row in dimension_regressions
            if row["augmentation_issue"] == "possible"
        ]
        adjusted_regression_count = (
            int(metrics["clean_success_noisy_failure"]) - len(possible_oracle_issues)
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
                "clean_success_noisy_failure": metrics[
                    "clean_success_noisy_failure"
                ],
                "conditional_failure_given_clean_success": metrics[
                    "conditional_failure_given_clean_success"
                ],
                "raw_regression_count": metrics["clean_success_noisy_failure"],
                "possible_oracle_issue_regressions": len(possible_oracle_issues),
                "possible_augmentation_issue_regressions": len(
                    possible_augmentation_issues
                ),
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
                "adjusted_regression_count excludes only rows with "
                "oracle_issue=possible"
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
            "benchmark_summary_csv": benchmark_summary_csv_path.relative_to(
                REPO_ROOT
            ).as_posix(),
            "benchmark_summary_json": benchmark_summary_json_path.relative_to(
                REPO_ROOT
            ).as_posix(),
            "flip_review_csv": flip_review_path.relative_to(REPO_ROOT).as_posix(),
            "regression_review_csv": regression_review_path.relative_to(
                REPO_ROOT
            ).as_posix(),
        },
    )
    print(f"Wrote {benchmark_summary_csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {benchmark_summary_json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {flip_review_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {regression_review_path.relative_to(REPO_ROOT)}")


def run_stage(stage: Stage, dry_run: bool) -> None:
    describe_stage(stage)
    if dry_run:
        return
    if stage.name == "prepare-subset":
        freeze_bfcl()
        return
    if stage.name == "augment":
        augment()
        return
    if stage.name == "run-bfcl":
        run_bfcl()
        return
    if stage.name == "analyze":
        analyze()
        return
    raise SystemExit(f"Stage implementation missing: {stage.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Realistic-BFCL research steps.")
    parser.add_argument("stage", nargs="?", help="Step name to inspect or run.")
    parser.add_argument("--list", action="store_true", help="List available steps.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the step without running it.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_stages()
        return
    if not args.stage:
        parser.error("provide a step name or --list")

    run_stage(stage_by_name(args.stage), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
