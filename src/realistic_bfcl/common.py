from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
OPENAI_MODEL = "gpt-5.4-nano"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_CONCURRENCY = 8
OPENAI_MAX_ATTEMPTS = 8
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
    "live_simple": (
        "BFCL_v4_live_simple.json",
        "possible_answer/BFCL_v4_live_simple.json",
    ),
    "live_multiple": (
        "BFCL_v4_live_multiple.json",
        "possible_answer/BFCL_v4_live_multiple.json",
    ),
    "live_parallel": (
        "BFCL_v4_live_parallel.json",
        "possible_answer/BFCL_v4_live_parallel.json",
    ),
    "live_parallel_multiple": (
        "BFCL_v4_live_parallel_multiple.json",
        "possible_answer/BFCL_v4_live_parallel_multiple.json",
    ),
}
DIMENSION_FILES = {
    "typos": "typos.jsonl",
    "cursing": "cursing.jsonl",
    "irrelevant_context": "irrelevant_context.jsonl",
    "removed_spaces": "removed_spaces.jsonl",
    "argumentative_challenge": "argumentative_challenge.jsonl",
    "llm_work_context": "llm_work_context.jsonl",
    "llm_prior_thread": "llm_prior_thread.jsonl",
    "llm_conversation_history": "llm_conversation_history.jsonl",
}


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


def compact_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
