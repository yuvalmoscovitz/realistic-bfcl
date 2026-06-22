from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
OPENAI_MODEL = "gpt-5.4-nano"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MESSAGE_BATCHES_URL = "https://api.anthropic.com/v1/messages/batches"
XAI_CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
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
    "profane_sandwich": "profane_sandwich.jsonl",
    "argumentative_sandwich": "argumentative_sandwich.jsonl",
    "distractor_sandwich": "distractor_sandwich.jsonl",
    "pasted_context_block": "pasted_context_block.jsonl",
    "telegraphic_request": "telegraphic_request.jsonl",
    "llm_work_context": "llm_work_context.jsonl",
    "llm_prior_thread": "llm_prior_thread.jsonl",
    "llm_conversation_history": "llm_conversation_history.jsonl",
    "llm_messy_pre_intent_history": "llm_messy_pre_intent_history.jsonl",
    "llm_profane_frustration": "llm_profane_frustration.jsonl",
    "llm_argumentative_challenge": "llm_argumentative_challenge.jsonl",
    "llm_frustrated_distractor_context": "llm_frustrated_distractor_context.jsonl",
    "llm_super_casual_abbreviations": "llm_super_casual_abbreviations.jsonl",
    "llm_frustrated_swearing": "llm_frustrated_swearing.jsonl",
    "llm_student_broke_context": "llm_student_broke_context.jsonl",
    "llm_typos_shorthand": "llm_typos_shorthand.jsonl",
    "llm_rambling_overexplaining": "llm_rambling_overexplaining.jsonl",
    "llm_impatient_direct_attitude": "llm_impatient_direct_attitude.jsonl",
    "llm_arguing_correcting_ai": "llm_arguing_correcting_ai.jsonl",
    "llm_confused_overwhelmed": "llm_confused_overwhelmed.jsonl",
    "llm_swearing_urgency_work": "llm_swearing_urgency_work.jsonl",
    "llm_vague_slightly_aggressive": "llm_vague_slightly_aggressive.jsonl",
}


@dataclass(frozen=True)
class ModelRun:
    id: str
    provider: str
    tier: str
    temperature: float

    @property
    def filename(self) -> str:
        return safe_model_filename(self.id)


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


def read_float_setting(path: Path, key: str, default: float) -> float:
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(prefix):
            return float(line.split(":", 1)[1].strip())
    return default


def read_model_setting(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    models: list[dict[str, str]] = []
    in_models = False
    current: dict[str, str] | None = None

    for line in lines:
        stripped = line.strip()
        if stripped == "models:":
            in_models = True
            current = None
            continue
        if not in_models:
            continue
        if stripped and not line.startswith(" ") and not stripped.startswith("- "):
            break
        if stripped.startswith("- "):
            if current:
                models.append(current)
            current = {}
            item = stripped[2:].strip()
            if item:
                if ":" in item:
                    key, value = item.split(":", 1)
                    current[key.strip()] = value.strip()
                else:
                    current["id"] = item
            continue
        if current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = value.strip()

    if current:
        models.append(current)
    return models


def safe_model_filename(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")


def configured_model_runs() -> list[ModelRun]:
    override = os.environ.get("REALISTIC_BFCL_MODELS")
    temperature = evaluation_temperature()
    if override:
        return [
            parse_model_override(item, temperature)
            for item in override.split(",")
            if item.strip()
        ]

    config_path = REPO_ROOT / "configs/project.yaml"
    rows = read_model_setting(config_path)
    if not rows:
        rows = [{"id": OPENAI_MODEL, "provider": "openai", "tier": "small"}]
    return [
        ModelRun(
            id=str(row.get("id", "")).strip(),
            provider=str(row.get("provider", "openai")).strip().lower(),
            tier=str(row.get("tier", "")).strip(),
            temperature=temperature,
        )
        for row in rows
        if str(row.get("id", "")).strip()
    ]


def parse_model_override(value: str, temperature: float) -> ModelRun:
    parts = [part.strip() for part in value.strip().split(":")]
    if len(parts) == 1:
        return ModelRun(id=parts[0], provider="openai", tier="", temperature=temperature)
    if len(parts) == 2:
        provider, model_id = parts
        return ModelRun(id=model_id, provider=provider.lower(), tier="", temperature=temperature)
    if len(parts) == 3:
        provider, model_id, tier = parts
        return ModelRun(id=model_id, provider=provider.lower(), tier=tier, temperature=temperature)
    raise SystemExit(
        "REALISTIC_BFCL_MODELS entries must be model_id, provider:model_id, "
        "or provider:model_id:tier."
    )


def evaluation_temperature() -> float:
    value = os.environ.get("REALISTIC_BFCL_TEMPERATURE")
    if value:
        return float(value)
    return read_float_setting(REPO_ROOT / "configs/project.yaml", "temperature", 0.0)


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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
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


def anthropic_api_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("CLAUDE_API_KEY"):
        return os.environ["CLAUDE_API_KEY"]

    candidates = [
        Path(os.environ["REALISTIC_BFCL_ENV_FILE"])
        if os.environ.get("REALISTIC_BFCL_ENV_FILE")
        else None,
        REPO_ROOT / ".env",
        REPO_ROOT.parent / "underlayer/.env",
    ]
    for path in candidates:
        if path is None:
            continue
        values = read_env_file(path)
        key = values.get("ANTHROPIC_API_KEY") or values.get("CLAUDE_API_KEY")
        if key:
            return key

    raise SystemExit(
        "Missing ANTHROPIC_API_KEY or CLAUDE_API_KEY. Set it in the environment or "
        "REALISTIC_BFCL_ENV_FILE."
    )


def xai_api_key() -> str:
    if os.environ.get("XAI_API_KEY"):
        return os.environ["XAI_API_KEY"]
    if os.environ.get("GROK_API_KEY"):
        return os.environ["GROK_API_KEY"]

    candidates = [
        Path(os.environ["REALISTIC_BFCL_ENV_FILE"])
        if os.environ.get("REALISTIC_BFCL_ENV_FILE")
        else None,
        REPO_ROOT / ".env",
        REPO_ROOT.parent / "underlayer/.env",
    ]
    for path in candidates:
        if path is None:
            continue
        values = read_env_file(path)
        key = values.get("XAI_API_KEY") or values.get("GROK_API_KEY")
        if key:
            return key

    raise SystemExit(
        "Missing XAI_API_KEY or GROK_API_KEY. Set it in the environment or "
        "REALISTIC_BFCL_ENV_FILE."
    )


def openrouter_api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if os.environ.get("OPEN_ROUTER_API_KEY"):
        return os.environ["OPEN_ROUTER_API_KEY"]

    candidates = [
        Path(os.environ["REALISTIC_BFCL_ENV_FILE"])
        if os.environ.get("REALISTIC_BFCL_ENV_FILE")
        else None,
        REPO_ROOT / ".env",
        REPO_ROOT.parent / "underlayer/.env",
    ]
    for path in candidates:
        if path is None:
            continue
        values = read_env_file(path)
        key = values.get("OPENROUTER_API_KEY") or values.get("OPEN_ROUTER_API_KEY")
        if key:
            return key

    raise SystemExit(
        "Missing OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY. Set it in the environment "
        "or REALISTIC_BFCL_ENV_FILE."
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
