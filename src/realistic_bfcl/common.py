from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
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
DEFAULT_ROUTER_MAX_OUTPUT_TOKENS = 256
ROUTER_MESSAGE_SERIALIZATION = "preserve_bfcl_turns_v1"
RETRYABLE_HTTP_STATUS = {408, 409, 429, 500, 502, 503, 504}
_EXPLICIT_ENV_FILE: Path | None = None
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


def realism_dimension_configs() -> dict[str, dict[str, object]]:
    path = REPO_ROOT / "configs/realism_dimensions.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("dimensions"), dict):
        raise ValueError(f"Invalid realism dimension config: {path}")

    dimensions: dict[str, dict[str, object]] = {}
    for name, raw_config in payload["dimensions"].items():
        if not isinstance(name, str) or not isinstance(raw_config, dict):
            raise ValueError(f"Invalid realism dimension entry in {path}")
        config = dict(raw_config)
        status = config.get("status")
        if status not in {"evaluated", "pilot"}:
            raise ValueError(f"Dimension {name} has invalid status: {status!r}.")
        article_facing = config.get("article_facing")
        if not isinstance(article_facing, bool):
            raise ValueError(f"Dimension {name} must declare article_facing as a boolean.")
        if article_facing and status != "evaluated":
            raise ValueError(f"Article-facing dimension {name} must have evaluated status.")
        if not article_facing and not str(config.get("article_exclusion_reason", "")).strip():
            raise ValueError(
                f"Non-article dimension {name} must declare article_exclusion_reason."
            )
        dimensions[name] = config
    return dimensions


def article_facing_dimensions() -> frozenset[str]:
    return frozenset(
        name
        for name, config in realism_dimension_configs().items()
        if config["article_facing"] is True
    )


@dataclass(frozen=True)
class ModelRun:
    name: str
    id: str
    provider: str
    tier: str
    temperature: float
    max_output_tokens: int
    input_cost_per_million_tokens: float | None = None
    output_cost_per_million_tokens: float | None = None
    pricing_source: str = ""
    provider_options: dict[str, object] | None = None
    article_primary: bool = False

    @property
    def filename(self) -> str:
        return safe_model_filename(self.id)

    @property
    def sampling_parameters(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
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


def safe_model_filename(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")


def _optional_float(value: object, field: str, model_name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Model '{model_name}' has invalid {field}: {value!r}.") from error


def _model_from_config(row: object) -> ModelRun:
    if not isinstance(row, dict):
        raise SystemExit("Each configs/models.yaml entry must be a mapping.")
    model_id = str(row.get("id", "")).strip()
    name = str(row.get("name", "")).strip()
    provider = str(row.get("provider", "")).strip().lower()
    if not name or not model_id or not provider:
        raise SystemExit("Each model requires non-empty name, id, and provider fields.")
    sampling = row.get("sampling", {})
    pricing = row.get("pricing_usd_per_million_tokens", {})
    provider_options = row.get("provider_options", {})
    if not isinstance(sampling, dict) or not isinstance(pricing, dict):
        raise SystemExit(f"Model '{name}' sampling and pricing fields must be mappings.")
    if not isinstance(provider_options, dict):
        raise SystemExit(f"Model '{name}' provider_options must be a mapping.")
    try:
        temperature = float(sampling.get("temperature", 0))
        max_output_tokens = int(sampling.get("max_output_tokens", DEFAULT_ROUTER_MAX_OUTPUT_TOKENS))
    except (TypeError, ValueError) as error:
        raise SystemExit(f"Model '{name}' has invalid sampling parameters.") from error
    if max_output_tokens < 1:
        raise SystemExit(f"Model '{name}' max_output_tokens must be >= 1.")
    return ModelRun(
        name=name,
        id=model_id,
        provider=provider,
        tier=str(row.get("tier", "")).strip(),
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input_cost_per_million_tokens=_optional_float(pricing.get("input"), "input price", name),
        output_cost_per_million_tokens=_optional_float(
            pricing.get("output"), "output price", name
        ),
        pricing_source=str(row.get("pricing_source", "")).strip(),
        provider_options=dict(provider_options),
        article_primary=row.get("article_primary") is True,
    )


def model_registry(path: Path | None = None) -> list[ModelRun]:
    config_path = path or REPO_ROOT / "configs/models.yaml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SystemExit(f"Could not read model registry {config_path}: {error}") from error
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise SystemExit("configs/models.yaml must contain a non-empty 'models' list.")
    models = [_model_from_config(row) for row in rows]
    for label, values in (
        ("name", [model.name for model in models]),
        ("id", [model.id for model in models]),
        ("output namespace", [model.filename for model in models]),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise SystemExit(f"Duplicate model {label}(s) in configs/models.yaml: {duplicates}")
    primary_models = [model for model in models if model.article_primary]
    if len(primary_models) != 1:
        raise SystemExit("configs/models.yaml must mark exactly one article_primary model.")
    return models


def article_primary_model() -> ModelRun:
    return next(model for model in model_registry() if model.article_primary)


def configured_model_runs(selected: list[str] | None = None) -> list[ModelRun]:
    override = os.environ.get("REALISTIC_BFCL_MODELS")
    if override:
        if selected:
            raise SystemExit("Use either --models or REALISTIC_BFCL_MODELS, not both.")
        selected = [item.strip() for item in override.split(",") if item.strip()]

    models = model_registry()
    if not selected:
        return models
    by_selector = {selector: model for model in models for selector in (model.name, model.id)}
    unknown = [selector for selector in selected if selector not in by_selector]
    if unknown:
        known = ", ".join(model.name for model in models)
        raise SystemExit(f"Unknown model selector(s) {unknown}. Known model names: {known}.")
    chosen = [by_selector[selector] for selector in selected]
    if len({model.id for model in chosen}) != len(chosen):
        raise SystemExit("--models selects the same model more than once.")
    return chosen


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
    if not path.exists():
        raise SystemExit(f"Environment file does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"Environment file is not a regular file: {path}")

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise SystemExit(f"Could not read environment file {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            raise SystemExit(
                f"Invalid environment file {path}: line {line_number} must be KEY=VALUE."
            )
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise SystemExit(
                f"Invalid environment file {path}: line {line_number} has an invalid key."
            )
        if key in values:
            raise SystemExit(
                f"Invalid environment file {path}: line {line_number} duplicates {key}."
            )
        values[key] = parse_env_value(value, path=path, line_number=line_number)
    return values


def parse_env_value(value: str, *, path: Path, line_number: int) -> str:
    value = value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        closing_index = value.find(quote, 1)
        if closing_index == -1:
            raise SystemExit(
                f"Invalid environment file {path}: line {line_number} has "
                "an unmatched quote."
            )
        trailing = value[closing_index + 1 :].strip()
        if trailing and not trailing.startswith("#"):
            raise SystemExit(
                f"Invalid environment file {path}: line {line_number} has text "
                "after a quoted value."
            )
        return value[1:closing_index]

    comment_match = re.search(r"\s+#", value)
    if comment_match:
        value = value[: comment_match.start()]
    return value.rstrip()


def set_explicit_env_file(path: Path | None) -> None:
    global _EXPLICIT_ENV_FILE
    _EXPLICIT_ENV_FILE = None
    if path is not None:
        read_env_file(path)
    _EXPLICIT_ENV_FILE = path


def api_key_from_config(key_names: tuple[str, ...]) -> str:
    env_file = os.environ.get("REALISTIC_BFCL_ENV_FILE")
    candidates = [_EXPLICIT_ENV_FILE]
    if env_file:
        configured_path = Path(env_file)
        if configured_path != _EXPLICIT_ENV_FILE:
            candidates.append(configured_path)

    for path in candidates:
        if path is None:
            continue
        values = read_env_file(path)
        for key_name in key_names:
            if values.get(key_name):
                return values[key_name]

    for key_name in key_names:
        if os.environ.get(key_name):
            return os.environ[key_name]

    names = " or ".join(key_names)
    raise SystemExit(
        f"Missing {names}. Provide it with --env-file, REALISTIC_BFCL_ENV_FILE, "
        "or the process environment."
    )


def openai_api_key() -> str:
    return api_key_from_config(("OPENAI_API_KEY",))


def anthropic_api_key() -> str:
    return api_key_from_config(("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"))


def xai_api_key() -> str:
    return api_key_from_config(("XAI_API_KEY", "GROK_API_KEY"))


def grok_api_key() -> str:
    return api_key_from_config(("GROK_API_KEY", "XAI_API_KEY"))


def openrouter_api_key() -> str:
    return api_key_from_config(("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"))


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
