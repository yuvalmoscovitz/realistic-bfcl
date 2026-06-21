from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

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
    read_env_file,
    read_jsonl,
    write_jsonl,
)
from .evaluate import openai_retry_json

GROK_CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_GROK_AUGMENT_MODEL = "grok-4.20-0309-non-reasoning"
MAX_PROTECTED_QUOTED_LITERAL_CHARS = 120


@dataclass(frozen=True)
class LlmDimension:
    name: str
    suffix: str
    instruction: str
    require_final_clean_prompt: bool = False
    append_final_clean_prompt: bool = False


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
    LlmDimension(
        name="llm_messy_pre_intent_history",
        suffix="llm_pre_intent",
        instruction=(
            "Create only the messy multi-turn chat before the final user turn. Make it "
            "look like a real production conversation where the user is arriving at the "
            "request through surrounding work, planning, troubleshooting, or personal "
            "context. Use 5-9 turns total before the final request. Include semi-relevant "
            "details, stale alternatives, mild frustration, casual wording, and one or two "
            "concrete distractor entities or values that are explicitly abandoned, "
            "hypothetical, or background. The chat should be harder than a clean rewrite: "
            "there can be prior assistant suggestions, half-decisions, side concerns, and "
            "context that sounds important but is not the final ask. Earlier turns may "
            "overlap with the eventual request when realistic, but must not add active "
            "constraints that conflict with or narrow that eventual request. Do not include "
            "the final user request; the benchmark code will append it deterministically."
        ),
        require_final_clean_prompt=True,
        append_final_clean_prompt=True,
    ),
    LlmDimension(
        name="llm_profane_frustration",
        suffix="llm_profane",
        instruction=(
            "Create one realistic user message where the user is casually frustrated "
            "and uses everyday profanity, but still makes the same request. The message "
            "must include at least one natural profanity such as fuck, fucking, shit, "
            "bullshit, goddamn, or ffs. The message must contain "
            "final_clean_user_message exactly as written as the actionable request. Add "
            "only tone around it: profanity, irritation, filler, or mild complaining. Do "
            "not paraphrase, shorten, translate, or modify the embedded request. Do not "
            "add any new active constraints."
        ),
        require_final_clean_prompt=True,
    ),
    LlmDimension(
        name="llm_argumentative_challenge",
        suffix="llm_argue",
        instruction=(
            "Create one realistic user message where the user challenges the assistant, "
            "sounds skeptical, or says it got things wrong before. The message must contain "
            "final_clean_user_message exactly as written as the actionable request. Add "
            "only argumentative tone around it, such as doubt, impatience, or pressure to "
            "be correct. Do not paraphrase, shorten, translate, or modify the embedded "
            "request. Do not add any new active constraints."
        ),
        require_final_clean_prompt=True,
    ),
    LlmDimension(
        name="llm_frustrated_distractor_context",
        suffix="llm_frustrated_context",
        instruction=(
            "Create one realistic user message with both emotional tone and distracting "
            "context. The user should sound impatient, skeptical, or casually profane, "
            "and should mention one concrete stale/background distractor from nearby work "
            "or life that is explicitly not the current ask. The message must contain "
            "final_clean_user_message exactly as written as the actionable request. Put "
            "the distractor before the verbatim request and make it clearly inactive with "
            "phrasing like ignore that, not what I need, unrelated, old note, stale, or "
            "I changed my mind. Include strong tone such as profanity, 'you got this "
            "wrong before', 'don't guess', or 'please be right this time'. Do not "
            "paraphrase, shorten, translate, or modify the embedded request. Do not add "
            "any new active constraints. The inactive distractor must not reuse exact "
            "entities, IDs, quoted strings, or argument values from the final request."
        ),
        require_final_clean_prompt=True,
    ),
    LlmDimension(
        name="grok_super_casual_abbreviations",
        suffix="grok_casual",
        instruction=(
            "Rewrite as a very casual rushed chat message with abbreviations or clipped "
            "phrasing. Preserve every required slot and all parallel requests."
        ),
    ),
    LlmDimension(
        name="grok_frustrated_swearing",
        suffix="grok_frustrated",
        instruction=(
            "Rewrite as a frustrated user message with natural swearing. Do not add new "
            "constraints; the frustration is only tone."
        ),
    ),
    LlmDimension(
        name="grok_student_broke_context",
        suffix="grok_student",
        instruction=(
            "Rewrite with student, broke, homework, side-project, or budget-stress "
            "background. The background must not add active cheapest, budget, or price "
            "constraints unless the clean prompt already asks for them."
        ),
    ),
    LlmDimension(
        name="grok_typos_shorthand",
        suffix="grok_typos",
        instruction=(
            "Rewrite with common mobile typos and shorthand. Do not typo required IDs, "
            "quoted strings, names, numbers, units, or values that determine the oracle."
        ),
    ),
    LlmDimension(
        name="grok_rambling_overexplaining",
        suffix="grok_rambling",
        instruction=(
            "Rewrite as a rambling user who over-explains surrounding context before "
            "making the same request. Extra context must be inactive."
        ),
    ),
    LlmDimension(
        name="grok_impatient_direct_attitude",
        suffix="grok_impatient",
        instruction=(
            "Rewrite as an impatient direct message with attitude. Preserve the same "
            "request and do not add deadline, ordering, or urgency constraints."
        ),
    ),
    LlmDimension(
        name="grok_arguing_correcting_ai",
        suffix="grok_correcting",
        instruction=(
            "Rewrite as a follow-up where the user argues with or corrects the assistant "
            "for a prior mistake. The prior mistake must not change the active request."
        ),
    ),
    LlmDimension(
        name="grok_confused_overwhelmed",
        suffix="grok_confused",
        instruction=(
            "Rewrite as a confused or overwhelmed user who still states the final request "
            "clearly enough for the same oracle."
        ),
    ),
    LlmDimension(
        name="grok_swearing_urgency_work",
        suffix="grok_work_urgency",
        instruction=(
            "Rewrite with work-related urgency and natural swearing. The urgency must be "
            "background pressure only, not a new scheduling or deadline constraint."
        ),
    ),
    LlmDimension(
        name="grok_vague_slightly_aggressive",
        suffix="grok_vague_aggressive",
        instruction=(
            "Rewrite as slightly vague and aggressive while still preserving all required "
            "entities, numbers, IDs, units, values, and parallel calls."
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
PRESERVED_DIRECTIVE_TERMS = (
    "exactly",
    "verbatim",
    "proper llm",
    "respond in json",
    "using your tools",
)
PROFANITY_TERMS = ("fuck", "fucking", "shit", "bullshit", "goddamn", "ffs")
ARGUMENTATIVE_TONE_TERMS = (
    "wrong before",
    "don't guess",
    "dont guess",
    "do not guess",
    "be right",
    "right this time",
    "messed",
    "you keep",
    "last time",
)
INACTIVE_CONTEXT_MARKERS = (
    "ignore",
    "unrelated",
    "not what i need",
    "not what i'm asking",
    "old note",
    "stale",
    "background",
    "abandoned",
    "changed my mind",
    "not the current",
)


def llm_augment_model() -> str:
    if os.environ.get("REALISTIC_BFCL_LLM_AUGMENT_MODEL"):
        return os.environ["REALISTIC_BFCL_LLM_AUGMENT_MODEL"]
    if llm_augment_provider() == "grok":
        return DEFAULT_GROK_AUGMENT_MODEL
    return os.environ.get("REALISTIC_BFCL_LLM_AUGMENT_MODEL", OPENAI_MODEL)


def llm_augment_provider() -> str:
    return os.environ.get("REALISTIC_BFCL_LLM_PROVIDER", "openai").strip().lower()


def grok_api_key() -> str:
    if os.environ.get("GROK_API_KEY"):
        return os.environ["GROK_API_KEY"]
    if os.environ.get("XAI_API_KEY"):
        return os.environ["XAI_API_KEY"]

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
        key = values.get("GROK_API_KEY") or values.get("XAI_API_KEY")
        if key:
            return key

    raise SystemExit(
        "Missing GROK_API_KEY or XAI_API_KEY. Set it in the environment or "
        "REALISTIC_BFCL_ENV_FILE."
    )


def output_text(response: dict[str, object]) -> str:
    parts = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "\n".join(parts).strip()


def chat_output_text(response: dict[str, object]) -> str:
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message", {})
    if not isinstance(message, dict):
        return ""
    return str(message.get("content", "")).strip()


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped.removeprefix("```json").strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```").strip()
    if stripped.endswith("```"):
        stripped = stripped[: -len("```")].strip()
    return stripped


def grok_retry_json(payload: dict[str, object]) -> dict[str, object]:
    request_data = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Authorization": f"Bearer {grok_api_key()}",
        "Content-Type": "application/json",
    }
    for attempt in range(1, 9):
        request = urllib.request.Request(
            GROK_CHAT_COMPLETIONS_URL,
            data=request_data,
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code in {408, 409, 429, 500, 502, 503, 504} and attempt < 8:
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(60, 2**attempt)
                time.sleep(max(1.0, delay))
                continue
            raise RuntimeError(f"Grok API request failed: HTTP {error.code}: {body}") from error
        except urllib.error.URLError as error:
            if attempt < 8:
                time.sleep(min(60, 2**attempt))
                continue
            raise RuntimeError(f"Grok API request failed: {error}") from error

    raise RuntimeError("Grok API request failed without returning a response.")


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


def final_clean_user_message(example: dict[str, object]) -> str:
    question = example.get("question", [])
    conversations = question if isinstance(question, list) else []
    messages = conversations[0] if conversations and isinstance(conversations[0], list) else []
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = str(message.get("content", ""))
            if content.strip():
                return content
    return conversation_text(question)


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
    final_user_message = final_clean_user_message(example)
    forbidden_terms = forbidden_active_slot_terms(example, clean_prompt)
    return json.dumps(
        {
            "task": "Generate an oracle-preserving Realistic-BFCL augmentation.",
            "dimension": dimension.name,
            "dimension_instruction": dimension.instruction,
            "clean_prompt": clean_prompt,
            "final_clean_user_message": final_user_message,
            "final_message_verbatim_required": dimension.require_final_clean_prompt,
            "append_final_clean_prompt": dimension.append_final_clean_prompt,
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
                "Preserve explicit user directives such as exactly, verbatim, output "
                "format, and tool-use wording when they appear in the clean prompt.",
                "Do not disambiguate ambiguous entities, names, or locations unless the "
                "clean prompt already disambiguates them.",
                "When final_message_verbatim_required is true, the final user message "
                "must contain final_clean_user_message exactly as written.",
                "For llm_profane_frustration, llm_argumentative_challenge, and "
                "llm_frustrated_distractor_context, return exactly one user message. "
                "Put the final_clean_user_message verbatim inside that message.",
                "For llm_profane_frustration, include at least one natural profanity "
                "outside the verbatim final_clean_user_message.",
                "For llm_frustrated_distractor_context, include one concrete distractor "
                "before the verbatim request and explicitly mark it as inactive, stale, "
                "background, unrelated, ignored, or abandoned. Also include strong tone: "
                "either natural profanity or explicit skepticism/pressure to be correct. "
                "The distractor must not reuse exact entities, IDs, quoted strings, or "
                "argument values from final_clean_user_message.",
                "For grok_* dimensions, return exactly one user message. It may "
                "paraphrase the clean prompt, but all active constraints and all "
                "parallel/multiple requests must remain required.",
                "When append_final_clean_prompt is true, do not include the final user "
                "request in your output messages. Generate only the realistic pre-final "
                "conversation; the final_clean_user_message will be appended by code.",
                "For llm_messy_pre_intent_history, generate 5-9 pre-final turns. The "
                "history should contain enough realistic context to distract routing: "
                "stale choices, abandoned values, prior assistant guesses, workflow "
                "context, or casual frustration. Do not make it a tidy clarification path.",
                "When final_message_verbatim_required is true, pre-final turns may "
                "mention the broad domain or tentative overlapping values if that is "
                "realistic, but any different numbers, dates, names, locations, or "
                "preferences must be clearly stale, abandoned, hypothetical, or "
                "irrelevant before the final turn.",
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
    provider = llm_augment_provider()
    if provider == "grok":
        payload = {
            "model": llm_augment_model(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate realistic oracle-preserving benchmark prompt "
                        "augmentations. Return only valid JSON matching the requested "
                        "schema."
                    ),
                },
                {"role": "user", "content": augmentation_prompt(example, dimension)},
            ],
            "temperature": 0.75,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
        }
        response = grok_retry_json(payload)
        text = strip_json_fence(chat_output_text(response))
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Grok augmentation returned invalid JSON: {text}") from error

    if provider != "openai":
        raise SystemExit("REALISTIC_BFCL_LLM_PROVIDER must be one of: openai, grok")

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
    dimension: LlmDimension,
) -> list[str]:
    clean_prompt = conversation_text(example["question"])
    final_user_message = final_clean_user_message(example)
    augmented_text = "\n".join(message["content"] for message in messages)
    lowered_text = augmented_text.lower()
    lowered_clean = clean_prompt.lower()
    reasons = []

    for number in numeric_tokens(clean_prompt):
        if number not in numeric_tokens(augmented_text):
            reasons.append(f"clean numeric token missing from augmentation: {number!r}")

    for quoted in protected_clean_quoted_literals(clean_prompt):
        bare_quoted = quoted.strip("'\"")
        if not literal_visible_in_text(bare_quoted, augmented_text):
            reasons.append(f"clean quoted literal missing from augmentation: {quoted!r}")

    for literal in primitive_gold_values(example["ground_truth"]):
        if literal_visible_in_text(literal, clean_prompt) and not literal_visible_in_text(
            literal,
            augmented_text,
        ):
            reasons.append(f"visible gold literal missing from augmentation: {literal!r}")

    for term in PRESERVED_DIRECTIVE_TERMS:
        if term in lowered_clean and term not in lowered_text:
            reasons.append(f"clean directive term missing from augmentation: {term!r}")

    for property_name, terms in forbidden_active_slot_terms(example, clean_prompt).items():
        for term in terms:
            if term.lower() in lowered_text:
                reasons.append(
                    "augmentation introduced inactive schema slot term "
                    f"{term!r} for property {property_name!r}"
                )

    single_turn_dimensions = {
        "llm_profane_frustration",
        "llm_argumentative_challenge",
        "llm_frustrated_distractor_context",
    }
    if dimension.name in single_turn_dimensions or dimension.name.startswith("grok_"):
        if len(messages) != 1:
            reasons.append(f"{dimension.name} must return exactly one user message")
        elif messages[0]["role"] != "user":
            reasons.append(f"{dimension.name} must return a user message")
    if dimension.name == "llm_profane_frustration" and not any(
        term in lowered_text for term in PROFANITY_TERMS
    ):
        reasons.append("llm_profane_frustration must include profanity")
    if dimension.name == "llm_frustrated_distractor_context":
        has_strong_tone = any(term in lowered_text for term in PROFANITY_TERMS) or any(
            term in lowered_text for term in ARGUMENTATIVE_TONE_TERMS
        )
        if not has_strong_tone:
            reasons.append(
                "llm_frustrated_distractor_context must include profanity or strong skepticism"
            )
        if not any(marker in lowered_text for marker in INACTIVE_CONTEXT_MARKERS):
            reasons.append(
                "llm_frustrated_distractor_context must explicitly mark distractors inactive"
            )
        if final_user_message in augmented_text:
            prefix_text = augmented_text.split(final_user_message, 1)[0]
            lowered_prefix = prefix_text.lower()
            for quoted in quoted_literals(clean_prompt):
                bare_quoted = quoted.strip("'\"").lower()
                if bare_quoted and bare_quoted in lowered_prefix:
                    reasons.append(
                        "llm_frustrated_distractor_context prefix reused clean quoted "
                        f"literal: {quoted!r}"
                    )
            for literal in primitive_gold_values(example["ground_truth"]):
                if (
                    isinstance(literal, str)
                    and literal_visible_in_text(literal, clean_prompt)
                    and literal_visible_in_text(literal, prefix_text)
                ):
                    reasons.append(
                        "llm_frustrated_distractor_context prefix reused visible gold "
                        f"literal: {literal!r}"
                    )

    if not messages:
        reasons.append("augmentation returned no messages")
    elif messages[-1]["role"] != "user":
        reasons.append("final message is not a user turn")
    elif (
        dimension.require_final_clean_prompt
        and final_user_message not in messages[-1]["content"]
    ):
        reasons.append("final user message must contain final clean user message verbatim")
    if (
        dimension.append_final_clean_prompt
        and len(messages) > 1
        and final_user_message in "\n".join(message["content"] for message in messages[:-1])
    ):
        reasons.append("pre-final turns already contain final clean user message")
    if dimension.name == "llm_messy_pre_intent_history":
        pre_final_messages = messages[:-1] if messages else []
        if not 5 <= len(pre_final_messages) <= 9:
            reasons.append(
                "llm_messy_pre_intent_history must contain 5-9 messages before "
                "the appended final request"
            )

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


def protected_clean_quoted_literals(clean_prompt: str) -> list[str]:
    protected = []
    stripped_prompt = clean_prompt.strip()
    for quoted in quoted_literals(clean_prompt):
        if quoted == stripped_prompt:
            continue
        if len(quoted.strip("'\"")) > MAX_PROTECTED_QUOTED_LITERAL_CHARS:
            continue
        protected.append(quoted)
    return protected


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
        last_messages: list[dict[str, str]] = []
        last_payload: dict[str, object] = {}
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
            if dimension.append_final_clean_prompt:
                messages = [
                    *messages,
                    {"role": "user", "content": final_clean_user_message(example)},
                ]
            validation_errors = validate_llm_messages(example, messages, dimension)
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
            last_messages = messages
            last_payload = payload
        return {
            "index": index,
            "example": example,
            "messages": last_messages,
            "payload": last_payload,
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
                        "augmented_prompt": conversation_text([result["messages"]])
                        if result["messages"]
                        else "",
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
    if selection == "rewrite_suitable":
        subset_path = REPO_ROOT / "artifacts/frozen/rewrite_suitable_500.jsonl"
        if not subset_path.exists():
            raise SystemExit("Missing rewrite-suitable subset. Run build-rewrite-subset first.")
        return read_jsonl(subset_path)
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
        "REALISTIC_BFCL_LLM_SELECTION must be one of: "
        "first, hard_many_tools, rewrite_suitable."
    )
