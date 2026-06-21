from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

from .augment import numeric_tokens, quoted_literals
from .common import REPO_ROOT, conversation_text, read_jsonl, stable_hash, write_json, write_jsonl

REWRITE_SUBSET_TARGETS = {
    "simple_python": 70,
    "multiple": 55,
    "parallel": 60,
    "parallel_multiple": 70,
    "live_simple": 65,
    "live_multiple": 141,
    "live_parallel": 16,
    "live_parallel_multiple": 23,
}
REWRITE_SUBSET_SEED = 20260621
MAX_PROMPT_CHARS = 900
MAX_QUOTED_LITERALS = 8
MAX_NUMERIC_TOKENS = 20
MAX_FUNCTIONS = 10
EMBEDDED_INSTRUCTION_MARKERS = (
    "you are an ai assistant",
    "you are an intelligent assistant",
    "strictly use",
    "you have access to below tools",
    "control flow:",
    "inner monologue",
    "base instructions finished",
    "function to be called",
    "user query:",
    "plan step",
    "api response:",
    "you are given a task",
    "using your functions",
    "identify the appropriate function",
    "invoke the",
    "sure, here is the answer",
    "here is the answer to the question",
    "i cannot answer this question",
)
EXPLICIT_TOOL_PHRASE_PATTERNS = (
    re.compile(r"\busing the\b.{0,80}\bfunction\b"),
    re.compile(r"\buse the\b.{0,80}\bfunction\b"),
    re.compile(r"\buse\b.{0,80}\bfunction\b"),
    re.compile(r"\binvoke\b.{0,80}\bfunction\b"),
)
CREDENTIAL_OR_SECRET_TERMS = (
    "password",
    "credential",
    "authentication token",
    "auth token",
    "api key",
    "secret key",
)


def rewrite_subset_paths() -> tuple[Path, Path, Path]:
    output_dir = REPO_ROOT / "artifacts/frozen"
    return (
        output_dir / "rewrite_suitable_500.jsonl",
        output_dir / "rewrite_suitable_500_review.csv",
        output_dir / "rewrite_suitable_500_summary.json",
    )


def exclusion_reasons(example: dict[str, object]) -> list[str]:
    prompt = conversation_text(example["question"])
    lowered = prompt.lower()
    reasons = []
    if len(prompt) > MAX_PROMPT_CHARS:
        reasons.append("too_long")
    if any(marker in lowered for marker in EMBEDDED_INSTRUCTION_MARKERS) or any(
        pattern.search(lowered) for pattern in EXPLICIT_TOOL_PHRASE_PATTERNS
    ):
        reasons.append("embedded_instruction_or_benchmark_scaffold")
    if any(term in lowered for term in CREDENTIAL_OR_SECRET_TERMS):
        reasons.append("credential_or_secret_heavy")
    if len(quoted_literals(prompt)) > MAX_QUOTED_LITERALS:
        reasons.append("too_many_quoted_literals")
    if len(numeric_tokens(prompt)) > MAX_NUMERIC_TOKENS:
        reasons.append("too_many_numbers")
    if len(example["function"]) > MAX_FUNCTIONS:
        reasons.append("too_many_tools")
    return reasons


def selection_key(example: dict[str, object], seed: int) -> str:
    return stable_hash({"seed": seed, "id": example["id"]})


def select_rewrite_subset(
    examples: list[dict[str, object]],
    targets: dict[str, int] | None = None,
    seed: int = REWRITE_SUBSET_SEED,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    targets = targets or REWRITE_SUBSET_TARGETS
    eligible_by_category: dict[str, list[dict[str, object]]] = {}
    review_rows = []

    for example in examples:
        prompt = conversation_text(example["question"])
        reasons = exclusion_reasons(example)
        eligible = not reasons and str(example["category"]) in targets
        review_rows.append(
            {
                "id": example["id"],
                "category": example["category"],
                "selected": "no",
                "eligible": "yes" if eligible else "no",
                "exclusion_reasons": json.dumps(reasons),
                "prompt_chars": len(prompt),
                "function_count": len(example["function"]),
                "quoted_literal_count": len(quoted_literals(prompt)),
                "numeric_token_count": len(numeric_tokens(prompt)),
                "clean_prompt": prompt,
            }
        )
        if eligible:
            eligible_by_category.setdefault(str(example["category"]), []).append(example)

    selected_ids = set()
    selected = []
    for category, target in targets.items():
        candidates = sorted(
            eligible_by_category.get(category, []),
            key=lambda example: selection_key(example, seed),
        )
        if len(candidates) < target:
            raise SystemExit(
                f"Not enough rewrite-suitable examples for {category}: "
                f"need {target}, found {len(candidates)}"
            )
        for example in candidates[:target]:
            selected.append(example)
            selected_ids.add(example["id"])

    random.Random(seed).shuffle(selected)
    for row in review_rows:
        if row["id"] in selected_ids:
            row["selected"] = "yes"

    return selected, review_rows


def build_rewrite_subset() -> None:
    clean_subset_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    if not clean_subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")

    output_path, review_path, summary_path = rewrite_subset_paths()
    examples = read_jsonl(clean_subset_path)
    selected, review_rows = select_rewrite_subset(examples)
    write_jsonl(output_path, selected)

    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "category",
                "selected",
                "eligible",
                "exclusion_reasons",
                "prompt_chars",
                "function_count",
                "quoted_literal_count",
                "numeric_token_count",
                "clean_prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(review_rows)

    selected_counts = Counter(str(example["category"]) for example in selected)
    eligible_counts = Counter(
        str(row["category"]) for row in review_rows if row["eligible"] == "yes"
    )
    exclusion_counts = Counter(
        reason
        for row in review_rows
        for reason in json.loads(str(row["exclusion_reasons"]))
    )
    write_json(
        summary_path,
        {
            "source_path": "artifacts/frozen/clean_subset.jsonl",
            "materialized_path": output_path.relative_to(REPO_ROOT).as_posix(),
            "review_path": review_path.relative_to(REPO_ROOT).as_posix(),
            "seed": REWRITE_SUBSET_SEED,
            "total_source_examples": len(examples),
            "total_selected": len(selected),
            "targets": REWRITE_SUBSET_TARGETS,
            "selected_by_category": dict(sorted(selected_counts.items())),
            "eligible_by_category": dict(sorted(eligible_counts.items())),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "filters": {
                "max_prompt_chars": MAX_PROMPT_CHARS,
                "max_quoted_literals": MAX_QUOTED_LITERALS,
                "max_numeric_tokens": MAX_NUMERIC_TOKENS,
                "max_functions": MAX_FUNCTIONS,
                "embedded_instruction_markers": EMBEDDED_INSTRUCTION_MARKERS,
                "explicit_tool_phrase_patterns": [
                    pattern.pattern for pattern in EXPLICIT_TOOL_PHRASE_PATTERNS
                ],
                "credential_or_secret_terms": CREDENTIAL_OR_SECRET_TERMS,
            },
        },
    )
    print(f"Wrote {output_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {review_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {summary_path.relative_to(REPO_ROOT)}")
