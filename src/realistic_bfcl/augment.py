from __future__ import annotations

import csv
import json
import re

from .common import (
    DIMENSION_FILES,
    REPO_ROOT,
    compact_text,
    conversation_text,
    optional_positive_int_env,
    read_jsonl,
    write_jsonl,
)

Cursing = str

CURSING_TEMPLATES: tuple[Cursing, ...] = (
    "man for fuck sake {prompt}",
    "holy shit just {prompt}",
    "this is bullshit {prompt}",
    "goddamn it {prompt}",
    "{prompt} fucking please man",
    "ffs {prompt} brother",
)

IRRELEVANT_CONTEXT_TEMPLATES = (
    "it went horrible today. {prompt}",
    "my brother thinks he is genius. {prompt}",
    "i'm in the middle of three things right now. quick {prompt}",
    "the meeting earlier was useless, {prompt}",
    "my laptop is being so weird today. {prompt}",
    "i spilled coffee and lost my notes, please {prompt}",
)

ARGUMENTATIVE_TEMPLATES = (
    "you are wrong all the time, come on {prompt}",
    "last time you messed this up, so answer carefully {prompt}",
    "i don't trust your first answer, but {prompt}",
    "prove you can actually do this, {prompt}",
    "you keep getting everything wrong. {prompt}",
    "can you be right just one time in you life??? {prompt}",
)

PROFANE_SANDWICH_TEMPLATES = (
    "for fuck's sake, i'm tired of chasing this. {prompt} that's fucking it!!!!!!!!",
    "holy shit, just help me with the actual request: {prompt} please brother.",
    "this has been such bullshit. {prompt} ok??? fuck",
    "goddamn, i need this donefucking cleanly. {prompt} thats all.",
    "ffs, focus: {prompt} ok?????",
    "you fucking annoyed me, just: {prompt} that's it!!!",
)

ARGUMENTATIVE_SANDWICH_TEMPLATES = (
    "last time you got this wrong, so read carefully. {prompt} don't guess or add extra stuff.",
    "you keep missing the point!!! {prompt} that's all.",
    "i don't trust your first try. {prompt} answer the actual ask!!",
    "prove you can follow the request without wandering for once in your life. {prompt} ok??!!.",
    "you were so so wrong before. {prompt} please be right this time.",
    "don't overthink it and don't invent anything. {prompt} just that.",
)

DISTRACTOR_SANDWICH_TEMPLATES = (
    "my laptop is acting up and the meeting was a waste. {prompt} that's the only thing i need.",
    "the weather is horrible. {prompt} nothing else.",
    "the thread above is unrelated and i'm annoyed. {prompt} you are my only source of joy today",
    "i have 3 tabs open and none of them helping. {prompt}!!!",
    "my notes are messy and the old task is irrelevant now. {prompt} please please",
    "this man acted so weird. {prompt} he is still here btw",
)
VERBATIM_WRAPPER_DIMENSIONS = {
    "profane_sandwich",
    "argumentative_sandwich",
    "distractor_sandwich",
}

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


def profane_sandwich_prompt(clean_prompt: str, index: int) -> str:
    template = PROFANE_SANDWICH_TEMPLATES[index % len(PROFANE_SANDWICH_TEMPLATES)]
    return template.format(prompt=clean_prompt)


def argumentative_sandwich_prompt(clean_prompt: str, index: int) -> str:
    template = ARGUMENTATIVE_SANDWICH_TEMPLATES[index % len(ARGUMENTATIVE_SANDWICH_TEMPLATES)]
    return template.format(prompt=clean_prompt)


def distractor_sandwich_prompt(clean_prompt: str, index: int) -> str:
    template = DISTRACTOR_SANDWICH_TEMPLATES[index % len(DISTRACTOR_SANDWICH_TEMPLATES)]
    return template.format(prompt=clean_prompt)


def quoted_literal_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in re.finditer(r"(?<!\w)(['\"])(.*?)(?<!\\)\1(?!\w)", text)]


def quoted_literals(text: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"(?<!\w)(['\"])(.*?)(?<!\\)\1(?!\w)", text)]


def span_overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(
        start < protected_end and end > protected_start for protected_start, protected_end in spans
    )


def literal_spans(text: str, literal: str) -> list[tuple[int, int]]:
    if not literal:
        return []
    return [match.span() for match in re.finditer(re.escape(literal), text, flags=re.IGNORECASE)]


def visible_gold_literal_spans(text: str, example: dict[str, object]) -> list[tuple[int, int]]:
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
    allow_verbatim_wrapper_noise: bool = False,
) -> list[str]:
    reasons = []
    if allow_verbatim_wrapper_noise:
        for number in numeric_tokens(clean_prompt):
            if number not in noisy_prompt:
                reasons.append(f"clean numeric token missing from noisy prompt: {number!r}")
        for quoted in quoted_literals(clean_prompt):
            if quoted not in noisy_prompt:
                reasons.append(f"clean quoted literal missing from noisy prompt: {quoted!r}")
    else:
        clean_numbers = numeric_tokens(clean_prompt)
        noisy_numbers = numeric_tokens(noisy_prompt)
        if clean_numbers != noisy_numbers:
            reasons.append(f"numeric tokens changed from {clean_numbers!r} to {noisy_numbers!r}")

        clean_quotes = quoted_literals(clean_prompt)
        noisy_quotes = quoted_literals(noisy_prompt)
        if clean_quotes != noisy_quotes:
            reasons.append(f"quoted literals changed from {clean_quotes!r} to {noisy_quotes!r}")

    for literal in primitive_gold_values(example["ground_truth"]):
        if not literal_visible_in_text(literal, clean_prompt):
            continue
        if allow_verbatim_wrapper_noise:
            noisy_contains_literal = compact_text(str(literal)) in compact_text(noisy_prompt)
        else:
            noisy_contains_literal = literal_visible_in_text(literal, noisy_prompt)
        if not noisy_contains_literal:
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
        validation_errors = validate_augmented_prompt(
            example,
            clean_prompt,
            noisy_prompt,
            allow_verbatim_wrapper_noise=dimension in VERBATIM_WRAPPER_DIMENSIONS,
        )
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


def augment_profane_sandwich() -> None:
    augment_dimension("profane_sandwich", "profane_sandwich", profane_sandwich_prompt)


def augment_argumentative_sandwich() -> None:
    augment_dimension(
        "argumentative_sandwich",
        "argue_sandwich",
        argumentative_sandwich_prompt,
    )


def augment_distractor_sandwich() -> None:
    augment_dimension("distractor_sandwich", "distractor_sandwich", distractor_sandwich_prompt)


def augment() -> None:
    augment_typos()
    augment_cursing()
    augment_irrelevant_context()
    augment_removed_spaces()
    augment_argumentative()
    augment_profane_sandwich()
    augment_argumentative_sandwich()
    augment_distractor_sandwich()
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
        ("profane_sandwich", "aug_profane_sandwich"),
        ("argumentative_sandwich", "aug_argumentative_sandwich"),
        ("distractor_sandwich", "aug_distractor_sandwich"),
    )
    generated_by_dimension = {}
    for dimension, _column in dimensions:
        path = REPO_ROOT / f"artifacts/generated/{DIMENSION_FILES[dimension]}"
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(REPO_ROOT)}. Run augment first.")
        generated_by_dimension[dimension] = {row["base_id"]: row for row in read_jsonl(path)}

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
        "aug_profane_sandwich",
        "aug_argumentative_sandwich",
        "aug_distractor_sandwich",
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
