from __future__ import annotations

from pathlib import Path

import pytest

import realistic_bfcl.augment as augment_module
from realistic_bfcl.augment import validate_augmented_prompt
from realistic_bfcl.common import conversation_text, read_jsonl


def example(prompt: str, value: object = "job A") -> dict[str, object]:
    return {
        "id": "triangle",
        "category": "simple_python",
        "question": [[{"role": "user", "content": prompt}]],
        "function": [
            {
                "name": "calculate_triangle_area",
                "parameters": {
                    "type": "dict",
                    "properties": {"unit": {"type": "string"}},
                    "required": ["unit"],
                },
            }
        ],
        "ground_truth": [{"calculate_triangle_area": {"unit": [value]}}],
    }


def test_public_augmenters_write_oracle_preserving_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template_count = max(
        len(templates)
        for templates in (
            augment_module.CURSING_TEMPLATES,
            augment_module.IRRELEVANT_CONTEXT_TEMPLATES,
            augment_module.ARGUMENTATIVE_TEMPLATES,
            augment_module.PROFANE_SANDWICH_TEMPLATES,
            augment_module.ARGUMENTATIVE_SANDWICH_TEMPLATES,
            augment_module.DISTRACTOR_SANDWICH_TEMPLATES,
            augment_module.PASTED_CONTEXT_BLOCK_TEMPLATES,
        )
    )
    clean = "Apply -12.5 and -12.5 to 'job A' and 'job A', then calculate area."
    examples = []
    for index in range(template_count):
        item = example(clean)
        item["id"] = f"case-{index}"
        item["ground_truth"] = [
            {
                "calculate_triangle_area": {
                    "request": {"adjustments": [-12.5, -12.5], "labels": ["job A", "job A"]}
                }
            }
        ]
        examples.append(item)

    monkeypatch.setattr(augment_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(augment_module, "selected_augmentation_examples", lambda: examples)
    augmenters = (
        ("typos", "typos", augment_module.augment_typos),
        ("cursing", "cursing", augment_module.augment_cursing),
        ("irrelevant_context", "context", augment_module.augment_irrelevant_context),
        ("removed_spaces", "spaces", augment_module.augment_removed_spaces),
        ("argumentative_challenge", "argue", augment_module.augment_argumentative),
        ("profane_sandwich", "profane_sandwich", augment_module.augment_profane_sandwich),
        (
            "argumentative_sandwich",
            "argue_sandwich",
            augment_module.augment_argumentative_sandwich,
        ),
        (
            "distractor_sandwich",
            "distractor_sandwich",
            augment_module.augment_distractor_sandwich,
        ),
        (
            "pasted_context_block",
            "pasted_context_block",
            augment_module.augment_pasted_context_block,
        ),
        (
            "telegraphic_request",
            "telegraphic_request",
            augment_module.augment_telegraphic_request,
        ),
    )
    for _dimension, _suffix, entry_point in augmenters:
        entry_point()

    originals = {item["id"]: item for item in examples}
    for dimension, suffix, _entry_point in augmenters:
        filename = augment_module.DIMENSION_FILES[dimension]
        rows = read_jsonl(tmp_path / "artifacts/generated" / filename)
        assert len(rows) == template_count
        for row in rows:
            original = originals[row["base_id"]]
            assert row["id"] == f"{row['base_id']}__{suffix}"
            assert row["category"] == original["category"]
            assert row["dimension"] == dimension
            assert row["function"] == original["function"]
            assert row["ground_truth"] == original["ground_truth"]
            assert row["oracle_preservation"] == {
                "function_schema_unchanged": True,
                "ground_truth_unchanged": True,
            }
            noisy = conversation_text(row["question"])
            wrapper = dimension in augment_module.VERBATIM_WRAPPER_DIMENSIONS
            assert (
                validate_augmented_prompt(
                    original,
                    clean,
                    noisy,
                    allow_verbatim_wrapper_noise=wrapper,
                    verbatim_source_text=clean if wrapper else None,
                )
                == []
            )


@pytest.mark.parametrize(
    ("clean", "noisy", "value", "reason"),
    [
        ("Use 12 units.", "Use 13 units.", 12, "numeric tokens changed"),
        ("Use 12 twice: 12", "Use 12 once.", 12, "numeric tokens changed"),
        ("Open 'report.csv'.", "Open 'report.tsv'.", "report.csv", "quoted literals changed"),
        ("Export the report as CSV.", "Export the report.", "CSV", "gold literal"),
    ],
)
def test_corrupted_prompts_are_rejected(clean: str, noisy: str, value: object, reason: str) -> None:
    errors = validate_augmented_prompt(example(clean, value), clean, noisy)
    assert any(reason in error for error in errors)


@pytest.mark.parametrize(
    ("clean", "noisy"),
    [("12", "312"), ("Use 12 and 12 units.", "Use 12 units.")],
)
def test_wrapper_requires_each_exact_numeric_token(clean: str, noisy: str) -> None:
    errors = validate_augmented_prompt(
        example(clean, 12),
        clean,
        noisy,
        allow_verbatim_wrapper_noise=True,
        verbatim_source_text=clean,
    )
    assert any("clean numeric token missing" in error for error in errors)


def test_wrapper_may_add_unrelated_numeric_context() -> None:
    clean = "Use 12 units."
    noisy = "I have 3 tabs open. Use 12 units."
    assert (
        validate_augmented_prompt(
            example(clean, 12),
            clean,
            noisy,
            allow_verbatim_wrapper_noise=True,
            verbatim_source_text=clean,
        )
        == []
    )
