from __future__ import annotations

import json

import pytest

from realistic_bfcl import evaluate
from realistic_bfcl.common import ModelRun, write_jsonl


def model(provider: str = "openai") -> ModelRun:
    return ModelRun("test", "test-model", provider, "test", 0, 128)


def example(example_id: str = "example-1") -> dict[str, object]:
    return {
        "id": example_id,
        "category": "simple_python",
        "question": [[{"role": "user", "content": "Weather in Haifa?"}]],
        "function": [
            {
                "name": "weather.lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"city name": {"type": "string"}},
                    "required": ["city name"],
                },
            }
        ],
        "ground_truth": [{"weather.lookup": {"city name": "Haifa"}}],
    }


@pytest.mark.parametrize(
    ("provider", "response"),
    [
        (
            "openai",
            {
                "output": [
                    {"type": "message"},
                    {
                        "type": "function_call",
                        "name": "weather_lookup_0",
                        "arguments": '{"city name": "Haifa"}',
                    },
                ]
            },
        ),
        (
            "openrouter",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "weather_lookup_0",
                                        "arguments": '{"city name": "Haifa"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        ),
        (
            "anthropic",
            {
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "name": "weather_lookup_0",
                        "input": {"city_name": "Haifa"},
                    },
                ]
            },
        ),
    ],
)
def test_provider_parsers_normalize_tool_calls(
    provider: str, response: dict[str, object]
) -> None:
    assert evaluate.function_calls(response, example(), model(provider)) == [
        {"name": "weather.lookup", "arguments": {"city name": "Haifa"}}
    ]


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_json_argument_parse_failure_is_preserved(provider: str) -> None:
    if provider == "openai":
        response = {
            "output": [
                {"type": "function_call", "name": "weather_lookup_0", "arguments": "{"}
            ]
        }
    else:
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "weather_lookup_0",
                                    "arguments": "{",
                                }
                            }
                        ]
                    }
                }
            ]
        }

    calls = evaluate.function_calls(response, example(), model(provider))
    assert calls[0]["arguments"] == {"__malformed_arguments__": "{"}


@pytest.mark.parametrize(
    ("provider", "response", "message"),
    [
        ("openai", {"output": {}}, "output must be a list"),
        ("openrouter", {"choices": [{"message": []}]}, "message must be an object"),
        ("anthropic", {"content": {}}, "content must be a list"),
    ],
)
def test_provider_parsers_reject_invalid_envelopes(
    provider: str, response: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate.function_calls(response, example(), model(provider))


def test_anthropic_restores_safe_names_inside_arrays() -> None:
    schema = {
        "type": "object",
        "properties": {
            "trip legs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"from city": {"type": "string"}},
                    "required": ["from city"],
                },
            }
        },
        "required": ["trip legs"],
    }
    converted, schema_map = evaluate.anthropic_safe_schema(schema)

    assert converted["required"] == ["trip_legs"]
    assert evaluate.restore_anthropic_arguments(
        {"trip_legs": [{"from_city": "Haifa"}]}, schema_map
    ) == {"trip legs": [{"from city": "Haifa"}]}

    tuple_schema = {
        "type": "array",
        "items": [
            {"type": "object", "properties": {"first key": {"type": "string"}}},
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"nested key": {"type": "string"}},
                },
            },
        ],
    }
    _converted, tuple_map = evaluate.anthropic_safe_schema(tuple_schema)
    assert evaluate.restore_anthropic_arguments(
        [{"first_key": "one"}, [{"nested_key": "two"}]], tuple_map
    ) == [{"first key": "one"}, [{"nested key": "two"}]]


def test_bfcl_adapter_forwards_the_exact_scoring_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: tuple[object, ...] = ()

    def checker(*args: object) -> dict[str, object]:
        nonlocal captured
        captured = args
        return {"valid": True}

    monkeypatch.setattr(evaluate, "load_bfcl_ast_checker", lambda: (checker, "python"))
    fixture = example()
    calls = [{"name": "weather.lookup", "arguments": {"city name": "Haifa"}}]

    assert evaluate.bfcl_ast_result(fixture, calls, model()) == {"valid": True}
    assert captured == (
        fixture["function"],
        [{"weather.lookup": {"city name": "Haifa"}}],
        fixture["ground_truth"],
        "python",
        fixture["category"],
        "test-model",
    )


@pytest.mark.parametrize("stale", [False, True])
def test_prediction_cache_reuses_only_matching_inputs(
    tmp_path, monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    fixture = example()
    configured_model = model()
    cache_path = tmp_path / "predictions.jsonl"
    fingerprint = evaluate.input_fingerprint(fixture, configured_model)
    write_jsonl(
        cache_path,
        [
            {
                "id": fixture["id"],
                "prediction": [],
                "input_fingerprint": "stale" if stale else fingerprint,
            }
        ],
    )
    calls = 0

    def run(_example: object, _model: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"id": fixture["id"], "prediction": [], "api_attempts": 1}

    monkeypatch.setattr(evaluate, "run_model_prediction", run)
    monkeypatch.setattr(evaluate, "bfcl_ast_result", lambda *_args: {"valid": True})
    monkeypatch.setattr(evaluate, "openai_concurrency", lambda: 1)

    predictions, invocation = evaluate.run_or_load_model_predictions(
        [fixture], cache_path, configured_model
    )

    assert calls == int(stale)
    assert invocation["cache_hits"] == int(not stale)
    assert predictions[0]["correct"] is True
    stored = [json.loads(line) for line in cache_path.read_text().splitlines()]
    assert len(stored) == 1
    assert stored[0]["input_fingerprint"] == fingerprint


def test_cache_schema_versions_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    original = evaluate.input_fingerprint(example(), model())
    monkeypatch.setattr(evaluate, "CACHE_SCHEMA_VERSION", 2)
    assert evaluate.input_fingerprint(example(), model()) != original


def test_duplicate_evaluation_ids_fail_before_model_calls(tmp_path) -> None:
    with pytest.raises(SystemExit, match="Duplicate id"):
        evaluate.run_or_load_model_predictions(
            [example("same"), example("same")], tmp_path / "predictions.jsonl", model()
        )


def test_paired_metrics_known_truth_table_and_strict_booleans() -> None:
    rows = [
        {"clean_correct": clean, "noisy_correct": noisy}
        for clean, noisy in [(True, True), (True, False), (False, True), (False, False)]
    ]
    metrics = evaluate.paired_metrics(rows)

    assert metrics["clean_accuracy"] == metrics["noisy_accuracy"] == 0.5
    assert metrics["absolute_degradation"] == 0
    assert metrics["clean_success_noisy_failure"] == 1
    assert metrics["clean_failure_noisy_success"] == 1
    assert metrics["both_correct"] == metrics["both_wrong"] == 1
    assert metrics["conditional_failure_given_clean_success"] == 0.5

    with pytest.raises(ValueError, match="must be booleans"):
        evaluate.paired_metrics([{"clean_correct": "false", "noisy_correct": False}])


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"id": "noisy", "base_id": "a", "dimension": "typos"},
            {"id": "noisy", "base_id": "b", "dimension": "typos"},
        ],
        [
            {"id": "one", "base_id": "a", "dimension": "typos"},
            {"id": "two", "base_id": "a", "dimension": "typos"},
        ],
        [{"id": "one", "base_id": "a", "dimension": "cursing"}],
    ],
)
def test_pair_integrity_fails_loudly(rows: list[dict[str, object]]) -> None:
    with pytest.raises(SystemExit, match="Duplicate|Mismatched"):
        evaluate.validate_noisy_pairs(rows, "typos")
