from __future__ import annotations

import json

import pytest

from realistic_bfcl import evaluate
from realistic_bfcl.common import configured_model_runs, model_registry
from realistic_bfcl.evaluate import estimated_cost_usd


def test_model_registry_spans_tiers_and_has_unique_namespaces() -> None:
    models = model_registry()

    assert {model.name for model in models} == {"nano", "haiku", "glm", "frontier"}
    assert "frontier" in {model.tier for model in models}
    assert len({model.filename for model in models}) == len(models)
    assert all(model.sampling_parameters["temperature"] == 0 for model in models)


def test_model_selection_accepts_names_and_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REALISTIC_BFCL_MODELS", raising=False)

    models = configured_model_runs(["haiku", "gpt-5.4-nano"])

    assert [model.name for model in models] == ["haiku", "nano"]


def test_model_selection_fails_loudly_for_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REALISTIC_BFCL_MODELS", raising=False)

    with pytest.raises(SystemExit, match="Unknown model selector"):
        configured_model_runs(["not-a-model"])


def test_estimated_cost_uses_provider_token_names() -> None:
    model = configured_model_runs(["nano"])[0]

    cost = estimated_cost_usd(model, {"prompt_tokens": 1_000_000, "completion_tokens": 2_000_000})

    assert cost == 2.70


def test_run_manifest_records_model_cost_and_wall_clock(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_dir = tmp_path / "artifacts/frozen"
    frozen_dir.mkdir(parents=True)
    (frozen_dir / "clean_subset.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaluate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluate,
        "clean_baseline",
        lambda models: {
            models[0].id: {
                "wall_clock_seconds": 1.25,
                "usage": {"input_tokens": 1_000_000.0},
            }
        },
    )
    monkeypatch.setattr(
        evaluate,
        "paired_eval",
        lambda models: {
            models[0].id: {
                "wall_clock_seconds": 2.75,
                "usage": {"output_tokens": 2_000_000.0},
            }
        },
    )
    monkeypatch.setattr(evaluate, "generated_dimensions", lambda: [])

    evaluate.run_bfcl(["nano"])

    manifests = list((tmp_path / "artifacts").glob("run-*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["models"][0]["estimated_cost_usd"] == 2.70
    assert manifest["models"][0]["wall_clock_seconds"] == 4.0
