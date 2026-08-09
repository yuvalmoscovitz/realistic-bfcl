from __future__ import annotations

import json
from pathlib import Path

import pytest

from realistic_bfcl import evaluate
from realistic_bfcl.common import (
    article_primary_model,
    configured_model_runs,
    model_registry,
    read_int_setting,
    read_list_setting,
)
from realistic_bfcl.evaluate import (
    estimated_cost_usd,
    estimated_invocation_cost_usd,
    input_fingerprint,
    model_execution_metadata,
)


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


def test_subset_settings_are_loaded_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "subset.yaml"
    config.write_text(
        "subset:\n  max_examples: 12\n  bfcl_categories: [simple_python, multiple]\n",
        encoding="utf-8",
    )

    assert read_int_setting(config, "max_examples") == 12
    assert read_list_setting(config, "bfcl_categories") == ["simple_python", "multiple"]


def test_subset_settings_reject_wrong_types(tmp_path: Path) -> None:
    config = tmp_path / "subset.yaml"
    config.write_text(
        "subset:\n  max_examples: 0\n  bfcl_categories: simple_python\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="positive integer"):
        read_int_setting(config, "max_examples")
    with pytest.raises(SystemExit, match="list of strings"):
        read_list_setting(config, "bfcl_categories")


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


def test_batch_cost_uses_anthropic_discount() -> None:
    model = configured_model_runs(["haiku"])[0]

    cost = estimated_cost_usd(
        model,
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        "anthropic_batch",
    )

    assert cost == 3.0


def test_mixed_batch_and_retry_costs_use_their_own_rates() -> None:
    model = configured_model_runs(["haiku"])[0]

    cost = estimated_invocation_cost_usd(
        model,
        {
            "anthropic_batch": {"input_tokens": 1_000_000},
            "synchronous_retry": {"output_tokens": 1_000_000},
        },
    )

    assert cost == 5.5


def test_openrouter_routing_is_part_of_cache_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = configured_model_runs(["glm"])[0]
    example = {"question": [], "function": [], "ground_truth": [], "id": "x"}
    original = input_fingerprint(example, model)
    monkeypatch.setenv("REALISTIC_BFCL_OPENROUTER_PROVIDER_ONLY", "AnotherProvider")

    assert input_fingerprint(example, model) != original


def test_manifest_execution_metadata_records_openrouter_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = configured_model_runs(["glm"])[0]
    monkeypatch.setenv("REALISTIC_BFCL_OPENROUTER_PROVIDER_ONLY", "AnotherProvider")

    metadata = model_execution_metadata(model)

    assert metadata["execution_backend"] == "synchronous"
    assert metadata["openrouter_routing"]["only"] == ["AnotherProvider"]


def test_article_primary_exposes_safe_output_namespace() -> None:
    primary = article_primary_model()

    assert primary.id == "gpt-5.4-nano"
    assert "/" not in primary.filename


def test_run_manifest_records_model_cost_and_wall_clock(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen_dir = tmp_path / "artifacts/frozen"
    frozen_dir.mkdir(parents=True)
    (frozen_dir / "clean_subset.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaluate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluate,
        "build_run_provenance",
        lambda: {
            "repository": {"git_sha": "a" * 40, "dirty": False},
            "bfcl": {"pinned_sha": "b" * 40, "verified_checkout_sha": "b" * 40},
            "configs": {},
            "library_versions": {"python": "test"},
        },
    )
    monkeypatch.setattr(evaluate, "result_files", lambda models, dimensions, suffix: [])
    monkeypatch.setattr(
        evaluate,
        "frozen_dataset_provenance",
        lambda dimensions: {"clean_subset_sha256": "frozen", "dimensions": {}},
    )
    monkeypatch.setattr(
        evaluate,
        "clean_baseline",
        lambda models: {
            models[0].id: {
                "wall_clock_seconds": 1.25,
                "usage": {"input_tokens": 1_000_000.0},
                "usage_by_execution": {
                    "synchronous": {"input_tokens": 1_000_000.0}
                },
                "api_calls": 1,
                "cache_hits": 0,
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
                "usage_by_execution": {
                    "synchronous": {"output_tokens": 2_000_000.0}
                },
                "api_calls": 1,
                "cache_hits": 0,
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
    assert manifest["models"][0]["api_calls"] == 2
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "complete"
    assert manifest["models"][0]["execution_backend"] == "synchronous"


def test_failed_evaluation_leaves_failed_manifest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluate,
        "build_run_provenance",
        lambda: {
            "repository": {"git_sha": "a" * 40, "dirty": False},
            "bfcl": {"pinned_sha": "b" * 40, "verified_checkout_sha": "b" * 40},
            "configs": {},
            "library_versions": {"python": "test"},
        },
    )
    monkeypatch.setattr(evaluate, "generated_dimensions", lambda: ["typos"])
    monkeypatch.setattr(
        evaluate,
        "frozen_dataset_provenance",
        lambda dimensions: {"clean_subset_sha256": "frozen", "dimensions": {"typos": "x"}},
    )

    def fail(_models: object) -> object:
        raise RuntimeError("offline failure")

    monkeypatch.setattr(evaluate, "clean_baseline", fail)

    with pytest.raises(RuntimeError, match="offline failure"):
        evaluate.run_bfcl(["nano"])

    manifests = list((tmp_path / "artifacts").glob("run-*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_type"] == "RuntimeError"


def test_provenance_failure_leaves_failed_manifest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "generated_dimensions", lambda: [])
    monkeypatch.setattr(
        evaluate,
        "frozen_dataset_provenance",
        lambda dimensions: {"clean_subset_sha256": "frozen", "dimensions": {}},
    )

    def fail() -> dict[str, object]:
        raise SystemExit("dirty checkout")

    monkeypatch.setattr(evaluate, "build_run_provenance", fail)

    with pytest.raises(SystemExit, match="dirty checkout"):
        evaluate.run_bfcl(["nano"])

    manifest_path = next((tmp_path / "artifacts").glob("run-*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_type"] == "SystemExit"


def test_result_hashing_failure_leaves_failed_manifest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluate, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluate,
        "build_run_provenance",
        lambda: {
            "repository": {"git_sha": "a" * 40, "dirty": False},
            "bfcl": {"pinned_sha": "b" * 40, "verified_checkout_sha": "b" * 40},
            "configs": {},
            "library_versions": {"python": "test"},
        },
    )
    monkeypatch.setattr(evaluate, "generated_dimensions", lambda: [])
    monkeypatch.setattr(
        evaluate,
        "frozen_dataset_provenance",
        lambda dimensions: {"clean_subset_sha256": "frozen", "dimensions": {}},
    )
    metrics = {
        "wall_clock_seconds": 0.0,
        "usage": {},
        "usage_by_execution": {},
        "api_calls": 0,
        "cache_hits": 0,
    }
    monkeypatch.setattr(
        evaluate, "clean_baseline", lambda models: {models[0].id: metrics}
    )
    monkeypatch.setattr(evaluate, "paired_eval", lambda models: {models[0].id: metrics})

    def fail_results(models: object, dimensions: object, suffix: object) -> object:
        raise SystemExit("missing result")

    monkeypatch.setattr(evaluate, "result_files", fail_results)

    with pytest.raises(SystemExit, match="missing result"):
        evaluate.run_bfcl(["nano"])

    manifest_path = next((tmp_path / "artifacts").glob("run-*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure_type"] == "SystemExit"
