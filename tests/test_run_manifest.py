from __future__ import annotations

import json
from pathlib import Path

import pytest

from realistic_bfcl import analyze, pipeline, run_manifest
from realistic_bfcl.common import BFCL_COMMIT, file_sha256


def write(path: Path, contents: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def test_bfcl_checkout_requires_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REALISTIC_BFCL_BFCL_ROOT", raising=False)

    with pytest.raises(SystemExit, match="REALISTIC_BFCL_BFCL_ROOT"):
        run_manifest.bfcl_checkout_root()


def valid_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(run_manifest, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_manifest, "git_sha", lambda repository: "a" * 40)
    monkeypatch.setattr(run_manifest, "git_dirty", lambda repository: False)

    subset_config = write(
        tmp_path / "configs/subsets/smoke.yaml",
        "subset:\n  selection:\n    seed: 20260616\n",
    )
    dimensions_config = write(tmp_path / "configs/realism_dimensions.yaml", "dimensions: {}\n")
    models_config = write(tmp_path / "configs/models.yaml", "models: []\n")
    review_labels = write(tmp_path / "configs/article_failure_review_labels.csv", "id,label\n")
    clean_subset = write(tmp_path / "artifacts/frozen/clean_subset.jsonl", "{}\n")
    generated = write(tmp_path / "artifacts/generated/typos.jsonl", "{}\n")
    paired = write(
        tmp_path / "artifacts/results/paired/typos/gpt-5.4-nano_paired.jsonl",
        "{}\n",
    )
    summary = write(
        tmp_path / "artifacts/results/paired/typos/gpt-5.4-nano_summary.json",
        "{}\n",
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_id": "run-test",
        "started_at": "2026-08-09T00:00:00+00:00",
        "completed_at": "2026-08-09T00:01:00+00:00",
        "repository": {"git_sha": "a" * 40, "dirty": False},
        "bfcl": {
            "pinned_sha": BFCL_COMMIT,
            "verified_checkout_sha": BFCL_COMMIT,
            "dirty": False,
        },
        "configs": {
            "subset": {
                "path": "configs/subsets/smoke.yaml",
                "sha256": file_sha256(subset_config),
                "selection_seed": 20260616,
            },
            "dimensions": {
                "path": "configs/realism_dimensions.yaml",
                "sha256": file_sha256(dimensions_config),
            },
            "models": {
                "path": "configs/models.yaml",
                "sha256": file_sha256(models_config),
            },
            "article_review_labels": {
                "path": "configs/article_failure_review_labels.csv",
                "sha256": file_sha256(review_labels),
            },
        },
        "library_versions": {
            "python": "3.12.0",
            "realistic_bfcl": "0.1.0",
            "numpy": "1.26.0",
            "pyyaml": "6.0.0",
        },
        "frozen_dataset": {
            "clean_subset_sha256": file_sha256(clean_subset),
            "dimensions": {"typos": file_sha256(generated)},
        },
        "models": [
            {
                "id": "gpt-5.4-nano",
                "provider": "openai",
                "tier": "small",
                "sampling": {"temperature": 0, "max_output_tokens": 256},
                "execution_backend": "synchronous",
                "openrouter_routing": None,
            }
        ],
        "results": {
            "result_suffix": "",
            "files": [
                {
                    "kind": "paired",
                    "model": "gpt-5.4-nano",
                    "dimension": "typos",
                    "path": paired.relative_to(tmp_path).as_posix(),
                    "sha256": file_sha256(paired),
                },
                {
                    "kind": "summary",
                    "model": "gpt-5.4-nano",
                    "dimension": "typos",
                    "path": summary.relative_to(tmp_path).as_posix(),
                    "sha256": file_sha256(summary),
                },
            ],
        },
    }
    manifest_path = tmp_path / "artifacts/run-test/manifest.json"
    write(manifest_path, json.dumps(manifest))
    return manifest_path


def test_validated_manifest_accepts_bound_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = valid_manifest(tmp_path, monkeypatch)

    manifest = run_manifest.load_validated_manifest(manifest_path)

    assert manifest["run_id"] == "run-test"


def test_explicit_manifest_is_required() -> None:
    with pytest.raises(SystemExit, match="requires --run-manifest"):
        run_manifest.load_validated_manifest()


def test_manifest_rejects_tampered_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = valid_manifest(tmp_path, monkeypatch)
    paired = tmp_path / "artifacts/results/paired/typos/gpt-5.4-nano_paired.jsonl"
    paired.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="result hash does not match"):
        run_manifest.load_validated_manifest(manifest_path)


def test_manifest_rejects_incomplete_result_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = valid_manifest(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["results"]["files"].pop()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="result-file set is incomplete"):
        run_manifest.load_validated_manifest(manifest_path)


def test_manifest_rejects_hashed_alternate_result_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = valid_manifest(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical = tmp_path / manifest["results"]["files"][0]["path"]
    alternate = write(tmp_path / "artifacts/results/paired/typos/alternate.jsonl", "{}\n")
    assert file_sha256(alternate) == file_sha256(canonical)
    manifest["results"]["files"][0]["path"] = alternate.relative_to(tmp_path).as_posix()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="result-file set is incomplete"):
        run_manifest.load_validated_manifest(manifest_path)


def test_manifest_rejects_repository_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = valid_manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(run_manifest, "git_sha", lambda repository: "b" * 40)

    with pytest.raises(SystemExit, match="repository state"):
        run_manifest.load_validated_manifest(manifest_path)


def test_manifest_rejects_model_registry_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = valid_manifest(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"][0]["provider"] = "different-provider"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SystemExit, match="model config does not match"):
        run_manifest.load_validated_manifest(manifest_path)


def test_explicit_manifest_must_stay_inside_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_manifest, "REPO_ROOT", tmp_path)

    with pytest.raises(SystemExit, match="escapes the repository"):
        run_manifest.load_validated_manifest(Path("../outside/manifest.json"))


def test_provenance_refuses_dirty_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_manifest, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_manifest, "git_sha", lambda repository: "a" * 40)
    monkeypatch.setattr(run_manifest, "git_dirty", lambda repository: True)

    with pytest.raises(SystemExit, match="checkout must be clean"):
        run_manifest.build_run_provenance()


def test_bfcl_provenance_refuses_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_manifest, "bfcl_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(run_manifest, "git_sha", lambda repository: BFCL_COMMIT)
    monkeypatch.setattr(run_manifest, "git_dirty", lambda repository: True)

    with pytest.raises(SystemExit, match="BFCL checkout must be clean"):
        run_manifest.verified_bfcl_metadata()


def test_atomic_json_write_leaves_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "artifacts/run-test/manifest.json"

    run_manifest.atomic_write_json(path, {"run_id": "run-test"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"run_id": "run-test"}
    assert list(path.parent.glob(".manifest.json.*.tmp")) == []


def test_article_bundle_does_not_overwrite_stability_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    article_dir = tmp_path / "artifacts/analysis/article"
    paths = [
        write(article_dir / name, "checked-in\n")
        for name in (
            "stability_repeat_runs.csv",
            "stability_repeat_summary.csv",
            "stability_repeat_summary.json",
        )
    ]
    monkeypatch.setattr(analyze, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(analyze, "article_review_labels", lambda: {})

    analyze.write_article_bundle([], [], [])

    assert all(path.read_text(encoding="utf-8") == "checked-in\n" for path in paths)


def test_analyze_refuses_before_reading_results_without_manifest() -> None:
    with pytest.raises(SystemExit, match="requires --run-manifest"):
        analyze.analyze()


def test_cli_passes_explicit_manifest_to_analyze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Path | None] = []
    monkeypatch.setattr(pipeline, "analyze", captured.append)

    pipeline.main(["analyze", "--run-manifest", "artifacts/run-test/manifest.json"])

    assert captured == [Path("artifacts/run-test/manifest.json")]
