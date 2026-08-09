from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy

from . import __version__
from .common import (
    BFCL_COMMIT,
    DIMENSION_FILES,
    REPO_ROOT,
    ModelRun,
    configured_model_runs,
    file_sha256,
    load_yaml_mapping,
)

MANIFEST_SCHEMA_VERSION = 1


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def git_output(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Git inspection failed for {repository}: {error}") from error
    return result.stdout.strip()


def git_sha(repository: Path) -> str:
    sha = git_output(repository, "rev-parse", "HEAD")
    if not sha:
        raise SystemExit(f"Git SHA is empty for {repository}.")
    return sha


def git_dirty(repository: Path) -> bool:
    return bool(git_output(repository, "status", "--porcelain"))


def bfcl_checkout_root() -> Path:
    configured = os.environ.get("REALISTIC_BFCL_BFCL_ROOT", "").strip()
    if not configured:
        raise SystemExit(
            "Set REALISTIC_BFCL_BFCL_ROOT to the pinned BFCL repository checkout."
        )
    return Path(configured).expanduser().resolve()


def verified_bfcl_metadata() -> dict[str, object]:
    checkout = bfcl_checkout_root()
    if not checkout.exists():
        raise SystemExit(
            "Cannot verify the pinned BFCL SHA: set REALISTIC_BFCL_BFCL_ROOT to the "
            "pinned gorilla checkout."
        )
    actual = git_sha(checkout)
    if actual != BFCL_COMMIT:
        raise SystemExit(
            f"BFCL checkout SHA mismatch: expected {BFCL_COMMIT}, found {actual}."
        )
    if git_dirty(checkout):
        raise SystemExit(f"BFCL checkout must be clean before evaluation: {checkout}")
    return {
        "pinned_sha": BFCL_COMMIT,
        "verified_checkout_sha": actual,
        "dirty": False,
    }


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "realistic_bfcl": __version__,
        "numpy": numpy.__version__,
        "pyyaml": importlib.metadata.version("PyYAML"),
    }


def repository_path(relative_path: str) -> Path:
    candidate = (REPO_ROOT / relative_path).resolve()
    root = REPO_ROOT.resolve()
    if candidate != root and root not in candidate.parents:
        raise SystemExit(f"Manifest path escapes the repository: {relative_path}")
    return candidate


def subset_config_metadata() -> dict[str, object]:
    frozen_path = REPO_ROOT / "artifacts/frozen/bfcl_manifest.json"
    if not frozen_path.exists():
        raise SystemExit("Missing artifacts/frozen/bfcl_manifest.json. Run prepare-subset first.")
    try:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid frozen BFCL manifest: {error}") from error
    if not isinstance(frozen, dict):
        raise SystemExit("Invalid frozen BFCL manifest: expected a JSON object.")
    clean_subset = frozen.get("clean_subset")
    if not isinstance(clean_subset, dict) or not isinstance(clean_subset.get("config_path"), str):
        raise SystemExit("Frozen BFCL manifest is missing clean_subset.config_path.")
    config_path = repository_path(clean_subset["config_path"])
    if not config_path.exists():
        raise SystemExit(f"Subset config is missing: {clean_subset['config_path']}")
    try:
        config = load_yaml_mapping(config_path)
        subset = config["subset"]
        if not isinstance(subset, dict):
            raise TypeError("subset must be a mapping")
        selection = subset["selection"]
        if not isinstance(selection, dict):
            raise TypeError("selection must be a mapping")
        seed = selection["seed"]
    except (TypeError, KeyError) as error:
        raise SystemExit(
            f"Subset config is missing subset.selection.seed: {config_path}"
        ) from error
    if not isinstance(seed, int):
        raise SystemExit(f"Subset selection seed must be an integer: {config_path}")
    return {
        "path": config_path.relative_to(REPO_ROOT).as_posix(),
        "sha256": file_sha256(config_path),
        "selection_seed": seed,
    }


def file_record(relative_path: str) -> dict[str, str]:
    path = repository_path(relative_path)
    if not path.exists():
        raise SystemExit(f"Required provenance file is missing: {relative_path}")
    return {"path": relative_path, "sha256": file_sha256(path)}


def build_run_provenance() -> dict[str, object]:
    dimensions_config = REPO_ROOT / "configs/realism_dimensions.yaml"
    models_config = REPO_ROOT / "configs/models.yaml"
    review_labels = REPO_ROOT / "configs/article_failure_review_labels.csv"
    repository_sha = git_sha(REPO_ROOT)
    if git_dirty(REPO_ROOT):
        raise SystemExit("Realistic-BFCL checkout must be clean before evaluation.")
    return {
        "repository": {
            "git_sha": repository_sha,
            "dirty": False,
        },
        "bfcl": verified_bfcl_metadata(),
        "configs": {
            "subset": subset_config_metadata(),
            "dimensions": file_record(dimensions_config.relative_to(REPO_ROOT).as_posix()),
            "models": file_record(models_config.relative_to(REPO_ROOT).as_posix()),
            "article_review_labels": file_record(
                review_labels.relative_to(REPO_ROOT).as_posix()
            ),
        },
        "library_versions": runtime_versions(),
    }


def frozen_dataset_provenance(dimensions: list[str]) -> dict[str, object]:
    clean_subset = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    if not clean_subset.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run prepare-subset first.")
    hashes = {}
    for dimension in dimensions:
        path = REPO_ROOT / "artifacts/generated" / DIMENSION_FILES[dimension]
        if not path.exists():
            raise SystemExit(f"Missing frozen dimension artifact: {dimension}")
        hashes[dimension] = file_sha256(path)
    return {
        "clean_subset_sha256": file_sha256(clean_subset),
        "dimensions": hashes,
    }


def result_files(
    models: list[ModelRun], dimensions: list[str], suffix: str
) -> list[dict[str, str]]:
    records = []
    for model in models:
        for dimension in dimensions:
            for kind in ("paired", "summary"):
                relative_path = canonical_result_path(model, dimension, kind, suffix)
                path = REPO_ROOT / relative_path
                if not path.exists():
                    raise SystemExit(f"Evaluation did not produce required result: {relative_path}")
                records.append(
                    {
                        "kind": kind,
                        "model": model.id,
                        "dimension": dimension,
                        "path": relative_path,
                        "sha256": file_sha256(path),
                    }
                )
    return records


def canonical_result_path(model: ModelRun, dimension: str, kind: str, suffix: str) -> str:
    result_dir = Path("artifacts/results/paired") / dimension
    if suffix:
        result_dir /= suffix
    extension = "paired.jsonl" if kind == "paired" else "summary.json"
    return (result_dir / f"{model.filename}_{extension}").as_posix()


def read_manifest(path: Path) -> dict[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid run manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise SystemExit(f"Invalid run manifest {path}: expected a JSON object.")
    return manifest


def validate_manifest_header(manifest: dict[str, object], path: Path) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SystemExit(f"Unsupported run manifest schema in {path}.")
    if manifest.get("status") != "complete":
        raise SystemExit(f"Run manifest is not complete: {path}")
    if not isinstance(manifest.get("started_at"), str) or not isinstance(
        manifest.get("completed_at"), str
    ):
        raise SystemExit("Run manifest is missing run timestamps.")
    versions = manifest.get("library_versions")
    if not isinstance(versions, dict) or not all(
        isinstance(versions.get(name), str)
        for name in ("python", "realistic_bfcl", "numpy", "pyyaml")
    ):
        raise SystemExit("Run manifest is missing relevant library versions.")


def validate_repository_provenance(manifest: dict[str, object]) -> None:
    repository = manifest.get("repository")
    if (
        not isinstance(repository, dict)
        or repository.get("git_sha") != git_sha(REPO_ROOT)
        or repository.get("dirty") is not False
        or git_dirty(REPO_ROOT)
    ):
        raise SystemExit("Run manifest repository state does not match the current checkout.")
    bfcl = manifest.get("bfcl")
    if (
        not isinstance(bfcl, dict)
        or bfcl.get("pinned_sha") != BFCL_COMMIT
        or bfcl.get("verified_checkout_sha") != BFCL_COMMIT
        or bfcl.get("dirty") is not False
    ):
        raise SystemExit("Run manifest BFCL SHA is missing or inconsistent with the pinned SHA.")


def validate_config_provenance(manifest: dict[str, object]) -> dict[str, object]:
    configs = manifest.get("configs")
    if not isinstance(configs, dict):
        raise SystemExit("Run manifest is missing config provenance.")
    for name in ("subset", "dimensions", "models", "article_review_labels"):
        record = configs.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise SystemExit(f"Run manifest is missing {name} config provenance.")
        config_path = repository_path(record["path"])
        if not config_path.exists() or record.get("sha256") != file_sha256(config_path):
            raise SystemExit(f"Run manifest {name} config hash does not match the current file.")
    subset_record = configs["subset"]
    if not isinstance(subset_record.get("selection_seed"), int):
        raise SystemExit("Run manifest is missing the subset selection seed.")
    return configs


def validate_frozen_provenance(manifest: dict[str, object]) -> dict[str, object]:
    frozen = manifest.get("frozen_dataset")
    if not isinstance(frozen, dict):
        raise SystemExit("Run manifest is missing frozen-dataset provenance.")
    clean_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    if not clean_path.exists() or frozen.get("clean_subset_sha256") != file_sha256(clean_path):
        raise SystemExit("Run manifest clean-subset hash does not match the current artifact.")
    dimension_hashes = frozen.get("dimensions")
    if not isinstance(dimension_hashes, dict):
        raise SystemExit("Run manifest is missing frozen dimension hashes.")
    for dimension, expected_hash in dimension_hashes.items():
        if dimension not in DIMENSION_FILES:
            raise SystemExit(f"Run manifest contains an unknown dimension: {dimension}")
        generated_path = REPO_ROOT / "artifacts/generated" / DIMENSION_FILES[dimension]
        if not generated_path.exists() or expected_hash != file_sha256(generated_path):
            raise SystemExit(f"Run manifest dimension hash does not match: {dimension}")
    return dimension_hashes


def validate_model_provenance(manifest: dict[str, object]) -> dict[str, ModelRun]:
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise SystemExit("Run manifest does not contain any evaluated models.")
    model_ids: list[str] = []
    for record in models:
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not record["id"]
            or not isinstance(record.get("sampling"), dict)
        ):
            raise SystemExit("Run manifest contains an invalid model record.")
        model_ids.append(record["id"])
    if len(set(model_ids)) != len(models):
        raise SystemExit("Run manifest contains duplicate model records.")
    configured_models = {model.id: model for model in configured_model_runs(model_ids)}
    for record in models:
        configured = configured_models[str(record["id"])]
        if (
            record.get("provider") != configured.provider
            or record.get("tier") != configured.tier
            or record.get("sampling") != configured.sampling_parameters
        ):
            raise SystemExit(f"Run manifest model config does not match: {configured.id}")
        backend = record.get("execution_backend")
        if backend not in {"synchronous", "anthropic_batch"}:
            raise SystemExit(f"Run manifest has invalid execution backend: {configured.id}")
        routing = record.get("openrouter_routing")
        if configured.provider == "openrouter" and not isinstance(routing, dict):
            raise SystemExit(f"Run manifest is missing OpenRouter routing: {configured.id}")
        if configured.provider != "openrouter" and routing is not None:
            raise SystemExit(f"Run manifest has unexpected OpenRouter routing: {configured.id}")
    return configured_models


def expected_result_paths(
    configured_models: dict[str, ModelRun], dimension_hashes: dict[str, object]
) -> dict[tuple[str, str, str], str]:
    return {
        (kind, model_id, dimension): canonical_result_path(model, dimension, kind, "")
        for model_id, model in configured_models.items()
        for dimension in dimension_hashes
        for kind in ("paired", "summary")
    }


def validate_result_record(
    record: object,
    expected_paths: dict[tuple[str, str, str], str],
    seen: set[tuple[object, object, object]],
) -> None:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise SystemExit("Run manifest contains an invalid result-file record.")
    key = (record.get("kind"), record.get("model"), record.get("dimension"))
    if key in seen or key not in expected_paths or record["path"] != expected_paths[key]:
        raise SystemExit("Run manifest result-file set is incomplete or inconsistent.")
    seen.add(key)
    result_path = repository_path(record["path"])
    if not result_path.exists() or record.get("sha256") != file_sha256(result_path):
        raise SystemExit(f"Run manifest result hash does not match: {record['path']}")


def validate_result_provenance(
    manifest: dict[str, object],
    configured_models: dict[str, ModelRun],
    dimension_hashes: dict[str, object],
) -> None:
    results = manifest.get("results")
    if not isinstance(results, dict) or not isinstance(results.get("files"), list):
        raise SystemExit("Run manifest is missing result-file provenance.")
    suffix = results.get("result_suffix")
    if suffix:
        raise SystemExit("Analyze currently requires an unsuffixed run manifest.")
    expected_paths = expected_result_paths(configured_models, dimension_hashes)
    files = results["files"]
    if len(files) != len(expected_paths):
        raise SystemExit("Run manifest result-file set is incomplete or inconsistent.")
    seen: set[tuple[object, object, object]] = set()
    for record in files:
        validate_result_record(record, expected_paths, seen)
    if seen != set(expected_paths):
        raise SystemExit("Run manifest result-file set is incomplete or inconsistent.")


def load_validated_manifest(explicit_path: Path | None = None) -> dict[str, object]:
    if explicit_path is None:
        raise SystemExit("Analyze requires --run-manifest with a completed evaluation manifest.")
    path = repository_path(str(explicit_path))
    if not path.exists():
        raise SystemExit(f"Run manifest does not exist: {path}")
    manifest = read_manifest(path)
    validate_manifest_header(manifest, path)
    validate_repository_provenance(manifest)
    validate_config_provenance(manifest)
    dimension_hashes = validate_frozen_provenance(manifest)
    configured_models = validate_model_provenance(manifest)
    validate_result_provenance(manifest, configured_models, dimension_hashes)
    return manifest
