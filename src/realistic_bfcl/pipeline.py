from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla"


@dataclass(frozen=True)
class Stage:
    name: str
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    next_action: str


STAGES: tuple[Stage, ...] = (
    Stage(
        name="freeze-bfcl",
        purpose="Pin BFCL source metadata and local subset configuration.",
        inputs=("configs/project.yaml", "configs/subsets/smoke.yaml"),
        outputs=("artifacts/frozen/bfcl_manifest.json",),
        next_action="Materialize the configured subset from the pinned BFCL commit.",
    ),
    Stage(
        name="clean-baseline",
        purpose="Reproduce BFCL-style clean scores before adding noise.",
        inputs=("artifacts/frozen/bfcl_manifest.json",),
        outputs=("artifacts/results/clean/",),
        next_action="Wire the BFCL evaluator adapter and run the selected models on clean prompts.",
    ),
    Stage(
        name="augment-overhang",
        purpose="Generate realistic irrelevant conversational context around clean requests.",
        inputs=("artifacts/frozen/bfcl_manifest.json", "configs/realism_dimensions.yaml"),
        outputs=("artifacts/generated/conversational_overhang.jsonl",),
        next_action=(
            "Implement the conversational overhang generator with oracle-preservation metadata."
        ),
    ),
    Stage(
        name="augment-incremental",
        purpose="Split clean requests across natural multi-turn slot revelation.",
        inputs=("artifacts/frozen/bfcl_manifest.json", "configs/realism_dimensions.yaml"),
        outputs=("artifacts/generated/incremental_slot_revelation.jsonl",),
        next_action=(
            "Implement multi-turn prompt construction without changing the final oracle."
        ),
    ),
    Stage(
        name="verify-noisy",
        purpose="Run invariant checks and realism audit before evaluation.",
        inputs=("artifacts/generated/",),
        outputs=("artifacts/audits/noisy_examples_audit.jsonl", "artifacts/accepted/"),
        next_action=(
            "Add deterministic schema and oracle checks, then attach human or LLM audit labels."
        ),
    ),
    Stage(
        name="paired-eval",
        purpose="Evaluate clean and noisy variants with identical models and schemas.",
        inputs=("artifacts/results/clean/", "artifacts/accepted/"),
        outputs=("artifacts/results/noisy/", "artifacts/results/paired/"),
        next_action=(
            "Run the evaluator for each accepted noisy variant and join results to clean runs."
        ),
    ),
    Stage(
        name="analyze",
        purpose="Compute degradation metrics and error taxonomy.",
        inputs=("artifacts/results/paired/",),
        outputs=("artifacts/analysis/metrics.json", "artifacts/analysis/error_taxonomy.csv"),
        next_action="Implement paired metrics and classify noisy failures.",
    ),
    Stage(
        name="defenses",
        purpose="Ablate simple defenses against realistic conversational noise.",
        inputs=("artifacts/accepted/", "artifacts/results/paired/"),
        outputs=("artifacts/analysis/defense_ablations/"),
        next_action=(
            "Run denoising, stricter tool-use instructions, schema variants, and decoding variants."
        ),
    ),
)


def stage_by_name(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    known = ", ".join(stage.name for stage in STAGES)
    raise SystemExit(f"Unknown stage '{name}'. Known stages: {known}")


def list_stages() -> None:
    for stage in STAGES:
        print(f"{stage.name}: {stage.purpose}")


def describe_stage(stage: Stage) -> None:
    print(f"Stage: {stage.name}")
    print(f"Purpose: {stage.purpose}")
    print("Inputs:")
    for item in stage.inputs:
        print(f"  - {item}")
    print("Outputs:")
    for item in stage.outputs:
        print(f"  - {item}")
    print(f"Next action: {stage.next_action}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_int_setting(path: Path, key: str) -> int:
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(prefix):
            return int(line.split(":", 1)[1].strip())
    raise SystemExit(f"Missing required setting '{key}' in {path.relative_to(REPO_ROOT)}")


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def freeze_bfcl() -> None:
    project_config = REPO_ROOT / "configs/project.yaml"
    subset_config = REPO_ROOT / "configs/subsets/smoke.yaml"
    manifest_path = REPO_ROOT / "artifacts/frozen/bfcl_manifest.json"
    reject_placeholders((project_config, subset_config))

    write_json(
        manifest_path,
        {
            "created_at": utc_now(),
            "bfcl": {
                "upstream_repository": BFCL_REPOSITORY,
                "dataset_commit": BFCL_COMMIT,
                "evaluator_version": f"gorilla@{BFCL_COMMIT}",
            },
            "clean_subset": {
                "config_path": "configs/subsets/smoke.yaml",
                "config_sha256": file_sha256(subset_config),
                "max_examples": read_int_setting(subset_config, "max_examples"),
                "status": "configured_not_materialized",
            },
            "local_configs": {
                "project_yaml_sha256": file_sha256(project_config),
                "subset_yaml_sha256": file_sha256(subset_config),
            },
            "model_list": {
                "status": "not_configured",
                "models": [],
            },
            "status": "source_pinned_subset_pending",
            "notes": [
                "This pins the BFCL upstream commit and hashes the local subset definition.",
                "The subset examples still need to be materialized from BFCL before "
                "model evaluation.",
            ],
        },
    )
    print(f"Wrote {manifest_path.relative_to(REPO_ROOT)}")


def clean_baseline() -> None:
    manifest_path = REPO_ROOT / "artifacts/frozen/bfcl_manifest.json"
    result_path = REPO_ROOT / "artifacts/results/clean/clean_baseline_summary.json"

    if not manifest_path.exists():
        raise SystemExit("Missing artifacts/frozen/bfcl_manifest.json. Run freeze-bfcl first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_json(
        result_path,
        {
            "created_at": utc_now(),
            "stage": "clean-baseline",
            "status": "not_run",
            "reason": (
                "BFCL subset materialization, evaluator adapter, and model list are not wired yet."
            ),
            "bfcl_manifest": "artifacts/frozen/bfcl_manifest.json",
            "bfcl_dataset_commit": manifest["bfcl"]["dataset_commit"],
            "models": [],
            "metrics": {
                "clean_total": 0,
                "clean_correct": 0,
                "clean_accuracy": None,
            },
            "next_required_work": [
                "Materialize the smoke subset from the pinned BFCL commit.",
                "Add the BFCL evaluator adapter.",
                "Configure the model list before recording clean accuracy.",
            ],
        },
    )
    print(f"Wrote {result_path.relative_to(REPO_ROOT)}")


def run_stage(stage: Stage, dry_run: bool) -> None:
    describe_stage(stage)
    if dry_run:
        return
    if stage.name == "freeze-bfcl":
        freeze_bfcl()
        return
    if stage.name == "clean-baseline":
        clean_baseline()
        return
    raise SystemExit(
        "Stage implementation pending. Use --dry-run for planning, then replace "
        "this placeholder once the corresponding artifact is implemented."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Realistic-BFCL research stages.")
    parser.add_argument("stage", nargs="?", help="Stage name to inspect or run.")
    parser.add_argument("--list", action="store_true", help="List available stages.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the stage without running it.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_stages()
        return
    if not args.stage:
        parser.error("provide a stage name or --list")

    run_stage(stage_by_name(args.stage), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
