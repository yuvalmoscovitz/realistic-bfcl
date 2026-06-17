from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
BFCL_CATEGORY_FILES = {
    "simple_python": (
        "BFCL_v4_simple_python.json",
        "possible_answer/BFCL_v4_simple_python.json",
    )
}


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


def read_list_setting(path: Path, key: str) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values: list[str] = []
    in_block = False
    prefix = f"{key}:"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            after_colon = stripped.split(":", 1)[1].strip()
            if after_colon == "[]":
                return []
            in_block = True
            continue
        if in_block:
            if stripped.startswith("- "):
                values.append(stripped[2:].strip())
                continue
            if stripped and not line.startswith(" "):
                break

    return values


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


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bfcl_data_root() -> Path:
    root = os.environ.get("REALISTIC_BFCL_BFCL_ROOT")
    if root:
        return Path(root) / "berkeley-function-call-leaderboard/bfcl_eval/data"
    default_root = Path(
        "/tmp/gorilla-bfcl-inspect/berkeley-function-call-leaderboard/bfcl_eval/data"
    )
    if default_root.exists():
        return default_root
    raise SystemExit(
        "Set REALISTIC_BFCL_BFCL_ROOT to a checkout of "
        "https://github.com/ShishirPatil/gorilla at the pinned commit."
    )


def materialize_smoke_subset(subset_config: Path, manifest_path: Path) -> Path:
    categories = read_list_setting(subset_config, "bfcl_categories")
    max_examples = read_int_setting(subset_config, "max_examples")
    data_root = bfcl_data_root()
    rows: list[dict[str, object]] = []

    for category in categories:
        question_file, answer_file = BFCL_CATEGORY_FILES[category]
        questions = read_jsonl(data_root / question_file)
        answers = {row["id"]: row for row in read_jsonl(data_root / answer_file)}
        for question in questions:
            answer = answers[question["id"]]
            rows.append(
                {
                    "id": question["id"],
                    "category": category,
                    "question": question["question"],
                    "function": question["function"],
                    "ground_truth": answer["ground_truth"],
                }
            )
            if len(rows) >= max_examples:
                break
        if len(rows) >= max_examples:
            break

    subset_path = manifest_path.parent / "clean_subset.jsonl"
    write_jsonl(subset_path, rows)
    return subset_path


def freeze_bfcl() -> None:
    project_config = REPO_ROOT / "configs/project.yaml"
    subset_config = REPO_ROOT / "configs/subsets/smoke.yaml"
    manifest_path = REPO_ROOT / "artifacts/frozen/bfcl_manifest.json"
    reject_placeholders((project_config, subset_config))
    categories = read_list_setting(subset_config, "bfcl_categories")
    subset_path = materialize_smoke_subset(subset_config, manifest_path)

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
                "categories": categories,
                "max_examples": read_int_setting(subset_config, "max_examples"),
                "materialized_path": "artifacts/frozen/clean_subset.jsonl",
                "materialized_sha256": file_sha256(subset_path),
                "materialized_total": len(read_jsonl(subset_path)),
                "status": "materialized",
            },
            "local_configs": {
                "project_yaml_sha256": file_sha256(project_config),
                "subset_yaml_sha256": file_sha256(subset_config),
            },
            "model_list": {
                "status": "not_configured",
                "models": [],
            },
            "status": "source_pinned_subset_materialized",
            "notes": [
                "This pins the BFCL upstream commit and materializes the local smoke subset.",
                "Model API evaluation is still pending; clean-baseline runs oracle replay only.",
            ],
        },
    )
    print(f"Wrote {manifest_path.relative_to(REPO_ROOT)}")


def clean_baseline() -> None:
    manifest_path = REPO_ROOT / "artifacts/frozen/bfcl_manifest.json"
    subset_path = REPO_ROOT / "artifacts/frozen/clean_subset.jsonl"
    result_path = REPO_ROOT / "artifacts/results/clean/clean_baseline_summary.json"
    predictions_path = REPO_ROOT / "artifacts/results/clean/oracle_replay_predictions.jsonl"

    if not manifest_path.exists():
        raise SystemExit("Missing artifacts/frozen/bfcl_manifest.json. Run freeze-bfcl first.")
    if not subset_path.exists():
        raise SystemExit("Missing artifacts/frozen/clean_subset.jsonl. Run freeze-bfcl first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = read_jsonl(subset_path)
    predictions = [
        {
            "id": example["id"],
            "model": "oracle_replay",
            "prediction": example["ground_truth"],
            "correct": True,
        }
        for example in examples
    ]
    write_jsonl(predictions_path, predictions)

    write_json(
        result_path,
        {
            "created_at": utc_now(),
            "stage": "clean-baseline",
            "status": "ran_oracle_replay",
            "reason": "Oracle replay validates subset loading, gold alignment, and result saving.",
            "bfcl_manifest": "artifacts/frozen/bfcl_manifest.json",
            "bfcl_dataset_commit": manifest["bfcl"]["dataset_commit"],
            "predictions": "artifacts/results/clean/oracle_replay_predictions.jsonl",
            "models": ["oracle_replay"],
            "metrics": {
                "clean_total": len(predictions),
                "clean_correct": len(predictions),
                "clean_accuracy": 1.0 if predictions else None,
            },
            "next_required_work": [
                "Add the BFCL evaluator adapter.",
                "Configure the real model list before recording model clean accuracy.",
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
