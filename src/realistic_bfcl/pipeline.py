from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


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
        purpose="Pin BFCL dataset, evaluator, model list, and clean subset.",
        inputs=("configs/project.yaml", "configs/subsets/smoke.yaml"),
        outputs=("artifacts/frozen/bfcl_manifest.json",),
        next_action="Download BFCL, record immutable commits, and write a clean subset manifest.",
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
        next_action="Implement the conversational overhang generator with oracle-preservation metadata.",
    ),
    Stage(
        name="augment-incremental",
        purpose="Split clean requests across natural multi-turn slot revelation.",
        inputs=("artifacts/frozen/bfcl_manifest.json", "configs/realism_dimensions.yaml"),
        outputs=("artifacts/generated/incremental_slot_revelation.jsonl",),
        next_action="Implement multi-turn prompt construction without changing the final oracle.",
    ),
    Stage(
        name="verify-noisy",
        purpose="Run invariant checks and realism audit before evaluation.",
        inputs=("artifacts/generated/",),
        outputs=("artifacts/audits/noisy_examples_audit.jsonl", "artifacts/accepted/"),
        next_action="Add deterministic schema and oracle checks, then attach human or LLM audit labels.",
    ),
    Stage(
        name="paired-eval",
        purpose="Evaluate clean and noisy variants with identical models and schemas.",
        inputs=("artifacts/results/clean/", "artifacts/accepted/"),
        outputs=("artifacts/results/noisy/", "artifacts/results/paired/"),
        next_action="Run the evaluator for each accepted noisy variant and join results to clean runs.",
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
        next_action="Run denoising, stricter tool-use instructions, schema variants, and decoding variants.",
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


def run_stage(stage: Stage, dry_run: bool) -> None:
    describe_stage(stage)
    if dry_run:
        return
    raise SystemExit(
        "Stage implementation pending. Use --dry-run for planning, then replace "
        "this placeholder once the corresponding artifact is implemented."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Realistic-BFCL research stages.")
    parser.add_argument("stage", nargs="?", help="Stage name to inspect or run.")
    parser.add_argument("--list", action="store_true", help="List available stages.")
    parser.add_argument("--dry-run", action="store_true", help="Describe the stage without running it.")
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
