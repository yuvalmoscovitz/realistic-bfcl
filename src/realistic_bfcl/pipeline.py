from __future__ import annotations

import argparse
from dataclasses import dataclass

from .analyze import analyze
from .augment import augment
from .evaluate import freeze_bfcl, run_bfcl
from .llm_augment import augment_llm_pilot
from .rewrite_subset import build_rewrite_subset


@dataclass(frozen=True)
class Stage:
    name: str
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    next_action: str


STAGES: tuple[Stage, ...] = (
    Stage(
        name="prepare-subset",
        purpose="Freeze the reproducible BFCL clean subset used as the substrate.",
        inputs=("configs/project.yaml", "configs/subsets/smoke.yaml"),
        outputs=(
            "artifacts/frozen/bfcl_manifest.json",
            "artifacts/frozen/clean_subset.jsonl",
        ),
        next_action="Run augment to construct the oracle-preserving noisy dataset.",
    ),
    Stage(
        name="augment",
        purpose="Construct the frozen noisy dataset with oracle-preserving transforms.",
        inputs=("artifacts/frozen/bfcl_manifest.json", "configs/realism_dimensions.yaml"),
        outputs=(
            "artifacts/generated/typos.jsonl",
            "artifacts/generated/cursing.jsonl",
            "artifacts/generated/irrelevant_context.jsonl",
            "artifacts/generated/removed_spaces.jsonl",
            "artifacts/generated/argumentative_challenge.jsonl",
            "artifacts/generated/profane_sandwich.jsonl",
            "artifacts/generated/argumentative_sandwich.jsonl",
            "artifacts/generated/distractor_sandwich.jsonl",
            "artifacts/generated/pasted_context_block.jsonl",
            "artifacts/generated/telegraphic_request.jsonl",
            "artifacts/generated/augmentation_review.csv",
        ),
        next_action="Run BFCL clean/noisy paired evaluation on the frozen dataset.",
    ),
    Stage(
        name="augment-llm-pilot",
        purpose="Generate saved LLM-based realistic augmentations for a small review pilot.",
        inputs=("artifacts/frozen/clean_subset.jsonl",),
        outputs=(
            "artifacts/generated/llm_work_context.jsonl",
            "artifacts/generated/llm_prior_thread.jsonl",
            "artifacts/generated/llm_conversation_history.jsonl",
            "artifacts/generated/llm_messy_pre_intent_history.jsonl",
            "artifacts/generated/llm_profane_frustration.jsonl",
            "artifacts/generated/llm_argumentative_challenge.jsonl",
            "artifacts/generated/llm_frustrated_distractor_context.jsonl",
            "artifacts/generated/llm_super_casual_abbreviations.jsonl",
            "artifacts/generated/llm_frustrated_swearing.jsonl",
            "artifacts/generated/llm_student_broke_context.jsonl",
            "artifacts/generated/llm_typos_shorthand.jsonl",
            "artifacts/generated/llm_rambling_overexplaining.jsonl",
            "artifacts/generated/llm_impatient_direct_attitude.jsonl",
            "artifacts/generated/llm_arguing_correcting_ai.jsonl",
            "artifacts/generated/llm_confused_overwhelmed.jsonl",
            "artifacts/generated/llm_swearing_urgency_work.jsonl",
            "artifacts/generated/llm_vague_slightly_aggressive.jsonl",
            "artifacts/generated/llm_work_context_review.csv",
            "artifacts/generated/llm_prior_thread_review.csv",
            "artifacts/generated/llm_conversation_history_review.csv",
            "artifacts/generated/llm_messy_pre_intent_history_review.csv",
            "artifacts/generated/llm_profane_frustration_review.csv",
            "artifacts/generated/llm_argumentative_challenge_review.csv",
            "artifacts/generated/llm_frustrated_distractor_context_review.csv",
            "artifacts/generated/llm_super_casual_abbreviations_review.csv",
            "artifacts/generated/llm_frustrated_swearing_review.csv",
            "artifacts/generated/llm_student_broke_context_review.csv",
            "artifacts/generated/llm_typos_shorthand_review.csv",
            "artifacts/generated/llm_rambling_overexplaining_review.csv",
            "artifacts/generated/llm_impatient_direct_attitude_review.csv",
            "artifacts/generated/llm_arguing_correcting_ai_review.csv",
            "artifacts/generated/llm_confused_overwhelmed_review.csv",
            "artifacts/generated/llm_swearing_urgency_work_review.csv",
            "artifacts/generated/llm_vague_slightly_aggressive_review.csv",
        ),
        next_action=(
            "Review accepted rows, then run paired evaluation for the LLM pilot dimensions."
        ),
    ),
    Stage(
        name="build-rewrite-subset",
        purpose="Build the 500-example rewrite-suitable subset for LLM augmentation.",
        inputs=("artifacts/frozen/clean_subset.jsonl",),
        outputs=(
            "artifacts/frozen/rewrite_suitable_500.jsonl",
            "artifacts/frozen/rewrite_suitable_500_review.csv",
            "artifacts/frozen/rewrite_suitable_500_summary.json",
        ),
        next_action=(
            "Run a 50-example LLM augmentation/eval pilot using "
            "REALISTIC_BFCL_LLM_SELECTION=rewrite_suitable."
        ),
    ),
    Stage(
        name="run-bfcl",
        purpose="Run clean and noisy BFCL-style evaluation with identical schemas.",
        inputs=("artifacts/frozen/clean_subset.jsonl", "artifacts/generated/"),
        outputs=(
            "artifacts/results/clean/",
            "artifacts/results/noisy/",
            "artifacts/results/paired/",
        ),
        next_action="Analyze paired clean-vs-noisy degradation and failure types.",
    ),
    Stage(
        name="analyze",
        purpose="Compute degradation metrics and error taxonomy.",
        inputs=("artifacts/results/paired/",),
        outputs=(
            "artifacts/analysis/benchmark_summary.csv",
            "artifacts/analysis/benchmark_summary.json",
            "artifacts/analysis/model_comparison.csv",
            "artifacts/analysis/flip_review.csv",
            "artifacts/analysis/regression_review.csv",
            "artifacts/analysis/article_failure_review.csv",
            "artifacts/analysis/article_failure_examples.csv",
            "artifacts/analysis/article/paired_stats.csv",
            "artifacts/analysis/article/realism_audit_summary.csv",
            "artifacts/analysis/article/review_filtering.csv",
            "artifacts/analysis/article/stability_repeat_summary.csv",
            "artifacts/analysis/article/stability_repeat_runs.csv",
            "artifacts/analysis/article/stability_repeat_summary.json",
        ),
        next_action=(
            "Use adjusted regression metrics to decide whether the pilot is ready to scale."
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
    print(f"Step: {stage.name}")
    print(f"Purpose: {stage.purpose}")
    print("Inputs:")
    for item in stage.inputs:
        print(f"  - {item}")
    print("Outputs:")
    for item in stage.outputs:
        print(f"  - {item}")
    print(f"Next action: {stage.next_action}")


def run_stage(stage: Stage, dry_run: bool, models: list[str] | None = None) -> None:
    describe_stage(stage)
    if dry_run:
        return
    if stage.name == "prepare-subset":
        freeze_bfcl()
        return
    if stage.name == "augment":
        augment()
        return
    if stage.name == "augment-llm-pilot":
        augment_llm_pilot()
        return
    if stage.name == "build-rewrite-subset":
        build_rewrite_subset()
        return
    if stage.name == "run-bfcl":
        run_bfcl(models)
        return
    if stage.name == "analyze":
        analyze()
        return
    raise SystemExit(f"Stage implementation missing: {stage.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Realistic-BFCL research steps.")
    parser.add_argument("stage", nargs="?", help="Step name to inspect or run.")
    parser.add_argument("--list", action="store_true", help="List available steps.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Describe the step without running it.",
    )
    parser.add_argument(
        "--models",
        help=(
            "Comma-separated model names or ids from configs/models.yaml. "
            "Only valid for run-bfcl."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        list_stages()
        return
    if not args.stage:
        parser.error("provide a step name or --list")

    selected_models = None
    if args.models:
        if args.stage != "run-bfcl":
            parser.error("--models is only valid for the run-bfcl stage")
        selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
        if not selected_models:
            parser.error("--models must contain at least one model name or id")

    run_stage(stage_by_name(args.stage), dry_run=args.dry_run, models=selected_models)


if __name__ == "__main__":
    main()
