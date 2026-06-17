from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import types
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
OPENAI_MODEL = "gpt-5.4-nano"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
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


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def openai_api_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]

    candidates = [
        Path(os.environ["REALISTIC_BFCL_ENV_FILE"])
        if os.environ.get("REALISTIC_BFCL_ENV_FILE")
        else None,
        REPO_ROOT.parent / "underlayer/.env",
    ]
    for path in candidates:
        if path is None:
            continue
        key = read_env_file(path).get("OPENAI_API_KEY")
        if key:
            return key

    raise SystemExit(
        "Missing OPENAI_API_KEY. Set it in the environment or REALISTIC_BFCL_ENV_FILE."
    )


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


def bfcl_eval_root() -> Path:
    root = os.environ.get("REALISTIC_BFCL_BFCL_ROOT")
    if root:
        return Path(root) / "berkeley-function-call-leaderboard"
    default_root = Path("/tmp/gorilla-bfcl-inspect/berkeley-function-call-leaderboard")
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


def openai_type(type_name: str) -> str:
    return {"dict": "object", "float": "number"}.get(type_name, type_name)


def normalize_json_schema(value: object) -> object:
    if isinstance(value, dict):
        normalized = {}
        for key, child in value.items():
            if key == "type" and isinstance(child, str):
                normalized[key] = openai_type(child)
            else:
                normalized[key] = normalize_json_schema(child)
        return normalized
    if isinstance(value, list):
        return [normalize_json_schema(item) for item in value]
    return value


def safe_tool_name(name: str, index: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return f"{safe}_{index}"


def openai_tool(function_doc: dict[str, object], name: str) -> dict[str, object]:
    parameters = normalize_json_schema(function_doc["parameters"])
    return {
        "type": "function",
        "name": name,
        "description": function_doc.get("description", ""),
        "parameters": parameters,
    }


def call_openai_tool_router(example: dict[str, object], api_key: str) -> dict[str, object]:
    messages = example["question"][0]
    user_content = "\n".join(message["content"] for message in messages)
    tools = []
    for index, function_doc in enumerate(example["function"]):
        tools.append(openai_tool(function_doc, safe_tool_name(str(function_doc["name"]), index)))
    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": (
                    "Call the provided tool that best satisfies the user request. "
                    "Do not answer in prose when a tool call is appropriate."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "tools": tools,
        "tool_choice": "required",
        "max_output_tokens": 256,
    }
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise SystemExit(f"OpenAI API request failed: HTTP {error.code}: {body}") from error


def tool_name_map(example: dict[str, object]) -> dict[str, str]:
    return {
        safe_tool_name(str(function_doc["name"]), index): str(function_doc["name"])
        for index, function_doc in enumerate(example["function"])
    }


def response_function_calls(
    response: dict[str, object], name_map: dict[str, str]
) -> list[dict[str, object]]:
    calls = []
    for item in response.get("output", []):
        if item.get("type") != "function_call":
            continue
        try:
            arguments = json.loads(item.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"__malformed_arguments__": item.get("arguments")}
        name = str(item.get("name"))
        calls.append({"name": name_map.get(name, name), "arguments": arguments})
    return calls


def load_bfcl_ast_checker() -> tuple[object, object]:
    eval_root = str(bfcl_eval_root())
    if eval_root not in sys.path:
        sys.path.insert(0, eval_root)

    # ast_checker only needs this mapping for dot/underscore function-name conversion.
    # Importing the full upstream model registry pulls every provider SDK.
    if "bfcl_eval.constants.model_config" not in sys.modules:
        model_config = types.ModuleType("bfcl_eval.constants.model_config")
        model_config.MODEL_CONFIG_MAPPING = {
            OPENAI_MODEL: types.SimpleNamespace(underscore_to_dot=False)
        }
        sys.modules["bfcl_eval.constants.model_config"] = model_config

    ast_checker = importlib.import_module(
        "bfcl_eval.eval_checker.ast_eval.ast_checker"
    ).ast_checker
    language = importlib.import_module("bfcl_eval.constants.enums").Language.PYTHON
    return ast_checker, language


def bfcl_model_output(calls: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{call["name"]: call["arguments"]} for call in calls]


def bfcl_ast_result(
    example: dict[str, object], calls: list[dict[str, object]]
) -> dict[str, object]:
    ast_checker, language = load_bfcl_ast_checker()
    return ast_checker(
        example["function"],
        bfcl_model_output(calls),
        example["ground_truth"],
        language,
        str(example["category"]),
        OPENAI_MODEL,
    )


def aggregate_usage(predictions: list[dict[str, object]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for prediction in predictions:
        usage = prediction.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


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
                "status": "configured",
                "models": [OPENAI_MODEL],
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
    oracle_predictions_path = REPO_ROOT / "artifacts/results/clean/oracle_replay_predictions.jsonl"
    model_predictions_path = REPO_ROOT / f"artifacts/results/clean/{OPENAI_MODEL}_predictions.jsonl"

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
    write_jsonl(oracle_predictions_path, predictions)

    if model_predictions_path.exists() and not os.environ.get("REALISTIC_BFCL_FORCE_MODEL_RUN"):
        model_predictions = read_jsonl(model_predictions_path)
        if len(model_predictions) != len(examples):
            raise SystemExit(
                f"Existing {model_predictions_path.relative_to(REPO_ROOT)} has "
                f"{len(model_predictions)} rows, expected {len(examples)}. "
                "Set REALISTIC_BFCL_FORCE_MODEL_RUN=1 to rerun model calls."
            )
        print(f"Reused {model_predictions_path.relative_to(REPO_ROOT)}")
    else:
        api_key = openai_api_key()
        model_predictions = []
        for example in examples:
            response = call_openai_tool_router(example, api_key)
            calls = response_function_calls(response, tool_name_map(example))
            eval_result = bfcl_ast_result(example, calls)
            model_predictions.append(
                {
                    "id": example["id"],
                    "model": OPENAI_MODEL,
                    "prediction": calls,
                    "correct": eval_result["valid"],
                    "evaluator": "bfcl_ast_checker",
                    "eval_result": eval_result,
                    "response_id": response.get("id"),
                    "usage": response.get("usage"),
                }
            )
            print(f"Ran {OPENAI_MODEL} on {example['id']}")
        write_jsonl(model_predictions_path, model_predictions)

    examples_by_id = {example["id"]: example for example in examples}
    rescored_model_predictions = []
    for prediction in model_predictions:
        calls = prediction["prediction"]
        eval_result = bfcl_ast_result(examples_by_id[prediction["id"]], calls)
        rescored_prediction = dict(prediction)
        rescored_prediction["correct"] = eval_result["valid"]
        rescored_prediction["evaluator"] = "bfcl_ast_checker"
        rescored_prediction["eval_result"] = eval_result
        rescored_model_predictions.append(rescored_prediction)
    model_predictions = rescored_model_predictions
    write_jsonl(model_predictions_path, model_predictions)
    model_correct = sum(1 for prediction in model_predictions if prediction["correct"])

    write_json(
        result_path,
        {
            "created_at": utc_now(),
            "stage": "clean-baseline",
            "status": "ran_model_baseline",
            "reason": "Ran oracle replay and a real OpenAI model baseline.",
            "bfcl_manifest": "artifacts/frozen/bfcl_manifest.json",
            "bfcl_dataset_commit": manifest["bfcl"]["dataset_commit"],
            "predictions": {
                "oracle_replay": "artifacts/results/clean/oracle_replay_predictions.jsonl",
                OPENAI_MODEL: f"artifacts/results/clean/{OPENAI_MODEL}_predictions.jsonl",
            },
            "models": ["oracle_replay", OPENAI_MODEL],
            "metrics": {
                "oracle_replay": {
                    "clean_total": len(predictions),
                    "clean_correct": len(predictions),
                    "clean_accuracy": 1.0 if predictions else None,
                    "usage": {},
                },
                OPENAI_MODEL: {
                    "clean_total": len(model_predictions),
                    "clean_correct": model_correct,
                    "clean_accuracy": model_correct / len(model_predictions)
                    if model_predictions
                    else None,
                    "usage": aggregate_usage(model_predictions),
                },
            },
            "next_required_work": [
                "Compare this clean baseline against noisy variants.",
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
