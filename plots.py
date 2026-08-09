from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "artifacts/analysis/significance.csv"
OUTPUT_DIR = ROOT / "docs/figures"

DIMENSIONS = (
    "telegraphic_request",
    "cursing",
    "irrelevant_context",
    "argumentative_challenge",
    "pasted_context_block",
    "removed_spaces",
    "typos",
)
DIMENSION_LABELS = {
    "telegraphic_request": "Telegraphic request",
    "cursing": "Cursing",
    "pasted_context_block": "Pasted context block",
    "argumentative_challenge": "Argumentative challenge",
    "irrelevant_context": "Irrelevant context",
    "removed_spaces": "Removed spaces",
    "typos": "Typos",
}
MODEL_ORDER = ("gpt-5.4-nano", "claude-haiku-4-5-20251001", "z-ai/glm-4.6")
MODEL_LABELS = {
    "gpt-5.4-nano": "GPT-5.4 nano",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "z-ai/glm-4.6": "GLM-4.6",
}
BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#6B7280"
ALPHA = 0.05


class SignificanceRow(TypedDict):
    model: str
    dimension: str
    clean_acc: float
    noisy_acc: float
    degradation: float
    ci_low: float
    ci_high: float
    b: int
    c: int
    n_pairs: int
    n_discordant: int
    p_value: float
    p_adjusted: float
    significant: bool


def parse_significance_row(source: dict[str, str]) -> SignificanceRow:
    significant_text = source["significant"].lower()
    if significant_text not in {"true", "false"}:
        raise ValueError("Significant must be True or False.")
    return {
        "model": source["model"],
        "dimension": source["dimension"],
        "clean_acc": float(source["clean_acc"]),
        "noisy_acc": float(source["noisy_acc"]),
        "degradation": float(source["degradation"]),
        "ci_low": float(source["ci_low"]),
        "ci_high": float(source["ci_high"]),
        "b": int(source["b"]),
        "c": int(source["c"]),
        "n_pairs": int(source["n_pairs"]),
        "n_discordant": int(source["n_discordant"]),
        "p_value": float(source["p_value"]),
        "p_adjusted": float(source["p_adjusted"]),
        "significant": significant_text == "true",
    }


def validate_significance_row(row: SignificanceRow) -> None:
    numeric = (
        row["degradation"],
        row["clean_acc"],
        row["noisy_acc"],
        row["ci_low"],
        row["ci_high"],
        row["p_value"],
        row["p_adjusted"],
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("Significance table contains a non-finite value.")
    validate_paired_values(row)
    validate_probability_values(row)


def validate_paired_values(row: SignificanceRow) -> None:
    b, c, n_pairs = int(row["b"]), int(row["c"]), int(row["n_pairs"])
    n_discordant = int(row["n_discordant"])
    if min(b, c, n_discordant) < 0 or n_pairs <= 0:
        raise ValueError("Significance counts must be non-negative with positive pairs.")
    if b + c != n_discordant or b + c > n_pairs:
        raise ValueError("Discordant count must equal b + c and not exceed n_pairs.")
    degradation = float(row["degradation"])
    if not math.isclose(degradation, (b - c) / n_pairs, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Degradation is inconsistent with paired counts.")
    if not math.isclose(
        degradation,
        float(row["clean_acc"]) - float(row["noisy_acc"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Degradation is inconsistent with reported accuracies.")
    if not float(row["ci_low"]) <= float(row["ci_high"]):
        raise ValueError("Confidence interval endpoints are reversed.")


def validate_probability_values(row: SignificanceRow) -> None:
    if not 0 <= row["clean_acc"] <= 1 or not 0 <= row["noisy_acc"] <= 1:
        raise ValueError("Accuracies must be between zero and one.")
    if not 0 <= row["p_value"] <= 1 or not 0 <= row["p_adjusted"] <= 1:
        raise ValueError("P-values must be between zero and one.")
    if float(row["p_adjusted"]) < float(row["p_value"]):
        raise ValueError("Adjusted p-value cannot be smaller than raw p-value.")
    if bool(row["significant"]) != (float(row["p_adjusted"]) <= ALPHA):
        raise ValueError("Significance flag is inconsistent with adjusted p-value.")


def validate_significance_table(rows: list[SignificanceRow]) -> None:
    keys = {(row["model"], row["dimension"]) for row in rows}
    expected = {(model, dimension) for model in MODEL_ORDER for dimension in DIMENSIONS}
    if keys != expected or len(rows) != len(expected):
        raise ValueError("Significance table must contain one row per model and dimension.")
    if len({int(row["n_pairs"]) for row in rows}) != 1:
        raise ValueError("Significance rows must use a common pair count.")
    for model in MODEL_ORDER:
        clean_accuracies = {float(row["clean_acc"]) for row in rows if row["model"] == model}
        if len(clean_accuracies) != 1:
            raise ValueError(f"Clean accuracy must be constant within model: {model}")


def read_significance(path: Path = INPUT) -> list[SignificanceRow]:
    required = {
        "model",
        "dimension",
        "clean_acc",
        "noisy_acc",
        "degradation",
        "ci_low",
        "ci_high",
        "b",
        "c",
        "n_pairs",
        "n_discordant",
        "p_value",
        "p_adjusted",
        "significant",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing significance columns: {', '.join(sorted(missing))}")
        rows = []
        for source in reader:
            row = parse_significance_row(source)
            validate_significance_row(row)
            rows.append(row)
    validate_significance_table(rows)
    return rows


def rows_by_model(rows: list[SignificanceRow]) -> dict[str, dict[str, SignificanceRow]]:
    return {
        model: {
            dimension: next(
                row for row in rows if row["model"] == model and row["dimension"] == dimension
            )
            for dimension in DIMENSIONS
        }
        for model in MODEL_ORDER
    }


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.6)
    axis.set_axisbelow(True)


def plot_degradation(rows: list[SignificanceRow], output: Path) -> None:
    grouped = rows_by_model(rows)
    figure, axes = plt.subplots(3, 1, figsize=(8, 8.4), sharex=True, constrained_layout=True)
    y = list(range(len(DIMENSIONS)))
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        for position, dimension in zip(y, DIMENSIONS, strict=True):
            row = grouped[model][dimension]
            estimate = float(row["degradation"]) * 100
            low = float(row["ci_low"]) * 100
            high = float(row["ci_high"]) * 100
            significant = bool(row["significant"])
            axis.errorbar(
                estimate,
                position,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                markersize=6,
                color=BLUE if significant else GRAY,
                markerfacecolor=BLUE if significant else "white",
                markeredgewidth=1.3,
                capsize=3,
                linewidth=1.4,
            )
        axis.axvline(0, color="#374151", linewidth=0.8)
        axis.set_yticks(y, [DIMENSION_LABELS[item] for item in DIMENSIONS])
        axis.invert_yaxis()
        axis.set_title(MODEL_LABELS[model], loc="left", fontsize=11, fontweight="bold", pad=7)
        style_axis(axis)
    figure.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, linestyle="none", label="Holm-significant"),
            Line2D(
                [],
                [],
                marker="o",
                color=GRAY,
                markerfacecolor="white",
                linestyle="none",
                label="Not significant",
            ),
        ],
        frameon=False,
        ncol=2,
        loc="outside upper center",
    )
    figure.supxlabel("Clean minus noisy accuracy (percentage points)")
    figure.savefig(
        output,
        format="png",
        dpi=160,
        facecolor="white",
        metadata={"Software": "realistic-bfcl"},
    )
    plt.close(figure)


def plot_discordance(rows: list[SignificanceRow], output: Path) -> None:
    grouped = rows_by_model(rows)
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    y = list(range(len(DIMENSIONS)))
    for axis, model in zip(axes, MODEL_ORDER, strict=True):
        regressions = [int(grouped[model][dimension]["b"]) for dimension in DIMENSIONS]
        recoveries = [int(grouped[model][dimension]["c"]) for dimension in DIMENSIONS]
        axis.barh([value - 0.18 for value in y], regressions, height=0.34, color=BLUE)
        axis.barh([value + 0.18 for value in y], recoveries, height=0.34, color=ORANGE)
        axis.set_yticks(y, [DIMENSION_LABELS[item] for item in DIMENSIONS])
        axis.invert_yaxis()
        axis.set_title(MODEL_LABELS[model], loc="left", fontsize=11, fontweight="bold", pad=7)
        style_axis(axis)
    figure.legend(
        handles=[
            Patch(color=BLUE, label="Clean correct, noisy wrong"),
            Patch(color=ORANGE, label="Clean wrong, noisy correct"),
        ],
        frameon=False,
        ncol=2,
        loc="outside upper center",
    )
    figure.supxlabel("Paired examples")
    figure.savefig(
        output,
        format="png",
        dpi=160,
        facecolor="white",
        metadata={"Software": "realistic-bfcl"},
    )
    plt.close(figure)


def generate_figures(input_path: Path = INPUT, output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    rows = read_significance(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    degradation = output_dir / "degradation_by_dimension.png"
    discordance = output_dir / "discordance_decomposition.png"
    plot_degradation(rows, degradation)
    plot_discordance(rows, discordance)
    return degradation, discordance


if __name__ == "__main__":
    for figure_path in generate_figures():
        print(figure_path.relative_to(ROOT))
