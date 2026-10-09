"""Графики "честности" к Статье 4 — валидация по нескольким разбиениям и
генерализация на другой источник данных (к требованиям медицинского
рецензента, medical_article_requirements.txt пп.5, 7). Та же цветовая
методика, что и в остальных графиках статьи.

Запуск (после experiment_article4_multi_split_validation.py и
experiment_article4_cross_institution.py):
    python experiments/generate_article4_reliability_charts.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ARTIFACTS_DIR = Path("artifacts")
MULTI_SPLIT_PATH = ARTIFACTS_DIR / "experiment_article4_multi_split_report.json"
CROSS_INST_PATH = ARTIFACTS_DIR / "experiment_article4_cross_institution_report.json"
ARCHITECTURES_PATH = ARTIFACTS_DIR / "experiment_article4_architectures_report.json"
ACCURACY_BOOST_PATH = ARTIFACTS_DIR / "experiment_article4_accuracy_boost_report.json"

COLOR_MEAN = "#2a78d6"
COLOR_MEDIAN = "#eb6834"
COLOR_MUTED = "#8a8a86"
GRID_COLOR = "#e3e2dd"
TEXT_COLOR = "#2b2b28"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "text.color": TEXT_COLOR,
    "axes.edgecolor": "#c7c6c0",
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.titlesize": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def chart_multi_split():
    report = json.loads(MULTI_SPLIT_PATH.read_text(encoding="utf-8"))
    runs = report["runs"]
    seeds = [str(r["split_seed"]) for r in runs]
    no_ctx = [r["no_context_dice_tumor"] for r in runs]
    meaningful = [r["meaningful_dice_tumor"] for r in runs]
    shuffled = [r["shuffled_dice_tumor"] for r in runs]

    x = np.arange(len(seeds))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=150)
    ax.bar(x - width, no_ctx, width * 0.92, label="Без контекста (K=1)", color=COLOR_MUTED, zorder=3)
    ax.bar(x, meaningful, width * 0.92, label="Физические соседи (K=5)", color=COLOR_MEAN, zorder=3)
    ax.bar(x + width, shuffled, width * 0.92, label="Случайные «соседи» (K=5)", color=COLOR_MEDIAN, zorder=3)

    summary = report["summary"]
    ax.axhline(summary["meaningful_dice_tumor"]["mean"], color=COLOR_MEAN, linewidth=1.2, linestyle="--", zorder=2)
    ax.axhline(summary["no_context_dice_tumor"]["mean"], color=COLOR_MUTED, linewidth=1.2, linestyle="--", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Dice (с опухолью)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(
        "Dice по 5 независимым разбиениям train/test",
        loc="left", fontweight="bold", pad=14,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = ARTIFACTS_DIR / "article4_unet_chart_multi_split.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def chart_cross_institution():
    report = json.loads(CROSS_INST_PATH.read_text(encoding="utf-8"))
    result = report["result"]
    baseline = report["baseline_random_split_comparison"]

    categories = ["Случайное разбиение\n(все источники вперемешку)", "Другой источник\n(train=CBICA, test=TCIA+2013)"]
    values = [baseline["mean_dice_with_tumor"], result["mean_dice_with_tumor"]]
    colors = [COLOR_MEAN, COLOR_MEDIAN]

    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=150)
    bars = ax.bar(categories, values, width=0.5, color=colors, zorder=3)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1%}", (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=TEXT_COLOR, fontweight="bold")

    gap = baseline["mean_dice_with_tumor"] - result["mean_dice_with_tumor"]
    ax.annotate(
        f"разрыв\n{gap*100:.1f} п.п.",
        xy=(0.5, (values[0] + values[1]) / 2), xycoords=("data", "data"),
        ha="center", va="center", fontsize=9, color=TEXT_COLOR, style="italic",
    )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean Dice (с опухолью)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Генерализация на необученный источник данных", loc="left", fontweight="bold", pad=14)
    fig.tight_layout()
    out_path = ARTIFACTS_DIR / "article4_unet_chart_cross_institution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def chart_accuracy_boost():
    arch_report = json.loads(ARCHITECTURES_PATH.read_text(encoding="utf-8"))
    boost_report = json.loads(ACCURACY_BOOST_PATH.read_text(encoding="utf-8"))

    best_single_modality = max(
        arch_report["results"].values(), key=lambda r: r["mean_dice_with_tumor"]
    )["mean_dice_with_tumor"]
    multimodal = boost_report["final_evaluation"]["mean_dice_with_tumor"]

    categories = ["Лучшая архитектура,\nодна модальность (FLAIR)", "Мультимодальный вход\n(FLAIR+T1+T1ce+T2)"]
    values = [best_single_modality, multimodal]
    colors = [COLOR_MUTED, COLOR_MEAN]

    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=150)
    bars = ax.bar(categories, values, width=0.5, color=colors, zorder=3)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1%}", (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, color=TEXT_COLOR, fontweight="bold")

    gain = multimodal - best_single_modality
    ax.annotate(
        f"+{gain*100:.1f} п.п.",
        xy=(0.5, (values[0] + values[1]) / 2), xycoords=("data", "data"),
        ha="center", va="center", fontsize=9, color=TEXT_COLOR, style="italic",
    )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean Dice (с опухолью)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Эффект мультимодального входа", loc="left", fontweight="bold", pad=14)
    fig.tight_layout()
    out_path = ARTIFACTS_DIR / "article4_unet_chart_accuracy_boost.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_multi_split()
    chart_cross_institution()
    chart_accuracy_boost()


if __name__ == "__main__":
    main()
