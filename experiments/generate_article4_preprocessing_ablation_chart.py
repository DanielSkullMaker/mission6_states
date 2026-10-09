"""График абляции предобработки (к разделу "Результаты эксперимента"
Статьи 4). Та же цветовая методика dataviz, что и в остальных графиках.

Запуск:
    python experiments/generate_article4_preprocessing_ablation_chart.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = ARTIFACTS_DIR / "experiment_article4_preprocessing_ablation_report.json"

COLOR_MEAN = "#2a78d6"
COLOR_MEDIAN = "#eb6834"
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

ORDER = ["full_pipeline", "no_centering", "strong_threshold", "clahe_contrast", "minmax_norm"]
LABELS = ["Полный\nконвейер", "Без центри-\nрования", "Порог\nфона 40", "CLAHE-\nконтраст", "Min-max\nнормализация"]


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    results = report["results"]

    mean_vals = [results[k]["mean_dice"] for k in ORDER]
    median_vals = [results[k]["median_dice"] for k in ORDER]

    x = np.arange(len(ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=150)
    bars1 = ax.bar(x - width / 2, mean_vals, width * 0.92, label="Mean Dice", color=COLOR_MEAN, zorder=3)
    bars2 = ax.bar(x + width / 2, median_vals, width * 0.92, label="Median Dice", color=COLOR_MEDIAN, zorder=3)

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1%}", (b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Dice")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Качество сегментации по вариантам предобработки (K=1)",
                 loc="left", fontweight="bold", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=9)

    fig.tight_layout()
    out_path = ARTIFACTS_DIR / "article4_unet_chart_preprocessing_ablation.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
