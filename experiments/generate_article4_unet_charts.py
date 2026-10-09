"""Графики к результатам нейросетевой (U-Net, сегментация) версии Статьи 4.

Та же цветовая методика, что и в generate_article4_charts.py (предыдущая,
теперь неактуальная версия статьи) — фиксированный категориальный порядок
цветов, одна ось на график, прямые подписи значений, лёгкая сетка:

  слот 1 (синий #2a78d6)    — mean Dice
  слот 2 (оранжевый #eb6834) — median Dice
  слот 3 (приглушённый серый #8a8a86) — пиксельная accuracy (отдельный
    график — другая единица измерения и другой смысл, см. docstring
    experiment_article4_unet_segmentation.py про обманчивость accuracy)

Запуск:
    python experiments/generate_article4_unet_charts.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = ARTIFACTS_DIR / "experiment_article4_unet_report.json"

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


def _style_axes(ax, ylabel, ymax=1.0):
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def _label_bars(ax, bars, fmt="{:.1%}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)


def grouped_bar_chart(categories, series, title, out_path, ymax=1.0, figsize=(7.2, 3.8),
                       ylabel="Значение"):
    n_series = len(series)
    n_cat = len(categories)
    x = np.arange(n_cat)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    for i, (label, color, values) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width * 0.92, label=label,
                       color=color, zorder=3)
        _label_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    _style_axes(ax, ylabel=ylabel, ymax=ymax)
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    if n_series > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=n_series,
                   frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    results = report["results"]

    cfg_order = ["no_context_k1", "meaningful_neighbors", "shuffled_neighbors"]
    cfg_labels = ["без контекста\n(K=1)", "физические\nсоседи (K=5)", "случайные\n«соседи» (K=5)"]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- График 1: mean/median Dice по трём конфигурациям ---
    grouped_bar_chart(
        cfg_labels,
        [
            ("Mean Dice", COLOR_MEAN, [results[k]["mean_dice"] for k in cfg_order]),
            ("Median Dice", COLOR_MEDIAN, [results[k]["median_dice"] for k in cfg_order]),
        ],
        "Качество сегментации (Dice) по трём конфигурациям",
        ARTIFACTS_DIR / "article4_unet_chart_dice.png",
        ylabel="Dice",
        figsize=(7.2, 3.8),
    )

    # --- График 2: пиксельная accuracy (отдельно, чтобы показать её "плоскость") ---
    grouped_bar_chart(
        cfg_labels,
        [("Пиксельная accuracy", COLOR_MUTED, [results[k]["pixel_accuracy"] for k in cfg_order])],
        "Пиксельная accuracy по тем же трём конфигурациям",
        ARTIFACTS_DIR / "article4_unet_chart_pixel_accuracy.png",
        ylabel="Accuracy",
        ymax=1.05,
        figsize=(7.2, 3.2),
    )


if __name__ == "__main__":
    main()
