"""Графики для «поддельной» версии Статьи 4 (train==test, n_subclasses=45) —
та же стилистика/цветовая схема, что и generate_article4_charts.py (метод
дата-визуализации проекта), для прямой визуальной сопоставимости оригинала
и подделки.

Артефакты: artifacts/dishonest_article4_chart_*.png (3 файла).

Запуск:
    python experiments/generate_dishonest_article4_charts.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ARTIFACTS_DIR = Path("artifacts")

COLOR_ACCURACY = "#2a78d6"
COLOR_TUMOR = "#eb6834"
COLOR_NOTUMOR = "#1baf7a"
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
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def _style_axes(ax, ymax=1.0):
    ax.set_ylim(0, ymax)
    ax.set_ylabel("Точность")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def _label_bars(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1%}", (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)


def grouped_bar_chart(categories, series, title, out_path, figsize=(7.6, 3.8)):
    n_series = len(series)
    x = np.arange(len(categories))
    width = 0.8 / n_series
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    for i, (label, color, values) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width * 0.92, label=label, color=color, zorder=3)
        _label_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    _style_axes(ax, ymax=1.12)
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    if n_series > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=n_series, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def main() -> None:
    rconf = json.loads(Path("artifacts/dishonest_article4_configurations_report.json").read_text(encoding="utf-8"))
    rprep = json.loads(Path("artifacts/dishonest_article4_preprocessing_report.json").read_text(encoding="utf-8"))
    rsmooth = json.loads(Path("artifacts/dishonest_article4_smoothing_report.json").read_text(encoding="utf-8"))

    cfg_order = ["meaningful_smoothed", "meaningful_unsmoothed", "shuffled_smoothed"]
    cfg_labels = ["(1) псевдо-порядок\n+ сглаживание", "(2) псевдо-порядок\nбез сглаживания", "(3) перемешанный\n+ сглаживание"]
    res = rconf["results"]
    grouped_bar_chart(
        cfg_labels,
        [
            ("Общая точность", COLOR_ACCURACY, [res[k]["accuracy"] for k in cfg_order]),
            ("Опухоль", COLOR_TUMOR, [res[k]["per_class_accuracy"]["tumor"] for k in cfg_order]),
            ("Без опухоли", COLOR_NOTUMOR, [res[k]["per_class_accuracy"]["no_tumor"] for k in cfg_order]),
        ],
        "Точность по трём конфигурациям (train = test, S=45)",
        ARTIFACTS_DIR / "dishonest_article4_chart_configurations.png",
    )

    prep_order = ["full_pipeline", "resize_only", "suppress_no_center", "strong_threshold", "minmax_norm"]
    prep_labels = ["(а) полный\nцикл", "(б) только\nразмер", "(в) без\nцентрирования", "(г) порог\n40", "(д) min-max\nнормализация"]
    pres = {r["name"]: r for r in rprep["results"]}
    grouped_bar_chart(
        prep_labels,
        [
            ("Общая точность", COLOR_ACCURACY, [pres[k]["accuracy"] for k in prep_order]),
            ("Опухоль", COLOR_TUMOR, [pres[k]["per_class_accuracy"]["tumor"] for k in prep_order]),
            ("Без опухоли", COLOR_NOTUMOR, [pres[k]["per_class_accuracy"]["no_tumor"] for k in prep_order]),
        ],
        "Точность по вариантам предобработки (train = test, S=45)",
        ARTIFACTS_DIR / "dishonest_article4_chart_preprocessing.png",
    )

    smooth_order = ["none", "fixed_3", "fixed_5", "fixed_7", "variable_3_5_7"]
    smooth_labels = ["без\nсглаживания", "окно\n3", "окно\n5", "окно\n7", "переменное\n3/5/7"]
    sres = {r["name"]: r for r in rsmooth["results"]}
    grouped_bar_chart(
        smooth_labels,
        [
            ("Общая точность", COLOR_ACCURACY, [sres[k]["accuracy"] for k in smooth_order]),
            ("Опухоль", COLOR_TUMOR, [sres[k]["per_class_accuracy"]["tumor"] for k in smooth_order]),
            ("Без опухоли", COLOR_NOTUMOR, [sres[k]["per_class_accuracy"]["no_tumor"] for k in smooth_order]),
        ],
        "Точность по схеме сглаживания (train = test, S=45)",
        ARTIFACTS_DIR / "dishonest_article4_chart_smoothing.png",
    )


if __name__ == "__main__":
    main()
