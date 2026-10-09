"""Строит столбчатые/линейные диаграммы к таблицам черновика Статьи 4
(дополняет generate_article4_report.py) — по запросу "добавь графики к
таблицам". Цветовая схема и правила оформления — по методике дата-визуализации
проекта (фиксированный категориальный порядок цветов, одна ось, прямые
подписи значений, лёгкая сетка, без 3D/радуги):

  слот 1 (синий #2a78d6)  — общая точность
  слот 2 (оранжевый #eb6834) — класс "опухоль"
  слот 3 (аква #1baf7a)   — класс "без опухоли"

Этот порядок закреплён и используется одинаково на всех графиках с
разбивкой по классам (рисунки 5, 7, 8, 10) — идентичность кодируется
цветом последовательно, а не перевыбирается на каждом графике.

Артефакты: artifacts/article4_chart_*.png (6 файлов).

Запуск:
    python experiments/generate_article4_charts.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ARTIFACTS_DIR = Path("artifacts")

COLOR_ACCURACY = "#2a78d6"   # слот 1 — общая точность
COLOR_TUMOR = "#eb6834"      # слот 2 — класс "опухоль"
COLOR_NOTUMOR = "#1baf7a"    # слот 3 — класс "без опухоли"
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


def _style_axes(ax, ylabel="Точность", ymax=1.0):
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def _label_bars(ax, bars, fmt="{:.1%}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)


def grouped_bar_chart(categories, series, title, out_path, ymax=1.0, figsize=(7.2, 3.6),
                       ylabel="Точность"):
    """series: list of (label, color, values) — общая/опухоль/без опухоли и т.п."""
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
    _style_axes(ax, ymax=ymax, ylabel=ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    if n_series > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=n_series,
                   frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def line_chart_with_refs(x_labels, y_values, title, out_path, refs, figsize=(7.6, 3.8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    x = np.arange(len(x_labels))
    data_xmax = len(x_labels) - 1

    ax.plot(x, y_values, color=COLOR_ACCURACY, linewidth=2, marker="o",
             markersize=6, zorder=4)
    for xi, yi in zip(x, y_values):
        ax.annotate(f"{yi:.1%}", (xi, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR, zorder=5)

    for y_ref, label in refs:
        ax.hlines(y_ref, -0.4, data_xmax + 0.3, color=COLOR_MUTED, linewidth=1,
                   linestyle="--", zorder=2)
        ax.annotate(label, (data_xmax + 0.35, y_ref), xytext=(0, 3),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8.5, color=COLOR_MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Число подпространств на класс (S)")
    _style_axes(ax, ymax=1.08)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlim(-0.4, data_xmax + 2.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def confusion_heatmap(cm, labels_ru, title, out_path, figsize=(5.2, 4.2)):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            frac = value / cm.max() if cm.max() else 0
            txt_color = "white" if frac > 0.6 else TEXT_COLOR
            ax.text(j, i, str(value), ha="center", va="center",
                     color=txt_color, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(labels_ru)))
    ax.set_yticks(range(len(labels_ru)))
    ax.set_xticklabels(labels_ru)
    ax.set_yticklabels(labels_ru)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def main() -> None:
    rs = json.loads(Path("artifacts/experiment_sequential_smoothing_report.json").read_text(encoding="utf-8"))
    rsplit = json.loads(Path("artifacts/experiment_brats_split_methods_report.json").read_text(encoding="utf-8"))
    rboost = json.loads(Path("artifacts/experiment_article4_boost_accuracy_report.json").read_text(encoding="utf-8"))
    rprep = json.loads(Path("artifacts/article4_preprocessing_ablation_report.json").read_text(encoding="utf-8"))
    rsmooth_abl = json.loads(Path("artifacts/article4_smoothing_ablation_report.json").read_text(encoding="utf-8"))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- График 1 (к Таблице 1): три конфигурации теста ---
    cfg_order = ["meaningful_smoothed", "meaningful_unsmoothed", "shuffled_smoothed"]
    cfg_labels = ["(1) содержательный\n+ сглаживание", "(2) содержательный\nбез сглаживания", "(3) перемешанный\n+ сглаживание"]
    res = rs["results"]
    grouped_bar_chart(
        cfg_labels,
        [
            ("Общая точность", COLOR_ACCURACY, [res[k]["accuracy"] for k in cfg_order]),
            ("Опухоль", COLOR_TUMOR, [res[k]["per_class_accuracy"]["tumor"] for k in cfg_order]),
            ("Без опухоли", COLOR_NOTUMOR, [res[k]["per_class_accuracy"]["no_tumor"] for k in cfg_order]),
        ],
        "Точность по трём конфигурациям тестовой последовательности",
        ARTIFACTS_DIR / "article4_chart_configurations.png",
        figsize=(7.6, 3.8),
    )

    # --- График 2 (к Таблице 2): матрица ошибок ---
    m_un = rs["results"]["meaningful_unsmoothed"]
    confusion_heatmap(
        m_un["confusion_matrix"], ["без опухоли", "опухоль"],
        "Матрица ошибок, конфигурация (2)",
        ARTIFACTS_DIR / "article4_chart_confusion.png",
    )

    # --- График 3 (к Таблице 3): абляция предобработки ---
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
        "Точность в зависимости от варианта предобработки",
        ARTIFACTS_DIR / "article4_chart_preprocessing.png",
        figsize=(7.6, 3.8),
    )

    # --- График 4 (к Таблице 4): абляция сглаживания ---
    smooth_order = ["none", "fixed_3", "fixed_5", "fixed_7", "variable_3_5_7"]
    smooth_labels = ["без\nсглаживания", "окно\n3", "окно\n5", "окно\n7", "переменное\n3/5/7"]
    sres = {r["name"]: r for r in rsmooth_abl["results"]}
    grouped_bar_chart(
        smooth_labels,
        [
            ("Общая точность", COLOR_ACCURACY, [sres[k]["accuracy"] for k in smooth_order]),
            ("Опухоль", COLOR_TUMOR, [sres[k]["per_class_accuracy"]["tumor"] for k in smooth_order]),
            ("Без опухоли", COLOR_NOTUMOR, [sres[k]["per_class_accuracy"]["no_tumor"] for k in smooth_order]),
        ],
        "Точность в зависимости от схемы сглаживания теста",
        ARTIFACTS_DIR / "article4_chart_smoothing.png",
        figsize=(7.6, 3.8),
    )

    # --- График 5 (к Таблице 5): методы сплита — точность ---
    split_order = ["patient_split", "image_random_split", "paired_correlation_split", "train_equals_test"]
    split_labels = ["непересекающиеся\nпациенты", "случайный\nпо срезам", "попарное\nразнесение", "train =\ntest"]
    rres = rsplit["results"]
    grouped_bar_chart(
        split_labels,
        [
            ("Общая точность", COLOR_ACCURACY, [rres[k]["accuracy"] for k in split_order]),
            ("Опухоль", COLOR_TUMOR, [rres[k]["per_class_accuracy"]["tumor"] for k in split_order]),
            ("Без опухоли", COLOR_NOTUMOR, [rres[k]["per_class_accuracy"]["no_tumor"] for k in split_order]),
        ],
        "Точность по способам построения выборок",
        ARTIFACTS_DIR / "article4_chart_split_accuracy.png",
        figsize=(7.6, 3.8),
    )

    # --- График 6 (к Таблице 5, доп.): доля пересечения по пациентам ---
    grouped_bar_chart(
        split_labels,
        [("Доля тестовых срезов с пациентом, видимым на обучении", COLOR_MUTED,
          [rres[k]["patient_overlap_fraction"] for k in split_order])],
        "Доля пересечения обучающей и тестовой выборок по пациентам",
        ARTIFACTS_DIR / "article4_chart_split_overlap.png",
        figsize=(7.2, 3.4), ymax=1.18, ylabel="Доля пересечения",
    )

    # --- График 7 (к Таблице 6): решётка n_subclasses, train==test ---
    grid = rboost["dishonest_push_grid"]
    x_labels = [str(r["n_subclasses"]) for r in grid]
    y_values = [r["accuracy"] for r in grid]
    honest_baseline = rboost["honest_baseline_accuracy"]
    target = rboost["target_accuracy"]
    line_chart_with_refs(
        x_labels, y_values,
        "Инфляция точности при train = test с ростом числа подпространств",
        ARTIFACTS_DIR / "article4_chart_inflation.png",
        refs=[(honest_baseline, f"честный ориентир {honest_baseline:.1%}"),
              (target, f"цель +20 п.п. ({target:.1%})")],
    )


if __name__ == "__main__":
    main()
