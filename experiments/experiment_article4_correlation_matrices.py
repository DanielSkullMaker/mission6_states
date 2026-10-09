"""Статья 4 — матрицы корреляции (косинусного сходства) между срезами,
визуализирующие саму предпосылку статьи: "соседние по индексу срезы
физически похожи" (условие корректности сглаживания, см. раздел
"Теория"). Использует уже существующую публичную функцию библиотеки
``core.metrics.cosine_similarity_matrix`` — та же, что метод применяет на
этапе поиска центров подклассов (фаза A.1) — здесь использована
диагностически, не для обучения.

Берётся один пациент (BraTS20_Training_001 — тот же, что на рисунках 1/2/5
черновика статьи), ВСЕ его срезы в содержательном порядке (по slice_idx) и
в перемешанном порядке (та же перестановка, что мог бы дать случайный
сплит). Косинусное сходство содержательного порядка переставляется той же
перестановкой — математически это ТА ЖЕ матрица с переставленными
строками/столбцами (sim_shuffled = sim_content[perm][:, perm]), что и
corrected demonstрирует: структура близости никуда не делась, просто
перестала быть видна вдоль диагонали после потери содержательного порядка.

Дополнительно считается средняя корреляция по "лагу" |i-j| — количественно
подтверждает затухание похожести с расстоянием, на котором основан выбор
окна 3/5/7 в разделе "Теория".

Артефакты:
  - artifacts/article4_chart_correlation_matrices.png (2 тепловые карты)
  - artifacts/article4_chart_correlation_decay.png (график затухания по лагу)
  - artifacts/article4_correlation_report.json (числа для текста статьи)

Запуск (после experiments/prepare_brats_dataset.py):
    python experiments/experiment_article4_correlation_matrices.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from subspace_conjugacy import cosine_similarity_matrix, load_and_vectorize_batch

DATASET_ROOT = Path("datasets/brats_flair_slices")
PATIENT_ID = "BraTS20_Training_001"
RANDOM_SEED = 42

ARTIFACTS_DIR = Path("artifacts")
FIG_MATRICES_PATH = ARTIFACTS_DIR / "article4_chart_correlation_matrices.png"
FIG_DECAY_PATH = ARTIFACTS_DIR / "article4_chart_correlation_decay.png"
REPORT_PATH = ARTIFACTS_DIR / "article4_correlation_report.json"

LAGS = [1, 2, 3, 5, 10, 20, 40, 80]

COLOR_ACCURACY = "#2a78d6"
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
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_patient_sequence(patient_id: str) -> np.ndarray:
    patient_dir = DATASET_ROOT / patient_id
    paths = sorted(patient_dir.glob(f"{patient_id}_slice*.png"),
                    key=lambda p: int(p.stem.split("slice")[-1]))
    return load_and_vectorize_batch(paths, method="horizontal")


def mean_similarity_by_lag(sim: np.ndarray, lags) -> dict:
    n = sim.shape[0]
    result = {}
    for lag in lags:
        if lag >= n:
            continue
        values = [sim[i, i + lag] for i in range(n - lag)]
        result[lag] = float(np.mean(values))
    return result


def plot_heatmaps(sim_content: np.ndarray, sim_shuffled: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), dpi=150)
    fig.suptitle(
        "Матрицы косинусного сходства между срезами одного пациента",
        x=0.07, y=0.98, ha="left", fontsize=11, fontweight="bold",
    )
    for ax, sim, title in zip(
        axes, [sim_content, sim_shuffled],
        ["Содержательный порядок (по индексу среза)", "Перемешанный порядок"],
    ):
        im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, loc="left", fontsize=9.5, pad=10)
        ax.set_xlabel("Индекс в последовательности")
        ax.set_ylabel("Индекс в последовательности")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(top=0.82, right=0.88, wspace=0.35)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.03, fraction=0.05,
                         label="Косинусное сходство")
    cbar.outline.set_visible(False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def plot_decay(decay_content: dict, decay_shuffled: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    lags = sorted(decay_content.keys())
    ax.plot(lags, [decay_content[l] for l in lags], color=COLOR_ACCURACY,
            marker="o", markersize=6, linewidth=2, label="Содержательный порядок", zorder=4)
    ax.plot(lags, [decay_shuffled[l] for l in lags], color=COLOR_MUTED,
            marker="s", markersize=5, linewidth=1.5, linestyle="--",
            label="Перемешанный порядок", zorder=3)
    ax.set_xlabel("Лаг |i − j| (расстояние между срезами по индексу)")
    ax.set_ylabel("Среднее косинусное сходство")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Затухание сходства срезов с расстоянием", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right" if decay_content[lags[0]] > decay_content[lags[-1]] else "lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def main() -> None:
    print(f"=== Матрицы корреляции: {PATIENT_ID} ===")
    X_content = load_patient_sequence(PATIENT_ID)
    n = X_content.shape[0]
    print(f"Срезов: {n}")

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)

    sim_content = cosine_similarity_matrix(X_content)
    sim_shuffled = sim_content[np.ix_(perm, perm)]

    decay_content = mean_similarity_by_lag(sim_content, LAGS)
    decay_shuffled = mean_similarity_by_lag(sim_shuffled, LAGS)

    off_diag_mask = ~np.eye(n, dtype=bool)
    overall_mean = float(sim_content[off_diag_mask].mean())

    print("Лаг | содержательный | перемешанный")
    for lag in LAGS:
        if lag in decay_content:
            print(f"{lag:>4} | {decay_content[lag]:.4f}         | {decay_shuffled[lag]:.4f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(sim_content, sim_shuffled, FIG_MATRICES_PATH)
    plot_decay(decay_content, decay_shuffled, FIG_DECAY_PATH)

    report = {
        "patient_id": PATIENT_ID,
        "n_slices": n,
        "random_seed": RANDOM_SEED,
        "overall_mean_similarity": overall_mean,
        "decay_content_order": decay_content,
        "decay_shuffled_order": decay_shuffled,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON-отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
