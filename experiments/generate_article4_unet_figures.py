"""Качественная иллюстрация сегментации: FLAIR-срез, разметка и предсказания
трёх конфигураций (без контекста / физические соседи / случайные "соседи")
для нескольких показательных тестовых срезов. Источник данных — .npz,
сохранённый `experiment_article4_unet_segmentation.py::save_qualitative_examples`
(содержит реальные бинарные предсказания обученных моделей, не иллюстрацию
"на глаз").

Запуск:
    python experiments/generate_article4_unet_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACTS_DIR = Path("artifacts")
EXAMPLES_PATH = ARTIFACTS_DIR / "experiment_article4_unet_examples.npz"
OUT_PATH = ARTIFACTS_DIR / "article4_unet_fig_examples.png"

TEXT_COLOR = "#2b2b28"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "text.color": TEXT_COLOR,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

COLUMN_TITLES = ["FLAIR (срез)", "Разметка (GT)", "Без контекста\n(K=1)", "Физические\nсоседи (K=5)", "Случайные\n«соседи» (K=5)"]


def main() -> None:
    data = np.load(EXAMPLES_PATH, allow_pickle=True)
    images = data["images"]
    gt = data["gt_masks"]
    pk1 = data["pred_k1"]
    pkn = data["pred_meaningful"]
    psh = data["pred_shuffled"]
    patient_ids = data["patient_ids"]

    n_rows = images.shape[0]
    n_cols = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.55 * n_cols, 1.7 * n_rows), dpi=150)
    if n_rows == 1:
        axes = axes[None, :]

    panels = [images, gt, pk1, pkn, psh]
    cmaps = ["gray", "gray", "gray", "gray", "gray"]

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.imshow(panels[col][row], cmap=cmaps[col], vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(COLUMN_TITLES[col], fontsize=8.5, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(str(patient_ids[row]), fontsize=7.5, rotation=90)

    fig.tight_layout()
    fig.savefig(OUT_PATH)
    plt.close(fig)
    print(f"Сохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()
