"""График кривых обучения (train vs test Dice по эпохам) для всех 8 моделей
сегментации — к `experiment_article4_overfitting_analysis.py`. Малая
мультипликация (4x2): train Dice (оранжевый, штрих) против test Dice "с
опухолью" (синий, сплошной, основная метрика статьи), пик теста отмечен
точкой, начало переобучения (если наблюдалось в пределах сетки эпох) —
вертикальной серой линией. Та же цветовая методика dataviz, что и в
остальных графиках статьи.

Запуск (после experiment_article4_overfitting_analysis.py):
    python experiments/generate_article4_overfitting_chart.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = ARTIFACTS_DIR / "experiment_article4_overfitting_report.json"

COLOR_TEST = "#2a78d6"
COLOR_TRAIN = "#eb6834"
COLOR_OVERFIT = "#8a8a84"
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
    "axes.titlesize": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

ORDER = [
    "unet_baseline", "plain_no_skip", "res_unet", "attention_unet",
    "segnet", "fcn8s", "multiresunet", "deeplabv3plus", "resnet18_unet_transfer",
]
NAME_RU = {
    "unet_baseline": "U-Net (базовый)",
    "plain_no_skip": "Plain encoder-decoder",
    "res_unet": "Residual U-Net",
    "attention_unet": "Attention U-Net",
    "segnet": "SegNet",
    "fcn8s": "FCN-8s",
    "multiresunet": "MultiResUNet",
    "deeplabv3plus": "DeepLabV3+ (lite)",
    "resnet18_unet_transfer": "ResNet18-UNet\n(transfer learning)",
}


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    history = report["history"]
    summary = {row["name"]: row for row in report["summary"]}
    epochs_full = report["epochs_full"]

    fig, axes = plt.subplots(3, 3, figsize=(12.0, 9.5), dpi=150, sharex=True, sharey=True)

    for ax, name in zip(axes.flat, ORDER):
        rows = history[name]
        s = summary[name]
        epochs = [r["epoch"] for r in rows]
        train_dice = [r["train_dice"] for r in rows]
        test_dice = [r["test_mean_dice_with_tumor"] for r in rows]

        ax.plot(epochs, train_dice, color=COLOR_TRAIN, linewidth=1.6, linestyle="--", label="Train Dice")
        ax.plot(epochs, test_dice, color=COLOR_TEST, linewidth=2.0, label="Test Dice (с опухолью)")

        best_epoch = s["best_epoch"]
        best_value = s["best_test_dice_with_tumor"]
        ax.scatter([best_epoch], [best_value], color=COLOR_TEST, s=42, zorder=5, edgecolor="white", linewidth=0.8)

        overfit_epoch = s["overfit_onset_epoch"]
        if overfit_epoch is not None:
            ax.axvline(best_epoch, color=COLOR_OVERFIT, linewidth=1.0, linestyle=":", zorder=1)
            subtitle = f"макс. {best_value:.3f} @ эп.{best_epoch} · переобуч. с эп.{overfit_epoch}"
        else:
            subtitle = f"макс. {best_value:.3f} @ эп.{best_epoch} · переобуч. не наблюдалось"

        ax.set_title(f"{NAME_RU[name]}\n{subtitle}", loc="left", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(1, epochs_full)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)

    for ax in axes[-1, :]:
        ax.set_xlabel("Эпоха")
    for ax in axes[:, 0]:
        ax.set_ylabel("Dice")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=10)
    fig.suptitle(
        "Кривые обучения по эпохам: переобучение и максимум качества (K=5, все модели с BatchNorm)",
        y=1.03, fontsize=11, fontweight="bold", x=0.02, ha="left",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = ARTIFACTS_DIR / "article4_unet_chart_overfitting.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
