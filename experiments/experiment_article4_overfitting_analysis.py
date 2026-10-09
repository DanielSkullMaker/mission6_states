"""Статья 4 — анализ переобучения: ЕДИНАЯ сетка эпох для всех 8 моделей
сегментации (базовый U-Net + 7 архитектур из experiment_article4_architectures.py,
ВСЕ теперь с BatchNorm — см. docstring conv_block/ASPP/AttentionGate), с
трекингом train/test Dice ПОСЛЕ КАЖДОЙ эпохи.

Зачем: `experiment_article4_architectures.py` обучает каждую архитектуру на
фиксированном бюджете (EPOCHS=6) и репортит только финальный результат —
нельзя сказать, достигла ли модель максимума качества к 6-й эпохе, или
переобучилась, или недообучилась. Этот скрипт обучает все 8 моделей на
ОДНОЙ и той же, более длинной сетке эпох (EPOCHS_FULL) и после каждой эпохи
замеряет test Dice — так можно явно увидеть (а) максимум качества модели и
эпоху, на которой он достигнут, (б) эпоху начала переобучения (test
Dice падает, пока train продолжает расти).

Определение "начала переобучения" (операциональное, не каноническое):
первая эпоха ПОСЛЕ эпохи максимума test_mean_dice_with_tumor (основная,
содержательная метрика статьи — см. project_article4_unet_segmentation в
памяти проекта: усреднение по всем срезам смешивает лёгкую подзадачу
"узнать пустой срез" с содержательной сегментацией). Если максимум
достигнут на последней эпохе сетки — переобучение в пределах EPOCHS_FULL
эпох не наблюдалось (это НЕ означает "модель никогда не переобучится",
только что это не видно в пределах заданного бюджета).

Методология: тот же сплит/датасет/гиперпараметры (кроме EPOCHS), K=5
(физические соседи) для ВСЕХ 8 моделей — базовый U-Net переобучается
здесь заново на этой сетке (НЕ берётся из experiment_article4_unet_report.json,
который использует EPOCHS=6), чтобы все 8 моделей прошли ИДЕНТИЧНЫЙ
протокол сравнения.

Запуск (после prepare_brats_dataset.py/prepare_brats_segmentation_masks.py):
    python experiments/experiment_article4_overfitting_analysis.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import experiment_article4_architectures as archs
import experiment_article4_unet_segmentation as base

EPOCHS_FULL = 15

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_overfitting_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_overfitting_results.md"

ARCHITECTURES_FULL = {"unet_baseline": base.SmallUNet, **archs.ARCHITECTURES}

NAME_RU = {
    "unet_baseline": "U-Net (базовый)",
    "plain_no_skip": "Plain encoder-decoder (без skip)",
    "res_unet": "Residual U-Net",
    "attention_unet": "Attention U-Net",
    "segnet": "SegNet",
    "fcn8s": "FCN-8s",
    "multiresunet": "MultiResUNet",
    "deeplabv3plus": "DeepLabV3+ (lite)",
    "resnet18_unet_transfer": "ResNet18-UNet (transfer learning)",
}


def train_with_epoch_tracking(
    train_groups,
    train_centers,
    test_groups,
    test_centers,
    model_factory,
    n_context: int,
    epochs: int,
) -> Tuple[torch.nn.Module, List[Dict[str, float]]]:
    torch.manual_seed(base.RANDOM_SEED)
    if base.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(base.RANDOM_SEED)

    train_dataset = base.SegStackDataset(train_groups, train_centers, n_context)
    loader = DataLoader(train_dataset, batch_size=base.BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataset = base.SegStackDataset(test_groups, test_centers, n_context)

    model = model_factory(n_context).to(base.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=base.LEARNING_RATE)

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        n_batches = 0
        for X, y in loader:
            X, y = X.to(base.DEVICE), y.to(base.DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss = base.combined_loss(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_dice += base.dice_coefficient(logits, y)
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        train_dice = epoch_dice / max(n_batches, 1)

        test_result = base.evaluate_model(model, test_dataset)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "test_mean_dice": test_result["mean_dice"],
            "test_mean_dice_with_tumor": test_result["mean_dice_with_tumor"],
        }
        history.append(record)
        print(
            f"      epoch {epoch}/{epochs}: train_dice={train_dice:.4f}  "
            f"test_dice={test_result['mean_dice']:.4f}  "
            f"test_dice_tumor={test_result['mean_dice_with_tumor']:.4f}"
        )
    return model, history


def analyze_history(history: List[Dict[str, float]]) -> Tuple[int, float, Optional[int]]:
    best_idx = max(range(len(history)), key=lambda i: history[i]["test_mean_dice_with_tumor"])
    best_epoch = history[best_idx]["epoch"]
    best_value = history[best_idx]["test_mean_dice_with_tumor"]
    overfit_epoch = history[best_idx + 1]["epoch"] if best_idx + 1 < len(history) else None
    return best_epoch, best_value, overfit_epoch


def main() -> None:
    start = time.perf_counter()
    print(f"=== Статья 4: анализ переобучения (единая сетка {EPOCHS_FULL} эпох, K=5, 8 моделей) ===")
    print(
        f"Устройство вычислений: {base.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})" if base.DEVICE.type == "cuda" else "")
    )

    rows = base.load_metadata()
    train_ids, test_ids = base.split_patients(rows)
    train_groups = base.group_by_patient(rows, train_ids)
    test_groups = base.group_by_patient(rows, test_ids)
    train_centers = base.build_center_list(train_groups, base.MAX_TRAIN_PER_CLASS, base.RANDOM_SEED)
    test_centers = base.all_centers(test_groups)
    print(f"n_train={len(train_centers)}, n_test={len(test_centers)}.")

    all_histories: Dict[str, List[Dict[str, float]]] = {}
    summary_rows: List[Dict[str, object]] = []
    for name, factory in ARCHITECTURES_FULL.items():
        print(f"\n--- {NAME_RU[name]} ---")
        t0 = time.perf_counter()
        model, history = train_with_epoch_tracking(
            train_groups, train_centers, test_groups, test_centers, factory, base.N_CONTEXT, EPOCHS_FULL,
        )
        fit_seconds = time.perf_counter() - t0
        best_epoch, best_value, overfit_epoch = analyze_history(history)
        n_params = sum(p.numel() for p in model.parameters())
        all_histories[name] = history
        summary_rows.append(
            {
                "name": name,
                "n_params": n_params,
                "fit_seconds": fit_seconds,
                "best_epoch": best_epoch,
                "best_test_dice_with_tumor": best_value,
                "final_epoch_test_dice_with_tumor": history[-1]["test_mean_dice_with_tumor"],
                "overfit_onset_epoch": overfit_epoch,
            }
        )
        print(
            f"   ИТОГ {name}: best_epoch={best_epoch}  best_test_dice_tumor={best_value:.4f}  "
            f"overfit_onset={overfit_epoch}  fit={fit_seconds:.0f}s  n_params={n_params:,}"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "epochs_full": EPOCHS_FULL,
        "n_context": base.N_CONTEXT,
        "device": str(base.DEVICE),
        "summary": summary_rows,
        "history": all_histories,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    lines = [
        "# Статья 4 — анализ переобучения (единая сетка эпох, BatchNorm во всех архитектурах)",
        "",
        f"{EPOCHS_FULL} эпох, K=5 (физические соседи), идентичный train/test сплит для ВСЕХ "
        "8 моделей (базовый U-Net переобучен заново на этой сетке, не взят из старого отчёта "
        "experiment_article4_unet_report.json). Устройство: "
        f"{base.DEVICE}" + (f" ({torch.cuda.get_device_name(0)})." if base.DEVICE.type == "cuda" else "."),
        "",
        "'Начало переобучения' — первая эпоха ПОСЛЕ пика test Dice (с опухолью, основная "
        "метрика статьи); '—' означает, что пик совпадает с последней эпохой сетки "
        f"(переобучение не наблюдалось в пределах {EPOCHS_FULL} эпох).",
        "",
        "| Модель | Макс. test Dice (с опухолью) | На эпохе | Dice на посл. эпохе | Начало переобучения | Параметров | Время |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary_rows:
        overfit_str = str(row["overfit_onset_epoch"]) if row["overfit_onset_epoch"] is not None else "—"
        n_params_str = f"{row['n_params']:,}".replace(",", " ")
        lines.append(
            f"| {NAME_RU[row['name']]} | {row['best_test_dice_with_tumor']:.4f} | {row['best_epoch']} | "
            f"{row['final_epoch_test_dice_with_tumor']:.4f} | {overfit_str} | {n_params_str} | "
            f"{row['fit_seconds']:.0f}s |"
        )
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
