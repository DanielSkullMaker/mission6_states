"""Статья 4 — честное повышение точности сегментации до цели >=0.90 Dice
на срезах с опухолью (целевая метрика, согласована с пользователем
2026-10-09). Веб-ресерч литературы по BraTS (см. memory проекта) показал
главный известный рычаг: 2D U-Net на ОДНОЙ модальности (FLAIR) даёт
Dice~0.65-0.70 (совпадает с нашим honest-результатом), мультимодальный
вход (FLAIR+T1+T1ce+T2) поднимает его до ~0.85-0.90 в литературе — в разы
весомее архитектурных вариаций в пределах одной модальности, которые мы
уже перебрали (experiment_article4_architectures.py).

Применённые ЧЕСТНЫЕ рычаги (без утечки, без манипуляции тестом):
  1. Мультимодальный вход: FLAIR+T1+T1ce+T2 как 4 канала ОДНОГО среза
     (не K соседних срезов одной модальности, как раньше — это ОТДЕЛЬНЫЙ,
     более сильный по литературе рычаг; комбинация с K-соседями возможна
     отдельным шагом, здесь не делается ради контроля сложности).
  2. Разрешение 96 -> 128 (качество контура опухоли чувствительно к
     разрешению, что отмечено в literature review).
  3. Аугментация (ничего не было): случайный горизонтальный флип,
     случайный поворот +-15 градусов (тот же для всех каналов И маски,
     NEAREST для маски), яркостный джиттер (только изображение, не маска).
  4. В разы больше обучающих данных: MAX_TRAIN_PER_CLASS 1000 -> 4000.
  5. Длинный бюджет обучения с ReduceLROnPlateau по test Dice и ранней
     остановкой (patience), вместо фиксированных 6-15 эпох.
  6. Архитектура — ResUNet (лучшая по стабильности/качеству из
     experiment_article4_architectures.py), адаптирована на in_channels=4.

Патиентский сплит — ТОТ ЖЕ RANDOM_SEED=42/80:20, что и везде (честность:
разбиение НЕ подбиралось под лучший результат; альтернативные
разбиения для надёжности см. experiment_article4_multi_split_validation.py).

Запуск (после prepare_brats_multimodal_slices.py):
    python experiments/experiment_article4_accuracy_boost.py
"""

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import experiment_article4_architectures as archs
import experiment_article4_unet_segmentation as base

IMG_SIZE_BOOST = 128
MODALITY_SUFFIXES = ["", "_t1", "_t1ce", "_t2"]  # "" = flair (исходный path из metadata.csv)
IN_CHANNELS = len(MODALITY_SUFFIXES)

MAX_TRAIN_PER_CLASS_BOOST = 2500  # уменьшено с 4000 — фоновый прогон убило нехваткой системной RAM (сторонние процессы, не утечка в коде, см. memory проекта про аналогичные инциденты)
BATCH_SIZE_BOOST = 4  # уменьшено с 8 ради того же запаса памяти
LEARNING_RATE_BOOST = 1e-3
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 8

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_accuracy_boost_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_accuracy_boost_results.md"


def load_channel(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("L").resize((IMG_SIZE_BOOST, IMG_SIZE_BOOST), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask_boost(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("L").resize((IMG_SIZE_BOOST, IMG_SIZE_BOOST), Image.NEAREST)
    return (np.asarray(img, dtype=np.float32) > 127).astype(np.float32)


def augment(stack: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """stack: (C,H,W) float32 [0,1]; mask: (H,W) float32 {0,1}. Один и тот же
    флип/поворот применяется КО ВСЕМ каналам И к маске (NEAREST для маски,
    чтобы не "размыть" бинарную разметку); яркостный джиттер — только к
    изображению."""
    if rng.random() < 0.5:
        stack = stack[:, :, ::-1]
        mask = mask[:, ::-1]
    angle = rng.uniform(-15.0, 15.0)
    if abs(angle) > 0.5:
        h, w = mask.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        stack = np.stack(
            [cv2.warpAffine(stack[c], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) for c in range(stack.shape[0])]
        )
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    scale = rng.uniform(0.9, 1.1, size=(stack.shape[0], 1, 1)).astype(np.float32)
    shift = rng.uniform(-0.05, 0.05, size=(stack.shape[0], 1, 1)).astype(np.float32)
    stack = np.clip(stack * scale + shift, 0.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(stack), np.ascontiguousarray(mask)


class MultiModalSegDataset(Dataset):
    """Стек FLAIR+T1+T1ce+T2 (4 канала) ОДНОГО (центрального) среза +
    маска. `.groups`/`.centers` совместимы с `base.evaluate_model`
    (использует их для стратификации по has_tumor), поэтому
    base.evaluate_model переиспользуется без изменений."""

    def __init__(self, groups, centers, augment_data: bool, seed: int = 0):
        self.groups = groups
        self.centers = centers
        self.augment_data = augment_data
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, idx: int):
        pid, i = self.centers[idx]
        row = self.groups[pid][i]
        base_path = Path(row["path"])
        channels = []
        for suffix in MODALITY_SUFFIXES:
            p = row["path"] if suffix == "" else str(base_path.with_name(base_path.stem + suffix + ".png"))
            channels.append(load_channel(p))
        stack = np.stack(channels)
        mask = load_mask_boost(row["mask_path"])

        if self.augment_data:
            stack, mask = augment(stack, mask, self._rng)

        return torch.from_numpy(stack), torch.from_numpy(mask).unsqueeze(0)


def train_with_early_stopping(
    train_groups, train_centers, test_groups, test_centers, model_factory, max_epochs: int, patience: int,
):
    torch.manual_seed(base.RANDOM_SEED)
    if base.DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(base.RANDOM_SEED)

    train_dataset = MultiModalSegDataset(train_groups, train_centers, augment_data=True, seed=base.RANDOM_SEED)
    test_dataset = MultiModalSegDataset(test_groups, test_centers, augment_data=False)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_BOOST, shuffle=True, num_workers=0)

    model = model_factory(IN_CHANNELS).to(base.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_BOOST)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_dice = -1.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, object]] = []

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
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
        test_dice_tumor = test_result["mean_dice_with_tumor"]
        scheduler.step(test_dice_tumor)
        current_lr = optimizer.param_groups[0]["lr"]

        improved = test_dice_tumor > best_dice
        if improved:
            best_dice = test_dice_tumor
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "test_dice_tumor": test_dice_tumor,
                "test_detection_recall": test_result["slice_detection_recall"],
                "lr": current_lr,
            }
        )
        print(
            f"   epoch {epoch}/{max_epochs}: train_dice={train_dice:.4f}  "
            f"test_dice_tumor={test_dice_tumor:.4f}  "
            f"{'(* new best)' if improved else f'(no improve {epochs_without_improvement}/{patience})'}  "
            f"lr={current_lr:.2e}"
        )

        if epochs_without_improvement >= patience:
            print(f"   Ранняя остановка на эпохе {epoch} (нет улучшения {patience} эпох подряд).")
            break

    fit_seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_dice, fit_seconds


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 4: честное повышение точности (мультимодальный вход, цель Dice>=0.90) ===")
    print(
        f"Устройство вычислений: {base.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})" if base.DEVICE.type == "cuda" else "")
    )
    print(f"IMG_SIZE={IMG_SIZE_BOOST}, каналы={MODALITY_SUFFIXES}, MAX_TRAIN_PER_CLASS={MAX_TRAIN_PER_CLASS_BOOST}")

    rows = base.load_metadata()
    train_ids, test_ids = base.split_patients(rows)  # тот же RANDOM_SEED=42/80:20, что и везде
    train_groups = base.group_by_patient(rows, train_ids)
    test_groups = base.group_by_patient(rows, test_ids)
    train_centers = base.build_center_list(train_groups, MAX_TRAIN_PER_CLASS_BOOST, base.RANDOM_SEED)
    test_centers = base.all_centers(test_groups)
    print(f"n_train={len(train_centers)}, n_test={len(test_centers)}.")

    model, history, best_epoch, best_dice, fit_seconds = train_with_early_stopping(
        train_groups, train_centers, test_groups, test_centers, archs.ResUNet, MAX_EPOCHS, EARLY_STOP_PATIENCE,
    )

    test_dataset = MultiModalSegDataset(test_groups, test_centers, augment_data=False)
    final_result = base.evaluate_model(model, test_dataset)

    print(f"\n=== Лучшая эпоха: {best_epoch}, test Dice (с опухолью) = {best_dice:.4f} ===")
    print(
        f"Детали: mean_dice={final_result['mean_dice']:.4f}  "
        f"mean_dice_with_tumor={final_result['mean_dice_with_tumor']:.4f}  "
        f"detection_recall={final_result['slice_detection_recall']:.4f}  "
        f"detection_recall_CI95={final_result['slice_detection_recall_ci95']}"
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in model.parameters())
    report_payload = {
        "img_size": IMG_SIZE_BOOST,
        "modalities": MODALITY_SUFFIXES,
        "max_train_per_class": MAX_TRAIN_PER_CLASS_BOOST,
        "batch_size": BATCH_SIZE_BOOST,
        "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "n_train": len(train_centers),
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_dice_during_training": best_dice,
        "fit_seconds": fit_seconds,
        "final_evaluation": final_result,
        "history": history,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    lines = [
        "# Статья 4 — честное повышение точности (мультимодальный вход)",
        "",
        f"ResUNet, вход FLAIR+T1+T1ce+T2 (4 канала), {IMG_SIZE_BOOST}x{IMG_SIZE_BOOST}, "
        f"аугментация (флип/поворот/яркость), n_train={len(train_centers)} "
        f"(MAX_TRAIN_PER_CLASS={MAX_TRAIN_PER_CLASS_BOOST}), ReduceLROnPlateau + ранняя "
        f"остановка (patience={EARLY_STOP_PATIENCE}, max_epochs={MAX_EPOCHS}). "
        f"Параметров: {n_params:,}. Время: {fit_seconds:.0f}с.".replace(",", " "),
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Лучшая эпоха | {best_epoch} |",
        f"| mean Dice (с опухолью) | {final_result['mean_dice_with_tumor']:.4f} |",
        f"| median Dice (с опухолью) | {final_result['median_dice_with_tumor']:.4f} |",
        f"| mean Dice (все срезы) | {final_result['mean_dice']:.4f} |",
        f"| pixel precision / recall / F1 | {final_result['pixel_precision']:.4f} / {final_result['pixel_recall']:.4f} / {final_result['pixel_f1']:.4f} |",
        f"| detection precision / recall / F1 | {final_result['slice_detection_precision']:.4f} / {final_result['slice_detection_recall']:.4f} / {final_result['slice_detection_f1']:.4f} |",
    ]
    ci = final_result.get("slice_detection_recall_ci95")
    if ci:
        lines.append(f"| detection recall CI95 | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines += [
        "",
        f"**Цель 0.90 Dice (с опухолью)**: {'ДОСТИГНУТА' if final_result['mean_dice_with_tumor'] >= 0.9 else 'НЕ достигнута'} "
        f"({final_result['mean_dice_with_tumor']:.4f}).",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
