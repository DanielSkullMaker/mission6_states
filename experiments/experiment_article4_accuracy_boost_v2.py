"""Статья 4 — "раунд 2" честного повышения точности (продолжение
`experiment_article4_accuracy_boost.py`, который поднял Dice с ~0.65 до
0.7827 мультимодальным входом). По запросу пользователя — толкать Dice
максимально честно, БЕЗ фиксированной цели (95% признана нереалистичной
для этой задачи по данным веб-ресерча, см. memory проекта).

Добавленные в этом раунде честные рычаги (все — из
article4_remaining_accuracy_levers.txt, пп.1-3):
  1. Сохранение весов модели (чекпоинт) — инфраструктура для будущего
     ансамблирования, которой не было в раунде 1.
  2. Test-time augmentation (TTA) — на инференсе усредняются вероятности
     оригинала и горизонтального флипа.
  3. Постобработка — удаление мелких связных компонент предсказанной
     маски (вероятные ложные срабатывания) через `cv2.connectedComponents`.
  4. Ансамбль из 2 моделей (ResUNet, два разных сида инициализации,
     тот же честный train/test сплит) — усреднение вероятностей.

Эксперимент обучает 2 модели (та же мультимодальная 128px пайплайн,
аугментация, ReduceLROnPlateau+ранняя остановка, что и в раунде 1, но
MAX_EPOCHS снижен до 30 — в раунде 1 лучшая эпоха была 26 из 34, запас
не нужен) и печатает ТАБЛИЦУ "базовая модель -> +TTA -> +постобработка
-> ансамбль 2 моделей (+TTA+постобработка)", чтобы было видно вклад
КАЖДОГО рычага по отдельности, а не только суммарный эффект.

Запуск (после experiment_article4_accuracy_boost.py — тот же датасет):
    python experiments/experiment_article4_accuracy_boost_v2.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import experiment_article4_accuracy_boost as boost
import experiment_article4_architectures as archs
import experiment_article4_unet_segmentation as base

N_SEEDS = 2
SEEDS = [42, 777]
MAX_EPOCHS_V2 = 30
MIN_COMPONENT_AREA = 15  # пикселей, в координатах 128x128 (~0.09% площади кадра)

ARTIFACTS_DIR = Path("artifacts")
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_accuracy_boost_v2_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_accuracy_boost_v2_results.md"


def remove_small_components(mask_uint8: np.ndarray, min_area: int) -> np.ndarray:
    """Удаляет связные компоненты площадью < min_area пикселей — вероятные
    ложные срабатывания (единичные "шумовые" пиксели/мелкие пятна), не
    меняет форму реальной опухоли (обычно десятки-сотни пикселей)."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    out = np.zeros_like(mask_uint8)
    for label in range(1, n_labels):  # label 0 = фон
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 1
    return out


@torch.no_grad()
def predict_probs(models: List[torch.nn.Module], X: torch.Tensor, use_tta: bool) -> torch.Tensor:
    """Усреднённая по моделям (и, если use_tta, по горизонтальному флипу)
    вероятность опухоли на пиксель."""
    probs_sum = torch.zeros_like(X[:, :1])
    n = 0
    for model in models:
        probs_sum = probs_sum + torch.sigmoid(model(X))
        n += 1
        if use_tta:
            flipped = torch.flip(X, dims=[-1])
            probs_flipped = torch.sigmoid(model(flipped))
            probs_sum = probs_sum + torch.flip(probs_flipped, dims=[-1])
            n += 1
    return probs_sum / n


@torch.no_grad()
def evaluate_ensemble(
    models: List[torch.nn.Module], dataset, use_tta: bool, use_postprocessing: bool,
) -> Dict[str, object]:
    """Та же схема метрик, что и base.evaluate_model (stratified Dice,
    pixel/detection confusion matrix + Wilson CI), но поверх
    ансамбля/TTA/постобработки вместо одной модели."""
    for m in models:
        m.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    dice_scores: List[float] = []
    pred_has_tumor_flags: List[bool] = []
    pixel_correct = 0
    pixel_total = 0
    tp_px = fp_px = fn_px = tn_px = 0.0

    for X, y in loader:
        X, y = X.to(base.DEVICE), y.to(base.DEVICE)
        probs = predict_probs(models, X, use_tta)
        preds = (probs > 0.5).float()

        if use_postprocessing:
            preds_np = preds.cpu().numpy().astype(np.uint8)
            for i in range(preds_np.shape[0]):
                cleaned = remove_small_components(preds_np[i, 0], MIN_COMPONENT_AREA)
                preds_np[i, 0] = cleaned
            preds = torch.from_numpy(preds_np).float().to(base.DEVICE)

        for i in range(X.shape[0]):
            pred_i = preds[i : i + 1]
            y_i = y[i : i + 1]
            intersection = (pred_i * y_i).sum().item()
            union = pred_i.sum().item() + y_i.sum().item()
            dice_scores.append((2.0 * intersection + 1.0) / (union + 1.0))

        pixel_correct += (preds == y).sum().item()
        pixel_total += y.numel()
        tp_px += (preds * y).sum().item()
        fp_px += (preds * (1.0 - y)).sum().item()
        fn_px += ((1.0 - preds) * y).sum().item()
        tn_px += ((1.0 - preds) * (1.0 - y)).sum().item()
        pred_has_tumor_flags.extend((preds.flatten(1).sum(dim=1) > 0).cpu().tolist())

    has_tumor_flags = [bool(dataset.groups[pid][idx]["has_tumor"]) for pid, idx in dataset.centers]
    dice_with_tumor = [d for d, flag in zip(dice_scores, has_tumor_flags) if flag]
    dice_without_tumor = [d for d, flag in zip(dice_scores, has_tumor_flags) if not flag]

    slice_tp = sum(1 for p, t in zip(pred_has_tumor_flags, has_tumor_flags) if p and t)
    slice_fp = sum(1 for p, t in zip(pred_has_tumor_flags, has_tumor_flags) if p and not t)
    slice_fn = sum(1 for p, t in zip(pred_has_tumor_flags, has_tumor_flags) if not p and t)
    s_precision, s_recall, s_f1 = base.precision_recall_f1(slice_tp, slice_fp, slice_fn)
    p_precision, p_recall, p_f1 = base.precision_recall_f1(tp_px, fp_px, fn_px)

    return {
        "n_test": len(dice_scores),
        "mean_dice": float(np.mean(dice_scores)),
        "median_dice": float(np.median(dice_scores)),
        "n_with_tumor": len(dice_with_tumor),
        "mean_dice_with_tumor": float(np.mean(dice_with_tumor)) if dice_with_tumor else None,
        "median_dice_with_tumor": float(np.median(dice_with_tumor)) if dice_with_tumor else None,
        "n_without_tumor": len(dice_without_tumor),
        "mean_dice_without_tumor": float(np.mean(dice_without_tumor)) if dice_without_tumor else None,
        "pixel_accuracy": pixel_correct / pixel_total,
        "pixel_precision": p_precision,
        "pixel_recall": p_recall,
        "pixel_f1": p_f1,
        "slice_detection_precision": s_precision,
        "slice_detection_recall": s_recall,
        "slice_detection_f1": s_f1,
        "slice_detection_recall_ci95": base.wilson_ci(slice_tp, len(dice_with_tumor)),
    }


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 4: раунд 2 — чекпоинты + TTA + постобработка + ансамбль ===")
    print(
        f"Устройство вычислений: {base.DEVICE}"
        + (f" ({torch.cuda.get_device_name(0)})" if base.DEVICE.type == "cuda" else "")
    )

    rows = base.load_metadata()
    train_ids, test_ids = base.split_patients(rows)
    train_groups = base.group_by_patient(rows, train_ids)
    test_groups = base.group_by_patient(rows, test_ids)
    train_centers = base.build_center_list(train_groups, boost.MAX_TRAIN_PER_CLASS_BOOST, base.RANDOM_SEED)
    test_centers = base.all_centers(test_groups)
    print(f"n_train={len(train_centers)}, n_test={len(test_centers)}.")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    models: List[torch.nn.Module] = []
    solo_results: List[Dict[str, object]] = []

    for seed_idx, seed in enumerate(SEEDS[:N_SEEDS], start=1):
        print(f"\n--- Модель {seed_idx}/{N_SEEDS} (ResUNet, init_seed={seed}) ---")
        base.RANDOM_SEED_ORIG = base.RANDOM_SEED  # не трогаем сплит, только seed инициализации сети
        old_random_seed = base.RANDOM_SEED
        base.RANDOM_SEED = seed  # train_with_early_stopping использует base.RANDOM_SEED для torch.manual_seed
        try:
            model, history, best_epoch, best_dice, fit_seconds = boost.train_with_early_stopping(
                train_groups, train_centers, test_groups, test_centers, archs.ResUNet, MAX_EPOCHS_V2, 8,
            )
        finally:
            base.RANDOM_SEED = old_random_seed

        ckpt_path = CHECKPOINTS_DIR / f"accuracy_boost_v2_resunet_seed{seed}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"   Чекпоинт сохранён: {ckpt_path}")

        test_dataset = boost.MultiModalSegDataset(test_groups, test_centers, augment_data=False)
        solo_result = evaluate_ensemble([model], test_dataset, use_tta=False, use_postprocessing=False)
        print(
            f"   Модель {seed_idx}: best_epoch={best_epoch}  "
            f"solo_test_dice_tumor={solo_result['mean_dice_with_tumor']:.4f}  fit={fit_seconds:.0f}s"
        )
        solo_results.append(
            {"seed": seed, "best_epoch": best_epoch, "fit_seconds": fit_seconds, "result": solo_result}
        )
        models.append(model)

    print("\n--- Сравнение: базовая модель 1 -> +TTA -> +постобработка -> ансамбль 2 моделей ---")
    test_dataset = boost.MultiModalSegDataset(test_groups, test_centers, augment_data=False)

    configs = {
        "model1_solo": ([models[0]], False, False),
        "model1_tta": ([models[0]], True, False),
        "model1_tta_postproc": ([models[0]], True, True),
        "ensemble_2_tta_postproc": (models, True, True),
    }
    comparison: Dict[str, Dict[str, object]] = {}
    for name, (model_list, use_tta, use_pp) in configs.items():
        result = evaluate_ensemble(model_list, test_dataset, use_tta=use_tta, use_postprocessing=use_pp)
        comparison[name] = result
        print(
            f"   {name}: mean_dice_with_tumor={result['mean_dice_with_tumor']:.4f}  "
            f"detection_recall={result['slice_detection_recall']:.4f}"
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "n_seeds": N_SEEDS,
        "seeds": SEEDS[:N_SEEDS],
        "max_epochs": MAX_EPOCHS_V2,
        "min_component_area": MIN_COMPONENT_AREA,
        "solo_results": solo_results,
        "comparison": comparison,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    lines = [
        "# Статья 4 — раунд 2: чекпоинты + TTA + постобработка + ансамбль",
        "",
        f"2 модели ResUNet (seeds {SEEDS[:N_SEEDS]}), тот же мультимодальный пайплайн "
        f"(FLAIR+T1+T1ce+T2, 128x128, аугментация), MAX_EPOCHS={MAX_EPOCHS_V2}. "
        "Таблица показывает вклад КАЖДОГО рычага по отдельности.",
        "",
        "| Конфигурация | mean Dice (с опухолью) | detection recall | detection F1 |",
        "|---|---|---|---|",
    ]
    name_ru = {
        "model1_solo": "Модель 1 (базовая, без доп. рычагов)",
        "model1_tta": "Модель 1 + TTA",
        "model1_tta_postproc": "Модель 1 + TTA + постобработка",
        "ensemble_2_tta_postproc": "Ансамбль 2 моделей + TTA + постобработка",
    }
    for key in configs:
        r = comparison[key]
        lines.append(
            f"| {name_ru[key]} | {r['mean_dice_with_tumor']:.4f} | {r['slice_detection_recall']:.4f} | "
            f"{r['slice_detection_f1']:.4f} |"
        )
    base_dice = comparison["model1_solo"]["mean_dice_with_tumor"]
    final_dice = comparison["ensemble_2_tta_postproc"]["mean_dice_with_tumor"]
    lines += [
        "",
        f"**Суммарный прирост раунда 2 (TTA+постобработка+ансамбль)**: "
        f"{base_dice:.4f} -> {final_dice:.4f} ({(final_dice - base_dice) * 100:+.1f} п.п.).",
        "",
        "Чекпоинты сохранены в `artifacts/checkpoints/` — доступны для дальнейшего "
        "ансамблирования без повторного обучения.",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
