"""Абляция методов предобработки для нейросетевой (U-Net, сегментация)
версии Статьи 4 — дополняет `experiment_article4_architectures.py`
(сравнение архитектур) ответом на смежный вопрос: насколько качество
сегментации зависит от конкретных решений на этапе подготовки данных
(`experiments/prepare_brats_dataset.py`), а не только от архитектуры сети.

Сравниваются 5 вариантов предобработки одного и того же FLAIR-среза:

  - full_pipeline     — текущий конвейер статьи: перцентильная нормализация
                        (1-99%) -> resize -> подавление фона (порог 10) ->
                        центрирование (см. раздел "Подготовка данных: от
                        NIfTI к PNG" в generate_article4_report.py).
  - no_centering      — без подавления фона и центрирования (только
                        нормализация + resize) — проверяет цену отказа от
                        геометрического выравнивания содержимого в кадре.
  - strong_threshold  — порог подавления фона увеличен с 10 до 40.
  - clahe_contrast    — адаптивное выравнивание гистограммы (CLAHE,
                        Zuiderveld, 1994) ПЕРЕД resize — распространённый
                        приём усиления локального контраста в медицинской
                        визуализации, в основном конвейере статьи не
                        используется.
  - minmax_norm       — простая min-max нормализация вместо перцентильного
                        клиппинга (чувствительнее к единичным выбросам
                        яркости).

Методология: чтобы изолировать эффект ИМЕННО предобработки (не контекста
соседних срезов), все варианты обучаются и оцениваются БЕЗ стека соседей
(K=1, один канал) — архитектура (SmallUNet) и протокол обучения те же,
что в experiment_article4_unet_segmentation.py. Для управляемого времени
выполнения (CPU, повторная загрузка сырых NIfTI-объёмов на каждый
вариант) используется сокращённая выборка пациентов (N_PATIENTS, не все
160) — это диагностический, а не основной эксперимент статьи (как и
аналогичная абляция предобработки в предыдущей, классической версии
статьи 4).

Запуск:
    python experiments/experiment_article4_preprocessing_ablation.py
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import experiment_article4_unet_segmentation as base
from subspace_conjugacy.preprocessing.centering import (
    DEFAULT_MIN_SHIFT,
    compute_horizontal_delta,
    compute_vertical_delta,
    shift_columns,
    shift_rows,
)
from subspace_conjugacy.preprocessing.normalization import suppress_background

BRATS_ROOT = Path(
    os.environ.get("BRATS_ROOT", "data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
)
MODALITY_SUFFIX = "flair"
SEG_SUFFIX = "seg"

N_PATIENTS = 15  # сокращённая выборка (не все 160) — см. docstring; после перехода на build_all_variant_datasets (один проход по пациентам вместо пяти) риск OOM заметно ниже, можно не жаться к 8
TEST_FRACTION = 0.2
MIN_BRAIN_FRACTION = 0.02
IMG_SIZE = 96
RANDOM_SEED = 42
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-3

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_article4_preprocessing_ablation_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_article4_preprocessing_ablation_results.md"

VARIANTS = ["full_pipeline", "no_centering", "strong_threshold", "clahe_contrast", "minmax_norm"]
VARIANT_LABELS_RU = {
    "full_pipeline": "Полный конвейер",
    "no_centering": "Без центрирования",
    "strong_threshold": "Усиленный порог фона (40)",
    "clahe_contrast": "CLAHE-контраст",
    "minmax_norm": "Min-max нормализация",
}


def find_patient_dirs(root: Path, n: int) -> List[Path]:
    candidates = sorted(
        {p.parent for p in list(root.rglob(f"*_{MODALITY_SUFFIX}.nii")) + list(root.rglob(f"*_{MODALITY_SUFFIX}.nii.gz"))}
    )
    return candidates[:n]


def find_volume_file(patient_dir: Path, suffix: str):
    for ext in (".nii.gz", ".nii"):
        matches = list(patient_dir.glob(f"*_{suffix}{ext}"))
        if matches:
            return matches[0]
    return None


def normalize_percentile(slice_2d: np.ndarray) -> np.ndarray:
    nonzero = slice_2d[slice_2d > 0]
    if nonzero.size == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    lo, hi = np.percentile(nonzero, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(slice_2d, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def normalize_minmax(slice_2d: np.ndarray) -> np.ndarray:
    nonzero = slice_2d[slice_2d > 0]
    if nonzero.size == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    lo, hi = float(nonzero.min()), float(nonzero.max())
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(slice_2d, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def process_variant(slice_2d: np.ndarray, variant: str) -> Tuple[np.ndarray, int, int, bool]:
    """Возвращает (обработанное изображение IMG_SIZExIMG_SIZE, h_delta,
    v_delta, применялся_ли_сдвиг) — deltas нужны, чтобы применить ТОТ ЖЕ
    сдвиг к маске."""
    norm = normalize_minmax(slice_2d) if variant == "minmax_norm" else normalize_percentile(slice_2d)

    if variant == "clahe_contrast":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = clahe.apply(norm)

    resized = cv2.resize(norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)

    if variant == "no_centering":
        return resized, 0, 0, False

    threshold = 40 if variant == "strong_threshold" else 10
    masked = suppress_background(resized, threshold)
    h_delta = compute_horizontal_delta(masked)
    v_delta = compute_vertical_delta(masked)
    shifted = shift_rows(masked, h_delta, min_shift=DEFAULT_MIN_SHIFT)
    shifted = shift_columns(shifted, v_delta, min_shift=DEFAULT_MIN_SHIFT)
    return shifted, h_delta, v_delta, True


def process_mask(mask_slice: np.ndarray, h_delta: int, v_delta: int, apply_shift: bool) -> np.ndarray:
    m = (mask_slice > 0).astype(np.uint8) * 255
    resized = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    if not apply_shift:
        return resized
    shifted = shift_rows(resized, h_delta, min_shift=DEFAULT_MIN_SHIFT)
    shifted = shift_columns(shifted, v_delta, min_shift=DEFAULT_MIN_SHIFT)
    return shifted


def build_all_variant_datasets(
    patient_dirs: List[Path],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Строит датасеты ВСЕХ вариантов за ОДИН проход по пациентам (не 5
    отдельных проходов — каждый объём NIfTI загружается ровно один раз,
    а не по разу на вариант). Решает повторный OOM: предыдущая версия
    (вариант снаружи, загрузка объёма внутри) открывала и декодировала
    одни и те же ~70МБ-объёмы 5 раз подряд, создавая 5 пиков нагрузки на
    и без того ограниченную свободную RAM в системе; при одном проходе
    пик — один."""
    import gc

    variant_images: Dict[str, List[np.ndarray]] = {v: [] for v in VARIANTS}
    variant_masks: Dict[str, List[np.ndarray]] = {v: [] for v in VARIANTS}
    variant_pids: Dict[str, List[str]] = {v: [] for v in VARIANTS}

    for patient_dir in patient_dirs:
        flair_path = find_volume_file(patient_dir, MODALITY_SUFFIX)
        seg_path = find_volume_file(patient_dir, SEG_SUFFIX)
        if flair_path is None or seg_path is None:
            continue
        # dtype=np.float32 (не float64 по умолчанию) — вдвое меньше памяти
        # на объём (240x240x155), важно при ограниченной свободной RAM (см.
        # OOM при первых прогонах этого скрипта).
        flair_vol = nib.load(str(flair_path)).get_fdata(dtype=np.float32)
        seg_vol = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
        for z in range(flair_vol.shape[2]):
            slice_2d = flair_vol[:, :, z]
            if float(np.mean(slice_2d > 0)) < MIN_BRAIN_FRACTION:
                continue
            seg_slice = seg_vol[:, :, z]
            for variant in VARIANTS:
                img, h_delta, v_delta, apply_shift = process_variant(slice_2d, variant)
                mask = process_mask(seg_slice, h_delta, v_delta, apply_shift)
                variant_images[variant].append(img.astype(np.float32) / 255.0)
                variant_masks[variant].append((mask > 127).astype(np.float32))
                variant_pids[variant].append(patient_dir.name)
        del flair_vol, seg_vol
        gc.collect()

    return {
        variant: (np.stack(variant_images[variant]), np.stack(variant_masks[variant]), variant_pids[variant])
        for variant in VARIANTS
    }


class InMemorySegDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray):
        self.images = images
        self.masks = masks

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx])[None, :, :]
        mask = torch.from_numpy(self.masks[idx])[None, :, :]
        return img, mask


def train_small_unet(train_ds: Dataset) -> Tuple[torch.nn.Module, float]:
    torch.manual_seed(RANDOM_SEED)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = base.SmallUNet(in_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    t0 = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for X, y in loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = base.combined_loss(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"      epoch {epoch + 1}/{EPOCHS}: loss={epoch_loss / max(n_batches, 1):.4f}")
    return model, time.perf_counter() - t0


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 4: абляция предобработки (K=1, SmallUNet) ===")

    if not BRATS_ROOT.is_dir():
        raise FileNotFoundError(f"BRATS_ROOT не найден: {BRATS_ROOT}.")

    patient_dirs = find_patient_dirs(BRATS_ROOT, N_PATIENTS)
    patient_ids_all = [p.name for p in patient_dirs]
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(patient_ids_all)
    n_test = max(1, round(len(shuffled) * TEST_FRACTION))
    test_ids = set(shuffled[:n_test].tolist())
    train_ids = set(shuffled[n_test:].tolist())
    print(f"Пациентов: {len(patient_dirs)} ({len(train_ids)} train / {len(test_ids)} test).")

    print("\nПостроение всех 5 вариантов предобработки за один проход по пациентам...")
    all_datasets = build_all_variant_datasets(patient_dirs)

    results: Dict[str, Dict] = {}
    for variant in VARIANTS:
        print(f"\n--- Вариант: {variant} ---")
        images, masks, pids = all_datasets[variant]
        pids_arr = np.array(pids)
        train_mask = np.isin(pids_arr, list(train_ids))
        test_mask = np.isin(pids_arr, list(test_ids))

        train_ds = InMemorySegDataset(images[train_mask], masks[train_mask])
        test_ds = InMemorySegDataset(images[test_mask], masks[test_mask])
        print(f"   n_train={len(train_ds)}, n_test={len(test_ds)}")

        model, fit_seconds = train_small_unet(train_ds)
        result = base.evaluate_model(model, test_ds)
        result["fit_seconds"] = fit_seconds
        result["n_train"] = len(train_ds)
        print(f"   {variant}: mean_dice={result['mean_dice']:.4f}  median_dice={result['median_dice']:.4f}")
        results[variant] = result

        del images, masks, train_ds, test_ds, model
        import gc
        gc.collect()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "random_seed": RANDOM_SEED,
        "n_patients": len(patient_dirs),
        "img_size": IMG_SIZE,
        "epochs": EPOCHS,
        "results": results,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    lines = [
        "# Статья 4 — абляция предобработки (K=1, SmallUNet, сокращённая выборка)",
        "",
        f"{len(patient_dirs)} пациентов ({len(train_ids)} train / {len(test_ids)} test), "
        f"{IMG_SIZE}x{IMG_SIZE}, {EPOCHS} эпох.",
        "",
        "| Вариант | Mean Dice | Median Dice | n_test |",
        "|---|---|---|---|",
    ]
    for variant in VARIANTS:
        r = results[variant]
        lines.append(f"| {VARIANT_LABELS_RU[variant]} | {r['mean_dice']:.4f} | {r['median_dice']:.4f} | {r['n_test']} |")
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
