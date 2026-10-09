"""Конвертирует BraTS2020 (NIfTI-объёмы, Kaggle: awsaf49/brats20-dataset-
training-validation) в PNG-срезы, совместимые с текущим препроцессинг-
пайплайном библиотеки (`subspace_conjugacy.preprocessing.ImagePreprocessor`).

Контекст (Статья 4 диссертационного плана, dissertation_articles_plan.txt):
нужен датасет с РЕАЛЬНЫМ (не суррогатным) физическим порядком срезов одного
пациента для эксперимента со сглаживанием смежных векторов
(`features.smoothing.smooth_adjacent_vectors`). Используемый в остальных
статьях Kaggle "Brain Tumor MRI Dataset" не хранит метаданные серии — это
отдельные анонимизированные снимки без привязки к пациенту/порядку.

BraTS2020 даёт для каждого пациента один NIfTI-объём (240, 240, 155) —
срезы по оси Z — это буквально реальная последовательность срезов одного
МРТ-исследования, не суррогат. Расплата: классы другие — не
glioma/meningioma/pituitary (как в Статье 1), а бинарная задача "срез
содержит опухоль" / "не содержит" (из маски сегментации `_seg.nii`) — план
статьи 4 явно допускает смену задачи ради метода Статьи 4 (сглаживание), не
наоборот.

Что делает скрипт (на пациента):
  1. Загружает один модальность-объём (по умолчанию FLAIR — лучший контраст
     опухоли) и объём сегментации (`_seg.nii`) через nibabel.
  2. Для каждого Z-среза: отбрасывает почти пустые срезы (доля ненулевых
     пикселей < MIN_BRAIN_FRACTION — аналог идеи LowInformativenessFilter,
     но на уровне кадра, не вектора).
  3. Нормализует интенсивность в uint8 (клиппинг по перцентилям ненулевых
     вокселей — устойчивее к выбросам, чем голый min-max).
  4. Прогоняет через `ImagePreprocessor.process()` — ТОТ ЖЕ resize ->
     подавление фона -> центрирование, что и для datasets/*_centered/.
  5. Сохраняет PNG с именем, сохраняющим порядок срезов
     ({patient_id}_slice{idx:03d}.png), и строку в metadata.csv
     (has_tumor, tumor_pixel_fraction — для классификации и для фильтрации
     признака задачи в последующем experiments/experiment_sequential_smoothing.py).

Запуск:
    BRATS_ROOT="D:/data/MICCAI_BraTS2020_TrainingData" python experiments/prepare_brats_dataset.py

Зависимость: nibabel (pip install -e ".[experiments]").
"""

import csv
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "nibabel не установлен — требуется для чтения NIfTI-объёмов BraTS. "
        "Установите: pip install -e \".[experiments]\" (или `pip install nibabel`)."
    ) from exc

from subspace_conjugacy.preprocessing import ImagePreprocessor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Конфигурация — правьте под свой путь/объём выборки.
# --------------------------------------------------------------------------

BRATS_ROOT = Path(
    os.environ.get("BRATS_ROOT", r"C:\Users\nazar\Downloads\MICCAI_BraTS2020_TrainingData")
)
OUTPUT_ROOT = Path(os.environ.get("BRATS_OUTPUT_ROOT", "datasets/brats_flair_slices"))
MODALITY_SUFFIX = "flair"  # "flair" | "t1" | "t1ce" | "t2"
SEG_SUFFIX = "seg"

# None = все найденные пациенты. BraTS2020 train = 369 пациентов — на полном
# наборе ~369 * ~80 содержательных срезов ~ 30k изображений, это избыточно
# для методологической демонстрации сглаживания (Статья 4); ограничиваем по
# умолчанию, как и весь остальной проект ограничивает объём эксперимента
# (main.py — 200/класс, не весь Kaggle-архив).
# Увеличено 40 -> 160 (2026-10-07): при 40 пациентах 20%-тестовая выборка
# (patient_split) даёт всего 8 тестовых пациентов — недостаточно для
# статистически значимой проверки H1 (см.
# experiment_article4_significance_check.py, 95% ДИ разницы accuracy
# включал ноль при 8 пациентах). 160 пациентов -> 32 тестовых, 128
# обучающих — на порядок больше мощности при том же протоколе честного
# разбиения по пациентам.
MAX_PATIENTS: Optional[int] = int(os.environ.get("BRATS_MAX_PATIENTS", "160"))

# Доля ненулевых ("мозговых") пикселей в срезе, ниже которой срез считается
# почти пустым (верхние/нижние срезы объёма, вне мозга) и отбрасывается.
MIN_BRAIN_FRACTION = float(os.environ.get("MIN_BRAIN_FRACTION", "0.02"))

TARGET_SIZE = (256, 256)  # DEFAULT_SIZE остального пайплайна


def _find_patient_dirs(root: Path) -> List[Path]:
    """Находит директории пациентов (содержат файл *_{MODALITY_SUFFIX}.nii[.gz])."""
    pattern_nii = f"*_{MODALITY_SUFFIX}.nii"
    pattern_nii_gz = f"*_{MODALITY_SUFFIX}.nii.gz"
    candidates = sorted(
        {
            p.parent
            for p in list(root.rglob(pattern_nii)) + list(root.rglob(pattern_nii_gz))
        }
    )
    return candidates


def _find_volume_file(patient_dir: Path, suffix: str) -> Optional[Path]:
    for ext in (".nii.gz", ".nii"):
        matches = list(patient_dir.glob(f"*_{suffix}{ext}"))
        if matches:
            return matches[0]
    return None


def _normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Клиппинг по 1-99 перцентилям ненулевых вокселей -> масштаб в [0, 255]."""
    nonzero = slice_2d[slice_2d > 0]
    if nonzero.size == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    lo, hi = np.percentile(nonzero, [1, 99])
    if hi <= lo:
        hi = lo + 1.0

    clipped = np.clip(slice_2d, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def convert_patient(
    patient_dir: Path, preprocessor: ImagePreprocessor
) -> List[Tuple[int, np.ndarray, bool, float, float]]:
    """Конвертирует один объём пациента в список (slice_idx, png_image, has_tumor, tumor_frac, brain_frac)."""
    modality_path = _find_volume_file(patient_dir, MODALITY_SUFFIX)
    seg_path = _find_volume_file(patient_dir, SEG_SUFFIX)

    if modality_path is None:
        logger.warning("   %s: не найден *_%s.nii[.gz], пропуск.", patient_dir.name, MODALITY_SUFFIX)
        return []

    volume = nib.load(str(modality_path)).get_fdata()
    seg_volume = nib.load(str(seg_path)).get_fdata() if seg_path is not None else None

    depth = volume.shape[2]
    results = []

    for z in range(depth):
        slice_2d = volume[:, :, z]
        brain_fraction = float(np.mean(slice_2d > 0))
        if brain_fraction < MIN_BRAIN_FRACTION:
            continue

        slice_uint8 = _normalize_to_uint8(slice_2d)
        processed = preprocessor.process(slice_uint8)

        if seg_volume is not None:
            seg_slice = seg_volume[:, :, z]
            tumor_fraction = float(np.mean(seg_slice > 0))
            has_tumor = tumor_fraction > 0.0
        else:
            tumor_fraction = float("nan")
            has_tumor = False

        results.append((z, processed, has_tumor, tumor_fraction, brain_fraction))

    return results


def main() -> None:
    if not BRATS_ROOT.is_dir():
        raise FileNotFoundError(
            f"BRATS_ROOT не найден: {BRATS_ROOT}. Скачайте BraTS2020 (Kaggle: "
            "awsaf49/brats20-dataset-training-validation) и укажите путь к "
            "распакованной MICCAI_BraTS2020_TrainingData через переменную "
            "окружения BRATS_ROOT."
        )

    patient_dirs = _find_patient_dirs(BRATS_ROOT)
    if not patient_dirs:
        raise FileNotFoundError(
            f"В {BRATS_ROOT} не найдено ни одного пациента (*_{MODALITY_SUFFIX}.nii[.gz]) — "
            "проверьте, что путь указывает на распакованную папку BraTS2020."
        )

    if MAX_PATIENTS is not None:
        patient_dirs = patient_dirs[:MAX_PATIENTS]

    logger.info(
        "Найдено пациентов: %d (modality=%s, лимит=%s).",
        len(patient_dirs), MODALITY_SUFFIX, MAX_PATIENTS,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    preprocessor = ImagePreprocessor(target_size=TARGET_SIZE)

    metadata_rows = []
    total_slices = 0

    for i, patient_dir in enumerate(patient_dirs, start=1):
        patient_id = patient_dir.name
        patient_out_dir = OUTPUT_ROOT / patient_id
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        slices = convert_patient(patient_dir, preprocessor)
        for slice_idx, image, has_tumor, tumor_fraction, brain_fraction in slices:
            filename = f"{patient_id}_slice{slice_idx:03d}.png"
            output_path = patient_out_dir / filename
            cv2.imwrite(str(output_path), image)

            metadata_rows.append(
                {
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "path": str(output_path.relative_to(OUTPUT_ROOT)),
                    "has_tumor": has_tumor,
                    "tumor_pixel_fraction": tumor_fraction,
                    "brain_pixel_fraction": brain_fraction,
                }
            )

        total_slices += len(slices)
        logger.info(
            "   [%d/%d] %s: %d срезов сохранено.",
            i, len(patient_dirs), patient_id, len(slices),
        )

    metadata_path = OUTPUT_ROOT / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id", "slice_idx", "path",
                "has_tumor", "tumor_pixel_fraction", "brain_pixel_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    n_tumor = sum(1 for r in metadata_rows if r["has_tumor"])
    logger.info(
        "\nГотово: %d пациентов, %d срезов (%d с опухолью, %d без) -> %s",
        len(patient_dirs), total_slices, n_tumor, total_slices - n_tumor, OUTPUT_ROOT,
    )
    logger.info("Метаданные: %s", metadata_path)


if __name__ == "__main__":
    main()
