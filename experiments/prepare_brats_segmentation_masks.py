"""Готовит бинарные маски сегментации (опухоль/фон) для уже существующих
срезов `datasets/brats_flair_slices/` — требуется для нейросетевой версии
Статьи 4 (бинарная СЕГМЕНТАЦИЯ опухоли, см. dissertation_articles_plan.txt),
которая заменила классификацию has_tumor на попиксельную сегментацию по
примеру `state4_scripts/3d-mri-brain-tumor-segmentation-u-net*.ipynb`.

Почему отдельный скрипт, а не перегенерация `prepare_brats_dataset.py`:
тот скрипт уже сохранил ОБРАБОТАННЫЕ (resize + подавление фона +
центрирование) FLAIR-срезы и их has_tumor/tumor_fraction в metadata.csv, но
НЕ сохранял сами маски сегментации как изображения — для классификации это
было не нужно. Чтобы маска была пиксельно выровнена с уже сохранённым FLAIR-
PNG, этот скрипт воспроизводит ТОТ ЖЕ геометрический путь (resize ->
подавление фона -> вычисление сдвига центрирования) из сырого NIfTI, но:
  - `resize_image` (LANCZOS) применяется к FLAIR, как и раньше;
  - для маски используется `cv2.INTER_NEAREST` при resize (LANCZOS размыл
    бы бинарную маску в полутона на границе опухоли — недопустимо для
    разметки);
  - сдвиг центрирования (h_delta/v_delta) вычисляется ОДИН РАЗ по FLAIR-
    срезу (как в `ImagePreprocessor`/`center_image`) и применяется ОБА раза
    — к FLAIR (уже сохранён, здесь не пересчитывается) и к маске — так оба
    изображения остаются пространственно выровненными.

Запуск (после того, как `prepare_brats_dataset.py` уже создал metadata.csv):
    BRATS_ROOT="<путь>/MICCAI_BraTS2020_TrainingData" \\
        python experiments/prepare_brats_segmentation_masks.py
"""

import csv
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "nibabel не установлен: pip install -e \".[experiments]\" (или `pip install nibabel`)."
    ) from exc

from subspace_conjugacy.preprocessing.normalization import suppress_background
from subspace_conjugacy.preprocessing.centering import (
    compute_horizontal_delta,
    compute_vertical_delta,
    shift_rows,
    shift_columns,
    DEFAULT_MIN_SHIFT,
)
from subspace_conjugacy.preprocessing.resize import DEFAULT_SIZE

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BRATS_ROOT = Path(
    os.environ.get("BRATS_ROOT", r"C:\Users\nazar\Downloads\MICCAI_BraTS2020_TrainingData")
)
DATASET_ROOT = Path(os.environ.get("BRATS_OUTPUT_ROOT", "datasets/brats_flair_slices"))
METADATA_PATH = DATASET_ROOT / "metadata.csv"
MODALITY_SUFFIX = "flair"
SEG_SUFFIX = "seg"
TARGET_SIZE = DEFAULT_SIZE
BACKGROUND_THRESHOLD = 10  # по умолчанию ImagePreprocessor — см. docstring модуля


def _find_volume_file(patient_dir: Path, suffix: str) -> Optional[Path]:
    for ext in (".nii.gz", ".nii"):
        matches = list(patient_dir.glob(f"*_{suffix}{ext}"))
        if matches:
            return matches[0]
    return None


def _normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Идентична prepare_brats_dataset.py::_normalize_to_uint8 — тот же путь,
    чтобы h_delta/v_delta совпали с уже сохранённым FLAIR-PNG."""
    nonzero = slice_2d[slice_2d > 0]
    if nonzero.size == 0:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    lo, hi = np.percentile(nonzero, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(slice_2d, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def load_metadata_by_patient() -> Dict[str, List[int]]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} не найден — сначала запустите prepare_brats_dataset.py."
        )
    by_patient: Dict[str, List[int]] = defaultdict(list)
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_patient[r["patient_id"]].append(int(r["slice_idx"]))
    return by_patient


def process_patient(patient_id: str, slice_indices: List[int]) -> int:
    patient_dir = BRATS_ROOT / patient_id
    flair_path = _find_volume_file(patient_dir, MODALITY_SUFFIX)
    seg_path = _find_volume_file(patient_dir, SEG_SUFFIX)
    if flair_path is None or seg_path is None:
        logger.warning("   %s: не найдены FLAIR/seg volume, пропуск.", patient_id)
        return 0

    flair_volume = nib.load(str(flair_path)).get_fdata()
    seg_volume = nib.load(str(seg_path)).get_fdata()

    out_dir = DATASET_ROOT / patient_id
    n_saved = 0
    for z in slice_indices:
        flair_uint8 = _normalize_to_uint8(flair_volume[:, :, z])
        resized_flair = cv2.resize(flair_uint8, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

        masked_gray = suppress_background(resized_flair, BACKGROUND_THRESHOLD)
        h_delta = compute_horizontal_delta(masked_gray)
        v_delta = compute_vertical_delta(masked_gray)

        mask_slice = (seg_volume[:, :, z] > 0).astype(np.uint8) * 255
        resized_mask = cv2.resize(mask_slice, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        shifted_mask = shift_rows(resized_mask, h_delta, min_shift=DEFAULT_MIN_SHIFT)
        shifted_mask = shift_columns(shifted_mask, v_delta, min_shift=DEFAULT_MIN_SHIFT)

        out_path = out_dir / f"{patient_id}_slice{z:03d}_mask.png"
        cv2.imwrite(str(out_path), shifted_mask)
        n_saved += 1
    return n_saved


def main() -> None:
    if not BRATS_ROOT.is_dir():
        raise FileNotFoundError(f"BRATS_ROOT не найден: {BRATS_ROOT}.")

    by_patient = load_metadata_by_patient()
    logger.info("Пациентов в metadata.csv: %d.", len(by_patient))

    total = 0
    for i, (patient_id, slice_indices) in enumerate(sorted(by_patient.items()), start=1):
        n_saved = process_patient(patient_id, slice_indices)
        total += n_saved
        logger.info("   [%d/%d] %s: %d масок сохранено.", i, len(by_patient), patient_id, n_saved)

    logger.info("\nГотово: %d масок сохранено в %s.", total, DATASET_ROOT)


if __name__ == "__main__":
    main()
