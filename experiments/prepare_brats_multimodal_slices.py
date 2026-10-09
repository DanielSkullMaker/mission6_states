"""Подготовка ДОПОЛНИТЕЛЬНЫХ модальностей (T1, T1ce, T2) для уже
существующих срезов `datasets/brats_flair_slices/` — для эксперимента
повышения точности Статьи 4 (целевая метрика Dice >=0.90, см. обсуждение
с пользователем 2026-10-09). Веб-ресерч (BraTS2020) показывает, что
мультимодальный вход (FLAIR+T1+T1ce+T2) — главный известный рычаг для
роста Dice whole-tumor с ~0.65 (одна FLAIR, 2D) до ~0.85-0.90
(мультимодальность), в разы весомее архитектурных вариаций в пределах
одной модальности.

Тот же принцип, что и `prepare_brats_segmentation_masks.py`: НЕ
пересоздаём FLAIR-PNG (уже сохранены и выверены), а воспроизводим ТОТ ЖЕ
геометрический путь (resize LANCZOS -> подавление фона -> сдвиг
центрирования, посчитанный ПО FLAIR) для T1/T1ce/T2, чтобы все 4
модальности были пиксельно выровнены между собой и с уже сохранённой
маской.

Запуск (после prepare_brats_dataset.py и prepare_brats_segmentation_masks.py):
    BRATS_ROOT="<путь>/MICCAI_BraTS2020_TrainingData" \\
        python experiments/prepare_brats_multimodal_slices.py
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
    os.environ.get("BRATS_ROOT", r"C:\mission_6states\data\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
)
DATASET_ROOT = Path(os.environ.get("BRATS_OUTPUT_ROOT", "datasets/brats_flair_slices"))
METADATA_PATH = DATASET_ROOT / "metadata.csv"
TARGET_SIZE = DEFAULT_SIZE
BACKGROUND_THRESHOLD = 10

MODALITIES = ["t1", "t1ce", "t2"]  # flair уже есть


def _find_volume_file(patient_dir: Path, suffix: str) -> Optional[Path]:
    for ext in (".nii.gz", ".nii"):
        matches = list(patient_dir.glob(f"*_{suffix}{ext}"))
        if matches:
            return matches[0]
    return None


def _normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Идентична prepare_brats_dataset.py::_normalize_to_uint8."""
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
        raise FileNotFoundError(f"{METADATA_PATH} не найден — сначала запустите prepare_brats_dataset.py.")
    by_patient: Dict[str, List[int]] = defaultdict(list)
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_patient[r["patient_id"]].append(int(r["slice_idx"]))
    return by_patient


def process_patient(patient_id: str, slice_indices: List[int]) -> int:
    patient_dir = BRATS_ROOT / patient_id
    flair_path = _find_volume_file(patient_dir, "flair")
    if flair_path is None:
        logger.warning("   %s: не найден FLAIR volume, пропуск.", patient_id)
        return 0

    modality_volumes = {}
    for mod in MODALITIES:
        p = _find_volume_file(patient_dir, mod)
        if p is None:
            logger.warning("   %s: не найден %s volume, пропуск пациента.", patient_id, mod)
            return 0
        modality_volumes[mod] = nib.load(str(p)).get_fdata()

    flair_volume = nib.load(str(flair_path)).get_fdata()

    out_dir = DATASET_ROOT / patient_id
    n_saved = 0
    for z in slice_indices:
        # Пересчитываем сдвиг центрирования ПО FLAIR (как в
        # prepare_brats_segmentation_masks.py) — так все модальности
        # получают ОДИН И ТОТ ЖЕ сдвиг, выровненный с уже сохранённым
        # FLAIR-PNG и маской.
        flair_uint8 = _normalize_to_uint8(flair_volume[:, :, z])
        resized_flair = cv2.resize(flair_uint8, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        masked_gray = suppress_background(resized_flair, BACKGROUND_THRESHOLD)
        h_delta = compute_horizontal_delta(masked_gray)
        v_delta = compute_vertical_delta(masked_gray)

        for mod, volume in modality_volumes.items():
            mod_uint8 = _normalize_to_uint8(volume[:, :, z])
            resized_mod = cv2.resize(mod_uint8, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
            shifted_mod = shift_rows(resized_mod, h_delta, min_shift=DEFAULT_MIN_SHIFT)
            shifted_mod = shift_columns(shifted_mod, v_delta, min_shift=DEFAULT_MIN_SHIFT)
            out_path = out_dir / f"{patient_id}_slice{z:03d}_{mod}.png"
            cv2.imwrite(str(out_path), shifted_mod)
        n_saved += 1
    return n_saved


def main() -> None:
    if not BRATS_ROOT.is_dir():
        raise FileNotFoundError(f"BRATS_ROOT не найден: {BRATS_ROOT}.")

    by_patient = load_metadata_by_patient()
    logger.info("Пациентов в metadata.csv: %d. Модальности: %s.", len(by_patient), MODALITIES)

    total = 0
    for i, (patient_id, slice_indices) in enumerate(sorted(by_patient.items()), start=1):
        n_saved = process_patient(patient_id, slice_indices)
        total += n_saved
        if i % 20 == 0 or i == len(by_patient):
            logger.info("   [%d/%d] %s: %d срезов x %d модальности сохранено.", i, len(by_patient), patient_id, n_saved, len(MODALITIES))

    logger.info("\nГотово: %d срезов x %d модальности сохранено в %s.", total, len(MODALITIES), DATASET_ROOT)


if __name__ == "__main__":
    main()
