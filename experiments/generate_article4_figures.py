"""Генерирует иллюстрации для черновика Статьи 4
(experiments/generate_article4_report.py -> artifacts/article4_draft.docx):

  - artifacts/article4_fig1_dataset_samples.png — образцы датасета BraTS2020
    ПОСЛЕ предобработки (срез без опухоли / срез с опухолью), 256x256.
  - artifacts/article4_fig2_preprocessing.png — демонстрация предобработки
    на одном срезе (пациент BraTS20_Training_001, slice 30): (1) сырой
    FLAIR-срез сразу после загрузки NIfTI и нормализации в uint8 (ДО
    resize/подавления фона/центрирования); (2) тот же срез с наложенной
    маской сегментации опухоли (_seg.nii) — визуальное обоснование метки
    has_tumor; (3) тот же срез ПОСЛЕ полного пайплайна ImagePreprocessor
    (resize -> подавление фона -> центрирование) — реальный файл из
    datasets/brats_flair_slices/, тот же, что использован в экспериментах.

Требует nibabel (pip install -e ".[experiments]") и доступа к сырым NIfTI-
объёмам BraTS2020 (BRATS_ROOT, по умолчанию как в prepare_brats_dataset.py).

Запуск:
    python experiments/generate_article4_figures.py
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from prepare_brats_dataset import _normalize_to_uint8, MODALITY_SUFFIX, SEG_SUFFIX

BRATS_ROOT = Path(
    os.environ.get(
        "BRATS_ROOT",
        r"C:\mission_6states\data\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    )
)
PATIENT_ID = "BraTS20_Training_001"
SLICE_IDX = 67  # has_tumor=True, tumor_pixel_fraction~8.8% (наглядно), metadata.csv

DATASET_ROOT = Path("datasets/brats_flair_slices")
PROCESSED_TUMOR_PATH = DATASET_ROOT / PATIENT_ID / f"{PATIENT_ID}_slice{SLICE_IDX:03d}.png"
PROCESSED_NOTUMOR_PATH = DATASET_ROOT / PATIENT_ID / f"{PATIENT_ID}_slice013.png"

ARTIFACTS_DIR = Path("artifacts")
FIG1_PATH = ARTIFACTS_DIR / "article4_fig1_dataset_samples.png"
FIG2_PATH = ARTIFACTS_DIR / "article4_fig2_preprocessing.png"

GAP = 12
BG = 255


def hconcat(images: list, gap: int = GAP, bg: int = BG) -> Image.Image:
    height = max(im.height for im in images)
    width = sum(im.width for im in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), (bg, bg, bg))
    x = 0
    for im in images:
        canvas.paste(im, (x, (height - im.height) // 2))
        x += im.width + gap
    return canvas


def find_volume(patient_dir: Path, suffix: str) -> Path:
    for ext in (".nii.gz", ".nii"):
        matches = list(patient_dir.glob(f"*_{suffix}{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"{suffix} не найден в {patient_dir}")


def main() -> None:
    patient_dir = BRATS_ROOT / PATIENT_ID
    flair_path = find_volume(patient_dir, MODALITY_SUFFIX)
    seg_path = find_volume(patient_dir, SEG_SUFFIX)

    flair_slice = nib.load(str(flair_path)).get_fdata()[:, :, SLICE_IDX]
    seg_slice = nib.load(str(seg_path)).get_fdata()[:, :, SLICE_IDX]

    raw_uint8 = _normalize_to_uint8(flair_slice)
    raw_img = Image.fromarray(raw_uint8, mode="L").convert("RGB")

    overlay_arr = np.stack([raw_uint8] * 3, axis=-1).copy()
    mask = seg_slice > 0
    overlay_arr[mask] = [255, 40, 40]
    overlay_img = Image.fromarray(overlay_arr, mode="RGB")

    processed_img = Image.open(PROCESSED_TUMOR_PATH).convert("RGB")

    fig2 = hconcat([raw_img, overlay_img, processed_img])
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig2.save(FIG2_PATH)
    print(f"Сохранено: {FIG2_PATH} ({fig2.width}x{fig2.height}, "
          f"панели: сырой {raw_img.size} | сырой+маска {overlay_img.size} | "
          f"обработанный {processed_img.size})")

    notumor_img = Image.open(PROCESSED_NOTUMOR_PATH).convert("RGB")
    tumor_img = Image.open(PROCESSED_TUMOR_PATH).convert("RGB")
    fig1 = hconcat([notumor_img, tumor_img])
    fig1.save(FIG1_PATH)
    print(f"Сохранено: {FIG1_PATH} ({fig1.width}x{fig1.height})")


if __name__ == "__main__":
    main()
