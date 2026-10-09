"""Статья 4 — два дополнительных абляционных эксперимента (по запросу
"добавь больше сравнений и больше таблиц: разные методы и разные
предобработки") + иллюстрация предобработки для черновика статьи.

1. ПРЕДОБРАБОТКА (preprocessing ablation): тот же честный пул (patient_split
   — те же 200 train / 1033 test, что и в experiment_brats_split_methods.py/
   experiment_article4_boost_accuracy.py, тот же RANDOM_SEED), но срезы
   извлекаются из сырых NIfTI-объёмов и обрабатываются ПЯТЬЮ разными
   способами (варьируется позже всего, после приведения к uint8), затем
   классификатор (S=8) обучается и оценивается заново на каждом варианте.
   Сравниваются: полный пайплайн (resize -> подавление фона -> центрирование,
   baseline), только resize (без подавления фона и центрирования), resize +
   подавление фона БЕЗ центрирующего сдвига, полный пайплайн с усиленным
   порогом подавления фона (40 вместо 10), полный пайплайн с простой
   min-max нормализацией интенсивности вместо перцентильного клиппинга.

2. СГЛАЖИВАНИЕ (smoothing-method ablation): ОДИН обученный классификатор
   (тот же train, что в experiment_sequential_smoothing.py), тестовая
   последовательность — содержательный порядок 8 отложенных пациентов;
   сравниваются 5 вариантов применяемого к тесту сглаживания: без
   сглаживания, фиксированное окно 3/5/7 и переменное окно 3/5/7
   (основной вариант статьи). Переобучение классификатора не требуется —
   сглаживание применяется только к тестовым векторам.

Артефакты:
  - artifacts/article4_preprocessing_ablation_report.json
  - artifacts/article4_smoothing_ablation_report.json
  - artifacts/article4_fig3_preprocessing_variants.png (иллюстрация пяти
    вариантов предобработки одного среза)

Требует nibabel и доступа к сырым NIfTI-объёмам BraTS2020 (BRATS_ROOT).

Запуск:
    python experiments/experiment_article4_ablations.py
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
from PIL import Image

from subspace_conjugacy import SubspaceConjugacyClassifier, smooth_adjacent_vectors
from subspace_conjugacy.evaluation.metrics import evaluate_classifier
from subspace_conjugacy.features.vectorization import vectorize_batch
from subspace_conjugacy.preprocessing.centering import center_image
from subspace_conjugacy.preprocessing.normalization import suppress_background
from subspace_conjugacy.preprocessing.resize import resize_image

from prepare_brats_dataset import MODALITY_SUFFIX, _normalize_to_uint8

BRATS_ROOT = Path(
    os.environ.get(
        "BRATS_ROOT",
        r"C:\mission_6states\data\BraTS2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData",
    )
)
DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42
N_SUBCLASSES = 8
TRAIN_PER_CLASS = 100

ARTIFACTS_DIR = Path("artifacts")
PREP_REPORT_PATH = ARTIFACTS_DIR / "article4_preprocessing_ablation_report.json"
SMOOTH_REPORT_PATH = ARTIFACTS_DIR / "article4_smoothing_ablation_report.json"
FIG3_PATH = ARTIFACTS_DIR / "article4_fig3_preprocessing_variants.png"
FIG4_PATH = ARTIFACTS_DIR / "article4_fig4_smoothing_effect.png"

DEMO_PATIENT_ID = "BraTS20_Training_001"
DEMO_SLICE_IDX = 67
SMOOTHING_DEMO_SLICE_RANGE = range(63, 68)  # 5 последовательных срезов, 63..67

Row = Dict[str, Any]


# --- Метаданные / сплит (идентично experiment_article4_boost_accuracy.py) ---

def load_metadata() -> List[Row]:
    rows: List[Row] = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "patient_id": r["patient_id"],
                "slice_idx": int(r["slice_idx"]),
                "label": "tumor" if r["has_tumor"] == "True" else "no_tumor",
            })
    return rows


def split_patients(rows: List[Row]) -> Tuple[List[str], List[str]]:
    patient_ids = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(patient_ids)
    n_test = max(1, round(len(shuffled) * 0.2))
    test_ids = sorted(shuffled[:n_test].tolist())
    train_ids = sorted(shuffled[n_test:].tolist())
    return train_ids, test_ids


def select_rows(rows: List[Row], patient_ids: Sequence[str], per_class: int, seed: int) -> List[Row]:
    pool = [r for r in rows if r["patient_id"] in set(patient_ids)]
    rng = np.random.default_rng(seed)
    selected: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in pool if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:per_class]
        selected.extend(cls_rows[i] for i in idx)
    return selected


# --- Доступ к сырым NIfTI-объёмам (с кэшем по пациенту) ---

_volume_cache: Dict[str, np.ndarray] = {}


def _find_volume(patient_dir: Path, suffix: str) -> Path:
    for ext in (".nii.gz", ".nii"):
        matches = list(patient_dir.glob(f"*_{suffix}{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"{suffix} не найден в {patient_dir}")


def get_raw_slice(patient_id: str, slice_idx: int) -> np.ndarray:
    if patient_id not in _volume_cache:
        path = _find_volume(BRATS_ROOT / patient_id, MODALITY_SUFFIX)
        _volume_cache[patient_id] = nib.load(str(path)).get_fdata()
    return _volume_cache[patient_id][:, :, slice_idx]


# --- Варианты предобработки ---

def _minmax_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Простая min-max нормализация (альтернатива перцентильному клиппингу)."""
    lo, hi = float(slice_2d.min()), float(slice_2d.max())
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((slice_2d - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def prep_full_pipeline(slice_2d: np.ndarray) -> np.ndarray:
    """Baseline: perc-clip -> resize -> подавление фона + центрирование (thr=10)."""
    uint8 = _normalize_to_uint8(slice_2d)
    resized = resize_image(uint8, (256, 256))
    return center_image(resized, background_threshold=10, min_shift=2)


def prep_resize_only(slice_2d: np.ndarray) -> np.ndarray:
    """Только resize — без подавления фона и без центрирования."""
    uint8 = _normalize_to_uint8(slice_2d)
    return resize_image(uint8, (256, 256))


def prep_suppress_no_shift(slice_2d: np.ndarray) -> np.ndarray:
    """Resize + подавление фона, БЕЗ центрирующего сдвига."""
    uint8 = _normalize_to_uint8(slice_2d)
    resized = resize_image(uint8, (256, 256))
    return suppress_background(resized, threshold=10)


def prep_strong_threshold(slice_2d: np.ndarray) -> np.ndarray:
    """Полный пайплайн с усиленным порогом подавления фона (40 вместо 10)."""
    uint8 = _normalize_to_uint8(slice_2d)
    resized = resize_image(uint8, (256, 256))
    return center_image(resized, background_threshold=40, min_shift=2)


def prep_minmax_normalization(slice_2d: np.ndarray) -> np.ndarray:
    """Полный пайплайн, но с простой min-max нормализацией вместо перцентильной."""
    uint8 = _minmax_to_uint8(slice_2d)
    resized = resize_image(uint8, (256, 256))
    return center_image(resized, background_threshold=10, min_shift=2)


PREPROCESSING_VARIANTS: List[Tuple[str, str, Any]] = [
    ("full_pipeline", "Полный пайплайн (resize + подавление фона + центрирование, порог 10)", prep_full_pipeline),
    ("resize_only", "Только resize (без подавления фона и центрирования)", prep_resize_only),
    ("suppress_no_center", "Resize + подавление фона, без центрирующего сдвига", prep_suppress_no_shift),
    ("strong_threshold", "Полный пайплайн, порог подавления фона 40", prep_strong_threshold),
    ("minmax_norm", "Полный пайплайн, min-max нормализация вместо перцентильной", prep_minmax_normalization),
]


def build_variant_matrix(rows: List[Row], prep_fn) -> np.ndarray:
    images = [prep_fn(get_raw_slice(r["patient_id"], r["slice_idx"])) for r in rows]
    return vectorize_batch(images, method="horizontal")


def run_preprocessing_ablation(train_rows: List[Row], test_rows: List[Row]) -> List[Dict[str, Any]]:
    y_train = np.array([r["label"] for r in train_rows])
    y_test = np.array([r["label"] for r in test_rows])

    results = []
    for name, description, prep_fn in PREPROCESSING_VARIANTS:
        print(f"\n=== Предобработка: {name} — {description} ===")
        X_train = build_variant_matrix(train_rows, prep_fn)
        X_test = build_variant_matrix(test_rows, prep_fn)

        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        report = evaluate_classifier(y_test, clf.predict(X_test))
        print(f"   fit={fit_seconds:.1f}s  accuracy={report['accuracy']:.4f}")
        results.append({
            "name": name,
            "description": description,
            "accuracy": report["accuracy"],
            "per_class_accuracy": {str(k): v for k, v in report["per_class_accuracy"].items()},
            "fit_seconds": fit_seconds,
        })
    return results


def run_smoothing_ablation(rows: List[Row], train_ids: List[str], test_ids: List[str]) -> Dict[str, Any]:
    train_rows = select_rows(rows, train_ids, TRAIN_PER_CLASS, RANDOM_SEED)
    X_train = build_variant_matrix(train_rows, prep_full_pipeline)
    y_train = np.array([r["label"] for r in train_rows])

    test_pool = [r for r in rows if r["patient_id"] in set(test_ids)]
    group_lengths: List[int] = []
    ordered_test_rows: List[Row] = []
    for pid in sorted(set(test_ids)):
        group = sorted((r for r in test_pool if r["patient_id"] == pid), key=lambda r: r["slice_idx"])
        ordered_test_rows.extend(group)
        group_lengths.append(len(group))
    X_test = build_variant_matrix(ordered_test_rows, prep_full_pipeline)
    y_test = np.array([r["label"] for r in ordered_test_rows])

    print(f"\n=== Обучение классификатора для абляции сглаживания (n_train={X_train.shape[0]}) ===")
    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    print(f"   fit={fit_seconds:.1f}s")

    def smooth_per_group(X: np.ndarray, half_widths) -> np.ndarray:
        blocks = []
        offset = 0
        for length in group_lengths:
            blocks.append(smooth_adjacent_vectors(X[offset:offset + length], half_widths=half_widths))
            offset += length
        return np.vstack(blocks)

    variants = [
        ("none", "Без сглаживания", None),
        ("fixed_3", "Фиксированное окно 3", (1,)),
        ("fixed_5", "Фиксированное окно 5", (2,)),
        ("fixed_7", "Фиксированное окно 7", (3,)),
        ("variable_3_5_7", "Переменное окно 3/5/7 (основной вариант статьи)", (1, 2, 3)),
    ]
    results = []
    for name, description, half_widths in variants:
        X_eval = X_test if half_widths is None else smooth_per_group(X_test, half_widths)
        report = evaluate_classifier(y_test, clf.predict(X_eval))
        print(f"   {name}: accuracy={report['accuracy']:.4f}")
        results.append({
            "name": name,
            "description": description,
            "accuracy": report["accuracy"],
            "per_class_accuracy": {str(k): v for k, v in report["per_class_accuracy"].items()},
        })
    return {"fit_seconds": fit_seconds, "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0]), "results": results}


def generate_figure() -> None:
    raw = get_raw_slice(DEMO_PATIENT_ID, DEMO_SLICE_IDX)
    panels = []
    for name, description, prep_fn in PREPROCESSING_VARIANTS:
        img = Image.fromarray(prep_fn(raw), mode="L").convert("RGB")
        panels.append(img)

    gap = 10
    height = max(im.height for im in panels)
    width = sum(im.width for im in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for im in panels:
        canvas.paste(im, (x, (height - im.height) // 2))
        x += im.width + gap
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(FIG3_PATH)
    print(f"\nСохранено: {FIG3_PATH} ({canvas.width}x{canvas.height}, {len(panels)} панелей)")


def generate_smoothing_figure() -> None:
    """Визуализирует эффект сглаживания на самих изображениях: 5
    последовательных срезов одного пациента до и после переменного
    сглаживания (окно 3/5/7) — наглядно дополняет формулу в разделе
    «Теория»."""
    slices = [get_raw_slice(DEMO_PATIENT_ID, z) for z in SMOOTHING_DEMO_SLICE_RANGE]
    processed = [prep_full_pipeline(s) for s in slices]
    X = vectorize_batch(processed, method="horizontal")
    X_smoothed = smooth_adjacent_vectors(X, half_widths=(1, 2, 3))

    h, w = processed[0].shape
    top = [Image.fromarray(img, mode="L").convert("RGB") for img in processed]
    bottom = [Image.fromarray(row.reshape(h, w).astype(np.uint8), mode="L").convert("RGB")
              for row in X_smoothed]

    gap = 8
    row_w = sum(im.width for im in top) + gap * (len(top) - 1)
    row_h = top[0].height
    canvas = Image.new("RGB", (row_w, row_h * 2 + gap), (255, 255, 255))
    for row_idx, imgs in enumerate([top, bottom]):
        x = 0
        y = row_idx * (row_h + gap)
        for im in imgs:
            canvas.paste(im, (x, y))
            x += im.width + gap
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(FIG4_PATH)
    print(f"Сохранено: {FIG4_PATH} ({canvas.width}x{canvas.height}, "
          f"срезы {list(SMOOTHING_DEMO_SLICE_RANGE)})")


def main() -> None:
    start = time.perf_counter()
    rows = load_metadata()
    train_ids, test_ids = split_patients(rows)

    print("=== Абляция предобработки ===")
    train_rows = select_rows(rows, train_ids, TRAIN_PER_CLASS, RANDOM_SEED)
    test_rows = [r for r in rows if r["patient_id"] in set(test_ids)]
    print(f"n_train={len(train_rows)}, n_test={len(test_rows)}")
    prep_results = run_preprocessing_ablation(train_rows, test_rows)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREP_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({"random_seed": RANDOM_SEED, "n_subclasses": N_SUBCLASSES, "results": prep_results},
                   f, ensure_ascii=False, indent=2)
    print(f"JSON-отчёт: {PREP_REPORT_PATH}")

    print("\n=== Абляция сглаживания ===")
    smooth_result = run_smoothing_ablation(rows, train_ids, test_ids)
    with open(SMOOTH_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({"random_seed": RANDOM_SEED, "n_subclasses": N_SUBCLASSES, **smooth_result},
                   f, ensure_ascii=False, indent=2)
    print(f"JSON-отчёт: {SMOOTH_REPORT_PATH}")

    print("\n=== Иллюстрация вариантов предобработки ===")
    generate_figure()

    print("\n=== Иллюстрация эффекта сглаживания ===")
    generate_smoothing_figure()

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
