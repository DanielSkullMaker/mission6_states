"""Идея 2 ("Есть ли идеи на счёт фильтрации датасета для увеличения
точности?"): более строгий порог MIN_BRAIN_FRACTION в конвертере
(0.02 -> 0.08) — отбрасывает больше срезов на границе мозга при конвертации
NIfTI -> PNG (experiments/prepare_brats_dataset.py, см. правку с
env-переменными MIN_BRAIN_FRACTION/BRATS_OUTPUT_ROOT).

Датасет сгенерирован отдельно (НЕ перезаписывая оригинал):
    BRATS_ROOT="<путь>" MIN_BRAIN_FRACTION=0.08 \\
        BRATS_OUTPUT_ROOT="datasets/brats_flair_slices_v2" \\
        python experiments/prepare_brats_dataset.py

⚠️ ПОБОЧНЫЙ ЭФФЕКТ, обнаруженный при генерации (важно для интерпретации
результата): новый порог отбрасывает ДИСПРОПОРЦИОНАЛЬНО больше no_tumor
срезов (они чаще оказываются на самой границе мозга, где опухоли физически
нет) — баланс классов на уровне ВСЕХ срезов сдвигается с ~50.7%/49.3%
(tumor/no_tumor) до ~59.6%/40.4%. Обучающий пул здесь всё равно сбалансирован
искусственно (100/класс), но TEST (все срезы held-out пациентов) — естественно
НЕТ, и теперь ещё более смещён в сторону tumor, чем раньше. Рост accuracy
(если будет) может частично объясняться этим сдвигом баланса, а не только
"чище" данными — это явно проверяется ниже сравнением per-class recall, не
только общей accuracy.

Патентный сплит — ТЕ ЖЕ 40 пациентов / те же 32 train (24 search_train + 8
validation) / 8 test ID, что и во всей серии (patient_id не зависит от
того, сколько срезов конвертер сохранил для этого пациента).

Артефакты: artifacts/improve_brats_accuracy_v4_{report.json,results.md}.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from subspace_conjugacy import (
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
    smooth_adjacent_vectors,
)
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

DATASET_ROOT_V1 = Path("datasets/brats_flair_slices")
DATASET_ROOT_V2 = Path("datasets/brats_flair_slices_v2")
RANDOM_SEED = 42
N_SUBCLASSES = 12
TRAIN_PER_CLASS = 100

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_v4_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_v4_results.md"

Row = Dict[str, Any]


def load_metadata(dataset_root: Path) -> List[Row]:
    rows: List[Row] = []
    with open(dataset_root / "metadata.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "patient_id": r["patient_id"],
                    "slice_idx": int(r["slice_idx"]),
                    "path": dataset_root / r["path"],
                    "label": "tumor" if r["has_tumor"] == "True" else "no_tumor",
                }
            )
    return rows


def vectorize_rows(rows: List[Row]) -> Tuple[np.ndarray, np.ndarray]:
    X = load_and_vectorize_batch([r["path"] for r in rows], method="horizontal")
    y = np.array([r["label"] for r in rows])
    return X, y


def split_patients(all_patient_ids: Sequence[str]) -> Tuple[List[str], List[str]]:
    """ТА ЖЕ логика/seed — набор patient_id идентичен в v1/v2 (конвертер не
    добавляет/удаляет пациентов, только срезы внутри каждого)."""
    patient_ids = sorted(all_patient_ids)
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(patient_ids)
    n_test = max(1, round(len(shuffled) * 0.2))
    test_ids = sorted(shuffled[:n_test].tolist())
    train_ids = sorted(shuffled[n_test:].tolist())
    return train_ids, test_ids


def split_search_validation(train_ids: List[str]) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(train_ids)
    n_val = round(len(shuffled) * 0.25)
    val_ids = sorted(shuffled[:n_val].tolist())
    search_train_ids = sorted(shuffled[n_val:].tolist())
    return search_train_ids, val_ids


def select_training_rows(rows: List[Row], patient_ids: Sequence[str], per_class: int, seed: int = RANDOM_SEED) -> List[Row]:
    pool = [r for r in rows if r["patient_id"] in set(patient_ids)]
    rng = np.random.default_rng(seed)
    selected: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in pool if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:per_class]
        selected.extend(cls_rows[i] for i in idx)
    return selected


def build_patient_grouped_rows(rows: List[Row], patient_ids: Sequence[str]) -> Tuple[List[Row], List[int]]:
    pool = [r for r in rows if r["patient_id"] in set(patient_ids)]
    ordered: List[Row] = []
    group_lengths: List[int] = []
    for pid in sorted(set(patient_ids)):
        group = sorted((r for r in pool if r["patient_id"] == pid), key=lambda r: r["slice_idx"])
        ordered.extend(group)
        group_lengths.append(len(group))
    return ordered, group_lengths


def smooth_per_patient_group(X: np.ndarray, group_lengths: List[int]) -> np.ndarray:
    blocks = []
    offset = 0
    for length in group_lengths:
        blocks.append(smooth_adjacent_vectors(X[offset : offset + length]))
        offset += length
    return np.vstack(blocks)


def main() -> None:
    start = time.perf_counter()
    print("=== Идея 2: более строгий MIN_BRAIN_FRACTION (0.02 -> 0.08) в конвертере ===")

    rows_v2 = load_metadata(DATASET_ROOT_V2)
    all_patient_ids = {r["patient_id"] for r in rows_v2}
    train_ids, test_ids = split_patients(all_patient_ids)
    search_train_ids, val_ids = split_search_validation(train_ids)
    print(f"Пациенты (v2): {len(search_train_ids)} search_train, {len(val_ids)} validation, {len(test_ids)} test.")
    print(f"Всего срезов v2: {len(rows_v2)} (было 5184 в v1).")

    val_rows, _ = build_patient_grouped_rows(rows_v2, val_ids)
    X_val, y_val = vectorize_rows(val_rows)
    test_rows, test_group_lengths = build_patient_grouped_rows(rows_v2, test_ids)
    X_test, y_test = vectorize_rows(test_rows)
    X_test_smoothed = smooth_per_patient_group(X_test, test_group_lengths)
    print(f"X_val.shape={X_val.shape}, X_test.shape={X_test.shape}")
    print(f"Баланс классов test (v2): tumor={int((y_test=='tumor').sum())}, no_tumor={int((y_test=='no_tumor').sum())}")

    print("\n--- Поиск (validation, search_train) ---")
    search_train_rows = select_training_rows(rows_v2, search_train_ids, TRAIN_PER_CLASS)
    X_search_train, y_search_train = vectorize_rows(search_train_rows)
    clf_search = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf_search.fit(X_search_train, y_search_train)
    search_fit_seconds = time.perf_counter() - t0
    val_acc = evaluate_classifier(y_val, clf_search.predict(X_val))["accuracy"]
    print(f"   fit={search_fit_seconds:.1f}s  val_accuracy={val_acc:.4f}  (сравнить с v1 baseline 0.6651)")

    print("\n--- Финал: переобучение на всех 32 train-пациентах (v2), одна оценка на test (v2) ---")
    final_train_rows = select_training_rows(rows_v2, train_ids, TRAIN_PER_CLASS)
    X_final_train, y_final_train = vectorize_rows(final_train_rows)
    final_clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    final_clf.fit(X_final_train, y_final_train)
    final_fit_seconds = time.perf_counter() - t0

    report = evaluate_classifier(y_test, final_clf.predict(X_test))
    report_smoothed = evaluate_classifier(y_test, final_clf.predict(X_test_smoothed))
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}
    per_class_smoothed = {str(k): v for k, v in report_smoothed["per_class_accuracy"].items()}
    print(f"   fit={final_fit_seconds:.1f}s  accuracy={report['accuracy']:.4f}  per_class={per_class}")
    print(f"   accuracy_smoothed={report_smoothed['accuracy']:.4f}  per_class_smoothed={per_class_smoothed}")

    print("\n" + "=" * 90)
    print("ИТОГ: идея 2 (MIN_BRAIN_FRACTION=0.08) vs v1 (0.02, baseline n_sub=12)")
    print("=" * 90)
    print(f"{'v1 baseline (0.02, из прошлых запусков)':55}0.6447  (0.6563 со сглаживанием)")
    print(f"{'v2 (0.08) validation (для сравнения с v1 0.6651)':55}{val_acc:.4f}")
    print(f"{'v2 (0.08) final test accuracy':55}{report['accuracy']:.4f}  ({report_smoothed['accuracy']:.4f} со сглаживанием)")
    print(f"{'v2 per-class recall':55}{per_class}")
    print("=" * 90)
    print(
        "\nВНИМАНИЕ: test v2 естественно смещён к tumor (баланс классов сдвинут при более "
        "строгой фильтрации) — рост общей accuracy, если есть, нужно проверять по per-class recall, "
        "не только по общей цифре."
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "min_brain_fraction": 0.08,
        "n_slices_v2": len(rows_v2),
        "test_class_balance": {"tumor": int((y_test == "tumor").sum()), "no_tumor": int((y_test == "no_tumor").sum())},
        "search_val_accuracy": val_acc,
        "search_fit_seconds": search_fit_seconds,
        "final_fit_seconds": final_fit_seconds,
        "final_accuracy": report["accuracy"],
        "final_accuracy_smoothed": report_smoothed["accuracy"],
        "final_per_class_accuracy": per_class,
        "final_per_class_accuracy_smoothed": per_class_smoothed,
        "v1_baseline_accuracy": 0.6447,
        "v1_baseline_accuracy_smoothed": 0.6563,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    md_lines = [
        "# Идея 2: MIN_BRAIN_FRACTION=0.08 (BraTS has_tumor)",
        "",
        f"v1 (0.02) baseline: 0.6447 (0.6563 со сглаживанием).",
        f"v2 (0.08): validation={val_acc:.4f}, final test accuracy={report['accuracy']:.4f} "
        f"({report_smoothed['accuracy']:.4f} со сглаживанием).",
        f"Per-class recall (v2, final): {per_class}.",
        "",
        "⚠️ Баланс классов test v2 смещён к tumor сильнее, чем v1 — интерпретировать рост "
        "общей accuracy с осторожностью, смотреть per-class recall.",
    ]
    REPORT_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
