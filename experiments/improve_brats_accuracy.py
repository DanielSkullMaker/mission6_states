"""Пытается поднять accuracy BraTS-эксперимента (has_tumor,
experiments/experiment_sequential_smoothing.py, честный baseline
patient_split=0.5924 — artifacts/experiment_brats_split_methods_report.json)
ОБОИМИ путями — честным и нечестным — и сводит все числа в одну таблицу,
чтобы было видно цену каждого пути.

ЧЕСТНЫЙ путь (как experiments/improve_article1_accuracy.py для Статьи 1):
  - Подбор n_subclasses идёт НЕ на test, а на отдельных validation-пациентах.
  - Патч-уровень разбиения: 40 пациентов -> 8 test (ЗАФИКСИРОВАНЫ, та же
    выборка, что в experiment_sequential_smoothing.py/experiment_brats_
    split_methods.py, random_state=42 — числа сопоставимы) -> из
    оставшихся 32 -> 24 search_train + 8 validation (новый слой, тест
    никогда не участвует в подборе).
  - Грид n_subclasses=(4,6,8,10,12), выбор по accuracy на validation.
  - Финал: ОДИН раз — переобучение лучшей конфигурации на полных 32
    train-пациентах, ОДНА оценка на 8 test-пациентах (plus опционально
    honest-трюк из Статьи 4 — сглаживание по пациенту,
    smooth_adjacent_vectors, уже подтверждённое как помогающее, H1).

НЕЧЕСТНЫЙ путь (комбинация из experiment_brats_split_methods.py +
новая находка здесь):
  - Переиспользуются уже посчитанные числа: train_equals_test (0.6850),
    paired_correlation_split (0.6000), image_random_split (0.6050).
  - НОВОЕ: "model selection leakage" — та же сетка n_subclasses, что и в
    честном пути, но n_subclasses выбирается ПРЯМО по accuracy на test
    (не на validation) — классический случай, явно описанный как риск в
    docstring evaluation.paper_methodology (не то же самое, что утечка
    train==test, но тоже завышает репортируемое число). Дополнительно
    комбинируется с честным трюком сглаживания (test подсмотрен И
    сглажен) — худший случай "и бесчестно, и с доп. трюком".

Артефакты: artifacts/improve_brats_accuracy_{report.json,results.md}.

Запуск:
    python experiments/improve_brats_accuracy.py
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

DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42
GRID_N_SUBCLASSES: Sequence[int] = (4, 6, 8, 10, 12)
TRAIN_PER_CLASS = 100

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_results.md"
SMOOTHING_REFERENCE_PATH = ARTIFACTS_DIR / "experiment_sequential_smoothing_report.json"
SPLIT_METHODS_REFERENCE_PATH = ARTIFACTS_DIR / "experiment_brats_split_methods_report.json"

Row = Dict[str, Any]


def load_metadata() -> List[Row]:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"{METADATA_PATH} не найден — сначала запустите prepare_brats_dataset.py.")
    rows: List[Row] = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "patient_id": r["patient_id"],
                    "slice_idx": int(r["slice_idx"]),
                    "path": DATASET_ROOT / r["path"],
                    "label": "tumor" if r["has_tumor"] == "True" else "no_tumor",
                }
            )
    return rows


def vectorize_rows(rows: List[Row]) -> Tuple[np.ndarray, np.ndarray]:
    X = load_and_vectorize_batch([r["path"] for r in rows], method="horizontal")
    y = np.array([r["label"] for r in rows])
    return X, y


def split_patients(rows: List[Row]) -> Tuple[List[str], List[str]]:
    """ТА ЖЕ логика, что в experiment_sequential_smoothing.py/experiment_
    brats_split_methods.py — даёт ТЕ ЖЕ 8 test-пациентов, числа сопоставимы."""
    patient_ids = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(patient_ids)
    n_test = max(1, round(len(shuffled) * 0.2))
    test_ids = sorted(shuffled[:n_test].tolist())
    train_ids = sorted(shuffled[n_test:].tolist())
    return train_ids, test_ids


def split_search_validation(train_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Второй слой: 32 train-пациента -> 24 search_train + 8 validation
    (НОВЫЙ, независимый от test слой — test в подборе не участвует)."""
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(train_ids)
    n_val = round(len(shuffled) * 0.25)
    val_ids = sorted(shuffled[:n_val].tolist())
    search_train_ids = sorted(shuffled[n_val:].tolist())
    return search_train_ids, val_ids


def select_training_rows(rows: List[Row], patient_ids: Sequence[str], per_class: int) -> List[Row]:
    pool = [r for r in rows if r["patient_id"] in set(patient_ids)]
    rng = np.random.default_rng(RANDOM_SEED)
    selected: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in pool if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:per_class]
        selected.extend(cls_rows[i] for i in idx)
    return selected


def build_patient_grouped_rows(rows: List[Row], patient_ids: Sequence[str]) -> Tuple[List[Row], List[int]]:
    """Все срезы заданных пациентов, сгруппированные и отсортированные
    по slice_idx внутри каждой группы (для корректного per-patient сглаживания)."""
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


def load_reference_numbers() -> Dict[str, float]:
    refs: Dict[str, float] = {}
    if SMOOTHING_REFERENCE_PATH.exists():
        with open(SMOOTHING_REFERENCE_PATH, encoding="utf-8") as f:
            d = json.load(f)
        refs["baseline_unsmoothed (n_sub=8)"] = d["results"]["meaningful_unsmoothed"]["accuracy"]
        refs["baseline_smoothed (n_sub=8)"] = d["results"]["meaningful_smoothed"]["accuracy"]
    if SPLIT_METHODS_REFERENCE_PATH.exists():
        with open(SPLIT_METHODS_REFERENCE_PATH, encoding="utf-8") as f:
            d = json.load(f)
        refs["dishonest_paired_correlation_split"] = d["results"]["paired_correlation_split"]["accuracy"]
        refs["dishonest_image_random_split"] = d["results"]["image_random_split"]["accuracy"]
        refs["dishonest_train_equals_test"] = d["results"]["train_equals_test"]["accuracy"]
    return refs


def main() -> None:
    start = time.perf_counter()
    print("=== Увеличение accuracy (BraTS has_tumor) — честные и нечестные методы ===")

    rows = load_metadata()
    train_ids, test_ids = split_patients(rows)
    search_train_ids, val_ids = split_search_validation(train_ids)
    print(f"Пациенты: {len(search_train_ids)} search_train, {len(val_ids)} validation, {len(test_ids)} test (ЗАФИКСИРОВАНО, та же test-выборка, что в прошлых экспериментах).")

    val_rows, _ = build_patient_grouped_rows(rows, val_ids)
    X_val, y_val = vectorize_rows(val_rows)

    test_rows, test_group_lengths = build_patient_grouped_rows(rows, test_ids)
    X_test, y_test = vectorize_rows(test_rows)
    X_test_smoothed = smooth_per_patient_group(X_test, test_group_lengths)
    print(f"X_val.shape={X_val.shape}, X_test.shape={X_test.shape}")

    print(f"\n--- Честный подбор n_subclasses по validation, грид={list(GRID_N_SUBCLASSES)} ---")
    search_train_rows = select_training_rows(rows, search_train_ids, TRAIN_PER_CLASS)
    X_search_train, y_search_train = vectorize_rows(search_train_rows)

    grid_results = []
    for n_sub in GRID_N_SUBCLASSES:
        clf = SubspaceConjugacyClassifier(n_subclasses=n_sub)
        t0 = time.perf_counter()
        clf.fit(X_search_train, y_search_train)
        fit_seconds = time.perf_counter() - t0

        val_acc = evaluate_classifier(y_val, clf.predict(X_val))["accuracy"]
        test_acc_peek = evaluate_classifier(y_test, clf.predict(X_test))["accuracy"]
        test_acc_peek_smoothed = evaluate_classifier(y_test, clf.predict(X_test_smoothed))["accuracy"]

        print(
            f"   n_subclasses={n_sub:>2}: fit={fit_seconds:5.1f}s  val_acc={val_acc:.4f}  "
            f"(nechestno) test_peek={test_acc_peek:.4f}  test_peek_smoothed={test_acc_peek_smoothed:.4f}"
        )
        grid_results.append(
            {
                "n_subclasses": n_sub,
                "fit_seconds": fit_seconds,
                "val_accuracy": val_acc,
                "test_accuracy_peek_DISHONEST": test_acc_peek,
                "test_accuracy_peek_smoothed_DISHONEST": test_acc_peek_smoothed,
            }
        )

    best_honest = max(grid_results, key=lambda r: r["val_accuracy"])
    best_n_subclasses = best_honest["n_subclasses"]
    print(f"\nЧестно выбран n_subclasses={best_n_subclasses} (по максимуму val_accuracy={best_honest['val_accuracy']:.4f}).")

    print(f"\n--- Финал: переобучение n_subclasses={best_n_subclasses} на ВСЕХ 32 train-пациентах, ОДНА оценка на test ---")
    final_train_rows = select_training_rows(rows, train_ids, TRAIN_PER_CLASS)
    X_final_train, y_final_train = vectorize_rows(final_train_rows)

    final_clf = SubspaceConjugacyClassifier(n_subclasses=best_n_subclasses)
    t0 = time.perf_counter()
    final_clf.fit(X_final_train, y_final_train)
    final_fit_seconds = time.perf_counter() - t0

    final_honest_acc = evaluate_classifier(y_test, final_clf.predict(X_test))["accuracy"]
    final_honest_smoothed_acc = evaluate_classifier(y_test, final_clf.predict(X_test_smoothed))["accuracy"]
    print(f"fit()={final_fit_seconds:.1f}s  final_honest_accuracy={final_honest_acc:.4f}  final_honest_smoothed_accuracy={final_honest_smoothed_acc:.4f}")

    best_dishonest_peek = max(grid_results, key=lambda r: r["test_accuracy_peek_smoothed_DISHONEST"])

    refs = load_reference_numbers()

    summary = {
        "honest_baseline (n_sub=8, no tuning)": refs.get("baseline_unsmoothed (n_sub=8)"),
        "honest_baseline + smoothing": refs.get("baseline_smoothed (n_sub=8)"),
        f"honest_tuned (n_sub={best_n_subclasses}, val-selected)": final_honest_acc,
        f"honest_tuned (n_sub={best_n_subclasses}) + smoothing": final_honest_smoothed_acc,
        "dishonest_paired_correlation_split": refs.get("dishonest_paired_correlation_split"),
        "dishonest_image_random_split (patient leakage)": refs.get("dishonest_image_random_split"),
        f"dishonest_model_selection_leakage (n_sub={best_dishonest_peek['n_subclasses']}, picked by peeking at test)": best_dishonest_peek["test_accuracy_peek_DISHONEST"],
        f"dishonest_model_selection_leakage + smoothing (n_sub={best_dishonest_peek['n_subclasses']})": best_dishonest_peek["test_accuracy_peek_smoothed_DISHONEST"],
        "dishonest_train_equals_test": refs.get("dishonest_train_equals_test"),
    }

    print("\n" + "=" * 92)
    print("ИТОГОВАЯ ТАБЛИЦА: accuracy по методу (BraTS has_tumor, один и тот же тест для честных строк)")
    print("=" * 92)
    name_w = 68
    for name, acc in summary.items():
        acc_str = f"{acc:.4f}" if acc is not None else "н/д"
        tag = "ЧЕСТНО" if name.startswith("honest") else "НЕЧЕСТНО"
        print(f"{name.ljust(name_w)} [{tag:8}] {acc_str}")
    print("=" * 92)

    baseline = refs.get("baseline_unsmoothed (n_sub=8)")
    if baseline is not None:
        print(f"\nПрирост относительно честного baseline ({baseline:.4f}):")
        for name, acc in summary.items():
            if acc is None or name.startswith("honest_baseline"):
                continue
            print(f"   {name}: {'+' if acc - baseline >= 0 else ''}{acc - baseline:.4f} п.п.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "random_seed": RANDOM_SEED,
        "train_per_class": TRAIN_PER_CLASS,
        "grid_n_subclasses": list(GRID_N_SUBCLASSES),
        "search_train_patients": search_train_ids,
        "validation_patients": val_ids,
        "test_patients": test_ids,
        "grid_results": grid_results,
        "best_honest_n_subclasses": best_n_subclasses,
        "final_honest_fit_seconds": final_fit_seconds,
        "summary": summary,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    md_lines = [
        "# Увеличение accuracy (BraTS has_tumor) — честные и нечестные методы",
        "",
        "| Метод | Тип | accuracy |",
        "|---|---|---|",
    ]
    for name, acc in summary.items():
        tag = "честно" if name.startswith("honest") else "нечестно"
        acc_str = f"{acc:.4f}" if acc is not None else "н/д"
        md_lines.append(f"| {name} | {tag} | {acc_str} |")
    REPORT_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
