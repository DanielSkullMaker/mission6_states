"""Проверяет идеи фильтрации ДАТАСЕТА (не гиперпараметров — см.
improve_brats_accuracy_v2.py), предложенные по запросу "Есть ли идеи на
счёт фильтрации датасета для увеличения точности?":

  1. filter_low_informativeness=True — другой фильтр библиотеки (находка
     №5 README, LowInformativenessFilter), ещё не пробовался на BraTS
     (ранее пробовался только filter_dependent).
  2. (MIN_BRAIN_FRACTION в конвертере) — отдельный скрипт/датасет, не здесь,
     см. datasets/brats_flair_slices_v2/ и improve_brats_accuracy_v4.py.
  3. Margin-фильтр ОБУЧАЮЩЕЙ выборки — has_tumor размечен как
     tumor_pixel_fraction > 0.0, то есть один пиксель опухоли на срез
     256x256 даёт метку "tumor". Эта граница класса заведомо шумная
     (p10 распределения tumor_pixel_fraction среди tumor-срезов = 0.0015,
     19% tumor-срезов имеют долю < 0.5%). Здесь исключаются из ОБУЧАЮЩЕГО
     пула (не из test!) tumor-срезы с 0 < tumor_pixel_fraction < MARGIN —
     гипотеза: модель учится на менее шумных примерах класса "tumor",
     test остаётся ОРИГИНАЛЬНЫЙ (честная оценка на исходной задаче).
  4. ⚠️ ОТДЕЛЬНО, НЕ как "улучшение точности" — демонстрация эффекта
     ПОДМЕНЫ ЗАДАЧИ: тот же margin-фильтр, применённый к TEST. Это меняет
     то, что вообще измеряется ("есть клинически заметная опухоль" вместо
     "есть хоть один пиксель опухоли") — честный результат на НОВОЙ,
     более лёгкой задаче, а не более точный результат на СТАРОЙ. Приведено
     для количественной иллюстрации разницы, с явной пометкой.

Методология — та же честная схема (n_subclasses=12 зафиксирован,
search_train/validation/test-пациенты те же 24/8/8, что во всей этой серии
экспериментов). Идеи 1 и 3 проверяются на validation; только если бьют
baseline (0.6651) — одна финальная проверка на test. Идея 4 — отдельная
секция, использует ПОВТОРНО обученный baseline (n_sub=12, без доп. флагов,
100/класс, все 32 train-пациента) для чистого "та же модель, другая
измеряемая популяция" сравнения.

Артефакты: artifacts/improve_brats_accuracy_v3_{report.json,results.md}.
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
N_SUBCLASSES = 12
TRAIN_PER_CLASS = 100
TUMOR_FRACTION_MARGIN = 0.005  # идея 3/4: "барахло" на границе класса tumor

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_v3_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_v3_results.md"
PREVIOUS_REPORT_PATH = ARTIFACTS_DIR / "improve_brats_accuracy_report.json"

Row = Dict[str, Any]


def load_metadata() -> List[Row]:
    rows: List[Row] = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "patient_id": r["patient_id"],
                    "slice_idx": int(r["slice_idx"]),
                    "path": DATASET_ROOT / r["path"],
                    "label": "tumor" if r["has_tumor"] == "True" else "no_tumor",
                    "tumor_fraction": float(r["tumor_pixel_fraction"]),
                }
            )
    return rows


def vectorize_rows(rows: List[Row]) -> Tuple[np.ndarray, np.ndarray]:
    X = load_and_vectorize_batch([r["path"] for r in rows], method="horizontal")
    y = np.array([r["label"] for r in rows])
    return X, y


def split_patients(rows: List[Row]) -> Tuple[List[str], List[str]]:
    patient_ids = sorted({r["patient_id"] for r in rows})
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


def select_training_rows_margin_filtered(
    rows: List[Row], patient_ids: Sequence[str], per_class: int,
    margin: float = TUMOR_FRACTION_MARGIN, seed: int = RANDOM_SEED,
) -> List[Row]:
    """Как select_training_rows, но из пула tumor исключены срезы с
    0 < tumor_fraction < margin (идея 3) — ТОЛЬКО для отбора train."""
    pool = [
        r for r in rows
        if r["patient_id"] in set(patient_ids)
        and not (r["label"] == "tumor" and 0.0 < r["tumor_fraction"] < margin)
    ]
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


def fit_and_eval(name: str, train_rows: List[Row], X_val: np.ndarray, y_val: np.ndarray, clf_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    X_train, y_train = vectorize_rows(train_rows)
    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, **clf_kwargs)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    val_acc = evaluate_classifier(y_val, clf.predict(X_val))["accuracy"]
    print(f"   {name}: n_train={X_train.shape[0]}  fit={fit_seconds:.1f}s  val_acc={val_acc:.4f}")
    return {"name": name, "val_accuracy": val_acc, "fit_seconds": fit_seconds, "n_train": int(X_train.shape[0])}


def final_confirm(name: str, train_rows: List[Row], X_test, y_test, X_test_smoothed, clf_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    X_train, y_train = vectorize_rows(train_rows)
    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES, **clf_kwargs)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    acc = evaluate_classifier(y_test, clf.predict(X_test))["accuracy"]
    acc_smoothed = evaluate_classifier(y_test, clf.predict(X_test_smoothed))["accuracy"]
    print(f"   [FINAL/test] {name}: fit={fit_seconds:.1f}s  accuracy={acc:.4f}  accuracy_smoothed={acc_smoothed:.4f}")
    return {"name": name, "accuracy": acc, "accuracy_smoothed": acc_smoothed, "fit_seconds": fit_seconds, "clf": clf}


def main() -> None:
    start = time.perf_counter()
    print("=== Идеи фильтрации датасета (BraTS has_tumor) ===")

    rows = load_metadata()
    train_ids, test_ids = split_patients(rows)
    search_train_ids, val_ids = split_search_validation(train_ids)
    print(f"Пациенты: {len(search_train_ids)} search_train, {len(val_ids)} validation, {len(test_ids)} test.")

    val_rows, _ = build_patient_grouped_rows(rows, val_ids)
    X_val, y_val = vectorize_rows(val_rows)
    test_rows, test_group_lengths = build_patient_grouped_rows(rows, test_ids)
    X_test, y_test = vectorize_rows(test_rows)
    X_test_smoothed = smooth_per_patient_group(X_test, test_group_lengths)

    baseline_val_acc = 0.6651
    if PREVIOUS_REPORT_PATH.exists():
        with open(PREVIOUS_REPORT_PATH, encoding="utf-8") as f:
            prev = json.load(f)
        baseline_val_acc = next(r["val_accuracy"] for r in prev["grid_results"] if r["n_subclasses"] == N_SUBCLASSES)
    print(f"Baseline val_accuracy (n_sub={N_SUBCLASSES}, случайный отбор): {baseline_val_acc:.4f}")

    print("\n--- Идея 1: filter_low_informativeness=True ---")
    random_train = select_training_rows(rows, search_train_ids, TRAIN_PER_CLASS)
    idea1 = fit_and_eval("1_filter_low_informativeness", random_train, X_val, y_val, {"filter_low_informativeness": True})

    print("\n--- Идея 3: margin-фильтр обучающей выборки (исключить 0<tumor_fraction<margin из TRAIN) ---")
    margin_train = select_training_rows_margin_filtered(rows, search_train_ids, TRAIN_PER_CLASS)
    idea3 = fit_and_eval("3_margin_filtered_train", margin_train, X_val, y_val, {})

    results: Dict[str, Any] = {"baseline_val_accuracy": baseline_val_acc, "idea1": idea1, "idea3": idea3}

    print("\n--- Финальная проверка на test (только для идей, превысивших baseline на validation) ---")
    if idea1["val_accuracy"] > baseline_val_acc:
        final_train = select_training_rows(rows, train_ids, TRAIN_PER_CLASS)
        r = final_confirm("idea1_filter_low_informativeness", final_train, X_test, y_test, X_test_smoothed, {"filter_low_informativeness": True})
        results["idea1_final"] = {k: v for k, v in r.items() if k != "clf"}
    else:
        print("   идея 1 не превысила baseline на validation — финальная проверка на test не выполняется.")
        results["idea1_final"] = None

    if idea3["val_accuracy"] > baseline_val_acc:
        final_train = select_training_rows_margin_filtered(rows, train_ids, TRAIN_PER_CLASS)
        r = final_confirm("idea3_margin_filtered_train", final_train, X_test, y_test, X_test_smoothed, {})
        results["idea3_final"] = {k: v for k, v in r.items() if k != "clf"}
    else:
        print("   идея 3 не превысила baseline на validation — финальная проверка на test не выполняется.")
        results["idea3_final"] = None

    print("\n--- Идея 4 (ВНИМАНИЕ: ПОДМЕНА ЗАДАЧИ, не улучшение) ---")
    print("   Переобучение baseline (n_sub=12, без флагов, 100/класс, все 32 train-пациента)...")
    baseline_final_train = select_training_rows(rows, train_ids, TRAIN_PER_CLASS)
    baseline_final = final_confirm("baseline_refit", baseline_final_train, X_test, y_test, X_test_smoothed, {})

    margin_mask_test = ~((np.array([r["label"] for r in test_rows]) == "tumor") & (np.array([r["tumor_fraction"] for r in test_rows]) > 0.0) & (np.array([r["tumor_fraction"] for r in test_rows]) < TUMOR_FRACTION_MARGIN))
    X_test_margin = X_test[margin_mask_test]
    y_test_margin = y_test[margin_mask_test]
    n_excluded_test = int((~margin_mask_test).sum())

    same_model = baseline_final["clf"]
    acc_margin_task = evaluate_classifier(y_test_margin, same_model.predict(X_test_margin))["accuracy"]
    print(f"   Та же модель, test с исключёнными {n_excluded_test} 'слабыми' tumor-срезами (ДРУГАЯ задача): accuracy={acc_margin_task:.4f}")
    results["idea4_task_redefinition"] = {
        "n_test_excluded": n_excluded_test,
        "n_test_remaining": int(margin_mask_test.sum()),
        "accuracy_on_redefined_task": acc_margin_task,
        "accuracy_on_original_task": baseline_final["accuracy"],
    }

    print("\n" + "=" * 92)
    print("ИТОГ: идеи фильтрации датасета (BraTS has_tumor)")
    print("=" * 92)
    print(f"{'baseline (повторно, честный ориентир)':60}{baseline_final['accuracy']:.4f}")
    print(f"{'baseline + сглаживание':60}{baseline_final['accuracy_smoothed']:.4f}")
    if results["idea1_final"]:
        print(f"{'идея 1: filter_low_informativeness':60}{results['idea1_final']['accuracy']:.4f}")
    else:
        print(f"{'идея 1: filter_low_informativeness':60}не прошла validation (val_acc={idea1['val_accuracy']:.4f} <= {baseline_val_acc:.4f})")
    if results["idea3_final"]:
        print(f"{'идея 3: margin-фильтр TRAIN':60}{results['idea3_final']['accuracy']:.4f}")
    else:
        print(f"{'идея 3: margin-фильтр TRAIN':60}не прошла validation (val_acc={idea3['val_accuracy']:.4f} <= {baseline_val_acc:.4f})")
    print(f"{'идея 4: ТА ЖЕ модель, ПОДМЕНЕННЫЙ test (не honest-improvement!)':60}{acc_margin_task:.4f}  (исключено {n_excluded_test} срезов из {len(y_test)})")
    print("=" * 92)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    md_lines = [
        "# Идеи фильтрации датасета (BraTS has_tumor)",
        "",
        f"baseline (повторно): {baseline_final['accuracy']:.4f} ({baseline_final['accuracy_smoothed']:.4f} со сглаживанием).",
        "",
        f"Идея 1 (filter_low_informativeness): validation={idea1['val_accuracy']:.4f} vs baseline {baseline_val_acc:.4f} — "
        + ("ПРОШЛА" if idea1["val_accuracy"] > baseline_val_acc else "не прошла") + ".",
        f"Идея 3 (margin-фильтр TRAIN): validation={idea3['val_accuracy']:.4f} vs baseline {baseline_val_acc:.4f} — "
        + ("ПРОШЛА" if idea3["val_accuracy"] > baseline_val_acc else "не прошла") + ".",
        "",
        f"⚠️ Идея 4 — ПОДМЕНА ЗАДАЧИ, не улучшение модели: та же обученная модель, {n_excluded_test} "
        f"'слабых' tumor-срезов исключены из TEST -> accuracy {acc_margin_task:.4f} на новой, более лёгкой задаче "
        f"(исходная задача: {baseline_final['accuracy']:.4f}).",
    ]
    REPORT_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown-отчёт: {REPORT_MD_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
