"""Статья 4 — проверяет, выдерживает ли главный результат (H1: сглаживание
+0.78 п.п. на содержательном порядке) статистическую критику.

Мотивация (по итогам вопроса "отвечает ли результат на вопрос статьи?"):
матрицы корреляций (experiment_article4_correlation_matrices.py) показали,
что соседние по индексу срезы ОДНОГО пациента почти идентичны (cos~0.96 на
лаге 1) — то есть 1033 тестовых среза НЕ являются 1033 независимыми
наблюдениями, их эффективное число ближе к числу пациентов (8). Проверка
значимости на уровне отдельных срезов (например, McNemar) опиралась бы на
неверное допущение независимости и завысила бы уверенность в выводе.
Здесь — два настоящих теста:

  1. McNemar (срез как единица, OПТИМИСТИЧНЫЙ, для сравнения) — что
     получилось бы, если (неверно) считать срезы независимыми.
  2. Разбивка по ПАЦИЕНТУ (честная единица анализа) + bootstrap по 8
     пациентам (resample with replacement, B=10000) — распределение
     разницы accuracy(смягчённый) − accuracy(несглаженный) по пациентам,
     95% CI, доля бутстрэп-выборок, где сглаживание проигрывает.

Переобучает ТОТ ЖЕ классификатор (тот же train, тот же seed), что и
experiment_sequential_smoothing.py — числа агрегированной accuracy должны
совпасть с уже сохранёнными в artifacts/experiment_sequential_smoothing_report.json
(сверяется в выводе).

Артефакты: artifacts/article4_significance_report.json

Запуск:
    python experiments/experiment_article4_significance_check.py
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import binomtest

from subspace_conjugacy import (
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
    smooth_adjacent_vectors,
)

DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42
N_SUBCLASSES = 8
TRAIN_PER_CLASS = 100

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = ARTIFACTS_DIR / "article4_significance_report.json"
N_BOOTSTRAP = 10_000

Row = Dict[str, Any]


def load_metadata() -> List[Row]:
    rows: List[Row] = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "patient_id": r["patient_id"],
                "slice_idx": int(r["slice_idx"]),
                "path": DATASET_ROOT / r["path"],
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


def select_training_rows(rows: List[Row], train_ids: List[str]) -> List[Row]:
    pool = [r for r in rows if r["patient_id"] in set(train_ids)]
    rng = np.random.default_rng(RANDOM_SEED)
    selected: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in pool if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:TRAIN_PER_CLASS]
        selected.extend(cls_rows[i] for i in idx)
    return selected


def vectorize_rows(rows: List[Row]) -> Tuple[np.ndarray, np.ndarray]:
    X = load_and_vectorize_batch([r["path"] for r in rows], method="horizontal")
    y = np.array([r["label"] for r in rows])
    return X, y


def main() -> None:
    rows = load_metadata()
    train_ids, test_ids = split_patients(rows)
    train_rows = select_training_rows(rows, train_ids)
    X_train, y_train = vectorize_rows(train_rows)

    print(f"Обучение классификатора (S={N_SUBCLASSES}, n_train={X_train.shape[0]})...")
    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    clf.fit(X_train, y_train)

    test_pool = [r for r in rows if r["patient_id"] in set(test_ids)]
    group_lengths: List[int] = []
    ordered_rows: List[Row] = []
    for pid in sorted(set(test_ids)):
        group = sorted((r for r in test_pool if r["patient_id"] == pid), key=lambda r: r["slice_idx"])
        ordered_rows.extend(group)
        group_lengths.append(len(group))

    X_test, y_test = vectorize_rows(ordered_rows)
    patient_ids_test = np.array([r["patient_id"] for r in ordered_rows])

    X_smoothed_blocks = []
    offset = 0
    for length in group_lengths:
        X_smoothed_blocks.append(smooth_adjacent_vectors(X_test[offset:offset + length]))
        offset += length
    X_smoothed = np.vstack(X_smoothed_blocks)

    pred_unsmoothed = clf.predict(X_test)
    pred_smoothed = clf.predict(X_smoothed)

    acc_unsmoothed = float(np.mean(pred_unsmoothed == y_test))
    acc_smoothed = float(np.mean(pred_smoothed == y_test))
    print(f"accuracy (без сглаживания) = {acc_unsmoothed:.4f}  "
          f"(ранее сохранено: 0.5924 — должно совпасть)")
    print(f"accuracy (сглаживание)     = {acc_smoothed:.4f}  "
          f"(ранее сохранено: 0.6002 — должно совпасть)")

    # --- 1. McNemar (срез как единица, ОПТИМИСТИЧНО — для сравнения) ---
    correct_un = (pred_unsmoothed == y_test)
    correct_sm = (pred_smoothed == y_test)
    n_only_un_correct = int(np.sum(correct_un & ~correct_sm))   # сглаживание ИСПОРТИЛО
    n_only_sm_correct = int(np.sum(correct_sm & ~correct_un))   # сглаживание ПОМОГЛО
    # Точный биномиальный вариант McNemar: при H0 "сглаживание не влияет"
    # число дискордантных пар, где выиграло сглаживание, ~ Binomial(b+c, 0.5).
    n_discordant = n_only_sm_correct + n_only_un_correct
    mcnemar_pvalue = (
        binomtest(n_only_sm_correct, n_discordant, 0.5).pvalue
        if n_discordant > 0 else 1.0
    )
    print(f"\nMcNemar (срез как единица, точный биномиальный): "
          f"помогло={n_only_sm_correct}, испортило={n_only_un_correct}, "
          f"p-value={mcnemar_pvalue:.4f}")

    # --- 2. Разбивка и bootstrap по ПАЦИЕНТУ (честная единица) ---
    unique_patients = sorted(set(patient_ids_test.tolist()))
    per_patient = {}
    for pid in unique_patients:
        mask = patient_ids_test == pid
        per_patient[pid] = {
            "n_slices": int(mask.sum()),
            "accuracy_unsmoothed": float(np.mean(correct_un[mask])),
            "accuracy_smoothed": float(np.mean(correct_sm[mask])),
        }
        diff = per_patient[pid]["accuracy_smoothed"] - per_patient[pid]["accuracy_unsmoothed"]
        print(f"  {pid}: n={per_patient[pid]['n_slices']:4d}  "
              f"unsmoothed={per_patient[pid]['accuracy_unsmoothed']:.3f}  "
              f"smoothed={per_patient[pid]['accuracy_smoothed']:.3f}  "
              f"diff={diff:+.3f}")

    n_patients_helped = sum(1 for p in per_patient.values()
                             if p["accuracy_smoothed"] > p["accuracy_unsmoothed"])
    n_patients_hurt = sum(1 for p in per_patient.values()
                           if p["accuracy_smoothed"] < p["accuracy_unsmoothed"])
    n_patients_same = len(per_patient) - n_patients_helped - n_patients_hurt
    print(f"\nПо пациентам: сглаживание помогло в {n_patients_helped}/{len(per_patient)}, "
          f"ухудшило в {n_patients_hurt}/{len(per_patient)}, "
          f"без изменений в {n_patients_same}/{len(per_patient)}.")

    rng = np.random.default_rng(RANDOM_SEED)
    patients_arr = np.array(unique_patients)
    diffs = np.array([
        per_patient[p]["accuracy_smoothed"] - per_patient[p]["accuracy_unsmoothed"]
        for p in unique_patients
    ])
    bootstrap_means = []
    for _ in range(N_BOOTSTRAP):
        sample_idx = rng.integers(0, len(patients_arr), size=len(patients_arr))
        bootstrap_means.append(diffs[sample_idx].mean())
    bootstrap_means = np.array(bootstrap_means)

    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    fraction_negative_or_zero = float(np.mean(bootstrap_means <= 0))
    print(f"\nBootstrap по 8 пациентам (B={N_BOOTSTRAP}): "
          f"среднее diff={diffs.mean():+.4f}, 95% CI=[{ci_low:+.4f}, {ci_high:+.4f}], "
          f"доля бутстрэп-выборок с diff<=0 = {fraction_negative_or_zero:.1%}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "random_seed": RANDOM_SEED,
        "accuracy_unsmoothed_recomputed": acc_unsmoothed,
        "accuracy_smoothed_recomputed": acc_smoothed,
        "mcnemar_slice_level": {
            "n_smoothing_helped": n_only_sm_correct,
            "n_smoothing_hurt": n_only_un_correct,
            "p_value": float(mcnemar_pvalue),
            "caveat": "срезы одного пациента сильно автокоррелированы (см. "
                      "article4_correlation_report.json) — это p-value "
                      "ОПТИМИСТИЧНО (неверно считает 1033 среза независимыми).",
        },
        "per_patient": per_patient,
        "n_patients_helped": n_patients_helped,
        "n_patients_hurt": n_patients_hurt,
        "n_patients_same": n_patients_same,
        "bootstrap_patient_level": {
            "n_bootstrap": N_BOOTSTRAP,
            "mean_diff": float(diffs.mean()),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "fraction_diff_leq_zero": fraction_negative_or_zero,
        },
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()
