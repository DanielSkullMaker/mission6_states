"""⚠️ Полный «поддельный» прогон Статьи 4 под нечестную ветку (train==test,
n_subclasses=45) — по прямому запросу "переделай статью под нечестную
ветку... перепиши результаты графиков и таблиц". Считает дишонест-аналоги
ВСЕХ количественных таблиц честной статьи (experiment_sequential_smoothing.py
/ experiment_article4_ablations.py), чтобы увидеть, как сильно меняются
выводы, если протокол оценки подменить на всём протяжении работы, а не
только в одном боковом эксперименте.

Ключевое методологическое ограничение (честно зафиксировано здесь, даже
если в самой "поддельной" статье это не будет подчёркнуто): пул
train==test — ТЕ ЖЕ 100/класс случайных срезов (patient_id игнорируется),
что и в dishonest_brats_capacity_report.py, НЕ физическая
пациент-сгруппированная последовательность. "Содержательный порядок"
поэтому здесь не более чем ПОРЯДОК СЛУЧАЙНОЙ ВЫБОРКИ — честная статья
такого подлога не делает (использует реальный физический индекс среза
внутри held-out пациентов). Это уже само по себе одна из причин, почему
"поддельная" статья не может честно воспроизвести методологию оригинала,
а вынуждена её подменить.

Три дишонест-аналога таблиц оригинала:
  1. "3 конфигурации" (как Таблица 2 оригинала) — train=test=весь пул,
     сравниваются (псевдо)содержательный+сглаживание / без сглаживания /
     перемешанный+сглаживание.
  2. Абляция предобработки (как Таблица 3/5 оригинала) — 5 вариантов,
     train=test=пул, n_subclasses=45.
  3. Абляция сглаживания (как Таблица 4/6 оригинала) — один классификатор
     (train=test=пул), 5 схем сглаживания применены к (псевдо)порядку.

Артефакты:
  - artifacts/dishonest_article4_configurations_report.json
  - artifacts/dishonest_article4_preprocessing_report.json
  - artifacts/dishonest_article4_smoothing_report.json

Запуск (после experiments/prepare_brats_dataset.py):
    python experiments/dishonest_article4_full_experiment.py
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from subspace_conjugacy import (
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
    smooth_adjacent_vectors,
)
from subspace_conjugacy.evaluation.metrics import evaluate_classifier
from subspace_conjugacy.features.vectorization import vectorize_batch

from experiment_article4_ablations import PREPROCESSING_VARIANTS, get_raw_slice

DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42
POOL_PER_CLASS = 100
N_SUBCLASSES = 45

ARTIFACTS_DIR = Path("artifacts")
CONFIG_REPORT_PATH = ARTIFACTS_DIR / "dishonest_article4_configurations_report.json"
PREP_REPORT_PATH = ARTIFACTS_DIR / "dishonest_article4_preprocessing_report.json"
SMOOTH_REPORT_PATH = ARTIFACTS_DIR / "dishonest_article4_smoothing_report.json"

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


def sample_pool(rows: List[Row], per_class: int, seed: int) -> List[Row]:
    rng = np.random.default_rng(seed)
    pool: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in rows if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:per_class]
        pool.extend(cls_rows[i] for i in idx)
    return pool


def vectorize_rows(rows: List[Row]):
    X = load_and_vectorize_batch([r["path"] for r in rows], method="horizontal")
    y = np.array([r["label"] for r in rows])
    return X, y


def run_configurations(pool: List[Row]) -> Dict[str, Any]:
    print("\n=== [1/3] Дишонест-аналог «3 конфигураций» (train=test=пул) ===")
    X, y = vectorize_rows(pool)
    print(f"X.shape={X.shape} (train=test, 100% пересечение)")

    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf.fit(X, y)
    fit_seconds = time.perf_counter() - t0
    print(f"fit()={fit_seconds:.1f}s")

    X_smoothed = smooth_adjacent_vectors(X)
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(X.shape[0])
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    X_shuffled_smoothed = smooth_adjacent_vectors(X_shuffled)

    def _eval(name, X_eval, y_eval):
        report = evaluate_classifier(y_eval, clf.predict(X_eval))
        print(f"   {name}: accuracy={report['accuracy']:.4f}")
        return {
            "n_test": int(X_eval.shape[0]),
            "accuracy": report["accuracy"],
            "per_class_accuracy": {str(k): v for k, v in report["per_class_accuracy"].items()},
            "confusion_matrix": report["confusion_matrix"].tolist(),
            "labels": [str(l) for l in report["labels"]],
        }

    results = {
        "meaningful_smoothed": _eval("(1) псевдо-порядок + сглаживание", X_smoothed, y),
        "meaningful_unsmoothed": _eval("(2) псевдо-порядок без сглаживания", X, y),
        "shuffled_smoothed": _eval("(3) перемешанный + сглаживание", X_shuffled_smoothed, y_shuffled),
    }
    return {
        "random_seed": RANDOM_SEED,
        "n_subclasses": N_SUBCLASSES,
        "n_train": int(X.shape[0]),
        "train_per_class": POOL_PER_CLASS,
        "fit_seconds": fit_seconds,
        "overlap_fraction": 1.0,
        "results": results,
    }


def run_preprocessing(pool: List[Row]) -> Dict[str, Any]:
    print("\n=== [2/3] Дишонест-аналог абляции предобработки (train=test=пул) ===")
    y = np.array([r["label"] for r in pool])
    results = []
    for name, description, prep_fn in PREPROCESSING_VARIANTS:
        images = [prep_fn(get_raw_slice(r["patient_id"], r["slice_idx"])) for r in pool]
        X = vectorize_batch(images, method="horizontal")

        clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
        t0 = time.perf_counter()
        clf.fit(X, y)
        fit_seconds = time.perf_counter() - t0

        report = evaluate_classifier(y, clf.predict(X))
        print(f"   {name}: accuracy={report['accuracy']:.4f}  fit={fit_seconds:.1f}s")
        results.append({
            "name": name,
            "description": description,
            "accuracy": report["accuracy"],
            "per_class_accuracy": {str(k): v for k, v in report["per_class_accuracy"].items()},
            "fit_seconds": fit_seconds,
        })
    return {"random_seed": RANDOM_SEED, "n_subclasses": N_SUBCLASSES, "results": results}


def run_smoothing(pool: List[Row]) -> Dict[str, Any]:
    print("\n=== [3/3] Дишонест-аналог абляции сглаживания (train=test=пул) ===")
    X, y = vectorize_rows(pool)

    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf.fit(X, y)
    fit_seconds = time.perf_counter() - t0
    print(f"fit()={fit_seconds:.1f}s")

    variants = [
        ("none", "Без сглаживания", None),
        ("fixed_3", "Фиксированное окно 3", (1,)),
        ("fixed_5", "Фиксированное окно 5", (2,)),
        ("fixed_7", "Фиксированное окно 7", (3,)),
        ("variable_3_5_7", "Переменное окно 3/5/7", (1, 2, 3)),
    ]
    results = []
    for name, description, half_widths in variants:
        X_eval = X if half_widths is None else smooth_adjacent_vectors(X, half_widths=half_widths)
        report = evaluate_classifier(y, clf.predict(X_eval))
        print(f"   {name}: accuracy={report['accuracy']:.4f}")
        results.append({
            "name": name,
            "description": description,
            "accuracy": report["accuracy"],
            "per_class_accuracy": {str(k): v for k, v in report["per_class_accuracy"].items()},
        })
    return {
        "random_seed": RANDOM_SEED, "n_subclasses": N_SUBCLASSES,
        "fit_seconds": fit_seconds, "n_train": int(X.shape[0]), "n_test": int(X.shape[0]),
        "results": results,
    }


def main() -> None:
    start = time.perf_counter()
    rows = load_metadata()
    pool = sample_pool(rows, POOL_PER_CLASS, RANDOM_SEED + 1)
    print(f"Пул train=test: {len(pool)} срезов ({POOL_PER_CLASS}/класс, patient_id игнорируется).")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    config_result = run_configurations(pool)
    with open(CONFIG_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(config_result, f, ensure_ascii=False, indent=2)
    print(f"JSON-отчёт: {CONFIG_REPORT_PATH}")

    prep_result = run_preprocessing(pool)
    with open(PREP_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(prep_result, f, ensure_ascii=False, indent=2)
    print(f"JSON-отчёт: {PREP_REPORT_PATH}")

    smooth_result = run_smoothing(pool)
    with open(SMOOTH_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(smooth_result, f, ensure_ascii=False, indent=2)
    print(f"JSON-отчёт: {SMOOTH_REPORT_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start):.1f} с ===")


if __name__ == "__main__":
    main()
