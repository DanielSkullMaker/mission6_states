"""Эксперимент: влияние находок из theory/code/*.m на точность классификации.

Сравнивает базовый SubspaceConjugacyClassifier (канон, без опциональных
находок) с версиями, использующими три опциональные фичи, реализованные из
MATLAB-кода (см. README, раздел "Опциональные находки из MATLAB-кода"):

  1. CorrelationCoreSelector — сокращает обучающий набор КАЖДОГО класса до
     "ядра" наиболее взаимно скоррелированных (типичных) векторов перед
     кластеризацией (DATA_100.m).
  2. SubspaceConjugacyClassifier(center_strategy="windowed") — "оконная"
     стратегия построения центров подклассов (DATA_100.m/Otbor_100.m).
  3. smooth_adjacent_vectors — сглаживание соседних ПО ИНДЕКСУ тестовых
     векторов переменным окном 3/5/7 перед predict() (Recogn_VSPU_S1.m).

Данные: datasets/{glioma,meningioma,pituitary}_centered/ (по 200 PNG
256x256 на класс, уже отцентрированных — см. README, раздел "Данные").

Масштаб эксперимента (важно): полный рост B.2 (ConjugacyClusterGrowth) —
O(M^2 * N) на чистом numpy, N=65536 (256x256), и растёт ВСЕГДА до конца
независимо от freeze_basis_at (см. docstring SubspaceConjugacyClassifier).
Эмпирически один FursovClusterer.fit() на M=140 (полный 70/30 split) занял
~95с — 4 тренировочные конфигурации x 3 класса на полном сплите заняли бы
> 15 минут. Чтобы эксперимент оставался практичным, обучающая выборка
намеренно уменьшена (см. TRAIN_PER_CLASS/TEST_PER_CLASS ниже) — ценой
немного меньшей статистической мощности, чем полный 70/30 split
(main.py, README, раздел "Известные ограничения").

Находка №3 предполагает содержательный порядок тестовых векторов (см.
docstring smooth_adjacent_vectors). Порядок файлов в datasets/*_centered/
НЕ задокументирован как последовательность срезов — поэтому тестовые
индексы КАЖДОГО класса здесь отсортированы по имени файла (устойчивый,
воспроизводимый порядок), но это не гарантирует содержательную
последовательность. Эксперимент честно измеряет эффект сглаживания в
таких условиях, а не подгоняет его под заведомо благоприятный сценарий.

Запуск:
    python experiments/experiment_matlab_findings.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from subspace_conjugacy import (
    CorrelationCoreSelector,
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
    smooth_adjacent_vectors,
)
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

logging.basicConfig(level=logging.WARNING)  # алгоритмы логируют на INFO/DEBUG — не засоряем вывод

DATASET_ROOT = Path("datasets")
CLASSES = ["glioma", "meningioma", "pituitary"]

TRAIN_PER_CLASS = 90
TEST_PER_CLASS = 40
CORE_FRACTION = 0.5  # находка №1: доля обучающих векторов класса, отбираемых в "ядро" (как в DATA_100.m: 100 из 200 = 0.5)
N_SUBCLASSES = 8
RANDOM_SEED = 42

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = ARTIFACTS_DIR / "experiment_matlab_findings_report.json"


def load_split() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Векторизует датасет и делит его на train/test по каждому классу.

    Обучающая выборка — случайное подмножество (порядок не важен: ни
    кластеризация, ни центры не зависят от порядка строк). Тестовая
    выборка — тоже случайное подмножество, но отсортированное по имени
    файла перед векторизацией, чтобы дать находке №3 (сглаживание) шанс
    опираться на устойчивый, воспроизводимый порядок соседей.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    X_train_by_class: Dict[str, np.ndarray] = {}
    X_test_by_class: Dict[str, np.ndarray] = {}

    for cls in CLASSES:
        paths = sorted((DATASET_ROOT / f"{cls}_centered").glob("*.png"))
        required = TRAIN_PER_CLASS + TEST_PER_CLASS
        if len(paths) < required:
            raise FileNotFoundError(
                f"{cls}: найдено {len(paths)} изображений в "
                f"{DATASET_ROOT / f'{cls}_centered'}, требуется {required}. "
                "Датасет не входит в репозиторий — см. README, раздел «Данные»."
            )

        order = rng.permutation(len(paths))
        train_idx = order[:TRAIN_PER_CLASS]
        test_idx = np.sort(order[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS])

        train_paths = [paths[i] for i in train_idx]
        test_paths = [paths[i] for i in test_idx]

        X_train_by_class[cls] = load_and_vectorize_batch(train_paths, method="horizontal")
        X_test_by_class[cls] = load_and_vectorize_batch(test_paths, method="horizontal")

    return X_train_by_class, X_test_by_class


def apply_core_selection(X_train_by_class: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Находка №1: сокращает обучающий набор каждого класса до "ядра"."""
    reduced = {}
    for cls, X in X_train_by_class.items():
        n_target = round(CORE_FRACTION * X.shape[0])
        selector = CorrelationCoreSelector(n_target=n_target)
        reduced[cls] = selector.fit_transform(X)
    return reduced


def smooth_per_class(X_by_class: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Применяет сглаживание ОТДЕЛЬНО к каждому классу — иначе окно у
    границы блоков в общей матрице примешало бы соседний класс (как и в
    оригинале, Recogn_VSPU_S1.m сглаживает внутри цикла по классам)."""
    return {cls: smooth_adjacent_vectors(X) for cls, X in X_by_class.items()}


def stack(X_by_class: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Собирает {class: X} в единые (X, y) в порядке CLASSES."""
    X = np.vstack([X_by_class[cls] for cls in CLASSES])
    y = np.concatenate([np.full(X_by_class[cls].shape[0], cls) for cls in CLASSES])
    return X, y


def run_config(
    name: str,
    X_train_by_class: Dict[str, np.ndarray],
    X_test_by_class: Dict[str, np.ndarray],
    X_test_smoothed_by_class: Dict[str, np.ndarray],
    y_test: np.ndarray,
    use_core: bool,
    use_windowed: bool,
) -> Dict:
    """Обучает одну конфигурацию и оценивает её с/без сглаживания теста."""
    X_train_cls = apply_core_selection(X_train_by_class) if use_core else X_train_by_class
    X_train, y_train = stack(X_train_cls)
    X_test_raw, _ = stack(X_test_by_class)
    X_test_smoothed, _ = stack(X_test_smoothed_by_class)

    clf = SubspaceConjugacyClassifier(
        n_subclasses=N_SUBCLASSES,
        center_strategy="windowed" if use_windowed else "default",
    )

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    eval_results = {}
    for label, X_eval in [("raw", X_test_raw), ("smoothed", X_test_smoothed)]:
        y_pred = clf.predict(X_eval)
        report = evaluate_classifier(y_test, y_pred)
        eval_results[label] = {
            "accuracy": report["accuracy"],
            "per_class_accuracy": {str(k): v for k, v in report["per_class_accuracy"].items()},
        }

    return {
        "config": name,
        "use_core_selector": use_core,
        "use_windowed_centers": use_windowed,
        "n_train_total": int(X_train.shape[0]),
        "fit_time_s": round(fit_time, 1),
        "eval": eval_results,
    }


def print_result(res: Dict) -> None:
    print(f"\n=== {res['config']} (n_train={res['n_train_total']}, fit={res['fit_time_s']}s) ===")
    for label, r in res["eval"].items():
        per_class = ", ".join(f"{k}={v:.3f}" for k, v in r["per_class_accuracy"].items())
        print(f"  test={label:<9} accuracy={r['accuracy']:.3f}  ({per_class})")


def main() -> None:
    overall_start = time.perf_counter()
    print("=== Эксперимент: находки из theory/code/*.m vs канон ===")
    print(f"Классы: {CLASSES}")
    print(f"train/class={TRAIN_PER_CLASS}, test/class={TEST_PER_CLASS}, n_subclasses={N_SUBCLASSES}")
    print(f"CorrelationCoreSelector: n_target = {CORE_FRACTION} * train/class = "
          f"{round(CORE_FRACTION * TRAIN_PER_CLASS)}")

    print("\nЗагрузка и векторизация датасета...")
    t0 = time.perf_counter()
    X_train_by_class, X_test_by_class = load_split()
    X_test_smoothed_by_class = smooth_per_class(X_test_by_class)
    _, y_test = stack(X_test_by_class)
    print(f"готово за {time.perf_counter() - t0:.1f}s "
          f"(train={sum(x.shape[0] for x in X_train_by_class.values())}, "
          f"test={sum(x.shape[0] for x in X_test_by_class.values())})")

    configs = [
        ("baseline (канон)", False, False),
        ("+ находка №1 (CorrelationCoreSelector)", True, False),
        ("+ находка №2 (center_strategy='windowed')", False, True),
        ("+ находки №1 и №2", True, True),
    ]

    all_results: List[Dict] = []
    for name, use_core, use_windowed in configs:
        res = run_config(
            name, X_train_by_class, X_test_by_class, X_test_smoothed_by_class,
            y_test, use_core, use_windowed,
        )
        all_results.append(res)
        print_result(res)

    # Находка №3 (сглаживание) оценена внутри КАЖДОЙ конфигурации (test=smoothed)
    # — так видно её эффект независимо от находок №1/№2.

    baseline_acc = all_results[0]["eval"]["raw"]["accuracy"]
    best = max(
        (
            (cfg["config"], label, r["accuracy"])
            for cfg in all_results
            for label, r in cfg["eval"].items()
        ),
        key=lambda item: item[2],
    )

    print("\n=== Итог ===")
    print(f"baseline (raw test, без находок):     accuracy = {baseline_acc:.3f}")
    print(f"лучшая конфигурация:                  {best[0]} / test={best[1]} "
          f"-> accuracy = {best[2]:.3f}  (delta = {best[2] - baseline_acc:+.3f})")

    elapsed = time.perf_counter() - overall_start
    print(f"\nВсего: {elapsed / 60:.1f} мин")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_per_class": TRAIN_PER_CLASS,
                "test_per_class": TEST_PER_CLASS,
                "core_fraction": CORE_FRACTION,
                "n_subclasses": N_SUBCLASSES,
                "random_seed": RANDOM_SEED,
                "baseline_accuracy": baseline_acc,
                "best_config": {"name": best[0], "test": best[1], "accuracy": best[2]},
                "results": all_results,
                "elapsed_s": round(elapsed, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
