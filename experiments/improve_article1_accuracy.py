"""Ищет конфигурацию SubspaceConjugacyClassifier, улучшающую честную accuracy
Статьи 1 (experiments/reproduce_article1.py, сценарий "70_30" — baseline
accuracy = 0.6111 на held-out test) — БЕЗ утечки: подбор ведётся на
train/val, финальная проверка выполняется ОДИН раз на held-out test.

Почему нельзя просто перебрать конфигурации и взять лучшую НА TEST — это
ровно то, о чём предупреждает evaluation.paper_methodology.find_best_n_subclasses
(model selection leakage, см. README): если один и тот же test использовать
и для подбора, и как финальный отчёт, число окажется мягко переобучённым
под конкретную тестовую выборку. Поэтому здесь три непересекающихся набора:

  1. test (180 изображений) — БУКВАЛЬНО тот же сплит, что в
     reproduce_article1.py::split_70_30 (train_test_split, test_size=0.3,
     stratify=y, random_state=42) — не участвует в поиске вообще, только в
     финальной оценке, один раз в самом конце.
  2. Из оставшихся 420 train-изображений — быстрая search-подвыборка
     (40/класс train_search + 10/класс val = 150 изображений, отдельный
     random_state) — НЕ вся обучающая выборка: B.2 растёт как O(M^2*N),
     полный fit на 420 занимает ~5-6 минут (см. reproduce_article1.py), а на
     150 — секунды, что позволяет перебрать много конфигураций дёшево. Плата
     за это — ранжирование конфигураций на подвыборке является разумным
     приближением к ранжированию на полном train, но не гарантией: финальный
     шаг 4 ниже подтверждает (или опровергает) находку на полном масштабе.
  3. Полный train (420) — используется один раз для финального fit() лучшей
     найденной конфигурации перед единственной оценкой на test.

Перебираемые конфигурации (варьируется один параметр относительно дефолта
SubspaceConjugacyClassifier за раз, чтобы результат был интерпретируем):
  - baseline                         (как в reproduce_article1.py)
  - filter_low_informativeness=True  (находка №5 сверки со статьёй)
  - filter_dependent=True            (находка №3 сверки со статьёй)
  - оба фильтра вместе
  - growth_strategy="master"         (NB7-аппроксимация)
  - center_strategy="windowed"       (находка №2 из MATLAB-кода)
  - freeze_basis_at="auto"           (находка №2 сверки со статьёй — растит
                                        подпространства без обрезки до k=2 +
                                        равнивает между классами; НЕ увеличивает
                                        стоимость fit() — B.2 и так растёт до
                                        конца всегда, freeze_basis_at влияет
                                        только на то, что остаётся на выходе)
Каждая конфигурация прогоняется по n_subclasses_grid=(4,6,8,10,12,16) —
evaluation.paper_methodology.find_best_n_subclasses делает это перебором и
возвращает лучшее n_subclasses по mean_per_class_accuracy на val.

Запуск:
    python experiments/improve_article1_accuracy.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from subspace_conjugacy import SubspaceConjugacyClassifier, load_and_vectorize_batch, save_model
from subspace_conjugacy.evaluation.metrics import evaluate_classifier
from subspace_conjugacy.evaluation.paper_methodology import (
    DEFAULT_N_SUBCLASSES_GRID,
    find_best_n_subclasses,
)

logging.basicConfig(level=logging.ERROR, force=True)  # подавляем WARNING о pinv-фолбэке (см. reproduce_article1.py)

DATASET_ROOT = Path("datasets")
CLASSES = ["glioma", "meningioma", "pituitary"]
RANDOM_SEED = 42
BASELINE_ACCURACY = 0.6111111111111112  # reproduce_article1.py, сценарий "70_30"

N_SEARCH_TRAIN_PER_CLASS = 40
N_SEARCH_VAL_PER_CLASS = 10

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = ARTIFACTS_DIR / "article1_accuracy_improvement.json"

CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {},
    "filter_low_informativeness": {"filter_low_informativeness": True},
    "filter_dependent": {"filter_dependent": True},
    "both_filters": {"filter_low_informativeness": True, "filter_dependent": True},
    "growth_strategy_master": {"growth_strategy": "master"},
    "center_strategy_windowed": {"center_strategy": "windowed"},
    "freeze_basis_at_auto": {"freeze_basis_at": "auto"},
}


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for cls in CLASSES:
        paths = sorted((DATASET_ROOT / f"{cls}_centered").glob("*.png"))
        if len(paths) != 200:
            raise FileNotFoundError(f"{cls}: найдено {len(paths)}, ожидалось 200.")
        X_list.append(load_and_vectorize_batch(paths, method="horizontal"))
        y_list.append(np.full(len(paths), cls))
    return np.vstack(X_list), np.concatenate(y_list)


def make_search_subsample(
    X_train_full: np.ndarray, y_train_full: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """40/класс train_search + 10/класс val — независимая случайная подвыборка
    ИЗ train_full (не из test!), для быстрого перебора конфигураций."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    idx_search, idx_val = [], []
    for cls in CLASSES:
        cls_idx = np.where(y_train_full == cls)[0].copy()
        rng.shuffle(cls_idx)
        idx_search.extend(cls_idx[:N_SEARCH_TRAIN_PER_CLASS].tolist())
        idx_val.extend(
            cls_idx[N_SEARCH_TRAIN_PER_CLASS:N_SEARCH_TRAIN_PER_CLASS + N_SEARCH_VAL_PER_CLASS].tolist()
        )
    idx_search, idx_val = np.array(idx_search), np.array(idx_val)
    return (
        X_train_full[idx_search], y_train_full[idx_search],
        X_train_full[idx_val], y_train_full[idx_val],
    )


def run_search(
    X_search: np.ndarray, y_search: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> Dict[str, Any]:
    all_configs_results = {}
    best_overall = None

    for config_name, kwargs in CONFIGS.items():
        print(f"\n--- Конфигурация: {config_name} ({kwargs or 'defaults'}) ---")
        t0 = time.perf_counter()
        result = find_best_n_subclasses(
            X_search, y_search, X_val, y_val,
            n_subclasses_grid=DEFAULT_N_SUBCLASSES_GRID,
            classifier_kwargs=kwargs,
        )
        elapsed = time.perf_counter() - t0
        print(
            f"   лучшее n_subclasses={result['best_n_subclasses']}, "
            f"val mean_per_class_accuracy={result['best_mean_per_class_accuracy']:.4f} "
            f"({elapsed:.1f}s)"
        )
        for point in result["all_results"]:
            print(
                f"      n_subclasses={point['n_subclasses']}: "
                f"mean={point['mean_per_class_accuracy']:.4f}, "
                f"per_class={ {k: round(v, 3) for k, v in point['per_class_accuracy'].items()} }"
            )

        all_configs_results[config_name] = {
            "kwargs": kwargs,
            "best_n_subclasses": result["best_n_subclasses"],
            "val_mean_per_class_accuracy": result["best_mean_per_class_accuracy"],
            "all_results": result["all_results"],
            "search_seconds": elapsed,
        }
        if best_overall is None or (
            result["best_mean_per_class_accuracy"] > best_overall["val_mean_per_class_accuracy"]
        ):
            best_overall = {
                "config_name": config_name,
                "kwargs": kwargs,
                "n_subclasses": result["best_n_subclasses"],
                "val_mean_per_class_accuracy": result["best_mean_per_class_accuracy"],
            }

    return {"all_configs": all_configs_results, "best_overall": best_overall}


def final_confirmation(
    best_overall: Dict[str, Any],
    X_train_full: np.ndarray, y_train_full: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
) -> Dict[str, Any]:
    """Переобучает лучшую найденную конфигурацию на ПОЛНОМ train (420) и
    оценивает ОДИН раз на held-out test (180, тот же, что в reproduce_article1.py)."""
    print(
        f"\n=== Финальное подтверждение: {best_overall['config_name']} "
        f"(n_subclasses={best_overall['n_subclasses']}) на полном train (420), "
        f"held-out test (180) ==="
    )
    clf = SubspaceConjugacyClassifier(
        n_subclasses=best_overall["n_subclasses"], **best_overall["kwargs"]
    )
    t0 = time.perf_counter()
    clf.fit(X_train_full, y_train_full)
    fit_seconds = time.perf_counter() - t0

    y_pred = clf.predict(X_test)
    report = evaluate_classifier(y_test, y_pred)
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}

    print(f"fit() заняло {fit_seconds:.1f}s")
    print(f"accuracy={report['accuracy']:.4f} (baseline 70/30 из reproduce_article1.py: {BASELINE_ACCURACY:.4f})")
    print(f"per_class_accuracy={per_class}")

    if report["accuracy"] > BASELINE_ACCURACY:
        print(f"УЛУЧШЕНИЕ: +{(report['accuracy'] - BASELINE_ACCURACY) * 100:.2f} п.п.")
    else:
        print(f"НЕ улучшилось: {(report['accuracy'] - BASELINE_ACCURACY) * 100:.2f} п.п. относительно baseline")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "article1_model_improved.pkl"
    save_model(clf, model_path)
    print(f"Модель сохранена: {model_path}")

    return {
        "config_name": best_overall["config_name"],
        "kwargs": best_overall["kwargs"],
        "n_subclasses": best_overall["n_subclasses"],
        "fit_seconds": fit_seconds,
        "accuracy": report["accuracy"],
        "per_class_accuracy": per_class,
        "confusion_matrix": report["confusion_matrix"].tolist(),
        "baseline_accuracy": BASELINE_ACCURACY,
        "improvement": report["accuracy"] - BASELINE_ACCURACY,
    }


def main() -> None:
    start = time.perf_counter()
    print("=== Поиск конфигурации, улучшающей accuracy Статьи 1 ===")

    print("\nЗагрузка datasets/*_centered...")
    X, y = load_data()

    # Тот же test, что и в reproduce_article1.py::split_70_30 — НЕ трогаем
    # до самого конца.
    idx = np.arange(X.shape[0])
    idx_train_full, idx_test = train_test_split(
        idx, test_size=0.3, stratify=y, random_state=RANDOM_SEED
    )
    X_train_full, y_train_full = X[idx_train_full], y[idx_train_full]
    X_test, y_test = X[idx_test], y[idx_test]
    print(f"train_full={X_train_full.shape[0]}, test={X_test.shape[0]} (held-out, не используется в поиске)")

    X_search, y_search, X_val, y_val = make_search_subsample(X_train_full, y_train_full)
    print(f"search-подвыборка: train_search={X_search.shape[0]}, val={X_val.shape[0]}")

    search_results = run_search(X_search, y_search, X_val, y_val)
    best = search_results["best_overall"]
    print(
        f"\n=== Лучшая конфигурация по val: {best['config_name']} "
        f"(n_subclasses={best['n_subclasses']}, "
        f"val mean_per_class_accuracy={best['val_mean_per_class_accuracy']:.4f}) ==="
    )

    final = final_confirmation(best, X_train_full, y_train_full, X_test, y_test)

    report = {
        "baseline_accuracy_70_30": BASELINE_ACCURACY,
        "search_train_per_class": N_SEARCH_TRAIN_PER_CLASS,
        "search_val_per_class": N_SEARCH_VAL_PER_CLASS,
        "n_subclasses_grid": list(DEFAULT_N_SUBCLASSES_GRID),
        "configs_tried": list(CONFIGS.keys()),
        "search": {
            name: {k: v for k, v in r.items()} for name, r in search_results["all_configs"].items()
        },
        "final_confirmation": final,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_PATH}")

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
