"""Сравнение канона (FursovClusterer) с подходами из theory/Теоретические
основы.docx (В.А. Фурсов, IAPR) — реализованными в этой же сессии как
subspace_conjugacy.algorithms.RecursiveBisectionClusterer и
CosineSecondVectorAttacher(criterion="max").

Контекст (см. предыдущий анализ в этом чате): статья в разделе 5 буквально
описывает построение подпространств РЕКУРСИВНОЙ бисекцией (2 -> 4 -> 8 -> ...,
"можно продолжить разбиение каждого из этих подклассов по той же схеме"), а
не плоским проходом, которым канон (ReferenceCenterBuilder, A.2-A.3) находит
все n_subclasses центров сразу. Отдельно — текст статьи указывает
"максимальные значения" для присоединения второго вектора к каждому центру
(B.1-аналог), тогда как канон использует МИНИМУМ косинуса; точные формулы
(9)-(10) встроены в документ как WMF-картинки и не читаются как текст, так
что направление "max" реализует текстуальное указание буквально, но не
подтверждено побитово по самой формуле.

Сравниваются 3 конфигурации кластеризации (на класс), остальное — общий
канонический SubspaceConjugacyClassifier через fit_from_subclass_bases():

  1. "canonical"              — FursovClusterer (A.1->A.3->B.1(min)->B.2), как в main.py.
  2. "recursive_bisection_min" — RecursiveBisectionClusterer, b1_criterion="min"
                                 (та же B.1-логика, что и канон, но дерево
                                 бисекции вместо плоского прохода — изолирует
                                 эффект именно структуры алгоритма).
  3. "recursive_bisection_max" — RecursiveBisectionClusterer, b1_criterion="max"
                                 (буквальная попытка воспроизвести раздел 5
                                 статьи целиком — и структуру, и критерий B.1).

Методология оценки — честный held-out (evaluation.paper_methodology.
paper_train_test_split, 70/30 — как в main.py/README, "70_30" сценарий,
accuracy 0.6111 для baseline), НЕСКОЛЬКО seed для устойчивости оценки
(те же 3 seed, что и в других honest/dishonest экспериментах этого проекта).

Датасет: datasets/{glioma,meningioma,pituitary}_centered/ (по 200 PNG на
класс, уже отцентрированных).

Артефакты:
  - artifacts/theory_vs_canon_report.json

Запуск:
    python experiments/compare_theory_approaches.py
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from subspace_conjugacy import (
    FursovClusterer,
    RecursiveBisectionClusterer,
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
)
from subspace_conjugacy.evaluation.metrics import evaluate_classifier
from subspace_conjugacy.evaluation.paper_methodology import paper_train_test_split

DATASET_ROOT = Path("datasets")
CLASSES = ["glioma", "meningioma", "pituitary"]
ARTIFACTS_DIR = Path("artifacts")

N_SUBCLASSES = 8
FREEZE_BASIS_AT = 2
TRAIN_FRACTION = 0.7  # README: сценарий "70_30", сопоставимый baseline = 0.6111
SEEDS = [42, 43, 44, 45, 46]
# Полный датасет (200 изображений/класс) первая попытка прогона убила систему
# по памяти за >40 минут без единой строки лога (см. обсуждение в чате) —
# рост базиса в ConjugacyClusterGrowth масштабируется квадратично по M.
# Подвыборка снижает M~4x, что даёт ~16x по времени/памяти на рост базиса.
SUBSAMPLE_PER_CLASS: Any = 60

APPROACHES: Dict[str, Dict[str, Any]] = {
    "canonical": {
        "clusterer_cls": FursovClusterer,
        "kwargs": {},
    },
    "recursive_bisection_min": {
        "clusterer_cls": RecursiveBisectionClusterer,
        "kwargs": {"b1_criterion": "min"},
    },
    "recursive_bisection_max": {
        "clusterer_cls": RecursiveBisectionClusterer,
        "kwargs": {"b1_criterion": "max"},
    },
}


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Загружает и векторизует datasets/{class}_centered/*.png (все классы).

    SUBSAMPLE_PER_CLASS (если задан) берёт первые N файлов на класс —
    снижает квадратичную по M стоимость роста базиса (B.2), см. комментарий
    у константы.
    """
    X_parts = []
    y_parts = []
    for cls in CLASSES:
        class_dir = DATASET_ROOT / f"{cls}_centered"
        paths = sorted(class_dir.glob("*.png"))
        if not paths:
            raise FileNotFoundError(
                f"Не найдены изображения в {class_dir} — ожидается "
                f"datasets/{{class}}_centered/*.png (см. README, раздел «Данные»)."
            )
        if SUBSAMPLE_PER_CLASS is not None:
            paths = paths[:SUBSAMPLE_PER_CLASS]
        X_cls = load_and_vectorize_batch(paths, method="horizontal")
        X_parts.append(X_cls)
        y_parts.append(np.full(len(paths), cls))
        print(f"   {cls}: {len(paths)} изображений, вектор N={X_cls.shape[1]}", flush=True)

    return np.vstack(X_parts), np.concatenate(y_parts)


def build_bases_by_class(
    X_train: np.ndarray, y_train: np.ndarray, clusterer_cls, kwargs: Dict[str, Any]
) -> Dict[str, List[np.ndarray]]:
    """Кластеризует КАЖДЫЙ класс независимо, возвращает {class: [Y_0, ..., Y_{S-1}]}."""
    bases_by_class: Dict[str, List[np.ndarray]] = {}
    for cls in CLASSES:
        X_cls = X_train[y_train == cls]
        clusterer = clusterer_cls(
            n_subclasses=N_SUBCLASSES, freeze_basis_at=FREEZE_BASIS_AT, **kwargs
        )
        clusterer.fit(X_cls)
        bases_by_class[cls] = clusterer.subspaces_
    return bases_by_class


def run_one_seed(X: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, Dict[str, Any]]:
    """Прогоняет все APPROACHES на одном честном train/test сплите."""
    X_train, X_test, y_train, y_test = paper_train_test_split(
        X, y, train_fraction=TRAIN_FRACTION, random_state=seed
    )

    results: Dict[str, Dict[str, Any]] = {}
    for name, spec in APPROACHES.items():
        t0 = time.perf_counter()
        try:
            bases_by_class = build_bases_by_class(
                X_train, y_train, spec["clusterer_cls"], spec["kwargs"]
            )
            clf = SubspaceConjugacyClassifier().fit_from_subclass_bases(bases_by_class)
            y_pred = clf.predict(X_test)
            report = evaluate_classifier(y_test, y_pred)
            elapsed = time.perf_counter() - t0
            results[name] = {
                "accuracy": report["accuracy"],
                "per_class_accuracy": report["per_class_accuracy"],
                "elapsed_s": elapsed,
                "error": None,
            }
            print(
                f"      {name:<26} accuracy={report['accuracy']:.4f} "
                f"({elapsed:.1f}s)"
            )
        except ValueError as exc:
            # RecursiveBisectionClusterer может упасть на некоторых классах/
            # seed'ах, если дерево бисекции не может продолжиться (см. его
            # docstring) — честно фиксируем это как результат, а не прячем.
            results[name] = {
                "accuracy": None,
                "per_class_accuracy": None,
                "elapsed_s": time.perf_counter() - t0,
                "error": str(exc),
            }
            print(f"      {name:<26} ОШИБКА: {exc}")

    return results


def summarize(all_seed_results: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    """Усредняет accuracy по seed'ам для каждого approach (где не было ошибок)."""
    summary: Dict[str, Any] = {}
    for name in APPROACHES:
        accuracies = [
            r[name]["accuracy"] for r in all_seed_results if r[name]["accuracy"] is not None
        ]
        summary[name] = {
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else None,
            "std_accuracy": float(np.std(accuracies)) if accuracies else None,
            "n_successful_seeds": len(accuracies),
            "n_total_seeds": len(all_seed_results),
        }
    return summary


def print_comparison_table(summary: Dict[str, Any]) -> None:
    print("\n=== Сравнение accuracy: канон vs. подходы из theory/Теоретические основы.docx ===")
    print(f"{'Подход':<28} | {'mean accuracy':>13} | {'std':>6} | seeds ok")
    print("-" * 65)
    for name, stats in summary.items():
        mean_str = f"{stats['mean_accuracy']:.4f}" if stats["mean_accuracy"] is not None else "—"
        std_str = f"{stats['std_accuracy']:.4f}" if stats["std_accuracy"] is not None else "—"
        print(
            f"{name:<28} | {mean_str:>13} | {std_str:>6} | "
            f"{stats['n_successful_seeds']}/{stats['n_total_seeds']}"
        )


def main() -> None:
    start = time.perf_counter()
    print("=== Загрузка датасета (datasets/*_centered) ===")
    X, y = load_dataset()

    print(
        f"\n=== Прогон {len(SEEDS)} seed(ов), train_fraction={TRAIN_FRACTION}, "
        f"n_subclasses={N_SUBCLASSES}, freeze_basis_at={FREEZE_BASIS_AT} ==="
    )
    all_seed_results = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        all_seed_results.append(run_one_seed(X, y, seed))

    summary = summarize(all_seed_results)
    print_comparison_table(summary)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "theory_vs_canon_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "n_subclasses": N_SUBCLASSES,
                    "freeze_basis_at": FREEZE_BASIS_AT,
                    "train_fraction": TRAIN_FRACTION,
                    "seeds": SEEDS,
                },
                "per_seed_results": {
                    str(seed): results for seed, results in zip(SEEDS, all_seed_results)
                },
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nОтчёт сохранён: {report_path}")
    print(f"Готово за {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
