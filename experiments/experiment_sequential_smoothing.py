"""Эксперимент Статьи 4 диссертационного плана (dissertation_articles_plan.txt):

  «Использование последовательной структуры смежных срезов медицинских
  изображений для повышения точности классификации»

Контекст (см. article4_notes.txt): используемый в Статьях 1/3
Kaggle-датасет glioma/meningioma/pituitary не хранит порядок срезов на
пациента, поэтому для этой статьи был выбран BraTS2020 (NIfTI-объёмы,
`experiments/prepare_brats_dataset.py` конвертирует их в PNG-срезы с
сохранением физического порядка по оси Z — `datasets/brats_flair_slices/`,
40 пациентов, 5184 срезов, задача — бинарная "срез содержит опухоль"
(has_tumor) / "не содержит", определяется по маске сегментации).

Методология (план, раздел 5):
  - Разбиение — ПО ПАЦИЕНТАМ, не по отдельным срезам: случайная перестановка
    40 patient_id (random_state=RANDOM_SEED), 80% (32) -> train-пул эталонов,
    20% (8) -> held-out test. Это принципиально важно именно для этой
    статьи: соседние срезы одного пациента физически похожи (это и есть
    предмет эксперимента), поэтому раздельный по пациентам сплит — не
    опция, а обязательное условие честности оценки (тот же класс утечки,
    о котором предупреждает Статья 6, только на уровне пациента, а не
    отдельного изображения).
  - Обучающий пул эталонов: из train-пациентов, независимо по каждому
    классу, случайно отобрано TRAIN_PER_CLASS изображений (без учёта
    порядка срезов — для фазы обучения порядок не используется, только
    для тестовой фазы, см. ниже) — классификатор
    `SubspaceConjugacyClassifier(n_subclasses=8)`, как в Статье 1, для
    сопоставимости baseline.
  - Тестовая последовательность: ВСЕ срезы 8 held-out пациентов,
    сгруппированные по patient_id и отсортированные внутри каждой группы
    по slice_idx — содержательный порядок (п. 1 ниже). Три конфигурации:
      1. meaningful_smoothed   — содержательный порядок + сглаживание
         (`smooth_adjacent_vectors`), применённое ОТДЕЛЬНО к каждой
         пациентской группе (не к сквозной конкатенации всех 8 пациентов
         подряд) — иначе граница между двумя разными пациентами внутри
         склеенного массива осталась бы несглаженной только в двух точках
         всего теста (первая/последняя строка целого массива), а не на
         каждой из 8 пациентских границ, и алгоритм усреднил бы срезы
         РАЗНЫХ людей друг с другом везде, где одна пациентская группа
         кончается, а следующая начинается, кроме самого первого и
         последнего пациента. Эта деталь — осознанное решение, не следствие
         API `smooth_adjacent_vectors` (функция не знает о пациентах и
         сглаживала бы globally, если передать ей весь тест целиком).
      2. meaningful_unsmoothed — тот же порядок, без сглаживания (baseline,
         предсказание `predict()` не зависит от порядка строк).
      3. shuffled_smoothed     — все те же тестовые срезы, но порядок строк
         случайно перемешан (теряется группировка по пациенту), сглаживание
         применено ОДНИМ глобальным вызовом на весь перемешанный массив —
         контроль: прямая демонстрация того, что документированное
         предупреждение `smooth_adjacent_vectors` ("если порядок
         случаен... усреднение смешивает признаки несвязанных объектов")
         справедливо на реальных данных, а не только в теории.
  - Гипотезы (план, раздел 6): H1: accuracy(1) > accuracy(2);
    H2: accuracy(3) <= accuracy(2).

Запуск:
    python experiments/experiment_sequential_smoothing.py

Предварительно (один раз): конвертировать BraTS2020 в срезы —
    BRATS_ROOT="<путь>/MICCAI_BraTS2020_TrainingData" \\
        python experiments/prepare_brats_dataset.py
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from subspace_conjugacy import (
    SubspaceConjugacyClassifier,
    load_and_vectorize_batch,
    smooth_adjacent_vectors,
)
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

logging.basicConfig(level=logging.WARNING, force=True)

DATASET_ROOT = Path("datasets/brats_flair_slices")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
RANDOM_SEED = 42
N_SUBCLASSES = 8  # план, раздел 5: "как в Статье 1" (baseline-конфигурация)
TEST_PATIENT_FRACTION = 0.2
TRAIN_PER_CLASS = 100  # масштаб сопоставим с другими экспериментами плана (m<=150)

ARTIFACTS_DIR = Path("artifacts")
REPORT_JSON_PATH = ARTIFACTS_DIR / "experiment_sequential_smoothing_report.json"
REPORT_MD_PATH = ARTIFACTS_DIR / "experiment_sequential_smoothing_results.md"

Row = Dict[str, Any]


def load_metadata() -> List[Row]:
    """Загружает metadata.csv, возвращает строки с разрешённым путём к PNG."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"{METADATA_PATH} не найден. Сначала запустите конвертер: "
            'BRATS_ROOT="<путь>/MICCAI_BraTS2020_TrainingData" '
            "python experiments/prepare_brats_dataset.py"
        )

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


def split_patients(rows: List[Row]) -> Tuple[List[str], List[str]]:
    """Случайный сплит УНИКАЛЬНЫХ patient_id (не отдельных срезов) 80/20."""
    patient_ids = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = rng.permutation(patient_ids)
    n_test = max(1, round(len(shuffled) * TEST_PATIENT_FRACTION))
    test_ids = sorted(shuffled[:n_test].tolist())
    train_ids = sorted(shuffled[n_test:].tolist())
    return train_ids, test_ids


def select_training_rows(rows: List[Row], train_patient_ids: List[str]) -> List[Row]:
    """Из train-пациентов — случайная подвыборка TRAIN_PER_CLASS на класс."""
    train_pool = [r for r in rows if r["patient_id"] in set(train_patient_ids)]
    rng = np.random.default_rng(RANDOM_SEED)

    selected: List[Row] = []
    for label in ("tumor", "no_tumor"):
        cls_rows = [r for r in train_pool if r["label"] == label]
        idx = rng.permutation(len(cls_rows))[:TRAIN_PER_CLASS]
        selected.extend(cls_rows[i] for i in idx)

    return selected


def vectorize_rows(rows: List[Row]) -> Tuple[np.ndarray, np.ndarray]:
    X = load_and_vectorize_batch([r["path"] for r in rows], method="horizontal")
    y = np.array([r["label"] for r in rows])
    return X, y


def build_meaningful_test_sequence(
    rows: List[Row], test_patient_ids: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Все срезы held-out пациентов, сгруппированные по patient_id, внутри
    группы — по slice_idx (содержательный порядок). Возвращает X, y и длины
    пациентских групп (для раздельного сглаживания каждой группы)."""
    test_pool = [r for r in rows if r["patient_id"] in set(test_patient_ids)]

    group_lengths: List[int] = []
    ordered_rows: List[Row] = []
    for pid in test_patient_ids:
        group = sorted((r for r in test_pool if r["patient_id"] == pid), key=lambda r: r["slice_idx"])
        ordered_rows.extend(group)
        group_lengths.append(len(group))

    X, y = vectorize_rows(ordered_rows)
    return X, y, group_lengths


def smooth_per_patient_group(X: np.ndarray, group_lengths: List[int]) -> np.ndarray:
    """Применяет smooth_adjacent_vectors ОТДЕЛЬНО к каждой пациентской группе
    (не к сквозной конкатенации) — см. docstring модуля, почему это важно."""
    blocks = []
    offset = 0
    for length in group_lengths:
        block = X[offset : offset + length]
        blocks.append(smooth_adjacent_vectors(block))
        offset += length
    return np.vstack(blocks)


def shuffle_and_smooth_globally(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Контроль H2: перемешивает тестовые строки (теряя группировку по
    пациенту) и сглаживает ОДНИМ глобальным вызовом весь перемешанный массив."""
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(X.shape[0])
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    return smooth_adjacent_vectors(X_shuffled), y_shuffled


def run_config(name: str, clf: SubspaceConjugacyClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    print(f"\n=== Конфигурация: {name} ===")
    print(f"   n_test={X_test.shape[0]}")

    y_pred = clf.predict(X_test)
    report = evaluate_classifier(y_test, y_pred)
    per_class = {str(k): v for k, v in report["per_class_accuracy"].items()}
    print(f"   accuracy={report['accuracy']:.4f}  per_class={per_class}")

    return {
        "n_test": int(X_test.shape[0]),
        "accuracy": report["accuracy"],
        "per_class_accuracy": per_class,
        "confusion_matrix": report["confusion_matrix"].tolist(),
        "labels": [str(label) for label in report["labels"]],
    }


def save_markdown_report(
    results: Dict[str, Dict[str, Any]],
    train_patient_ids: List[str],
    test_patient_ids: List[str],
    fit_seconds: float,
    n_train: int,
) -> None:
    lines = [
        "# Статья 4 — результаты эксперимента (сглаживание смежных срезов)",
        "",
        "Сгенерировано `experiments/experiment_sequential_smoothing.py`. "
        f"Датасет: `datasets/brats_flair_slices/` (BraTS2020, FLAIR, "
        f"{len(train_patient_ids) + len(test_patient_ids)} пациентов). "
        f"Классификатор: `SubspaceConjugacyClassifier(n_subclasses={N_SUBCLASSES})`, "
        f"обучен на {n_train} срезах из {len(train_patient_ids)} train-пациентов "
        f"(fit() заняло {fit_seconds:.1f}s), оценён на ВСЕХ срезах "
        f"{len(test_patient_ids)} held-out тестовых пациентов (сплит по "
        "пациентам, не по отдельным срезам — см. docstring модуля).",
        "",
        "| Конфигурация | n_test | accuracy | tumor | no_tumor |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        pc = r["per_class_accuracy"]
        lines.append(
            f"| {name} | {r['n_test']} | {r['accuracy']:.4f} | "
            f"{pc.get('tumor', float('nan')):.3f} | {pc.get('no_tumor', float('nan')):.3f} |"
        )

    acc = {name: r["accuracy"] for name, r in results.items()}
    h1 = acc["meaningful_smoothed"] > acc["meaningful_unsmoothed"]
    h2 = acc["shuffled_smoothed"] <= acc["meaningful_unsmoothed"]
    lines += [
        "",
        f"**H1** (сглаживание помогает при содержательном порядке: "
        f"meaningful_smoothed > meaningful_unsmoothed): "
        f"{acc['meaningful_smoothed']:.4f} > {acc['meaningful_unsmoothed']:.4f} -> **{h1}**.",
        f"**H2** (сглаживание случайного порядка не даёт выигрыша: "
        f"shuffled_smoothed <= meaningful_unsmoothed): "
        f"{acc['shuffled_smoothed']:.4f} <= {acc['meaningful_unsmoothed']:.4f} -> **{h2}**.",
        "",
        "`meaningful_smoothed` сглаживается ОТДЕЛЬНО по каждому пациенту "
        "(физические границы между пациентами не смешиваются), "
        "`shuffled_smoothed` — одним глобальным вызовом на перемешанный "
        "массив (контроль, воспроизводит некорректное применение метода).",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nMarkdown-отчёт: {REPORT_MD_PATH}")


def main() -> None:
    start = time.perf_counter()
    print("=== Статья 4: сглаживание смежных срезов (BraTS2020) ===")

    rows = load_metadata()
    n_patients = len({r["patient_id"] for r in rows})
    print(f"Загружено {len(rows)} срезов, {n_patients} пациентов.")

    train_patient_ids, test_patient_ids = split_patients(rows)
    print(f"Пациенты: {len(train_patient_ids)} train, {len(test_patient_ids)} test (сплит по пациентам).")

    print("\nФормирование обучающего пула эталонов...")
    train_rows = select_training_rows(rows, train_patient_ids)
    X_train, y_train = vectorize_rows(train_rows)
    print(f"X_train.shape={X_train.shape}, классы={sorted(set(y_train.tolist()))}")

    clf = SubspaceConjugacyClassifier(n_subclasses=N_SUBCLASSES)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    print(f"fit() заняло {fit_seconds:.1f}s")

    print("\nФормирование тестовой последовательности (содержательный порядок)...")
    X_meaningful, y_meaningful, group_lengths = build_meaningful_test_sequence(rows, test_patient_ids)
    print(f"X_test.shape={X_meaningful.shape}, пациентских групп={len(group_lengths)}")

    X_meaningful_smoothed = smooth_per_patient_group(X_meaningful, group_lengths)
    X_shuffled_smoothed, y_shuffled = shuffle_and_smooth_globally(X_meaningful, y_meaningful)

    results = {
        "meaningful_smoothed": run_config("meaningful_smoothed", clf, X_meaningful_smoothed, y_meaningful),
        "meaningful_unsmoothed": run_config("meaningful_unsmoothed", clf, X_meaningful, y_meaningful),
        "shuffled_smoothed": run_config("shuffled_smoothed", clf, X_shuffled_smoothed, y_shuffled),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    acc = {name: r["accuracy"] for name, r in results.items()}
    report_payload = {
        "random_seed": RANDOM_SEED,
        "n_subclasses": N_SUBCLASSES,
        "train_per_class": TRAIN_PER_CLASS,
        "n_train": int(X_train.shape[0]),
        "fit_seconds": fit_seconds,
        "train_patients": train_patient_ids,
        "test_patients": test_patient_ids,
        "results": results,
        "hypotheses": {
            "H1_smoothing_helps_meaningful_order": acc["meaningful_smoothed"] > acc["meaningful_unsmoothed"],
            "H2_smoothing_shuffled_no_better_than_baseline": acc["shuffled_smoothed"] <= acc["meaningful_unsmoothed"],
        },
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт: {REPORT_JSON_PATH}")

    save_markdown_report(results, train_patient_ids, test_patient_ids, fit_seconds, int(X_train.shape[0]))

    print(f"\n=== Завершено за {(time.perf_counter() - start) / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
