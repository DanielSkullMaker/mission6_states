"""Полный исследовательский пайплайн: классификация МРТ методом Фурсова
на Kaggle "Brain Tumor MRI Dataset" (archive/Training + archive/Testing).

В отличие от предыдущей версии скрипта (train_test_split на одном заранее
отцентрированном датасете datasets/*_centered), здесь весь путь проходит
от СЫРЫХ .jpg произвольного размера до отчёта о точности, через оркестратор
FursovPipeline (Фаза 6):

    archive/Training/{class}/*.jpg  --\
                                        +-> run_preprocessing (NB1 resize + NB2 center)
    archive/Testing/{class}/*.jpg   --/         |
                                                 v
                                    run_all_classes (NB3 vectorize -> канон A+B cluster -> NB6-7 export)
                                                 |
                                                 v
                                       build_classifier (Фаза C)
                                                 |
                                                 v
                                  classify_test (реальные метки из структуры archive/Testing/{class},
                                                  а не позиционный NB8-фолбэк)

Датасет не входит в репозиторий — путь задаётся переменной окружения
BRAIN_MRI_ARCHIVE_ROOT (по умолчанию — путь на машине автора). Класс
"notumor" присутствует в архиве, но вне скоупа проекта (см. README) и не
используется.
"""

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report

from subspace_conjugacy import FursovPipeline, save_model, save_pipeline_artifact
from subspace_conjugacy.config import DatasetConfig
from subspace_conjugacy.evaluation.metrics import confidence_summary, evaluate_classifier

# --------------------------------------------------------------------------
# Конфигурация эксперимента — правьте эти константы под свою задачу.
# --------------------------------------------------------------------------

ARCHIVE_ROOT = Path(
    os.environ.get("BRAIN_MRI_ARCHIVE_ROOT", r"C:\Users\nazar\Downloads\archive")
)
# Рабочая директория пайплайна (raw/resized/centered/vectors/CSV) —
# полностью пересоздаётся при каждом запуске, ничего вручную туда не кладите.
WORK_ROOT = Path("data/brain_tumor_mri")
ARTIFACTS_DIR = Path("artifacts")

CLASSES = ["glioma", "meningioma", "pituitary"]  # "notumor" вне скоупа проекта

N_SUBCLASSES = 8
FREEZE_BASIS_AT = 2
RANDOM_SEED = 42

# В archive/Training по 1400 файлов на класс, в archive/Testing — по 400.
# Полная кластеризация (Фаза B.2) растёт как O(M^2 * N) на чистом numpy —
# 300 изображений на класс это уже несколько минут на класс. Уменьшите эти
# константы для быстрой проверки кода, увеличьте (вплоть до 1400/400) для
# полноценного финального прогона.
N_TRAIN_PER_CLASS = 200
N_TEST_PER_CLASS = 100


def _sample_class_files(
    class_dir: Path, count: Optional[int], rng: random.Random
) -> List[Path]:
    """Случайная (без повторов, воспроизводимая) выборка файлов класса.

    В archive/Training часть классов смешивает оригинальные и аугментированные
    файлы под разными префиксами (например, meningioma: "Tr-me_*" и
    "Tr-aug-me_*", 1300 против 100 файлов) — простой sorted()[:count] взял бы
    только один префикс целиком из-за лексикографической сортировки строк.
    Случайная выборка с фиксированным seed даёт репрезентативный срез и
    воспроизводимость между запусками.
    """
    files = sorted(class_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"Не найдено .jpg файлов в {class_dir}")
    if count is None or count >= len(files):
        return files
    return rng.sample(files, count)


def prepare_raw_directories(config: DatasetConfig) -> np.ndarray:
    """Копирует сырые изображения archive/ в рабочую структуру DatasetConfig.

    Train — из archive/Training/{class} (метка не нужна для порядка файлов,
    вся папка получает одну метку). Test — из archive/Testing/{class}, но
    имена файлов при копировании дополняются нулями (test0001.jpg, ...),
    чтобы sorted() при resize (лексикографическая сортировка строк) не
    перепутал порядок с числовым — иначе индекс в y_test разъедется с
    файлом, которому он должен соответствовать.

    Returns
    -------
    y_test : np.ndarray
        Истинные метки тестовых объектов, в том же порядке, в котором они
        будут пронумерованы на стадии "resize" (test1.png, test2.png, ...).
    """
    if not (ARCHIVE_ROOT / "Training").is_dir():
        raise FileNotFoundError(
            f"Brain Tumor MRI Dataset не найден в {ARCHIVE_ROOT}. Скачайте "
            "датасет (Kaggle: Brain Tumor MRI Dataset, Masoud Nickparvar) и "
            "укажите путь через переменную окружения BRAIN_MRI_ARCHIVE_ROOT."
        )

    rng = random.Random(RANDOM_SEED)
    config.create_directories(stages=["raw"])

    print("1. Подготовка сырых данных (archive -> рабочая директория)...")
    total_train = 0
    for cls in CLASSES:
        files = _sample_class_files(ARCHIVE_ROOT / "Training" / cls, N_TRAIN_PER_CLASS, rng)
        dst_dir = config.paths[cls]["raw"]
        for f in files:
            shutil.copy(f, dst_dir / f.name)
        total_train += len(files)
        print(f"   train/{cls}: {len(files)} изображений (archive/Training)")

    test_files_by_class = {
        cls: _sample_class_files(ARCHIVE_ROOT / "Testing" / cls, N_TEST_PER_CLASS, rng)
        for cls in CLASSES
    }
    total_test = sum(len(files) for files in test_files_by_class.values())
    width = len(str(total_test))

    test_raw_dir = config.paths["test"]["raw"]
    test_raw_dir.mkdir(parents=True, exist_ok=True)

    y_test: List[str] = []
    counter = 0
    for cls in CLASSES:
        files = test_files_by_class[cls]
        for f in files:
            counter += 1
            shutil.copy(f, test_raw_dir / f"test{counter:0{width}d}.jpg")
            y_test.append(cls)
        print(f"   test/{cls}: {len(files)} изображений (archive/Testing)")

    print(f"   Итого: {total_train} train, {total_test} test\n")
    return np.array(y_test)


def run_preprocessing(pipeline: FursovPipeline) -> None:
    """NB1 (resize) + NB2 (center) для всех классов и test."""
    print("2. Препроцессинг (NB1 resize + NB2 center)...")
    for cls in list(CLASSES) + ["test"]:
        t0 = time.perf_counter()
        pipeline.run_preprocessing(cls, raw_pattern="*.jpg")
        print(f"   {cls}: готово за {time.perf_counter() - t0:.1f}s")
    print()


def run_clustering(pipeline: FursovPipeline) -> None:
    """NB3 (vectorize) -> канон A+B (cluster) -> NB6-7 (export_subspaces)."""
    print("3. Векторизация + кластеризация (канон) + экспорт базисов...")
    for cls in CLASSES:
        t0 = time.perf_counter()
        clusterer = pipeline.run_class(
            cls, n_subclasses=N_SUBCLASSES, freeze_basis_at=FREEZE_BASIS_AT
        )
        sizes = clusterer.get_subclass_sizes()
        elapsed = time.perf_counter() - t0
        print(
            f"   {cls}: {len(clusterer.subspaces_)} подклассов, "
            f"размеры={list(sizes)}, {elapsed:.1f}s"
        )
    print()


def run_evaluation(
    pipeline: FursovPipeline, y_test: np.ndarray
) -> Tuple[dict, np.ndarray]:
    """Векторизует тест, классифицирует, считает метрики (Фаза C / NB8)."""
    print("4. Сборка классификатора и оценка на тестовой выборке...")
    classifier = pipeline.build_classifier()

    X_test = pipeline.run_stage(
        "vectorize", class_name="test", stage="centered", save=True
    )
    y_pred = classifier.predict(X_test)
    report = evaluate_classifier(y_test, y_pred)
    confidence = confidence_summary(classifier.predict_confidence_ratio(X_test))

    print("\n" + "=" * 60)
    print(f"  Точность модели (Accuracy): {report['accuracy'] * 100:.2f}%")
    print("=" * 60)
    for cls, acc in report["per_class_accuracy"].items():
        print(f"    {cls}: {acc * 100:.2f}%")

    print("\nПодробный отчёт (sklearn classification_report):")
    print(classification_report(y_test, y_pred, digits=4))

    print("Матрица ошибок (порядок классов см. report['labels']):")
    print(report["confusion_matrix"])
    print(f"\nУверенность предсказаний (NB8 confidence ratio): {confidence}")

    return {"report": report, "confidence": confidence}, y_pred


def save_artifacts(pipeline: FursovPipeline, summary: dict) -> None:
    """Сохраняет модель (pickle + переносимый CSV/JSON) и сводку эксперимента."""
    print("\n5. Сохранение артефактов...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "subspace_model.pkl"
    save_model(pipeline.classifier_, model_path)
    print(f"   Модель (pickle): {model_path}")

    export_config = DatasetConfig(
        root=ARTIFACTS_DIR / "pipeline",
        classes=CLASSES,
        n_subclasses=N_SUBCLASSES,
        subclass_factor=FREEZE_BASIS_AT,
    )
    save_pipeline_artifact(pipeline.classifier_, export_config)
    print(f"   Модель (CSV+JSON, переносимая): {export_config.root}")

    summary_path = ARTIFACTS_DIR / "experiment_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": summary["report"]["accuracy"],
                "per_class_accuracy": {
                    str(k): v for k, v in summary["report"]["per_class_accuracy"].items()
                },
                "confidence": summary["confidence"],
                "n_train_per_class": N_TRAIN_PER_CLASS,
                "n_test_per_class": N_TEST_PER_CLASS,
                "n_subclasses": N_SUBCLASSES,
                "freeze_basis_at": FREEZE_BASIS_AT,
                "random_seed": RANDOM_SEED,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"   Сводка эксперимента (JSON): {summary_path}")


def main() -> None:
    """Запускает полный исследовательский пайплайн от сырых снимков до отчёта."""
    start = time.perf_counter()
    print("=== Fursov method: Brain Tumor MRI Dataset (archive/) ===\n")
    print(f"Источник данных: {ARCHIVE_ROOT}")
    print(f"Рабочая директория: {WORK_ROOT} (будет пересоздана)")
    print(f"Классы: {CLASSES} (n_subclasses={N_SUBCLASSES}, freeze_basis_at={FREEZE_BASIS_AT})\n")

    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    config = DatasetConfig(
        root=WORK_ROOT,
        classes=CLASSES,
        n_subclasses=N_SUBCLASSES,
        subclass_factor=FREEZE_BASIS_AT,
    )

    y_test = prepare_raw_directories(config)

    pipeline = FursovPipeline(config)
    run_preprocessing(pipeline)
    run_clustering(pipeline)
    summary, _ = run_evaluation(pipeline, y_test)
    save_artifacts(pipeline, summary)

    elapsed = time.perf_counter() - start
    print(f"\n=== Эксперимент завершён за {elapsed / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
