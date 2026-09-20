"""Повторение экспериментов 2 и 3 статьи (theory/, Korshikov & Fursov,
"Pathology Recognition Based on Conjugacy Criteria with Subspaces of
Reference Images") на Kaggle "Brain Tumor MRI Dataset".

Статья описывает 4 эксперимента ("Analysis of Experimental Results"):

  1. Определение проекции (axial/sagittal/coronal) на Otsu-бинаризованных
     изображениях — 100 train + 50 test изображений НА КАЖДУЮ проекцию.
  2. Определение типа опухоли (glioma/meningioma/pituitary) в AXIAL проекции
     на выборках размера 300/375/450 изображений (80% train / 20% test) —
     для каждого размера ищется число подклассов, дающее максимальную
     точность (аналог графика Fig. 7); результат — Table I.
  3. То же, что 2, но с дополнительным фильтром малоинформативных
     изображений (доля "белых" элементов < 50% от среднего по выборке) —
     Table II показывает прирост точности от фильтра.
  4. То же, что 3, но для sagittal и coronal проекций — Table III.

Этот скрипт воспроизводит ЭКСПЕРИМЕНТЫ 2 И 3 буквально (Table I / Table II)
на реальном Kaggle "Brain Tumor MRI Dataset" (archive/Training + Testing,
классы glioma/meningioma/pituitary, "notumor" вне скоупа проекта):

  - Для каждого SAMPLE_SIZE из статьи (300/375/450) собирается независимая
    выборка (80/20 train/test, как в статье) и перебирается число подклассов
    (N_SUBCLASSES_GRID) — берётся лучшая по средней accuracy точка (Fig. 5/7).
  - Каждый размер прогоняется ДВА раза: без фильтра (эксперимент 2,
    filter_low_informativeness=False) и с фильтром (эксперимент 3,
    filter_low_informativeness=True, informativeness_min_fraction=0.5 —
    буквально "50%" из статьи).
  - freeze_basis_at="auto" (не фиксированное k=2) — статья растит
    подпространства до фактического размера и лишь усекает их до общего
    минимума между классами для честного сравнения R(x,Y)
    (refactoring_plan.txt, раздел 10, находка №2; "Description of the
    Clustering Method": "...only the first n elements corresponding to the
    number of vectors of the smallest space are taken for each vector").

ВНЕ СКОУПА этого скрипта — эксперименты 1 и 4 (классификация ПРОЕКЦИИ и её
sagittal/coronal варианты): они требуют датасет с явной разметкой по
проекциям (axial/sagittal/coronal), которого нет в стандартном Kaggle
"Brain Tumor MRI Dataset" (там классы — только glioma/meningioma/notumor/
pituitary, без разметки по проекциям). Библиотека это поддерживает —
preprocessing.binarization.otsu_binarize (Otsu-бинаризация, статья, "Data
Preprocessing") + models.sequential_classifier.SequentialClassifier
(двухэтапная классификация "the result obtained at the previous stage of
classification becomes a set of data for the next stage") — см. README,
раздел "Двухэтапная классификация: проекция → тип опухоли". Если у вас есть
датасет с проекционной разметкой, эти два примитива достаточно скомбинировать
с уже написанными здесь функциями preprocess_and_vectorize/run_accuracy_sweep.

Датасет не входит в репозиторий — путь задаётся переменной окружения
BRAIN_MRI_ARCHIVE_ROOT (по умолчанию — путь на машине автора).
"""

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from subspace_conjugacy import FursovPipeline, SubspaceConjugacyClassifier, save_model
from subspace_conjugacy.config import DatasetConfig
from subspace_conjugacy.evaluation.metrics import evaluate_classifier

# --------------------------------------------------------------------------
# Конфигурация эксперимента — правьте эти константы под свою задачу.
# --------------------------------------------------------------------------

ARCHIVE_ROOT = Path(
    os.environ.get("BRAIN_MRI_ARCHIVE_ROOT", r"C:\Users\nazar\Downloads\archive")
)
# Рабочая директория пайплайна (raw/resized/centered) — полностью
# пересоздаётся перед каждым размером выборки, ничего вручную туда не кладите.
WORK_ROOT = Path("data/brain_tumor_mri")
ARTIFACTS_DIR = Path("artifacts")

CLASSES = ["glioma", "meningioma", "pituitary"]  # "notumor" вне скоупа проекта
RANDOM_SEED = 42

# Статья, Table I/II: "independent sets containing 300, 375 and 450 images
# were generated. Each set was divided into samples: 80% training sample
# and 20% test sample."
SAMPLE_SIZES = [300, 375, 450]
TRAIN_FRACTION = 0.8

# Статья ищет число подпространств, дающее максимум accuracy (Fig. 5/7), не
# фиксирует его заранее. Полный B.2 растёт как O(M^2*N) на чистом numpy —
# каждое значение здесь — отдельный fit() из 3 независимых кластеризаций
# (по одной на класс). Расширьте сетку для более точного повторения графика
# статьи, уменьшите — для быстрой проверки кода.
N_SUBCLASSES_GRID = [4, 6, 8, 10]

# Статья, 3-й эксперимент: "images with the number of white pixels less than
# 50% of the average number of white pixels in the images of the set are
# cut off" — совпадает с дефолтом LowInformativenessFilter.
INFORMATIVENESS_MIN_FRACTION = 0.5

# "auto" = полный рост подпространств + равнение до общего минимума между
# классами (refactoring_plan.txt, раздел 10, находка №2) — не фиксированное
# k=2, как в легаси-ноутбуках.
FREEZE_BASIS_AT = "auto"


def _pool_files(cls: str) -> List[Path]:
    """Объединяет archive/Training/{cls} и archive/Testing/{cls} в один пул.

    Статья формирует "independent sets" заданного размера и сама делит их
    80/20 — не использует готовый train/test сплит Kaggle-архива (у которого
    к тому же сильно разное число файлов на класс). Поэтому здесь оба
    подкаталога архива объединяются в общий пул, из которого набирается
    выборка нужного размера, а train/test сплит делается заново.
    """
    train_dir = ARCHIVE_ROOT / "Training" / cls
    test_dir = ARCHIVE_ROOT / "Testing" / cls
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Brain Tumor MRI Dataset не найден в {ARCHIVE_ROOT}. Скачайте "
            "датасет (Kaggle: Brain Tumor MRI Dataset, Masoud Nickparvar) и "
            "укажите путь через переменную окружения BRAIN_MRI_ARCHIVE_ROOT."
        )
    return sorted(train_dir.glob("*.jpg")) + sorted(test_dir.glob("*.jpg"))


def prepare_sample_set(
    config: DatasetConfig, per_class_count: int, rng: random.Random
) -> np.ndarray:
    """Собирает независимую выборку размера ``per_class_count`` на класс,
    разбитую 80/20 (статья, Table I/II), в рабочую структуру DatasetConfig.

    Returns
    -------
    y_test : np.ndarray
        Истинные метки тестовых объектов, в порядке, в котором они будут
        пронумерованы на стадии "resize" (test1.png, test2.png, ...).
    """
    shutil.rmtree(WORK_ROOT, ignore_errors=True)
    config.create_directories(stages=["raw"])

    test_raw_dir = config.paths["test"]["raw"]
    test_raw_dir.mkdir(parents=True, exist_ok=True)

    n_train = int(round(per_class_count * TRAIN_FRACTION))
    y_test: List[str] = []
    counter = 0
    for cls in CLASSES:
        pool = _pool_files(cls)
        if len(pool) < per_class_count:
            raise ValueError(
                f"В archive/{{Training,Testing}}/{cls} только {len(pool)} "
                f"изображений, требуется {per_class_count}."
            )
        chosen = rng.sample(pool, per_class_count)  # уже случайный порядок
        train_files, test_files = chosen[:n_train], chosen[n_train:]

        for f in train_files:
            shutil.copy(f, config.paths[cls]["raw"] / f.name)
        for f in test_files:
            counter += 1
            shutil.copy(f, test_raw_dir / f"test{counter:04d}.jpg")
            y_test.append(cls)

        print(
            f"      {cls}: {len(train_files)} train, {len(test_files)} test "
            f"(из пула {len(pool)})"
        )

    return np.array(y_test)


def preprocess_and_vectorize(
    pipeline: FursovPipeline,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """NB1+NB2 (resize+center) и NB3 (vectorize) для классов и test.

    Векторизация выполняется ОДИН раз на весь sample_size — цикл по
    N_SUBCLASSES_GRID ниже переиспользует эти векторы, обучая заново только
    сам классификатор (дешевле, чем повторять препроцессинг на каждое
    значение n_subclasses).
    """
    for cls in list(CLASSES) + ["test"]:
        pipeline.run_preprocessing(cls, raw_pattern="*.jpg")

    X_by_class = {
        cls: pipeline.run_stage("vectorize", class_name=cls, stage="centered", save=False)
        for cls in CLASSES
    }
    X_test = pipeline.run_stage(
        "vectorize", class_name="test", stage="centered", save=False
    )
    return X_by_class, X_test


def run_accuracy_sweep(
    X_by_class: Dict[str, np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    filter_low_informativeness: bool,
) -> Dict[str, Any]:
    """Перебирает N_SUBCLASSES_GRID, возвращает точку с лучшей mean accuracy.

    Аналог того, как статья "identified the number of subspaces at which
    the highest accuracy was achieved" (Fig. 5/7) — среди подпространств
    любого фиксированного размера выбирается число подклассов с наилучшей
    средней по классам точностью распознавания.
    """
    X_train = np.vstack([X_by_class[cls] for cls in CLASSES])
    y_train = np.concatenate(
        [np.full(X_by_class[cls].shape[0], cls) for cls in CLASSES]
    )

    all_results = []
    best = None
    for n_subclasses in N_SUBCLASSES_GRID:
        clf = SubspaceConjugacyClassifier(
            n_subclasses=n_subclasses,
            freeze_basis_at=FREEZE_BASIS_AT,
            filter_low_informativeness=filter_low_informativeness,
            informativeness_min_fraction=INFORMATIVENESS_MIN_FRACTION,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = evaluate_classifier(y_test, y_pred)

        per_class_accuracy = {str(k): v for k, v in report["per_class_accuracy"].items()}
        mean_accuracy = float(np.mean(list(per_class_accuracy.values())))

        point = {
            "n_subclasses": n_subclasses,
            "per_class_accuracy": per_class_accuracy,
            "mean_accuracy": mean_accuracy,
        }
        all_results.append(point)
        print(
            f"      n_subclasses={n_subclasses}: "
            f"{per_class_accuracy}, mean={mean_accuracy:.3f}"
        )

        if best is None or mean_accuracy > best["mean_accuracy"]:
            best = {**point, "classifier": clf}

    return {
        "best_n_subclasses": best["n_subclasses"],
        "per_class_accuracy": best["per_class_accuracy"],
        "mean_accuracy": best["mean_accuracy"],
        "classifier": best["classifier"],
        "all_results": all_results,
    }


def run_experiment(sample_size: int) -> Dict[str, Any]:
    """Экспериметы 2 (baseline) и 3 (filter_low_informativeness) для одного
    размера выборки — статья, "the aim of the second experiment.../third
    experiment..."."""
    per_class_count = sample_size // len(CLASSES)
    print(f"\n--- Размер выборки: {sample_size} ({per_class_count} на класс) ---")

    rng = random.Random(RANDOM_SEED + sample_size)
    config = DatasetConfig(root=WORK_ROOT, classes=CLASSES)

    print("   Сборка независимой выборки (80/20 train/test)...")
    y_test = prepare_sample_set(config, per_class_count, rng)

    print("   Препроцессинг + векторизация (NB1-NB3)...")
    pipeline = FursovPipeline(config)
    t0 = time.perf_counter()
    X_by_class, X_test = preprocess_and_vectorize(pipeline)
    print(f"   готово за {time.perf_counter() - t0:.1f}s")

    print("   Эксперимент 2 (без фильтра малоинформативных изображений):")
    baseline = run_accuracy_sweep(X_by_class, X_test, y_test, filter_low_informativeness=False)

    print("   Эксперимент 3 (с фильтром малоинформативных изображений, статья: <50% от среднего):")
    filtered = run_accuracy_sweep(X_by_class, X_test, y_test, filter_low_informativeness=True)

    return {"sample_size": sample_size, "baseline": baseline, "filtered": filtered}


def print_comparison_tables(results: List[Dict[str, Any]]) -> None:
    """Печатает Table I (эксперимент 2) и Table II (эксперимент 3) статьи."""
    header = f"{'Sample size':>12} | " + " | ".join(f"{cls:>11}" for cls in CLASSES) + " | mean"

    for label, key in [
        ("Table I — без фильтра (эксперимент 2)", "baseline"),
        ("Table II — с фильтром малоинформативных изображений (эксперимент 3)", "filtered"),
    ]:
        print(f"\n{label}")
        print(header)
        for r in results:
            point = r[key]
            row = f"{r['sample_size']:>12} | "
            row += " | ".join(
                f"{point['per_class_accuracy'].get(cls, float('nan')):>11.3f}" for cls in CLASSES
            )
            row += f" | {point['mean_accuracy']:.3f}  (n_subclasses={point['best_n_subclasses']})"
            print(row)


def save_artifacts(results: List[Dict[str, Any]]) -> None:
    """Сохраняет сводку эксперимента (JSON) и лучшую по mean accuracy модель."""
    print("\nСохранение артефактов...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    best_overall = max(
        (r[key] for r in results for key in ("baseline", "filtered")),
        key=lambda point: point["mean_accuracy"],
    )
    model_path = ARTIFACTS_DIR / "subspace_model.pkl"
    save_model(best_overall["classifier"], model_path)
    print(f"   Лучшая модель (mean_accuracy={best_overall['mean_accuracy']:.3f}): {model_path}")

    summary_path = ARTIFACTS_DIR / "experiment_report.json"
    summary = [
        {
            "sample_size": r["sample_size"],
            "baseline": {
                "best_n_subclasses": r["baseline"]["best_n_subclasses"],
                "per_class_accuracy": r["baseline"]["per_class_accuracy"],
                "mean_accuracy": r["baseline"]["mean_accuracy"],
                "all_results": [
                    {k: v for k, v in point.items()}
                    for point in r["baseline"]["all_results"]
                ],
            },
            "filtered": {
                "best_n_subclasses": r["filtered"]["best_n_subclasses"],
                "per_class_accuracy": r["filtered"]["per_class_accuracy"],
                "mean_accuracy": r["filtered"]["mean_accuracy"],
                "all_results": [
                    {k: v for k, v in point.items()}
                    for point in r["filtered"]["all_results"]
                ],
            },
        }
        for r in results
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sample_sizes": SAMPLE_SIZES,
                "n_subclasses_grid": N_SUBCLASSES_GRID,
                "informativeness_min_fraction": INFORMATIVENESS_MIN_FRACTION,
                "freeze_basis_at": FREEZE_BASIS_AT,
                "random_seed": RANDOM_SEED,
                "results": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"   Сводка эксперимента (JSON): {summary_path}")


def main() -> None:
    """Повторяет эксперименты 2 и 3 статьи для всех SAMPLE_SIZES."""
    start = time.perf_counter()
    print("=== Fursov method: повторение экспериментов 2-3 статьи (theory/) ===")
    print(f"Источник данных: {ARCHIVE_ROOT}")
    print(f"Классы: {CLASSES}")
    print(f"Размеры выборки (статья, Table I/II): {SAMPLE_SIZES}")
    print(f"Сетка n_subclasses: {N_SUBCLASSES_GRID}")

    results = [run_experiment(size) for size in SAMPLE_SIZES]

    print_comparison_tables(results)
    save_artifacts(results)

    elapsed = time.perf_counter() - start
    print(f"\n=== Эксперимент завершён за {elapsed / 60:.1f} мин ===")


if __name__ == "__main__":
    main()
