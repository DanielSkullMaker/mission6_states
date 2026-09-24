"""Два независимых эксперимента поверх subspace_conjugacy.

Запуск:
    python main.py                    # эксперимент "article" (по умолчанию)
    python main.py --experiment article
    python main.py --experiment draft-method

================================================================================
Эксперимент "article" (run_article_experiment) — см. его собственный docstring
ниже: повторение экспериментов 2 и 3 ОПУБЛИКОВАННОЙ статьи (theory/, Korshikov
& Fursov, "Pathology Recognition Based on Conjugacy Criteria with Subspaces of
Reference Images") на Kaggle "Brain Tumor MRI Dataset".

================================================================================
Эксперимент "draft-method" (run_draft_method_experiment) — новый: проверяет
"Первый этап" из ЧЕРНОВИКА ДРУГОЙ, неопубликованной статьи
(theory/Макет новой статьи.docx; см. refactoring_plan.txt, раздел 11) —
CorrelatedPairSplitter (SubspaceConjugacyClassifier(split_correlated_pairs=
True)) — на уже отцентрированных изображениях datasets/{class}_centered/
(200 PNG 256x256 на класс, эквивалент выхода NB1+NB2 / входа NB3 — этот
пайплайн НЕ включает препроцессинг, только векторизацию и классификацию).

Сравнивает три семейства подходов на одном и том же train/test сплите:
  - SubspaceConjugacyClassifier — расширенная сетка конфигураций
    (SUBSPACE_CONFIGS): baseline, отдельные фильтры статьи (находки №3/№5),
    метод из черновика (подмножества A/B) отдельно и в комбинации с другими
    возможностями библиотеки (growth_strategy="master", freeze_basis_at=
    "auto"), перебор n_subclasses;
  - классические методы ML (sklearn: логрегрессия, линейный SVM, random
    forest, kNN) поверх той же векторизации + PCA;
  - несколько сознательно УПРОЩЁННЫХ CNN (PyTorch, CPU, десятки тысяч
    параметров, 1-2 свёрточных блока).
Требует ``pip install torch`` (не входит в основные зависимости библиотеки —
только в extras "cnn-experiment" в pyproject.toml), импортируется лениво
только внутри run_draft_method_experiment().
"""

import argparse
import json
import itertools
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import randint

from subspace_conjugacy import (
    FursovPipeline,
    SubspaceConjugacyClassifier,
    save_model,
    random_search_classifier,
    summarize_search_results,
)
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


def run_article_experiment() -> None:
    """Повторяет эксперименты 2 и 3 ОПУБЛИКОВАННОЙ статьи для всех SAMPLE_SIZES.

    Статья описывает 4 эксперимента ("Analysis of Experimental Results"):

      1. Определение проекции (axial/sagittal/coronal) на Otsu-бинаризованных
         изображениях — 100 train + 50 test изображений НА КАЖДУЮ проекцию.
      2. Определение типа опухоли (glioma/meningioma/pituitary) в AXIAL
         проекции на выборках размера 300/375/450 изображений (80% train /
         20% test) — для каждого размера ищется число подклассов, дающее
         максимальную точность (аналог графика Fig. 7); результат — Table I.
      3. То же, что 2, но с дополнительным фильтром малоинформативных
         изображений (доля "белых" элементов < 50% от среднего по выборке) —
         Table II показывает прирост точности от фильтра.
      4. То же, что 3, но для sagittal и coronal проекций — Table III.

    Эта функция воспроизводит ЭКСПЕРИМЕНТЫ 2 И 3 буквально (Table I / Table II)
    на реальном Kaggle "Brain Tumor MRI Dataset" (archive/Training + Testing,
    классы glioma/meningioma/pituitary, "notumor" вне скоупа проекта):

      - Для каждого SAMPLE_SIZE из статьи (300/375/450) собирается независимая
        выборка (80/20 train/test, как в статье) и перебирается число
        подклассов (N_SUBCLASSES_GRID) — берётся лучшая по средней accuracy
        точка (Fig. 5/7).
      - Каждый размер прогоняется ДВА раза: без фильтра (эксперимент 2,
        filter_low_informativeness=False) и с фильтром (эксперимент 3,
        filter_low_informativeness=True, informativeness_min_fraction=0.5 —
        буквально "50%" из статьи).
      - freeze_basis_at="auto" (не фиксированное k=2) — статья растит
        подпространства до фактического размера и лишь усекает их до общего
        минимума между классами для честного сравнения R(x,Y)
        (refactoring_plan.txt, раздел 10, находка №2; "Description of the
        Clustering Method": "...only the first n elements corresponding to
        the number of vectors of the smallest space are taken for each
        vector").

    ВНЕ СКОУПА — эксперименты 1 и 4 (классификация ПРОЕКЦИИ и её
    sagittal/coronal варианты): они требуют датасет с явной разметкой по
    проекциям (axial/sagittal/coronal), которого нет в стандартном Kaggle
    "Brain Tumor MRI Dataset" (там классы — только glioma/meningioma/notumor/
    pituitary, без разметки по проекциям). Библиотека это поддерживает —
    preprocessing.binarization.otsu_binarize (Otsu-бинаризация, статья, "Data
    Preprocessing") + models.sequential_classifier.SequentialClassifier
    (двухэтапная классификация "the result obtained at the previous stage of
    classification becomes a set of data for the next stage") — см. README,
    раздел "Двухэтапная классификация: проекция → тип опухоли". Если у вас
    есть датасет с проекционной разметкой, эти два примитива достаточно
    скомбинировать с уже написанными здесь функциями
    preprocess_and_vectorize/run_accuracy_sweep.

    Датасет не входит в репозиторий — путь задаётся переменной окружения
    BRAIN_MRI_ARCHIVE_ROOT (по умолчанию — путь на машине автора).
    """
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


# ==============================================================================
# Эксперимент "draft-method": метод из черновика (theory/Макет новой статьи.docx,
# "Первый этап" -> CorrelatedPairSplitter, refactoring_plan.txt раздел 11) на
# уже отцентрированных изображениях datasets/{class}_centered/ + сравнение с
# несколькими свёрточными нейросетями (PyTorch, CPU).
# ==============================================================================

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch опционален, только для этого эксперимента
    torch = None
    nn = None

from subspace_conjugacy.features.vectorization import load_and_vectorize_batch
from PIL import Image as PILImage

CENTERED_DATASET_ROOT = Path("datasets")
CENTERED_CLASSES = ["glioma", "meningioma", "pituitary"]
CENTERED_N_PER_CLASS = 200  # весь датасет datasets/{class}_centered/
CENTERED_TEST_FRACTION = 0.2  # 80/20 train/test, как в статье (Table I/II)
CENTERED_RANDOM_SEED = 42

# Гиперпараметры SubspaceConjugacyClassifier — БАЗОВЫЕ значения, общие для
# всех конфигураций из SUBSPACE_CONFIGS ниже; каждая конфигурация переопределяет
# только те параметры, которые она реально исследует (см. BASE_SUBSPACE_PARAMS).
SUBSPACE_N_SUBCLASSES = 8
SUBSPACE_FREEZE_BASIS_AT = 2
SUBSPACE_GROWTH_STRATEGY = "default"
SUBSPACE_REG_PARAM = 1e-8

BASE_SUBSPACE_PARAMS: Dict[str, Any] = dict(
    n_subclasses=SUBSPACE_N_SUBCLASSES,
    freeze_basis_at=SUBSPACE_FREEZE_BASIS_AT,
    growth_strategy=SUBSPACE_GROWTH_STRATEGY,
    reg_param=SUBSPACE_REG_PARAM,
    filter_dependent=False,
    dependency_threshold=0.999,
    filter_low_informativeness=False,
    split_correlated_pairs=False,
    correlated_pairs_subset="a",
)

# Расширенная сетка конфигураций — от чистого канона (baseline) через
# отдельные проверенные публикацией фильтры (находки №3, №5 из
# refactoring_plan.txt, раздел 10) до метода из черновика (раздел 11) в
# комбинации с другими опциональными возможностями библиотеки
# (growth_strategy="master", freeze_basis_at="auto"/equalize) и с перебором
# n_subclasses — показывает библиотеку куда шире, чем просто "включен/выключен
# один флаг".
SUBSPACE_N_SUBCLASSES_GRID = [4, 6, 12, 16]  # доп. точки; 8 уже в основных конфигах

SUBSPACE_CONFIGS: List[Dict[str, Any]] = [
    {
        "key": "subspace_baseline",
        "name": "Baseline (канон, без фильтров/черновика)",
        "params_override": {},
    },
    {
        "key": "subspace_filter_dependent",
        "name": "+ filter_dependent (статья, находка №3)",
        "params_override": {"filter_dependent": True, "dependency_threshold": 0.999},
    },
    {
        "key": "subspace_filter_low_informativeness",
        "name": "+ filter_low_informativeness (статья, находка №5)",
        "params_override": {"filter_low_informativeness": True},
    },
    {
        "key": "subspace_draft_subset_a",
        "name": "+ черновик, подмножество A",
        "params_override": {"split_correlated_pairs": True, "correlated_pairs_subset": "a"},
    },
    {
        "key": "subspace_draft_subset_b",
        "name": "+ черновик, подмножество B",
        "params_override": {"split_correlated_pairs": True, "correlated_pairs_subset": "b"},
    },
    {
        "key": "subspace_draft_b_plus_both_filters",
        "name": "+ черновик (B) + оба фильтра статьи (№3+№5)",
        "params_override": {
            "split_correlated_pairs": True, "correlated_pairs_subset": "b",
            "filter_dependent": True, "filter_low_informativeness": True,
        },
    },
    {
        "key": "subspace_draft_b_master_growth",
        "name": "+ черновик (B) + growth_strategy='master' (NB7)",
        "params_override": {
            "split_correlated_pairs": True, "correlated_pairs_subset": "b",
            "growth_strategy": "master",
        },
    },
    {
        "key": "subspace_draft_b_freeze_auto",
        "name": "+ черновик (B) + freeze_basis_at='auto' (статья, находка №2)",
        "params_override": {
            "split_correlated_pairs": True, "correlated_pairs_subset": "b",
            "freeze_basis_at": "auto",
        },
    },
] + [
    {
        "key": f"subspace_draft_b_n{n}",
        "name": f"+ черновик (B), n_subclasses={n}",
        "params_override": {
            "split_correlated_pairs": True, "correlated_pairs_subset": "b",
            "n_subclasses": n,
        },
    }
    for n in SUBSPACE_N_SUBCLASSES_GRID
]

# Гиперпараметры обучения CNN (PyTorch, CPU) — общие для обеих архитектур.
# Архитектуры сознательно УПРОЩЕНЫ (1-2 свёрточных блока, без BatchNorm/
# Dropout/скрытых FC-слоёв) — параметров на 2 порядка меньше, чем в первой
# версии этого эксперимента (millions -> десятки тысяч), чтобы "объём модели"
# CNN не был заведомо несравним с числом эталонных векторов метода
# сопряжённости, а также чтобы сеть требовала работы с осмысленным объёмом
# данных, а не побеждала за счёт избыточной ёмкости.
CNN_IMG_SIZE = 64
CNN_EPOCHS = 30
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 1e-3
CNN_WEIGHT_DECAY = 1e-4
CNN_RANDOM_SEED = 42


if torch is not None:

    class TinyCNN(nn.Module):
        """Минимальная CNN: 1 свёрточный блок (8 каналов), агрессивный пулинг
        (/4 за один MaxPool) и линейный классификатор напрямую — без скрытого
        FC-слоя. in_channels — 3 для RGB (МРТ), 1 для grayscale (MNIST)."""

        def __init__(
            self, num_classes: int, img_size: int = CNN_IMG_SIZE, in_channels: int = 3,
        ) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(4),
            )
            flat_dim = 8 * (img_size // 4) ** 2
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, num_classes))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.features(x))

    class SmallCNN(nn.Module):
        """Простая CNN: 2 свёрточных блока (16, 32 канала), линейный
        классификатор напрямую — без скрытого FC-слоя, без BatchNorm/Dropout.
        in_channels — 3 для RGB (МРТ), 1 для grayscale (MNIST)."""

        def __init__(
            self, num_classes: int, img_size: int = CNN_IMG_SIZE, in_channels: int = 3,
        ) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )
            flat_dim = 32 * (img_size // 4) ** 2
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, num_classes))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.features(x))

    CNN_ARCHITECTURES = {
        "TinyCNN": TinyCNN,
        "SmallCNN": SmallCNN,
    }
else:  # pragma: no cover
    CNN_ARCHITECTURES = {}


# ------------------------------------------------------------------------------
# Классические методы машинного обучения (sklearn) — та же векторизация
# (65536 признаков), что и у метода сопряжённости, но с PCA-понижением
# размерности до CLASSICAL_ML_PCA_COMPONENTS: без него kNN/SVM/RandomForest на
# 65536 "сырых" признаках при 480 обучающих объектах либо непрактично
# медленны, либо статистически неустойчивы — PCA перед классическим ML это
# стандартная практика, а не подгонка под желаемый результат (в отличие от
# метода сопряжённости и CNN, которые работают с исходными пиксельными
# представлениями без понижения размерности).
# ------------------------------------------------------------------------------

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

CLASSICAL_ML_PCA_COMPONENTS = 100
CLASSICAL_ML_RANDOM_SEED = 42

CLASSICAL_ML_MODELS = {
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=2000, random_state=CLASSICAL_ML_RANDOM_SEED,
    ),
    "LinearSVM": lambda: SVC(kernel="linear", random_state=CLASSICAL_ML_RANDOM_SEED),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=CLASSICAL_ML_RANDOM_SEED, n_jobs=-1,
    ),
    "kNN_k5": lambda: KNeighborsClassifier(n_neighbors=5),
}


def _sorted_centered_class_images(class_name: str) -> List[Path]:
    """Пути к PNG класса в datasets/{class}_centered/, отсортированные по
    числовому суффиксу имени файла (glioma1.png, glioma2.png, ...) — тот же
    порядок, что и в исходных ноутбуках (tests/test_parity/conftest.py)."""
    class_dir = CENTERED_DATASET_ROOT / f"{class_name}_centered"
    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"{class_dir} не найдена. Ожидается уже отцентрированный датасет "
            f"datasets/{{class}}_centered/*.png (200 PNG на класс — см. README, "
            f"раздел 'Данные')."
        )
    return sorted(
        class_dir.glob(f"{class_name}*.png"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )


def prepare_centered_split() -> Dict[str, Dict[str, Any]]:
    """Строит один и тот же 80/20 train/test сплит на класс (фиксированный
    random_state), общий для метода подпространств И для CNN — чтобы
    сравнение точности было честным (одни и те же тестовые изображения для
    обоих подходов).

    Returns
    -------
    split : Dict[str, Dict[str, Any]]
        {class_name: {"paths": [...], "train_idx": np.ndarray, "test_idx": np.ndarray}}
    """
    rng = np.random.default_rng(CENTERED_RANDOM_SEED)
    n_test = int(round(CENTERED_N_PER_CLASS * CENTERED_TEST_FRACTION))
    split: Dict[str, Dict[str, Any]] = {}
    for cls in CENTERED_CLASSES:
        paths = _sorted_centered_class_images(cls)
        if len(paths) < CENTERED_N_PER_CLASS:
            raise ValueError(
                f"В datasets/{cls}_centered/ найдено только {len(paths)} "
                f"изображений, требуется {CENTERED_N_PER_CLASS}."
            )
        paths = paths[:CENTERED_N_PER_CLASS]
        perm = rng.permutation(CENTERED_N_PER_CLASS)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        split[cls] = {"paths": paths, "train_idx": train_idx, "test_idx": test_idx}
        print(f"   {cls}: {len(train_idx)} train, {len(test_idx)} test (из {len(paths)})")
    return split


def build_subspace_vectors(
    split: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Векторизует все изображения класса (NB3: горизонтальная развёртка,
    65536 признаков) и разрезает на train (по классу)/test (объединённый)
    согласно ``split`` из prepare_centered_split()."""
    X_train_by_class: Dict[str, np.ndarray] = {}
    X_test_list, y_test_list = [], []
    for cls, info in split.items():
        X_all = load_and_vectorize_batch(info["paths"], method="horizontal")
        X_train_by_class[cls] = X_all[info["train_idx"]]
        X_test_list.append(X_all[info["test_idx"]])
        y_test_list.append(np.full(len(info["test_idx"]), cls))
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    return X_train_by_class, X_test, y_test


def build_cnn_image_pool_by_class(
    split: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Загружает изображения как RGB, приводит к (CNN_IMG_SIZE, CNN_IMG_SIZE, 3)
    в [0, 1] и разрезает на train (СГРУППИРОВАННЫЙ по классу — нужно для
    hyperparameter_tuning.run_data_efficiency_sweep, которая берёт срезы
    переменного размера на класс)/test (объединённый, как у
    build_subspace_vectors()) ТЕМИ ЖЕ индексами train_idx/test_idx.

    Читает CNN_IMG_SIZE из глобальной области видимости НА МОМЕНТ ВЫЗОВА (не
    как значение по умолчанию параметра — то фиксировалось бы при определении
    функции и разошлось бы с реальным размером, который видят CNN-модели,
    если константу переопределить после импорта модуля)."""
    img_size = CNN_IMG_SIZE
    X_train_by_class: Dict[str, np.ndarray] = {}
    X_test_list, y_test_list = [], []
    for cls, info in split.items():
        imgs = np.stack([
            np.asarray(
                PILImage.open(p).convert("RGB").resize((img_size, img_size)),
                dtype=np.float32,
            ) / 255.0
            for p in info["paths"]
        ])
        X_train_by_class[cls] = imgs[info["train_idx"]]
        X_test_list.append(imgs[info["test_idx"]])
        y_test_list.append(np.full(len(info["test_idx"]), cls))
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    return X_train_by_class, X_test, y_test


def build_cnn_tensors(
    split: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Тонкая обёртка над build_cnn_image_pool_by_class(): конкатенирует
    train по всем классам в единый (X_train, y_train) — формат, нужный
    run_cnn_configs() для одноразового прогона (без среза по объёму данных)."""
    X_train_by_class, X_test, y_test = build_cnn_image_pool_by_class(split)
    classes = list(split.keys())
    X_train = np.vstack([X_train_by_class[c] for c in classes])
    y_train = np.concatenate([np.full(X_train_by_class[c].shape[0], c) for c in classes])
    return X_train, y_train, X_test, y_test


def run_subspace_configs(
    X_train_by_class: Dict[str, np.ndarray], X_test: np.ndarray, y_test: np.ndarray,
    classes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Обучает и оценивает все SUBSPACE_CONFIGS на одном и том же train/test.

    ``classes`` — список меток классов в порядке, определяющем train-выборку
    (по умолчанию CENTERED_CLASSES — датасет МРТ; для MNIST передаётся
    MNIST_CLASSES). Функция не зависит от предметной области — работает с
    любым числом классов и признаков.
    """
    classes = list(classes) if classes is not None else CENTERED_CLASSES
    X_train = np.vstack([X_train_by_class[c] for c in classes])
    y_train = np.concatenate(
        [np.full(X_train_by_class[c].shape[0], c) for c in classes]
    )

    results = []
    for cfg in SUBSPACE_CONFIGS:
        params = {**BASE_SUBSPACE_PARAMS, **cfg["params_override"]}
        print(f"\n   [{cfg['key']}] {cfg['name']}")
        clf = SubspaceConjugacyClassifier(**params)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        y_pred = clf.predict(X_test)
        report = evaluate_classifier(y_test, y_pred)
        per_class_accuracy = {str(k): v for k, v in report["per_class_accuracy"].items()}
        mean_accuracy = float(np.mean(list(per_class_accuracy.values())))

        if clf.excluded_indices_by_class_ is not None:
            n_effective_train = {
                cls: int(
                    X_train_by_class[cls].shape[0]
                    - len(clf.excluded_indices_by_class_.get(cls, []))
                )
                for cls in classes
            }
        else:
            n_effective_train = {
                cls: int(X_train_by_class[cls].shape[0]) for cls in classes
            }

        print(
            f"      accuracy: {per_class_accuracy}, mean={mean_accuracy:.3f}, "
            f"fit_time={fit_time:.2f}s, эталонных векторов/класс после метода="
            f"{n_effective_train}"
        )

        results.append({
            "key": cfg["key"],
            "name": cfg["name"],
            "type": "subspace_conjugacy",
            "params": params,
            "n_effective_train_vectors_by_class": n_effective_train,
            "fit_time_seconds": fit_time,
            "per_class_accuracy": per_class_accuracy,
            "mean_accuracy": mean_accuracy,
            "confusion_matrix": report["confusion_matrix"].tolist(),
        })
    return results


def run_classical_ml_configs(
    X_train_by_class: Dict[str, np.ndarray], X_test: np.ndarray, y_test: np.ndarray,
    classes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Обучает и оценивает CLASSICAL_ML_MODELS (sklearn) на той же
    векторизации, что и метод сопряжённости — с PCA-понижением размерности
    до CLASSICAL_ML_PCA_COMPONENTS (см. комментарий у CLASSICAL_ML_MODELS) и
    масштабированием в [0, 1] (/255).

    ``classes`` — см. run_subspace_configs (по умолчанию CENTERED_CLASSES)."""
    classes = list(classes) if classes is not None else CENTERED_CLASSES
    X_train = np.vstack([X_train_by_class[c] for c in classes]) / 255.0
    y_train = np.concatenate(
        [np.full(X_train_by_class[c].shape[0], c) for c in classes]
    )
    X_test_scaled = X_test / 255.0

    results = []
    for name, make_model in CLASSICAL_ML_MODELS.items():
        print(f"\n   [{name}] обучение (PCA -> {CLASSICAL_ML_PCA_COMPONENTS} компонент)...")
        pipeline = Pipeline([
            ("pca", PCA(n_components=CLASSICAL_ML_PCA_COMPONENTS, random_state=CLASSICAL_ML_RANDOM_SEED)),
            ("clf", make_model()),
        ])
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        y_pred = pipeline.predict(X_test_scaled)
        report = evaluate_classifier(y_test, y_pred)
        per_class_accuracy = {str(k): v for k, v in report["per_class_accuracy"].items()}
        mean_accuracy = float(np.mean(list(per_class_accuracy.values())))

        clf_params = {
            k: v for k, v in pipeline.named_steps["clf"].get_params().items()
            if isinstance(v, (int, float, str, bool)) or v is None
        }
        print(
            f"      accuracy: {per_class_accuracy}, mean={mean_accuracy:.3f}, "
            f"fit_time={fit_time:.2f}s"
        )

        results.append({
            "key": f"classical_{name.lower()}",
            "name": f"Классический ML: {name} (PCA-{CLASSICAL_ML_PCA_COMPONENTS})",
            "type": "classical_ml",
            "params": {
                "model": name,
                "pca_components": CLASSICAL_ML_PCA_COMPONENTS,
                "model_params": clf_params,
                "random_seed": CLASSICAL_ML_RANDOM_SEED,
            },
            "fit_time_seconds": fit_time,
            "per_class_accuracy": per_class_accuracy,
            "mean_accuracy": mean_accuracy,
            "confusion_matrix": report["confusion_matrix"].tolist(),
        })
    return results


def train_cnn_model(
    model_cls,
    X_train: np.ndarray, y_train_idx: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    in_channels: int = 3,
    use_hflip_augmentation: bool = True,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
) -> Dict[str, Any]:
    """Обучает одну CNN-архитектуру на CPU, возвращает метрики и предсказания.

    X_* — (N, H, W, in_channels) float32 в [0, 1]; y_train_idx — целочисленные
    метки. Единственная аугментация — случайное горизонтальное отражение
    (p=0.5) каждого объекта батча на train (use_hflip_augmentation=False
    отключает её — для MNIST горизонтальное отражение меняет смысл цифры,
    в отличие от МРТ, где оно анатомически чаще допустимо).

    ``epochs``/``learning_rate``/``weight_decay`` — явные значения (нужны для
    подбора гиперпараметров — tune_cnn_hyperparameters); None -> берутся из
    CNN_EPOCHS/CNN_LEARNING_RATE/CNN_WEIGHT_DECAY НА МОМЕНТ ВЫЗОВА (не как
    default параметра — та же причина позднего связывания, что и у
    CNN_IMG_SIZE в build_cnn_tensors)."""
    epochs = CNN_EPOCHS if epochs is None else epochs
    learning_rate = CNN_LEARNING_RATE if learning_rate is None else learning_rate
    weight_decay = CNN_WEIGHT_DECAY if weight_decay is None else weight_decay

    torch.manual_seed(CNN_RANDOM_SEED)

    model = model_cls(num_classes=num_classes, img_size=CNN_IMG_SIZE, in_channels=in_channels)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    X_train_t = torch.from_numpy(X_train.transpose(0, 3, 1, 2)).float()
    y_train_t = torch.from_numpy(y_train_idx).long()
    X_test_t = torch.from_numpy(X_test.transpose(0, 3, 1, 2)).float()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    n_train = X_train_t.shape[0]
    history = []
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, CNN_BATCH_SIZE):
            idx = perm[start:start + CNN_BATCH_SIZE]
            xb, yb = X_train_t[idx].clone(), y_train_t[idx]
            if use_hflip_augmentation:
                flip_mask = torch.rand(xb.shape[0]) < 0.5
                xb[flip_mask] = torch.flip(xb[flip_mask], dims=[3])

            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]

        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_t).argmax(dim=1)
            train_acc = (train_pred == y_train_t).float().mean().item()
        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / n_train,
            "train_accuracy": train_acc,
        })

    training_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        test_pred_idx = model(X_test_t).argmax(dim=1).numpy()

    return {
        "n_params": int(n_params),
        "training_time_seconds": training_time,
        "final_train_accuracy": history[-1]["train_accuracy"],
        "history": history,
        "test_pred_idx": test_pred_idx,
    }


def run_cnn_configs(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    in_channels: int = 3,
    use_hflip_augmentation: bool = True,
) -> List[Dict[str, Any]]:
    """Обучает и оценивает все CNN_ARCHITECTURES на одном и том же train/test.

    in_channels — 3 для RGB (МРТ), 1 для grayscale (MNIST).
    use_hflip_augmentation — см. train_cnn_model (для MNIST следует передавать
    False: горизонтальное отражение меняет смысл цифры)."""
    if torch is None:
        raise ImportError(
            "PyTorch не установлен — сравнение с CNN недоступно. Установите: "
            "pip install torch (или pip install -e '.[cnn-experiment]')."
        )

    classes_sorted = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    y_train_idx = np.array([class_to_idx[c] for c in y_train])

    results = []
    for name, model_cls in CNN_ARCHITECTURES.items():
        print(f"\n   [{name}] обучение ({CNN_EPOCHS} эпох, {CNN_IMG_SIZE}x{CNN_IMG_SIZE}, CPU)...")
        outcome = train_cnn_model(
            model_cls, X_train, y_train_idx, X_test, num_classes=len(classes_sorted),
            in_channels=in_channels, use_hflip_augmentation=use_hflip_augmentation,
        )
        y_pred = np.array([classes_sorted[i] for i in outcome["test_pred_idx"]])
        report = evaluate_classifier(y_test, y_pred)
        per_class_accuracy = {str(k): v for k, v in report["per_class_accuracy"].items()}
        mean_accuracy = float(np.mean(list(per_class_accuracy.values())))

        print(
            f"      params={outcome['n_params']:,}, "
            f"training_time={outcome['training_time_seconds']:.1f}s, "
            f"train_accuracy(последняя эпоха)={outcome['final_train_accuracy']:.3f}, "
            f"mean test accuracy={mean_accuracy:.3f}"
        )

        results.append({
            "key": f"cnn_{name.lower()}",
            "name": name,
            "type": "cnn",
            "params": {
                "architecture": name,
                "n_trainable_parameters": outcome["n_params"],
                "img_size": CNN_IMG_SIZE,
                "in_channels": in_channels,
                "epochs": CNN_EPOCHS,
                "batch_size": CNN_BATCH_SIZE,
                "learning_rate": CNN_LEARNING_RATE,
                "weight_decay": CNN_WEIGHT_DECAY,
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss",
                "augmentation": (
                    "random horizontal flip (p=0.5) на train" if use_hflip_augmentation else "нет"
                ),
                "device": "cpu",
                "random_seed": CNN_RANDOM_SEED,
            },
            "training_time_seconds": outcome["training_time_seconds"],
            "final_train_accuracy": outcome["final_train_accuracy"],
            "per_class_accuracy": per_class_accuracy,
            "mean_accuracy": mean_accuracy,
            "confusion_matrix": report["confusion_matrix"].tolist(),
        })
    return results


_RESULT_TYPE_LABELS = {
    "subspace_conjugacy": "сопряжённость",
    "classical_ml": "классич. ML",
    "cnn": "CNN",
}


def print_comparison_table(all_results: List[Dict[str, Any]], classes: List[str]) -> None:
    """Печатает единую таблицу сравнения всех трёх подходов.

    Колонка "размер" показывает величину РАЗНОЙ природы в зависимости от типа
    (не для прямого числового сравнения между строками разных типов — эти
    величины принципиально несопоставимы):
      - subspace_conjugacy: суммарное число ЭТАЛОННЫХ векторов, реально
        использованных после фильтрации, по всем классам;
      - classical_ml: число компонент PCA, поданных на вход классификатору
        (сам классификатор — не нейросеть, "параметров" в смысле CNN не имеет);
      - cnn: число обучаемых параметров модели.
    Колонка "тип" — чтобы это несоответствие природы величин не терялось из
    вида при чтении таблицы.

    При len(classes) > 5 (например, 10 цифр MNIST) полные per-class колонки
    сделали бы таблицу нечитаемой — вместо них печатаются mean/min/max по
    классам (полные per-class значения всегда доступны в JSON-отчёте).
    """
    show_per_class_columns = len(classes) <= 5
    if show_per_class_columns:
        header = (
            f"{'Метод':<55} | {'тип':<14} | "
            + " | ".join(f"{cls:>11}" for cls in classes)
            + " |  mean |        размер |  время"
        )
    else:
        header = (
            f"{'Метод':<55} | {'тип':<14} |  mean |   min |   max |        размер |  время"
        )
    print("\n" + header)
    print("-" * len(header))
    for r in all_results:
        row = f"{r['name']:<55} | {_RESULT_TYPE_LABELS[r['type']]:<14} | "
        if show_per_class_columns:
            row += " | ".join(
                f"{r['per_class_accuracy'].get(cls, float('nan')):>11.3f}" for cls in classes
            )
            row += f" | {r['mean_accuracy']:.3f}"
        else:
            vals = list(r["per_class_accuracy"].values())
            row += f"{r['mean_accuracy']:.3f} | {min(vals):.3f} | {max(vals):.3f}"
        if r["type"] == "subspace_conjugacy":
            capacity = sum(r["n_effective_train_vectors_by_class"].values())
            elapsed = r["fit_time_seconds"]
        elif r["type"] == "classical_ml":
            capacity = r["params"]["pca_components"]
            elapsed = r["fit_time_seconds"]
        else:
            capacity = r["params"]["n_trainable_parameters"]
            elapsed = r["training_time_seconds"]
        row += f" | {capacity:>14,} | {elapsed:>6.1f}s"
        print(row)


def save_experiment_artifacts(
    all_results: List[Dict[str, Any]], dataset_info: Dict[str, Any], report_filename: str,
) -> None:
    """Сохраняет полный JSON-отчёт (все параметры + метрики всех подходов).

    ``dataset_info`` — произвольный словарь с описанием датасета/сплита,
    полностью специфичный для конкретного эксперимента (МРТ или MNIST).
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset": dataset_info, "results": all_results},
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"\n   Отчёт эксперимента (JSON): {report_path}")


def save_draft_method_artifacts(
    all_results: List[Dict[str, Any]], split_summary: Dict[str, Any],
) -> None:
    """Сохраняет JSON-отчёт эксперимента на датасете МРТ (тонкая обёртка над
    save_experiment_artifacts с МРТ-специфичным dataset_info)."""
    save_experiment_artifacts(
        all_results,
        dataset_info={
            "root": str(CENTERED_DATASET_ROOT),
            "classes": CENTERED_CLASSES,
            "n_per_class": CENTERED_N_PER_CLASS,
            "test_fraction": CENTERED_TEST_FRACTION,
            "random_seed": CENTERED_RANDOM_SEED,
            "split_summary": split_summary,
        },
        report_filename="draft_method_experiment_report.json",
    )


def run_draft_method_experiment() -> None:
    """Сравнивает метод из черновика (theory/Макет новой статьи.docx,
    "Первый этап" -> CorrelatedPairSplitter) с классическим ML и с CNN на уже
    отцентрированных изображениях datasets/{class}_centered/.

    Пайплайн (БЕЗ препроцессинга — изображения уже прошли эквивалент NB1+NB2):
      1. Один и тот же 80/20 train/test сплит на класс — общий для ВСЕХ трёх
         подходов (одинаковые тестовые изображения, честное сравнение).
      2. SubspaceConjugacyClassifier — SUBSPACE_CONFIGS (12 конфигураций):
         baseline, отдельно оба фильтра статьи (находки №3/№5), метод из
         черновика (подмножества A/B), его комбинация с обоими фильтрами
         статьи, с growth_strategy="master" (NB7), с freeze_basis_at="auto"
         (находка №2) и перебор n_subclasses — единая база гиперпараметров
         BASE_SUBSPACE_PARAMS, каждая конфигурация переопределяет только то,
         что она исследует.
      3. Классические ML-модели (sklearn, CLASSICAL_ML_MODELS) — та же
         векторизация 65536 признаков, что и метод сопряжённости, с PCA
         понижением размерности до CLASSICAL_ML_PCA_COMPONENTS.
      4. Несколько CNN (PyTorch, CPU, CNN_ARCHITECTURES) — сознательно
         УПРОЩЁННЫЕ архитектуры (1-2 свёрточных блока, десятки тысяч
         параметров вместо миллионов), обучаются на исходных RGB-
         изображениях (без векторизации — это специфика метода
         сопряжённости/классического ML).
      5. Таблица сравнения (печать) + JSON-отчёт со ВСЕМИ параметрами всех
         трёх подходов (artifacts/draft_method_experiment_report.json).

    Requires
    --------
    PyTorch (``pip install torch``) — не входит в основные зависимости
    библиотеки, только в extras "cnn-experiment" (pyproject.toml).
    """
    if torch is None:
        raise ImportError(
            "Эксперимент 'draft-method' требует PyTorch для сравнения с CNN. "
            "Установите: pip install torch (или pip install -e '.[cnn-experiment]')."
        )

    start = time.perf_counter()
    print("=== Метод из черновика (theory/Макет новой статьи.docx) vs классический ML vs CNN ===")
    print(f"Датасет: {CENTERED_DATASET_ROOT}/{{class}}_centered/, классы: {CENTERED_CLASSES}")

    print("\n1. Формирование train/test сплита (80/20, общий для всех подходов)...")
    split = prepare_centered_split()
    split_summary = {
        cls: {"n_train": len(info["train_idx"]), "n_test": len(info["test_idx"])}
        for cls, info in split.items()
    }

    print("\n2. Векторизация для метода сопряжённости и классического ML (NB3, горизонтальная развёртка)...")
    X_train_by_class, X_test_vec, y_test = build_subspace_vectors(split)

    print(f"\n3. Загрузка изображений для CNN (resize -> {CNN_IMG_SIZE}x{CNN_IMG_SIZE}, RGB, [0,1])...")
    X_train_img, y_train_img, X_test_img, y_test_img = build_cnn_tensors(split)
    np.testing.assert_array_equal(y_test, y_test_img)  # физически тот же тестовый сплит

    print(f"\n4. Метод подпространственной сопряжённости ({len(SUBSPACE_CONFIGS)} конфигураций)...")
    subspace_results = run_subspace_configs(X_train_by_class, X_test_vec, y_test)

    print(f"\n5. Классические методы ML ({len(CLASSICAL_ML_MODELS)} моделей)...")
    classical_results = run_classical_ml_configs(X_train_by_class, X_test_vec, y_test)

    print(f"\n6. Свёрточные нейросети (PyTorch, CPU, {len(CNN_ARCHITECTURES)} упрощённые архитектуры)...")
    cnn_results = run_cnn_configs(X_train_img, y_train_img, X_test_img, y_test_img)

    all_results = subspace_results + classical_results + cnn_results
    print("\n7. Итоговое сравнение:")
    print_comparison_table(all_results, classes=CENTERED_CLASSES)
    save_draft_method_artifacts(all_results, split_summary)

    elapsed = time.perf_counter() - start
    print(f"\n=== Эксперимент завершён за {elapsed / 60:.1f} мин ===")


# ==============================================================================
# Эксперимент "mnist": ТОТ ЖЕ эксперимент (SUBSPACE_CONFIGS + CLASSICAL_ML_MODELS
# + CNN_ARCHITECTURES), что и run_draft_method_experiment(), но на MNIST (10
# классов цифр, 28x28 grayscale) — проверка, обобщается ли картина,
# наблюдавшаяся на МРТ (раздел 11 refactoring_plan.txt), на другой домен.
# MNIST не хранится как файлы на диске — скачивается/кэшируется через
# sklearn.datasets.fetch_openml (использует уже имеющуюся зависимость
# scikit-learn, без torchvision).
# ==============================================================================

MNIST_CLASSES = [str(d) for d in range(10)]
MNIST_N_PER_CLASS = 200  # тот же объём на класс, что и в МРТ-эксперименте
MNIST_TEST_FRACTION = 0.2
MNIST_RANDOM_SEED = 42


def fetch_mnist_split() -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Скачивает/кэширует MNIST (sklearn.datasets.fetch_openml, кэш —
    ~/scikit_learn_data/ после первого запуска) и строит тот же 80/20
    train/test сплит на класс (200 изображений/класс, seed=42), что и
    prepare_centered_split() для МРТ.

    Returns
    -------
    X_train_by_class : Dict[str, np.ndarray]
        {digit: (160, 784) float64 в [0, 255]} — те же векторы используются
        и методом сопряжённости, и классическим ML, и (после reshape) CNN.
    X_test, y_test : np.ndarray
        Тестовая выборка (400, 784) и метки (400,), объединённая по всем
        классам, в том же формате, что и build_subspace_vectors() для МРТ.
    """
    from sklearn.datasets import fetch_openml

    print("   Загрузка MNIST (sklearn.datasets.fetch_openml, кэшируется локально)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X_all = mnist.data.astype(np.float64)
    y_all = mnist.target.astype(str)

    rng = np.random.default_rng(MNIST_RANDOM_SEED)
    n_test = int(round(MNIST_N_PER_CLASS * MNIST_TEST_FRACTION))

    X_train_by_class: Dict[str, np.ndarray] = {}
    X_test_list, y_test_list = [], []
    for digit in MNIST_CLASSES:
        idx_all = np.where(y_all == digit)[0]
        if len(idx_all) < MNIST_N_PER_CLASS:
            raise ValueError(
                f"MNIST: класс '{digit}' содержит только {len(idx_all)} "
                f"изображений, требуется {MNIST_N_PER_CLASS}."
            )
        chosen = rng.choice(idx_all, size=MNIST_N_PER_CLASS, replace=False)
        perm = rng.permutation(MNIST_N_PER_CLASS)
        test_local, train_local = perm[:n_test], perm[n_test:]

        X_train_by_class[digit] = X_all[chosen[train_local]]
        X_test_list.append(X_all[chosen[test_local]])
        y_test_list.append(np.full(len(test_local), digit))
        print(
            f"      {digit}: {len(train_local)} train, {len(test_local)} test "
            f"(из {len(idx_all)} доступных)"
        )

    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    return X_train_by_class, X_test, y_test


def _mnist_vector_to_image(vec: np.ndarray, img_size: int) -> np.ndarray:
    """Reshape(28, 28) + resize -> (img_size, img_size, 1) в [0, 1]."""
    arr28 = vec.reshape(28, 28).astype(np.uint8)
    img = PILImage.fromarray(arr28, mode="L").resize((img_size, img_size))
    return (np.asarray(img, dtype=np.float32) / 255.0)[:, :, None]


def build_mnist_cnn_image_pool_by_class(
    X_train_by_class: Dict[str, np.ndarray], X_test: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Переводит ТЕ ЖЕ 784-мерные векторы (что и у метода сопряжённости и
    классического ML) в изображения (CNN_IMG_SIZE, CNN_IMG_SIZE, 1) через
    reshape(28, 28) + resize — одна и та же исходная пиксельная информация
    для всех трёх подходов. Train — СГРУППИРОВАННЫЙ по классу (нужно
    hyperparameter_tuning.run_data_efficiency_sweep для срезов переменного
    размера), test — объединённый (аналог build_cnn_image_pool_by_class()
    для МРТ, но без чтения файлов с диска — MNIST уже в памяти)."""
    img_size = CNN_IMG_SIZE
    X_train_by_class_img = {
        digit: np.stack([_mnist_vector_to_image(v, img_size) for v in X_cls])
        for digit, X_cls in X_train_by_class.items()
    }
    X_test_img = np.stack([_mnist_vector_to_image(v, img_size) for v in X_test])
    return X_train_by_class_img, X_test_img


def build_mnist_cnn_tensors(
    X_train_by_class: Dict[str, np.ndarray], X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Тонкая обёртка над build_mnist_cnn_image_pool_by_class(): конкатенирует
    train по всем классам — формат, нужный run_cnn_configs()."""
    X_train_by_class_img, X_test_img = build_mnist_cnn_image_pool_by_class(X_train_by_class, X_test)
    classes = list(X_train_by_class.keys())
    X_train_img = np.vstack([X_train_by_class_img[c] for c in classes])
    y_train_img = np.concatenate([np.full(X_train_by_class_img[c].shape[0], c) for c in classes])
    return X_train_img, y_train_img, X_test_img


def run_mnist_experiment() -> None:
    """ТОТ ЖЕ эксперимент, что и run_draft_method_experiment() (SUBSPACE_CONFIGS
    + CLASSICAL_ML_MODELS + CNN_ARCHITECTURES на одном train/test сплите), но
    на MNIST вместо МРТ опухолей мозга — проверка обобщаемости выводов раздела
    11 refactoring_plan.txt на другой домен.

    Отличия от run_draft_method_experiment(), обусловленные природой MNIST
    (не изменения метода/сравнения — та же сетка конфигураций):
      - 10 классов (цифры 0-9) вместо 3 — SubspaceConjugacyClassifier и
        классический ML работают как есть (не зависят от числа классов);
        таблица сравнения печатает mean/min/max по классам вместо 10 колонок
        (полные значения — в JSON-отчёте).
      - N=784 признака (28x28) вместо 65536 (256x256) — рост подпространств
        (B.2, O(M^2*N)) на 1-2 порядка дешевле, поэтому даже "полные"
        конфигурации (baseline, filter_dependent, filter_low_informativeness)
        укладываются в секунды, а не в минуты, как на МРТ.
      - CNN — 1 входной канал (grayscale) вместо 3 (RGB), без горизонтального
        отражения при аугментации (для цифр это меняет смысл класса, в
        отличие от анатомически чаще допустимого отражения МРТ).
      - Датасет скачивается через sklearn.datasets.fetch_openml (первый
        запуск требует интернет; далее используется локальный кэш) вместо
        файлов datasets/{class}_centered/.

    Requires
    --------
    PyTorch (см. run_draft_method_experiment) + интернет при первом запуске
    (для скачивания MNIST; далее данные кэшируются локально).
    """
    if torch is None:
        raise ImportError(
            "Эксперимент 'mnist' требует PyTorch для сравнения с CNN. "
            "Установите: pip install torch (или pip install -e '.[cnn-experiment]')."
        )

    start = time.perf_counter()
    print("=== Тот же эксперимент (метод из черновика vs классический ML vs CNN) на MNIST ===")

    print("\n1. Загрузка MNIST + формирование train/test сплита (80/20 на класс)...")
    X_train_by_class, X_test, y_test = fetch_mnist_split()
    split_summary = {
        digit: {
            "n_train": int(X_train_by_class[digit].shape[0]),
            "n_test": int(np.sum(y_test == digit)),
        }
        for digit in MNIST_CLASSES
    }

    print(
        f"\n2. Подготовка изображений для CNN (reshape 28x28 -> resize "
        f"{CNN_IMG_SIZE}x{CNN_IMG_SIZE}, grayscale)..."
    )
    X_train_img, y_train_img, X_test_img = build_mnist_cnn_tensors(X_train_by_class, X_test)

    print(f"\n3. Метод подпространственной сопряжённости ({len(SUBSPACE_CONFIGS)} конфигураций)...")
    subspace_results = run_subspace_configs(
        X_train_by_class, X_test, y_test, classes=MNIST_CLASSES,
    )

    print(f"\n4. Классические методы ML ({len(CLASSICAL_ML_MODELS)} моделей)...")
    classical_results = run_classical_ml_configs(
        X_train_by_class, X_test, y_test, classes=MNIST_CLASSES,
    )

    print(f"\n5. Свёрточные нейросети (PyTorch, CPU, {len(CNN_ARCHITECTURES)} упрощённые архитектуры)...")
    cnn_results = run_cnn_configs(
        X_train_img, y_train_img, X_test_img, y_test,
        in_channels=1, use_hflip_augmentation=False,
    )

    all_results = subspace_results + classical_results + cnn_results
    print("\n6. Итоговое сравнение:")
    print_comparison_table(all_results, classes=MNIST_CLASSES)
    save_experiment_artifacts(
        all_results,
        dataset_info={
            "name": "MNIST (sklearn.datasets.fetch_openml mnist_784)",
            "classes": MNIST_CLASSES,
            "n_per_class": MNIST_N_PER_CLASS,
            "test_fraction": MNIST_TEST_FRACTION,
            "random_seed": MNIST_RANDOM_SEED,
            "split_summary": split_summary,
        },
        report_filename="mnist_experiment_report.json",
    )

    elapsed = time.perf_counter() - start
    print(f"\n=== Эксперимент завершён за {elapsed / 60:.1f} мин ===")


# ==============================================================================
# Эксперимент "tuned": ЧЕСТНЫЙ подбор гиперпараметров для всех трёх подходов
# (holdout-валидация из train, тест НЕ участвует в подборе — методология
# одинакова для всех трёх сторон, чтобы сравнение оставалось справедливым) +
# кривая эффективности по объёму обучающих данных (data efficiency curve).
#
# Зачем кривая по объёму данных, а не просто ещё один прогон на полных
# данных: Introduction обоих документов theory/ утверждает, что преимущество
# метода сопряжённости — работоспособность на МАЛОМ числе эталонных
# изображений, тогда как CNN/классическому ML нужно больше данных. Прогон на
# фиксированном полном объёме (main.py --experiment draft-method/mnist) это
# не проверяет — там метод сопряжённости УЖЕ уступал по чистой accuracy на
# полных данных (raздел 11 refactoring_plan.txt). Кривая по объёму данных —
# прямая, не подогнанная проверка именно этого тезиса: если он верен, разрыв
# в accuracy между подходами должен СОКРАЩАТЬСЯ (или менять знак) при
# уменьшении обучающей выборки. Результат публикуется как есть, без
# гарантии предопределённого исхода.
# ==============================================================================

TUNING_VAL_FRACTION = 0.2
TUNING_RANDOM_SEED = 42

# Подпространственный метод: ЧЕСТНЫЙ поиск через встроенный в библиотеку
# random_search_classifier (subspace_conjugacy.model_selection.search) —
# StratifiedKFold-кросс-валидация вместо рукописного координатного
# (фаза 1 n_subclasses x growth_strategy -> фаза 2 донастройка фильтров)
# holdout-перебора предыдущей версии этого эксперимента. Два содержательных
# изменения:
#   1. cv=SUBSPACE_TUNING_CV фолдов вместо одного 80/20 holdout-сплита —
#      устойчивее к шуму от конкретного разбиения на малых per-class
#      выборках (~130-160 объектов/класс).
#   2. Фильтры (filter_dependent/filter_low_informativeness) и разбиение на
#      похожие пары (split_correlated_pairs) сэмплируются в СОБСТВЕННЫХ
#      группах пространства поиска (_subspace_param_distributions), каждая
#      со своим, заведомо совместимым с её сокращением данных диапазоном
#      n_subclasses — а не донастройкой поверх уже выбранного максимального
#      n_subclasses=32 из отдельной "фазы 1". В предыдущей версии это
#      оставляло фильтрам нулевой бюджет на МРТ: n_subclasses=32 +
#      split_correlated_pairs=True уже требует M>=64, фильтру уже нечего
#      отсечь — все 3 конфигурации фазы 2 падали с ValueError и молча
#      выпадали из перебора (artifacts/tuned_comparison_mri_report.json —
#      24 кандидата вместо ожидаемых 27, ни одного из фазы 2).
#   sklearn ParameterSampler с list-of-dicts сначала равновероятно выбирает
#   ГРУППУ, затем сэмплирует внутри неё — штатный способ sklearn задать
#   условные гиперпараметры (dependency_threshold имеет смысл только при
#   filter_dependent=True и т.п.), не тратя бюджет n_iter на бессмысленные
#   комбинации. Провалившиеся (ValueError) комбинации получают
#   mean_test_score=NaN (error_score=np.nan в random_search_classifier) и
#   сами исключаются из выбора лучшей — без ручного try/except.
#
#   Бюджет (cv/n_iter) подобран под РЕАЛЬНО измеренную стоимость на МРТ
#   (N=65536): один fit() без split_correlated_pairs, n_subclasses=16,
#   M=160/класс — 365с (B.2 — O(M^2*N), сравните: с split_correlated_pairs=True
#   тот же fit — 32с, т.к. M вдвое меньше). При cv=3/n_iter=24 полный
#   перебор занял бы часы; cv=2/n_iter=16 — компромисс, оставляющий
#   поиск завершаемым за разумное время сессии ценой не самой узкой
#   доверительной оценки. При большем бюджете времени/вычислений оба
#   значения стоит поднять обратно.
SUBSPACE_TUNING_CV = 2
SUBSPACE_TUNING_N_ITER = 16
SUBSPACE_TUNING_RANDOM_STATE = 42


def _subspace_param_distributions(max_n_subclasses_full: int) -> List[Dict[str, Any]]:
    """Пространство поиска SubspaceConjugacyClassifier как список условных
    групп — покрывает ВСЕ гиперпараметры классификатора, кроме reg_param
    (оставлен на дефолте 1e-8 — ни статья, ни README не отмечают его как
    объект подбора, а лишняя ось только развела бы бюджет n_iter тоньше):
    n_subclasses, growth_strategy, freeze_basis_at (2 или "auto" —
    находка №2 сверки со статьёй), filter_dependent (+dependency_threshold),
    filter_low_informativeness (+informativeness_min_fraction),
    split_correlated_pairs (+ОБА подмножества "a"/"b" — предыдущая версия
    фиксировала только "b").

    ``max_n_subclasses_full`` — верхняя граница n_subclasses для группы БЕЗ
    фильтров/разбиения (использует почти весь train); группы, сокращающие M
    (split и/или фильтры), используют пропорционально уменьшенные диапазоны,
    чтобы у фильтров оставался реальный запас векторов для отсечения, а не
    гарантированный ValueError.
    """
    hi_full = max(4, max_n_subclasses_full)
    hi_split = max(4, hi_full // 2)            # split_correlated_pairs делит M пополам
    hi_filtered = max(2, hi_full // 4)         # один фильтр поверх — вдвое меньше запаса
    hi_both_filters = max(2, hi_full // 6)     # оба фильтра статьи вместе
    hi_split_filtered = max(2, hi_full // 10)  # split + оба фильтра одновременно

    growth = ["default", "master"]
    freeze = [2, "auto"]

    return [
        {  # 1. baseline: без фильтров, без разбиения на похожие пары
            "n_subclasses": randint(4, hi_full + 1),
            "growth_strategy": growth,
            "freeze_basis_at": freeze,
            "filter_dependent": [False],
            "filter_low_informativeness": [False],
            "split_correlated_pairs": [False],
        },
        {  # 2. только разбиение на похожие пары (черновик, ОБА подмножества)
            "n_subclasses": randint(4, hi_split + 1),
            "growth_strategy": growth,
            "freeze_basis_at": freeze,
            "filter_dependent": [False],
            "filter_low_informativeness": [False],
            "split_correlated_pairs": [True],
            "correlated_pairs_subset": ["a", "b"],
        },
        {  # 3. только filter_dependent (статья, находка №3)
            "n_subclasses": randint(2, hi_filtered + 1),
            "growth_strategy": growth,
            "freeze_basis_at": freeze,
            "filter_dependent": [True],
            "dependency_threshold": [0.99, 0.995, 0.999, 0.9999],
            "filter_low_informativeness": [False],
            "split_correlated_pairs": [False],
        },
        {  # 4. только filter_low_informativeness (статья, находка №5)
            "n_subclasses": randint(2, hi_filtered + 1),
            "growth_strategy": growth,
            "freeze_basis_at": freeze,
            "filter_dependent": [False],
            "filter_low_informativeness": [True],
            "informativeness_min_fraction": [0.3, 0.4, 0.5, 0.6, 0.7],
            "split_correlated_pairs": [False],
        },
        {  # 5. оба фильтра статьи вместе, без разбиения
            "n_subclasses": randint(2, hi_both_filters + 1),
            "growth_strategy": growth,
            "freeze_basis_at": freeze,
            "filter_dependent": [True],
            "dependency_threshold": [0.99, 0.999],
            "filter_low_informativeness": [True],
            "informativeness_min_fraction": [0.4, 0.5, 0.6],
            "split_correlated_pairs": [False],
        },
        {  # 6. "кухонная раковина": разбиение + оба фильтра статьи вместе
            "n_subclasses": randint(2, hi_split_filtered + 1),
            "growth_strategy": growth,
            "freeze_basis_at": freeze,
            "filter_dependent": [True],
            "dependency_threshold": [0.99, 0.999],
            "filter_low_informativeness": [True],
            "informativeness_min_fraction": [0.4, 0.5, 0.6],
            "split_correlated_pairs": [True],
            "correlated_pairs_subset": ["a", "b"],
        },
    ]

# Классический ML: расширенная сетка на модель — полный GridSearchCV с
# k-fold был бы дороже без явной необходимости, holdout-валидация уже даёт
# честный, воспроизводимый выбор при единой методологии со остальными двумя
# подходами (та же валидационная выборка, что и у подпространственного
# метода и CNN).
CLASSICAL_ML_TUNING_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "LogisticRegression": {"pca_components": [30, 50, 75, 100, 150], "C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "LinearSVM": {"pca_components": [30, 50, 75, 100, 150], "C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "RandomForest": {
        "pca_components": [50, 100], "n_estimators": [100, 200, 300, 500], "max_depth": [None, 10, 20, 30],
    },
    "kNN_k5": {"pca_components": [30, 50, 75, 100, 150], "n_neighbors": [3, 5, 7, 9, 11, 15]},
}

CNN_TUNING_EPOCHS = 15  # меньше CNN_EPOCHS — тюнинг-прогон, не финальная оценка
CNN_TUNING_LEARNING_RATES = [1e-3, 5e-4, 3e-4, 1e-4]
CNN_TUNING_WEIGHT_DECAYS = [1e-4, 1e-5, 0.0]

# Объёмы обучающей выборки на класс для кривой эффективности — плотнее
# предыдущей версии (5 -> 8 точек) для более гладкой кривой в отчёте.
DATA_EFFICIENCY_SIZES = [10, 20, 30, 40, 60, 80, 120, 160]


def _mean_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    report = evaluate_classifier(y_true, y_pred)
    return float(np.mean(list(report["per_class_accuracy"].values())))


def _max_feasible_n_subclasses(n_per_class: int, split_correlated_pairs: bool) -> int:
    """Максимальный n_subclasses, для которого FursovClusterer.fit() не
    бросит ValueError на классе размера ``n_per_class`` (после опционального
    вдвое-сокращения фазой 0c) — см. FursovClusterer._validate_input:
    n_subclasses*2 <= M."""
    m = n_per_class // 2 if split_correlated_pairs else n_per_class
    return max(2, m // 2)


def carve_validation_split(
    pool_by_class: Dict[str, np.ndarray], val_fraction: float = TUNING_VAL_FRACTION,
    seed: int = TUNING_RANDOM_SEED,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Отделяет holdout-валидацию от train НЕЗАВИСИМО для каждого класса
    (сохраняет баланс классов). Работает и для векторов (M, N), и для
    изображений (M, H, W, C) — стекуется по оси 0 в обоих случаях.

    Returns
    -------
    subtrain_by_class : тот же формат, что и pool_by_class, но без валидации.
    X_val, y_val : объединённая по всем классам валидационная выборка.
    """
    rng = np.random.default_rng(seed)
    subtrain_by_class: Dict[str, np.ndarray] = {}
    X_val_list, y_val_list = [], []
    for cls, X in pool_by_class.items():
        n = X.shape[0]
        n_val = max(1, int(round(n * val_fraction)))
        perm = rng.permutation(n)
        val_idx, sub_idx = perm[:n_val], perm[n_val:]
        subtrain_by_class[cls] = X[sub_idx]
        X_val_list.append(X[val_idx])
        y_val_list.append(np.full(n_val, cls))
    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list)
    return subtrain_by_class, X_val, y_val


def tune_subspace_hyperparameters(
    vector_train_by_class: Dict[str, np.ndarray], classes: List[str],
) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
    """Честный подбор гиперпараметров SubspaceConjugacyClassifier через
    ВСТРОЕННЫЙ в библиотеку random_search_classifier (StratifiedKFold-кросс-
    валидация) — см. комментарий у SUBSPACE_TUNING_*/_subspace_param_
    distributions выше за тем, что изменилось относительно предыдущей
    (рукописной, holdout, двухфазной) версии и почему.

    В отличие от предыдущей версии не делает отдельный holdout-сплит —
    cv-фолды random_search_classifier сами берут на себя роль валидации,
    честно на ВСЕХ vector_train_by_class (тест по-прежнему не участвует,
    он используется только позже, в _fit_eval_full_data).

    Returns
    -------
    best_params : dict
        search.best_params_ — передаётся напрямую в SubspaceConjugacyClassifier(**...).
    best_val_accuracy : float
        search.best_score_ — средняя accuracy по SUBSPACE_TUNING_CV фолдам
        (не одно holdout-число, как в предыдущей версии).
    candidates : list
        Все SUBSPACE_TUNING_N_ITER опробованные конфигурации (через
        summarize_search_results) — для отчёта/воспроизводимости.
    """
    X_train = np.vstack([vector_train_by_class[c] for c in classes])
    y_train = np.concatenate([np.full(vector_train_by_class[c].shape[0], c) for c in classes])

    min_class_size = min(vector_train_by_class[c].shape[0] for c in classes)
    # Внутри одного cv-фолда обучающая часть класса — примерно (cv-1)/cv от
    # полного train по классу (StratifiedKFold делит без замены).
    fold_train_size = int(min_class_size * (SUBSPACE_TUNING_CV - 1) / SUBSPACE_TUNING_CV)
    max_n_subclasses_full = _max_feasible_n_subclasses(fold_train_size, split_correlated_pairs=False)

    param_distributions = _subspace_param_distributions(max_n_subclasses_full)
    print(
        f"      Пространство поиска: {len(param_distributions)} условных групп "
        f"(baseline/split/filter_dependent/filter_low_informativeness/оба фильтра/"
        f"split+оба фильтра), n_iter={SUBSPACE_TUNING_N_ITER}, cv={SUBSPACE_TUNING_CV}, "
        f"max_n_subclasses(baseline)={max_n_subclasses_full}."
    )

    search = random_search_classifier(
        X_train, y_train,
        param_distributions=param_distributions,
        n_iter=SUBSPACE_TUNING_N_ITER,
        cv=SUBSPACE_TUNING_CV,
        scoring="accuracy",
        random_state=SUBSPACE_TUNING_RANDOM_STATE,
    )

    candidates = summarize_search_results(search, top_n=len(search.cv_results_["params"]))
    n_failed = sum(1 for c in candidates if np.isnan(c["mean_test_score"]))
    print(
        f"      Готово: {len(candidates)} конфигураций опробовано, "
        f"{n_failed} неприменимы (ValueError -> NaN), "
        f"лучшая cv-accuracy={search.best_score_:.3f}."
    )
    return search.best_params_, float(search.best_score_), candidates


def _make_classical_estimator(name: str, **hyperparams: Any):
    """Строит sklearn-классификатор ``name`` (без PCA-шага) с заданными
    гиперпараметрами — общая часть tune_classical_ml_models() и
    run_data_efficiency_sweep()."""
    if name == "LogisticRegression":
        return LogisticRegression(
            max_iter=2000, random_state=CLASSICAL_ML_RANDOM_SEED, C=hyperparams["C"],
        )
    if name == "LinearSVM":
        return SVC(kernel="linear", random_state=CLASSICAL_ML_RANDOM_SEED, C=hyperparams["C"])
    if name == "RandomForest":
        return RandomForestClassifier(
            random_state=CLASSICAL_ML_RANDOM_SEED, n_jobs=-1,
            n_estimators=hyperparams["n_estimators"], max_depth=hyperparams["max_depth"],
        )
    if name == "kNN_k5":
        return KNeighborsClassifier(n_neighbors=hyperparams["n_neighbors"])
    raise ValueError(f"Неизвестная классическая модель: {name}")


def tune_classical_ml_models(
    vector_train_by_class: Dict[str, np.ndarray], classes: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Честный подбор PCA-компонент + гиперпараметров КАЖДОЙ из
    CLASSICAL_ML_MODELS через ТОТ ЖЕ holdout-сплит, что и у подпространственного
    метода (единая методология).

    Returns
    -------
    best_by_model : Dict[str, {"hyperparams": dict, "val_accuracy": float,
        "all_candidates": list}] — все опробованные комбинации сохраняются в
        "all_candidates" (нужно для подробного отчёта — reporting/word_report.py
        показывает размер и распределение результатов всей сетки, не только
        победителя).
    """
    subtrain_by_class, X_val, y_val = carve_validation_split(vector_train_by_class)
    X_subtrain = np.vstack([subtrain_by_class[c] for c in classes]) / 255.0
    y_subtrain = np.concatenate([np.full(subtrain_by_class[c].shape[0], c) for c in classes])
    X_val_scaled = X_val / 255.0

    best_by_model: Dict[str, Dict[str, Any]] = {}
    for name, grid in CLASSICAL_ML_TUNING_GRIDS.items():
        keys = list(grid.keys())
        candidates = []
        for combo in itertools.product(*grid.values()):
            hyperparams = dict(zip(keys, combo))
            pca_components = min(
                hyperparams["pca_components"], X_subtrain.shape[0] - 1, X_subtrain.shape[1],
            )
            pipeline = Pipeline([
                ("pca", PCA(n_components=pca_components, random_state=CLASSICAL_ML_RANDOM_SEED)),
                ("clf", _make_classical_estimator(
                    name, **{k: v for k, v in hyperparams.items() if k != "pca_components"}
                )),
            ])
            pipeline.fit(X_subtrain, y_subtrain)
            val_accuracy = _mean_accuracy(y_val, pipeline.predict(X_val_scaled))
            candidates.append({
                "hyperparams": {**hyperparams, "pca_components": pca_components},
                "val_accuracy": val_accuracy,
            })
        best = dict(max(candidates, key=lambda c: c["val_accuracy"]))
        best["all_candidates"] = candidates
        best_by_model[name] = best
        print(
            f"      [{name}] перебрано {len(candidates)} конфигураций, лучшая: "
            f"{best['hyperparams']}, val_accuracy={best['val_accuracy']:.3f}"
        )

    return best_by_model


def tune_cnn_hyperparameters(
    image_train_by_class: Dict[str, np.ndarray], classes: List[str],
    in_channels: int, use_hflip_augmentation: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Честный подбор архитектуры + learning rate через ТОТ ЖЕ holdout-сплит
    (единая методология с двумя другими подходами). Использует
    CNN_TUNING_EPOCHS (меньше CNN_EPOCHS) — только для ранжирования
    конфигураций; финальная оценка лучшей конфигурации обучается заново на
    полном train с CNN_EPOCHS (см. run_tuned_comparison_experiment)."""
    subtrain_by_class, X_val, y_val = carve_validation_split(image_train_by_class)
    classes_sorted = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}

    X_subtrain = np.concatenate([subtrain_by_class[c] for c in classes], axis=0)
    y_subtrain_labels = np.concatenate(
        [np.full(subtrain_by_class[c].shape[0], c) for c in classes]
    )
    y_subtrain_idx = np.array([class_to_idx[c] for c in y_subtrain_labels])

    candidates = []
    for arch_name, model_cls in CNN_ARCHITECTURES.items():
        for lr in CNN_TUNING_LEARNING_RATES:
            for wd in CNN_TUNING_WEIGHT_DECAYS:
                outcome = train_cnn_model(
                    model_cls, X_subtrain, y_subtrain_idx, X_val,
                    num_classes=len(classes_sorted), in_channels=in_channels,
                    use_hflip_augmentation=use_hflip_augmentation,
                    epochs=CNN_TUNING_EPOCHS, learning_rate=lr, weight_decay=wd,
                )
                y_pred = np.array([classes_sorted[i] for i in outcome["test_pred_idx"]])
                val_accuracy = _mean_accuracy(y_val, y_pred)
                candidates.append({
                    "architecture": arch_name, "learning_rate": lr, "weight_decay": wd,
                    "val_accuracy": val_accuracy,
                })
                print(f"      [{arch_name}, lr={lr}, weight_decay={wd}] val_accuracy={val_accuracy:.3f}")

    best = max(candidates, key=lambda c: c["val_accuracy"])
    return best, candidates


def run_data_efficiency_sweep(
    vector_train_by_class: Dict[str, np.ndarray], image_train_by_class: Dict[str, np.ndarray],
    classes: List[str], X_test_vec: np.ndarray, X_test_img: np.ndarray, y_test: np.ndarray,
    best_subspace_params: Dict[str, Any], best_classical_name: str,
    best_classical_hyperparams: Dict[str, Any], best_cnn_arch: str, best_cnn_lr: float,
    in_channels: int, use_hflip_augmentation: bool, best_cnn_weight_decay: float = 1e-4,
) -> List[Dict[str, Any]]:
    """Обучает все ТРИ подхода (с уже подобранными гиперпараметрами) на
    срезах обучающей выборки нарастающего размера (DATA_EFFICIENCY_SIZES) и
    оценивает на ОДНОМ И ТОМ ЖЕ полном тесте — прямая проверка тезиса
    theory/ о работоспособности метода сопряжённости на малых выборках.

    n_subclasses и pca_components подпространственного/классического метода
    АДАПТИВНО уменьшаются для малых срезов (иначе fit() бросил бы ValueError
    при нехватке векторов/сэмплов) — capped, не переподобранные заново:
    это тот же принцип гиперпараметров, просто применённый к меньшему
    объёму данных, а не отдельный тюнинг на каждый размер (что размыло бы
    сравнение "тот же метод, меньше данных").
    """
    max_available = min(X.shape[0] for X in vector_train_by_class.values())
    sizes = [s for s in DATA_EFFICIENCY_SIZES if s <= max_available]
    classes_sorted = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}

    records = []
    for k in sizes:
        print(f"\n   -- Объём обучающей выборки: {k}/класс ({k * len(classes)} всего) --")
        vec_k = {c: vector_train_by_class[c][:k] for c in classes}
        img_k = {c: image_train_by_class[c][:k] for c in classes}
        X_train = np.vstack([vec_k[c] for c in classes])
        y_train = np.concatenate([np.full(vec_k[c].shape[0], c) for c in classes])

        # --- Сопряжённость ---
        # n_subclasses ограничен исходя из split_correlated_pairs, но фильтры
        # статьи (если оказались в best_subspace_params по итогам фазы 2
        # тюнинга) могут сократить M ЕЩЁ СИЛЬНЕЕ на малых срезах —
        # непредсказуемо заранее (зависит от содержимого конкретного среза).
        # При ValueError прогрессивно уменьшаем n_subclasses, а не падаем —
        # честный перебор может упереться в границу применимости метода на
        # малых данных, это результат сам по себе, а не баг.
        n_subclasses_used = min(
            best_subspace_params["n_subclasses"],
            _max_feasible_n_subclasses(k, best_subspace_params.get("split_correlated_pairs", False)),
        )
        acc_sub, t_sub = None, None
        while n_subclasses_used >= 2:
            params_k = {**best_subspace_params, "n_subclasses": n_subclasses_used}
            try:
                clf = SubspaceConjugacyClassifier(**params_k)
                t0 = time.perf_counter()
                clf.fit(X_train, y_train)
                t_sub = time.perf_counter() - t0
            except ValueError:
                n_subclasses_used -= 1
                continue
            acc_sub = _mean_accuracy(y_test, clf.predict(X_test_vec))
            break
        if acc_sub is None:
            print(f"      Сопряжённость: НЕПРИМЕНИМО при {k}/класс (недостаточно векторов даже для n_subclasses=2).")
            acc_sub, t_sub, n_subclasses_used = float("nan"), 0.0, 0
        else:
            print(f"      Сопряжённость (n_subclasses={n_subclasses_used}): accuracy={acc_sub:.3f}, time={t_sub:.1f}s")

        # --- Классический ML ---
        pca_components_used = min(
            best_classical_hyperparams["pca_components"], X_train.shape[0] - 1, X_train.shape[1],
        )
        pipeline = Pipeline([
            ("pca", PCA(n_components=pca_components_used, random_state=CLASSICAL_ML_RANDOM_SEED)),
            ("clf", _make_classical_estimator(
                best_classical_name,
                **{k2: v for k2, v in best_classical_hyperparams.items() if k2 != "pca_components"},
            )),
        ])
        X_train_scaled = X_train / 255.0
        t0 = time.perf_counter()
        pipeline.fit(X_train_scaled, y_train)
        t_cls = time.perf_counter() - t0
        acc_cls = _mean_accuracy(y_test, pipeline.predict(X_test_vec / 255.0))
        print(
            f"      {best_classical_name} (PCA-{pca_components_used}): "
            f"accuracy={acc_cls:.3f}, time={t_cls:.1f}s"
        )

        # --- CNN ---
        X_train_img = np.concatenate([img_k[c] for c in classes], axis=0)
        y_train_img_labels = np.concatenate([np.full(img_k[c].shape[0], c) for c in classes])
        y_train_img_idx = np.array([class_to_idx[c] for c in y_train_img_labels])
        outcome = train_cnn_model(
            CNN_ARCHITECTURES[best_cnn_arch], X_train_img, y_train_img_idx, X_test_img,
            num_classes=len(classes_sorted), in_channels=in_channels,
            use_hflip_augmentation=use_hflip_augmentation,
            epochs=CNN_EPOCHS, learning_rate=best_cnn_lr, weight_decay=best_cnn_weight_decay,
        )
        y_pred_cnn = np.array([classes_sorted[i] for i in outcome["test_pred_idx"]])
        acc_cnn = _mean_accuracy(y_test, y_pred_cnn)
        print(
            f"      {best_cnn_arch} (lr={best_cnn_lr}): "
            f"accuracy={acc_cnn:.3f}, time={outcome['training_time_seconds']:.1f}s"
        )

        records.append({
            "n_train_per_class": k,
            "n_train_total": int(X_train.shape[0]),
            "subspace": {
                "n_subclasses": n_subclasses_used, "mean_accuracy": acc_sub, "fit_time_seconds": t_sub,
            },
            "classical_ml": {
                "model": best_classical_name, "pca_components": pca_components_used,
                "mean_accuracy": acc_cls, "fit_time_seconds": t_cls,
            },
            "cnn": {
                "architecture": best_cnn_arch, "learning_rate": best_cnn_lr,
                "mean_accuracy": acc_cnn, "training_time_seconds": outcome["training_time_seconds"],
            },
        })
    return records


def print_data_efficiency_table(records: List[Dict[str, Any]]) -> None:
    header = (
        f"{'N train/класс':>14} | {'Сопряжённость (acc/врем)':>26} | "
        f"{'Классич. ML (acc/врем)':>26} | {'CNN (acc/врем)':>26}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in records:
        s, c, n = r["subspace"], r["classical_ml"], r["cnn"]
        row = (
            f"{r['n_train_per_class']:>14} | "
            f"{s['mean_accuracy']:.3f} / {s['fit_time_seconds']:>5.1f}s".rjust(26) + " | "
            + f"{c['mean_accuracy']:.3f} / {c['fit_time_seconds']:>5.1f}s".rjust(26) + " | "
            + f"{n['mean_accuracy']:.3f} / {n['training_time_seconds']:>5.1f}s".rjust(26)
        )
        print(row)


def print_tuned_verdict(
    records: List[Dict[str, Any]],
    full_data_accuracy: Dict[str, float], full_data_time: Dict[str, float],
) -> None:
    """Печатает честную сводку по фактическим числам — БЕЗ предопределённого
    вывода: какая сторона выигрывает по accuracy/скорости на полных и на
    минимальных данных решают сами цифры, а не заранее заданный нарратив."""
    print("\n=== Сводка (по фактическим числам, без подгонки под ожидания) ===")
    print("\nПолный объём обучающих данных (после подбора гиперпараметров):")
    for name in ("subspace", "classical_ml", "cnn"):
        print(f"   {name:<14}: accuracy={full_data_accuracy[name]:.3f}, время={full_data_time[name]:.1f}s")

    smallest = records[0]
    print(f"\nМинимальный проверенный объём ({smallest['n_train_per_class']}/класс):")
    for name in ("subspace", "classical_ml", "cnn"):
        rec = smallest[name]
        acc = rec["mean_accuracy"]
        t = rec.get("fit_time_seconds", rec.get("training_time_seconds"))
        print(f"   {name:<14}: accuracy={acc:.3f}, время={t:.1f}s")

    print("\nИзменение разрыва accuracy (сопряжённость - конкурент) при уменьшении данных:")
    largest = records[-1]
    for name in ("classical_ml", "cnn"):
        gap_full = largest["subspace"]["mean_accuracy"] - largest[name]["mean_accuracy"]
        gap_small = smallest["subspace"]["mean_accuracy"] - smallest[name]["mean_accuracy"]
        direction = "сокращается" if gap_small > gap_full else "растёт" if gap_small < gap_full else "не меняется"
        print(
            f"   vs {name}: на {largest['n_train_per_class']}/класс = {gap_full:+.3f}, "
            f"на {smallest['n_train_per_class']}/класс = {gap_small:+.3f} (разрыв {direction})"
        )


def _fit_eval_full_data(
    vector_train_by_class: Dict[str, np.ndarray], image_train_by_class: Dict[str, np.ndarray],
    classes: List[str], X_test_vec: np.ndarray, X_test_img: np.ndarray, y_test: np.ndarray,
    best_subspace_params: Dict[str, Any], best_classical_name: str,
    best_classical_hyperparams: Dict[str, Any], best_cnn_arch: str, best_cnn_lr: float,
    in_channels: int, use_hflip_augmentation: bool, best_cnn_weight_decay: float = 1e-4,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Финальная оценка ТЮНИНГОВАННЫХ конфигураций на ПОЛНОМ train (тест не
    участвовал ни в подборе гиперпараметров, ни здесь — используется только
    для итоговой оценки). Возвращает {подход: accuracy}, {подход: время}."""
    X_train = np.vstack([vector_train_by_class[c] for c in classes])
    y_train = np.concatenate([np.full(vector_train_by_class[c].shape[0], c) for c in classes])

    clf = SubspaceConjugacyClassifier(**best_subspace_params)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_sub = time.perf_counter() - t0
    acc_sub = _mean_accuracy(y_test, clf.predict(X_test_vec))
    print(f"   Сопряжённость (tuned): accuracy={acc_sub:.3f}, fit_time={t_sub:.1f}s")

    pipeline = Pipeline([
        ("pca", PCA(
            n_components=best_classical_hyperparams["pca_components"],
            random_state=CLASSICAL_ML_RANDOM_SEED,
        )),
        ("clf", _make_classical_estimator(
            best_classical_name,
            **{k: v for k, v in best_classical_hyperparams.items() if k != "pca_components"},
        )),
    ])
    X_train_scaled = X_train / 255.0
    t0 = time.perf_counter()
    pipeline.fit(X_train_scaled, y_train)
    t_cls = time.perf_counter() - t0
    acc_cls = _mean_accuracy(y_test, pipeline.predict(X_test_vec / 255.0))
    print(f"   {best_classical_name} (tuned): accuracy={acc_cls:.3f}, fit_time={t_cls:.1f}s")

    classes_sorted = sorted(classes)
    class_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    X_train_img = np.concatenate([image_train_by_class[c] for c in classes], axis=0)
    y_train_img_labels = np.concatenate(
        [np.full(image_train_by_class[c].shape[0], c) for c in classes]
    )
    y_train_img_idx = np.array([class_to_idx[c] for c in y_train_img_labels])
    outcome = train_cnn_model(
        CNN_ARCHITECTURES[best_cnn_arch], X_train_img, y_train_img_idx, X_test_img,
        num_classes=len(classes_sorted), in_channels=in_channels,
        use_hflip_augmentation=use_hflip_augmentation,
        epochs=CNN_EPOCHS, learning_rate=best_cnn_lr, weight_decay=best_cnn_weight_decay,
    )
    y_pred_cnn = np.array([classes_sorted[i] for i in outcome["test_pred_idx"]])
    acc_cnn = _mean_accuracy(y_test, y_pred_cnn)
    t_cnn = outcome["training_time_seconds"]
    print(f"   {best_cnn_arch} (tuned): accuracy={acc_cnn:.3f}, time={t_cnn:.1f}s")

    return (
        {"subspace": acc_sub, "classical_ml": acc_cls, "cnn": acc_cnn},
        {"subspace": t_sub, "classical_ml": t_cls, "cnn": t_cnn},
    )


def run_tuned_comparison_experiment(dataset: str = "mri") -> None:
    """Честный подбор гиперпараметров для ВСЕХ трёх подходов + кривая
    эффективности по объёму обучающих данных, на МРТ (``dataset="mri"``) или
    MNIST (``dataset="mnist"``).

    Методология (одинакова для всех трёх подходов и обоих датасетов):
      1. holdout-валидация (20% train, TUNING_VAL_FRACTION) для подбора
         гиперпараметров — тест НЕ используется на этом шаге.
      2. Лучшая по val_accuracy конфигурация каждого подхода переобучается на
         ПОЛНОМ train и оценивается на test — это единственный момент,
         когда test вообще используется.
      3. Та же тройка (уже с фиксированными гиперпараметрами) переобучается
         на срезах train нарастающего размера (DATA_EFFICIENCY_SIZES) —
         прямая проверка тезиса theory/ о работоспособности метода
         сопряжённости на малых выборках (см. комментарий у блока
         "Эксперимент 'tuned'" выше).

    Requires
    --------
    PyTorch (см. run_draft_method_experiment).
    """
    if torch is None:
        raise ImportError(
            "Эксперимент 'tuned' требует PyTorch для сравнения с CNN. "
            "Установите: pip install torch (или pip install -e '.[cnn-experiment]')."
        )
    if dataset not in ("mri", "mnist"):
        raise ValueError(f"dataset должен быть 'mri' или 'mnist', получено {dataset!r}.")

    start = time.perf_counter()
    print(f"=== Честный подбор гиперпараметров + кривая эффективности по данным ({dataset}) ===")

    if dataset == "mri":
        split = prepare_centered_split()
        vector_train_by_class, X_test_vec, y_test = build_subspace_vectors(split)
        image_train_by_class, X_test_img, y_test_img = build_cnn_image_pool_by_class(split)
        np.testing.assert_array_equal(y_test, y_test_img)
        classes = CENTERED_CLASSES
        in_channels, use_hflip = 3, True
        dataset_info = {
            "name": "МРТ (datasets/{class}_centered/)", "classes": classes,
            "n_per_class": CENTERED_N_PER_CLASS, "test_fraction": CENTERED_TEST_FRACTION,
        }
        report_filename = "tuned_comparison_mri_report.json"
    else:
        vector_train_by_class, X_test_vec, y_test = fetch_mnist_split()
        image_train_by_class, X_test_img = build_mnist_cnn_image_pool_by_class(
            vector_train_by_class, X_test_vec,
        )
        classes = MNIST_CLASSES
        in_channels, use_hflip = 1, False
        dataset_info = {
            "name": "MNIST (sklearn.datasets.fetch_openml mnist_784)", "classes": classes,
            "n_per_class": MNIST_N_PER_CLASS, "test_fraction": MNIST_TEST_FRACTION,
        }
        report_filename = "tuned_comparison_mnist_report.json"

    print("\n1. Подбор гиперпараметров подпространственного метода (holdout-валидация)...")
    best_subspace_params, best_subspace_val_acc, subspace_candidates = tune_subspace_hyperparameters(
        vector_train_by_class, classes,
    )
    print(
        f"   Лучшая: n_subclasses={best_subspace_params['n_subclasses']}, "
        f"growth_strategy={best_subspace_params['growth_strategy']}, "
        f"val_accuracy={best_subspace_val_acc:.3f}"
    )

    print("\n2. Подбор гиперпараметров классических ML-моделей (тот же holdout-сплит)...")
    best_classical_by_model = tune_classical_ml_models(vector_train_by_class, classes)
    best_classical_name = max(
        best_classical_by_model, key=lambda name: best_classical_by_model[name]["val_accuracy"],
    )
    best_classical_hyperparams = best_classical_by_model[best_classical_name]["hyperparams"]
    print(
        f"   Лучшая модель: {best_classical_name} {best_classical_hyperparams}, "
        f"val_accuracy={best_classical_by_model[best_classical_name]['val_accuracy']:.3f}"
    )

    print("\n3. Подбор гиперпараметров CNN (тот же holdout-сплит)...")
    best_cnn, cnn_candidates = tune_cnn_hyperparameters(
        image_train_by_class, classes, in_channels, use_hflip,
    )
    print(
        f"   Лучшая: {best_cnn['architecture']}, lr={best_cnn['learning_rate']}, "
        f"weight_decay={best_cnn['weight_decay']}, val_accuracy={best_cnn['val_accuracy']:.3f}"
    )

    print("\n4. Финальная оценка тюнингованных конфигураций на полном train (test впервые используется здесь)...")
    full_data_accuracy, full_data_time = _fit_eval_full_data(
        vector_train_by_class, image_train_by_class, classes, X_test_vec, X_test_img, y_test,
        best_subspace_params, best_classical_name, best_classical_hyperparams,
        best_cnn["architecture"], best_cnn["learning_rate"], in_channels, use_hflip,
        best_cnn_weight_decay=best_cnn["weight_decay"],
    )

    print("\n5. Кривая эффективности по объёму обучающих данных...")
    sweep_records = run_data_efficiency_sweep(
        vector_train_by_class, image_train_by_class, classes, X_test_vec, X_test_img, y_test,
        best_subspace_params, best_classical_name, best_classical_hyperparams,
        best_cnn["architecture"], best_cnn["learning_rate"], in_channels, use_hflip,
        best_cnn_weight_decay=best_cnn["weight_decay"],
    )

    print("\n6. Итоговый отчёт:")
    print_data_efficiency_table(sweep_records)
    print_tuned_verdict(sweep_records, full_data_accuracy, full_data_time)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_info,
                "tuning": {
                    "val_fraction": TUNING_VAL_FRACTION,
                    "random_seed": TUNING_RANDOM_SEED,
                    "subspace_candidates": subspace_candidates,
                    "best_subspace_params": best_subspace_params,
                    "classical_ml_candidates_by_model": best_classical_by_model,
                    "best_classical_name": best_classical_name,
                    "cnn_candidates": cnn_candidates,
                    "best_cnn": best_cnn,
                },
                "full_data_evaluation": {
                    "accuracy": full_data_accuracy, "time_seconds": full_data_time,
                },
                "data_efficiency_sweep": sweep_records,
            },
            f, ensure_ascii=False, indent=2, default=str,
        )
    print(f"\n   Отчёт эксперимента (JSON): {report_path}")

    elapsed = time.perf_counter() - start
    print(f"\n=== Эксперимент завершён за {elapsed / 60:.1f} мин ===")


def main() -> None:
    """Точка входа: выбирает эксперимент через --experiment (см. docstring модуля)."""
    parser = argparse.ArgumentParser(
        description="Эксперименты поверх subspace_conjugacy (см. docstring модуля main.py)."
    )
    parser.add_argument(
        "--experiment",
        choices=["article", "draft-method", "mnist", "tuned"],
        default="article",
        help=(
            "'article' (по умолчанию) — эксперименты 2-3 опубликованной статьи "
            "на Kaggle archive/ (BRAIN_MRI_ARCHIVE_ROOT). 'draft-method' — метод "
            "из черновика (theory/Макет новой статьи.docx) vs классический ML vs "
            "CNN на datasets/{class}_centered/. 'mnist' — тот же эксперимент, что "
            "'draft-method', но на MNIST (10 классов цифр). 'tuned' — честный "
            "подбор гиперпараметров для всех трёх подходов (holdout-валидация) + "
            "кривая эффективности по объёму обучающих данных, см. --dataset."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["mri", "mnist"],
        default="mri",
        help="Только для --experiment tuned: 'mri' (по умолчанию) или 'mnist'.",
    )
    args = parser.parse_args()

    if args.experiment == "article":
        run_article_experiment()
    elif args.experiment == "draft-method":
        run_draft_method_experiment()
    elif args.experiment == "mnist":
        run_mnist_experiment()
    else:
        run_tuned_comparison_experiment(dataset=args.dataset)


if __name__ == "__main__":
    main()
