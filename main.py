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
        FC-слоя."""

        def __init__(self, num_classes: int, img_size: int = CNN_IMG_SIZE) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(4),
            )
            flat_dim = 8 * (img_size // 4) ** 2
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, num_classes))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.features(x))

    class SmallCNN(nn.Module):
        """Простая CNN: 2 свёрточных блока (16, 32 канала), линейный
        классификатор напрямую — без скрытого FC-слоя, без BatchNorm/Dropout."""

        def __init__(self, num_classes: int, img_size: int = CNN_IMG_SIZE) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
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


def build_cnn_tensors(
    split: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Загружает изображения как RGB, приводит к (CNN_IMG_SIZE, CNN_IMG_SIZE, 3)
    в [0, 1] и разрезает на train/test ТЕМИ ЖЕ индексами train_idx/test_idx,
    что и build_subspace_vectors() — тестовая выборка физически совпадает
    с той, что видит метод подпространств.

    Читает CNN_IMG_SIZE из глобальной области видимости НА МОМЕНТ ВЫЗОВА (не
    как значение по умолчанию параметра — то фиксировалось бы при определении
    функции и разошлось бы с реальным размером, который видят CNN-модели,
    если константу переопределить после импорта модуля)."""
    img_size = CNN_IMG_SIZE
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    for cls, info in split.items():
        imgs = np.stack([
            np.asarray(
                PILImage.open(p).convert("RGB").resize((img_size, img_size)),
                dtype=np.float32,
            ) / 255.0
            for p in info["paths"]
        ])
        X_train_list.append(imgs[info["train_idx"]])
        y_train_list.append(np.full(len(info["train_idx"]), cls))
        X_test_list.append(imgs[info["test_idx"]])
        y_test_list.append(np.full(len(info["test_idx"]), cls))
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    return X_train, y_train, X_test, y_test


def run_subspace_configs(
    X_train_by_class: Dict[str, np.ndarray], X_test: np.ndarray, y_test: np.ndarray,
) -> List[Dict[str, Any]]:
    """Обучает и оценивает все SUBSPACE_CONFIGS на одном и том же train/test."""
    X_train = np.vstack([X_train_by_class[c] for c in CENTERED_CLASSES])
    y_train = np.concatenate(
        [np.full(X_train_by_class[c].shape[0], c) for c in CENTERED_CLASSES]
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
                for cls in CENTERED_CLASSES
            }
        else:
            n_effective_train = {
                cls: int(X_train_by_class[cls].shape[0]) for cls in CENTERED_CLASSES
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
) -> List[Dict[str, Any]]:
    """Обучает и оценивает CLASSICAL_ML_MODELS (sklearn) на той же
    векторизации (65536 признаков), что и метод сопряжённости — с PCA-
    понижением размерности до CLASSICAL_ML_PCA_COMPONENTS (см. комментарий
    у CLASSICAL_ML_MODELS) и масштабированием в [0, 1] (/255)."""
    X_train = np.vstack([X_train_by_class[c] for c in CENTERED_CLASSES]) / 255.0
    y_train = np.concatenate(
        [np.full(X_train_by_class[c].shape[0], c) for c in CENTERED_CLASSES]
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
) -> Dict[str, Any]:
    """Обучает одну CNN-архитектуру на CPU, возвращает метрики и предсказания.

    X_* — (N, H, W, 3) float32 в [0, 1]; y_train_idx — целочисленные метки.
    Единственная аугментация — случайное горизонтальное отражение (p=0.5)
    каждого объекта батча на train, чтобы не усложнять сравнение архитектур
    лишними гиперпараметрами.
    """
    torch.manual_seed(CNN_RANDOM_SEED)

    model = model_cls(num_classes=num_classes, img_size=CNN_IMG_SIZE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    X_train_t = torch.from_numpy(X_train.transpose(0, 3, 1, 2)).float()
    y_train_t = torch.from_numpy(y_train_idx).long()
    X_test_t = torch.from_numpy(X_test.transpose(0, 3, 1, 2)).float()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CNN_LEARNING_RATE, weight_decay=CNN_WEIGHT_DECAY,
    )
    loss_fn = nn.CrossEntropyLoss()

    n_train = X_train_t.shape[0]
    history = []
    t0 = time.perf_counter()
    for epoch in range(CNN_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, CNN_BATCH_SIZE):
            idx = perm[start:start + CNN_BATCH_SIZE]
            xb, yb = X_train_t[idx].clone(), y_train_t[idx]
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
) -> List[Dict[str, Any]]:
    """Обучает и оценивает все CNN_ARCHITECTURES на одном и том же train/test."""
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
        )
        y_pred = np.array([classes_sorted[i] for i in outcome["test_pred_idx"]])
        report = evaluate_classifier(y_test, y_pred)
        per_class_accuracy = {str(k): v for k, v in report["per_class_accuracy"].items()}
        mean_accuracy = float(np.mean(list(per_class_accuracy.values())))

        print(
            f"      params={outcome['n_params']:,}, "
            f"training_time={outcome['training_time_seconds']:.1f}s, "
            f"train_accuracy(последняя эпоха)={outcome['final_train_accuracy']:.3f}, "
            f"test accuracy: {per_class_accuracy}, mean={mean_accuracy:.3f}"
        )

        results.append({
            "key": f"cnn_{name.lower()}",
            "name": name,
            "type": "cnn",
            "params": {
                "architecture": name,
                "n_trainable_parameters": outcome["n_params"],
                "img_size": CNN_IMG_SIZE,
                "epochs": CNN_EPOCHS,
                "batch_size": CNN_BATCH_SIZE,
                "learning_rate": CNN_LEARNING_RATE,
                "weight_decay": CNN_WEIGHT_DECAY,
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss",
                "augmentation": "random horizontal flip (p=0.5) на train",
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


def print_draft_method_comparison(all_results: List[Dict[str, Any]]) -> None:
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
    """
    header = (
        f"{'Метод':<55} | {'тип':<14} | "
        + " | ".join(f"{cls:>11}" for cls in CENTERED_CLASSES)
        + " |  mean |        размер |  время"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in all_results:
        row = f"{r['name']:<55} | {_RESULT_TYPE_LABELS[r['type']]:<14} | "
        row += " | ".join(
            f"{r['per_class_accuracy'].get(cls, float('nan')):>11.3f}" for cls in CENTERED_CLASSES
        )
        if r["type"] == "subspace_conjugacy":
            capacity = sum(r["n_effective_train_vectors_by_class"].values())
            elapsed = r["fit_time_seconds"]
        elif r["type"] == "classical_ml":
            capacity = r["params"]["pca_components"]
            elapsed = r["fit_time_seconds"]
        else:
            capacity = r["params"]["n_trainable_parameters"]
            elapsed = r["training_time_seconds"]
        row += f" | {r['mean_accuracy']:.3f} | {capacity:>14,} | {elapsed:>6.1f}s"
        print(row)


def save_draft_method_artifacts(
    all_results: List[Dict[str, Any]], split_summary: Dict[str, Any],
) -> None:
    """Сохраняет полный JSON-отчёт (все параметры + метрики обоих подходов)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACTS_DIR / "draft_method_experiment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": {
                    "root": str(CENTERED_DATASET_ROOT),
                    "classes": CENTERED_CLASSES,
                    "n_per_class": CENTERED_N_PER_CLASS,
                    "test_fraction": CENTERED_TEST_FRACTION,
                    "random_seed": CENTERED_RANDOM_SEED,
                    "split_summary": split_summary,
                },
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"\n   Отчёт эксперимента (JSON): {report_path}")


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
    print_draft_method_comparison(all_results)
    save_draft_method_artifacts(all_results, split_summary)

    elapsed = time.perf_counter() - start
    print(f"\n=== Эксперимент завершён за {elapsed / 60:.1f} мин ===")


def main() -> None:
    """Точка входа: выбирает эксперимент через --experiment (см. docstring модуля)."""
    parser = argparse.ArgumentParser(
        description="Эксперименты поверх subspace_conjugacy (см. docstring модуля main.py)."
    )
    parser.add_argument(
        "--experiment",
        choices=["article", "draft-method"],
        default="article",
        help=(
            "'article' (по умолчанию) — эксперименты 2-3 опубликованной статьи "
            "на Kaggle archive/ (BRAIN_MRI_ARCHIVE_ROOT). 'draft-method' — метод "
            "из черновика (theory/Макет новой статьи.docx) vs CNN на "
            "datasets/{class}_centered/."
        ),
    )
    args = parser.parse_args()

    if args.experiment == "article":
        run_article_experiment()
    else:
        run_draft_method_experiment()


if __name__ == "__main__":
    main()
