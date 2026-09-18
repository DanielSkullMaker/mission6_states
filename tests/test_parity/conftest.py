"""Fixtures for notebook parity tests using the real MRI dataset.

Датасет: c:\\mission_6states\\datasets\\{class}_centered\\{class}{i}.png
(3 класса, по 200 уже отцентрированных 256x256 PNG-изображений на класс —
эквивалент выхода NB2 / входа NB3).

Все fixtures используют pytest.skip, если реальный датасет недоступен,
как того требует план (раздел 8: "@pytest.mark.notebook_parity (CSV, skipif
no dataset)").
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest

from subspace_conjugacy.features.vectorization import load_and_vectorize_batch

# datasets/ лежит в корне репозитория, tests/test_parity/conftest.py -> ../../datasets
REAL_DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets"
REAL_CLASSES = ["glioma", "meningioma", "pituitary"]

# Полный размер класса в датасете и уменьшенное подмножество для тестов,
# в которых используется B.2 (наполнение кластеров) — этот шаг растёт как
# O(M^2 * N) на чистом numpy, и на 200 изображениях по 65536 признаков
# один прогон занимает больше минуты. 60 изображений — тот же реальный
# препроцессинг, но пайплайн укладывается в единицы секунд.
FULL_CLASS_SIZE = 200
GROWTH_SUBSET_SIZE = 60


def _class_dir(class_name: str) -> Path:
    return REAL_DATASET_ROOT / f"{class_name}_centered"


def _sorted_class_images(class_name: str) -> List[Path]:
    """Возвращает пути к PNG класса, отсортированные по числовому суффиксу.

    Имена файлов вида ``glioma12.png`` — сортировка по номеру, а не по
    строке, чтобы порядок совпадал с исходной нумерацией ноутбуков.
    """
    class_dir = _class_dir(class_name)
    paths = sorted(
        class_dir.glob(f"{class_name}*.png"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0),
    )
    return paths


def real_dataset_available() -> bool:
    return all(_class_dir(cls).is_dir() for cls in REAL_CLASSES)


@pytest.fixture(scope="session")
def skip_if_no_real_dataset():
    """Пропускает тест, если реальный PNG-датасет недоступен."""
    if not real_dataset_available():
        pytest.skip(
            f"Реальный датасет не найден в {REAL_DATASET_ROOT}. "
            "Parity-тесты требуют datasets/{class}_centered/*.png."
        )


@pytest.fixture(scope="session")
def load_real_class_vectors(skip_if_no_real_dataset):
    """Factory fixture: загружает и векторизует первые ``count`` PNG класса.

    Использует тот же метод векторизации, что и NB3 (horizontal —
    построчный обход, эквивалент ``img.flatten()``).

    Returns
    -------
    Callable[[str, int], np.ndarray]
        Функция ``(class_name, count) -> X`` формы ``(count, 65536)``.
    """

    def _load(class_name: str, count: int = FULL_CLASS_SIZE) -> np.ndarray:
        if class_name not in REAL_CLASSES:
            raise ValueError(f"Неизвестный класс: {class_name}")

        paths = _sorted_class_images(class_name)[:count]
        if len(paths) < count:
            pytest.skip(
                f"В {_class_dir(class_name)} найдено только {len(paths)} "
                f"изображений, требуется {count}."
            )

        return load_and_vectorize_batch(paths, method="horizontal")

    return _load


@pytest.fixture(scope="session")
def glioma_vectors_full(load_real_class_vectors) -> np.ndarray:
    """Полный набор реальных векторов glioma (200, 65536)."""
    return load_real_class_vectors("glioma", FULL_CLASS_SIZE)


@pytest.fixture(scope="session")
def glioma_vectors_subset(load_real_class_vectors) -> np.ndarray:
    """Подмножество реальных векторов glioma для B.2-тестов (60, 65536)."""
    return load_real_class_vectors("glioma", GROWTH_SUBSET_SIZE)
