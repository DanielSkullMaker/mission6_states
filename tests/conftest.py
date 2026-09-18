"""Pytest configuration and shared fixtures for subspace_conjugacy tests.

Предоставляет fixtures для:
- Синтетических данных (synthetic data для theory tests)
- Загрузки CSV из ноутбуков (notebook parity tests)
- Конфигурации датасета
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Tuple

# Импорты из библиотеки
try:
    from subspace_conjugacy.config import DatasetConfig
    from subspace_conjugacy.core.metrics import (
        conjugate_criterion,
        cosine_similarity_matrix,
    )
except ImportError:
    # Fallback для изолированного запуска тестов
    DatasetConfig = None
    conjugate_criterion = None
    cosine_similarity_matrix = None


# ============================================================================
# SYNTHETIC DATA FIXTURES (для theory tests)
# ============================================================================


@pytest.fixture
def random_seed():
    """Фиксированный seed для воспроизводимости тестов."""
    return 42


@pytest.fixture
def small_random_vectors(random_seed) -> np.ndarray:
    """Малый батч векторов для быстрых unit-тестов.

    Returns
    -------
    np.ndarray
        Матрица (20, 64) — 20 векторов, размерность 64.
    """
    np.random.seed(random_seed)
    return np.random.randn(20, 64)


@pytest.fixture
def medium_random_vectors(random_seed) -> np.ndarray:
    """Средний батч векторов для integration-тестов.

    Returns
    -------
    np.ndarray
        Матрица (100, 512) — 100 векторов, размерность 512.
    """
    np.random.seed(random_seed)
    return np.random.randn(100, 512)


@pytest.fixture
def synthetic_image_256(random_seed) -> np.ndarray:
    """Синтетическое изображение 256×256 для тестов векторизации.

    Returns
    -------
    np.ndarray
        Greyscale image, shape (256, 256), dtype uint8.
    """
    np.random.seed(random_seed)
    return (np.random.rand(256, 256) * 255).astype(np.uint8)


@pytest.fixture
def synthetic_subspaces(random_seed) -> Tuple[np.ndarray, np.ndarray]:
    """Два синтетических подпространства для тестов классификации.

    Returns
    -------
    Y1 : np.ndarray
        Базис первого подпространства (64, 2).
    Y2 : np.ndarray
        Базис второго подпространства (64, 2).
    """
    np.random.seed(random_seed)
    Y1 = np.random.randn(64, 2)
    Y2 = np.random.randn(64, 2)
    return Y1, Y2


@pytest.fixture
def three_class_dataset(random_seed) -> Tuple[np.ndarray, np.ndarray]:
    """Синтетический датасет с 3 классами для multi-class classification.

    Классы имеют разные центроиды в пространстве признаков.

    Returns
    -------
    X : np.ndarray
        Матрица (90, 32) — 30 векторов × 3 класса.
    y : np.ndarray
        Метки классов (90,) — [0, 0, ..., 1, 1, ..., 2, 2, ...].
    """
    np.random.seed(random_seed)
    n_samples_per_class = 30
    n_features = 32

    X_list = []
    y_list = []

    for cls_idx in range(3):
        # Смещение центроида на cls_idx * 3.0
        X_cls = np.random.randn(n_samples_per_class, n_features) + cls_idx * 3.0
        X_list.append(X_cls)
        y_list.append(np.full(n_samples_per_class, cls_idx))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y


# ============================================================================
# DATASET CONFIG FIXTURES
# ============================================================================


@pytest.fixture
def default_dataset_config() -> DatasetConfig:
    """DatasetConfig с путями по умолчанию (data/).

    Returns
    -------
    DatasetConfig
        Конфигурация с root="data", 3 класса, n_subclasses=8.
    """
    if DatasetConfig is None:
        pytest.skip("DatasetConfig not available")
    return DatasetConfig(root="data")


@pytest.fixture
def temp_dataset_config(tmp_path) -> DatasetConfig:
    """DatasetConfig с временной директорией (для тестов с IO).

    Parameters
    ----------
    tmp_path : Path
        Pytest built-in fixture для временной директории.

    Returns
    -------
    DatasetConfig
        Конфигурация с root=tmp_path.
    """
    if DatasetConfig is None:
        pytest.skip("DatasetConfig not available")
    return DatasetConfig(root=tmp_path)


# ============================================================================
# NOTEBOOK CSV FIXTURES (для parity tests)
# ============================================================================


def _load_csv_if_exists(path: Path) -> np.ndarray:
    """Загружает CSV файл, если он существует, иначе пропускает тест.

    Parameters
    ----------
    path : Path
        Путь к CSV файлу.

    Returns
    -------
    np.ndarray
        Загруженная матрица из CSV.

    Raises
    ------
    pytest.skip
        Если файл не существует (датасет не загружен).
    """
    if not path.exists():
        pytest.skip(f"CSV file not found: {path} (dataset not available)")

    return np.loadtxt(path, delimiter=",", dtype=np.float64)


@pytest.fixture
def glioma_horizontal_vectors(default_dataset_config) -> np.ndarray:
    """Векторы glioma из NB3 (horizontal vectorization).

    Returns
    -------
    np.ndarray
        Матрица (100, 65536) из CSV.
    """
    csv_path = default_dataset_config.get_vector_csv_path("glioma", "horizontal")
    return _load_csv_if_exists(csv_path)


@pytest.fixture
def glioma_subclass_bases(default_dataset_config) -> np.ndarray:
    """Базисы подклассов glioma из NB6-7.

    Returns
    -------
    np.ndarray
        Матрица (16, 65536) — 8 подклассов × 2 вектора на подкласс.
        Каждая пара строк [2*s : 2*s+2] — базис Y_s.
    """
    csv_path = default_dataset_config.get_subclass_bases_path("glioma")
    return _load_csv_if_exists(csv_path)


@pytest.fixture
def test_horizontal_vectors(default_dataset_config) -> np.ndarray:
    """Тестовые векторы из NB3 (75 изображений).

    Returns
    -------
    np.ndarray
        Матрица (75, 65536) из CSV.
    """
    csv_path = default_dataset_config.get_vector_csv_path("test", "horizontal")
    return _load_csv_if_exists(csv_path)


# ============================================================================
# PYTEST MARKERS
# ============================================================================


def pytest_configure(config):
    """Регистрирует кастомные pytest markers."""
    config.addinivalue_line(
        "markers",
        "theory: Tests based on canonical algorithm theory (synthetic data)",
    )
    config.addinivalue_line(
        "markers",
        "notebook_parity: Tests comparing with notebook CSV outputs (requires dataset)",
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests (e.g., full pipeline)"
    )


# ============================================================================
# SKIP CONDITIONS
# ============================================================================


@pytest.fixture
def skip_if_no_dataset(default_dataset_config):
    """Пропускает тест, если датасет не найден.

    Проверяет наличие хотя бы одного класса в root директории.
    """
    glioma_raw = default_dataset_config.paths["glioma"]["raw"]
    if not glioma_raw.exists():
        pytest.skip(
            f"Dataset not found at {default_dataset_config.root}. "
            "Parity tests require downloaded dataset."
        )
