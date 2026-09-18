"""Высокоуровневые функции для извлечения признаков из датасета."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    from subspace_conjugacy.config import DatasetConfig
    from subspace_conjugacy.features.vectorization import (
        load_and_vectorize_batch,
        VectorizationType,
    )
except ImportError:
    DatasetConfig = None
    load_and_vectorize_batch = None
    VectorizationType = None


def extract_class_vectors(
    config: "DatasetConfig",
    class_name: str,
    stage: str = "centered",
    method: str = "horizontal",
    count: Optional[int] = None,
) -> np.ndarray:
    """Извлекает векторы признаков для одного класса.

    Parameters
    ----------
    config : DatasetConfig
        Конфигурация датасета с путями.
    class_name : str
        Название класса ("glioma", "meningioma", "pituitary", "test").
    stage : str, default="centered"
        Этап препроцессинга: "raw", "resized", "centered".
    method : {"horizontal", "vertical"}, default="horizontal"
        Метод векторизации.
    count : int, optional
        Количество изображений. Если None, используется значение по умолчанию
        (100 для классов, 75 для test).

    Returns
    -------
    X : np.ndarray
        Матрица векторов признаков (M, N), где M — количество изображений,
        N — размерность вектора (обычно 65536 для 256×256).

    Examples
    --------
    >>> from subspace_conjugacy.config import DatasetConfig
    >>> config = DatasetConfig(root="data")
    >>> X_glioma = extract_class_vectors(config, "glioma", stage="centered")
    >>> X_glioma.shape
    (100, 65536)
    """
    image_paths = config.get_image_paths(class_name, stage, count)
    return load_and_vectorize_batch(image_paths, method=method)


def extract_all_classes(
    config: "DatasetConfig",
    stage: str = "centered",
    method: str = "horizontal",
    include_test: bool = False,
) -> Dict[str, np.ndarray]:
    """Извлекает векторы для всех классов датасета.

    Parameters
    ----------
    config : DatasetConfig
        Конфигурация датасета.
    stage : str, default="centered"
        Этап препроцессинга.
    method : {"horizontal", "vertical"}, default="horizontal"
        Метод векторизации.
    include_test : bool, default=False
        Включать ли тестовую выборку в результат.

    Returns
    -------
    vectors_dict : Dict[str, np.ndarray]
        Словарь {class_name: X}, где X — матрица (M, N).

    Examples
    --------
    >>> config = DatasetConfig(root="data")
    >>> all_vectors = extract_all_classes(config, include_test=True)
    >>> all_vectors.keys()
    dict_keys(['glioma', 'meningioma', 'pituitary', 'test'])
    """
    vectors_dict = {}

    classes_to_extract = config.classes.copy()
    if include_test:
        classes_to_extract.append("test")

    for cls in classes_to_extract:
        vectors_dict[cls] = extract_class_vectors(
            config, cls, stage=stage, method=method
        )

    return vectors_dict


def extract_training_data(
    config: "DatasetConfig",
    stage: str = "centered",
    method: str = "horizontal",
) -> Tuple[np.ndarray, np.ndarray]:
    """Извлекает обучающую выборку (X, y) для классификатора.

    Parameters
    ----------
    config : DatasetConfig
        Конфигурация датасета.
    stage : str, default="centered"
        Этап препроцессинга.
    method : {"horizontal", "vertical"}, default="horizontal"
        Метод векторизации.

    Returns
    -------
    X : np.ndarray
        Матрица признаков (M, N), где M = 100 × len(classes).
    y : np.ndarray
        Вектор меток классов (M,).

    Examples
    --------
    >>> config = DatasetConfig(root="data")
    >>> X_train, y_train = extract_training_data(config)
    >>> X_train.shape, y_train.shape
    ((300, 65536), (300,))
    >>> np.unique(y_train)
    array(['glioma', 'meningioma', 'pituitary'], dtype='<U10')
    """
    X_list = []
    y_list = []

    for cls in config.classes:
        X_cls = extract_class_vectors(config, cls, stage=stage, method=method)
        X_list.append(X_cls)
        y_list.append(np.full(X_cls.shape[0], cls))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y


if __name__ == "__main__":
    print("=== Демонстрация extraction.py ===\n")

    if DatasetConfig is None:
        print("⚠ DatasetConfig недоступен. Установите пакет: pip install -e .")
        exit(1)

    # Используем temp директорию для демонстрации
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        config = DatasetConfig(root=tmpdir)

        print("1. Конфигурация датасета:")
        print(f"   Root: {config.root}")
        print(f"   Classes: {config.classes}")
        print(f"   N features: {config.n_features}")

        print("\n2. Структура API:")
        print("   extract_class_vectors(config, 'glioma') → (100, 65536)")
        print("   extract_all_classes(config) → {'glioma': X, 'meningioma': X, ...}")
        print("   extract_training_data(config) → (X, y)")

        print("\n3. Пример использования (требует датасет):")
        print("   ```python")
        print("   config = DatasetConfig(root='data')")
        print("   X_train, y_train = extract_training_data(config)")
        print("   X_test = extract_class_vectors(config, 'test')")
        print("   ```")

        print("\n✓ Демонстрация завершена!")
