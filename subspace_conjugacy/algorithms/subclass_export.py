"""Subclass Export: экспорт базисов подклассов в формат CSV ноутбуков.

Этот модуль предоставляет функции для экспорта результатов кластеризации
(базисов подпространств Y_s) в формат, совместимый с NB6-7 и NB8.

Формат файла 8_{class}_subclasses_vectors.csv:
  - Каждые k строк (обычно k=2) соответствуют базису одного подкласса
  - Строки [2*s : 2*s+2] — базис подкласса s
  - Всего S*k строк для S подклассов
  - Каждая строка — вектор размерности N (обычно 65536)

Связь с другими модулями:
  - Входные данные: FursovClusterer.subspaces_ (список матриц N×k)
  - Выходные данные: CSV для SubspaceConjugacyClassifier (фаза C, NB8)
  - IO функции: save_subclass_bases, load_subclass_bases из io/vectors.py
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


def flatten_subspace_bases(
    subspaces: List[np.ndarray],
    expected_basis_size: Optional[int] = None,
) -> np.ndarray:
    """Преобразует список базисов Y_s в плоскую матрицу для экспорта в CSV.

    Parameters
    ----------
    subspaces : list[np.ndarray]
        Список из S базисных матриц размерности (N, k_s).
        Обычно k_s = 2 (freeze_basis_at=2 из ConjugacyClusterGrowth).
    expected_basis_size : int or None, default=None
        Ожидаемое количество векторов в базисе (обычно 2).
        Если задано, проверяется, что все базисы имеют эту размерность.

    Returns
    -------
    flattened : np.ndarray
        Плоская матрица размерности (S*k, N), где строки идут последовательно:
        [Y_0[0], Y_0[1], Y_1[0], Y_1[1], ..., Y_{S-1}[0], Y_{S-1}[1]]

    Raises
    ------
    ValueError
        Если базисы имеют разную размерность признаков N или разное k.

    Examples
    --------
    >>> import numpy as np
    >>> # 3 подкласса × 2 вектора
    >>> Y_0 = np.random.randn(256, 2)
    >>> Y_1 = np.random.randn(256, 2)
    >>> Y_2 = np.random.randn(256, 2)
    >>> subspaces = [Y_0, Y_1, Y_2]
    >>>
    >>> flat = flatten_subspace_bases(subspaces, expected_basis_size=2)
    >>> flat.shape
    (6, 256)  # 3*2 строк, 256 признаков
    >>> # Проверка: первые 2 строки = Y_0.T
    >>> np.allclose(flat[0:2], Y_0.T)
    True

    Notes
    -----
    Формат совместим с NB7:
    - Каждый базис Y_s (N, k) транспонируется в (k, N)
    - Строки всех базисов конкатенируются вертикально
    - Итоговый CSV содержит S*k строк
    """
    if not subspaces:
        raise ValueError("Список подпространств пуст.")

    n_subclasses = len(subspaces)
    n_features = subspaces[0].shape[0]
    basis_sizes = [Y.shape[1] for Y in subspaces]

    # Проверка консистентности размерности признаков
    if not all(Y.shape[0] == n_features for Y in subspaces):
        shapes = [Y.shape for Y in subspaces]
        raise ValueError(
            f"Все базисы должны иметь одинаковую размерность признаков N. "
            f"Получено: {shapes}"
        )

    # Проверка консистентности размера базиса
    if expected_basis_size is not None:
        invalid = [
            (i, Y.shape[1])
            for i, Y in enumerate(subspaces)
            if Y.shape[1] != expected_basis_size
        ]
        if invalid:
            raise ValueError(
                f"Ожидалось {expected_basis_size} векторов в каждом базисе. "
                f"Неверные базисы: {invalid}"
            )
    else:
        # Без явного expected_basis_size: все базисы должны иметь одинаковый k
        if len(set(basis_sizes)) > 1:
            raise ValueError(
                f"Все базисы должны иметь одинаковое количество векторов k. "
                f"Получено: {basis_sizes}"
            )

    # Транспонируем каждый базис (N, k) → (k, N) и конкатенируем
    transposed_bases = [Y.T for Y in subspaces]  # Каждый теперь (k, N)
    flattened = np.vstack(transposed_bases)  # (S*k, N)
    logger.debug(
        "flatten_subspace_bases: %d базисов (N=%d, k=%d) -> flattened.shape=%s.",
        n_subclasses, n_features, basis_sizes[0], flattened.shape,
    )

    return flattened


def unflatten_subspace_bases(
    flattened: np.ndarray,
    n_subclasses: int,
    basis_size: int = 2,
) -> List[np.ndarray]:
    """Обратное преобразование: плоская матрица → список базисов Y_s.

    Parameters
    ----------
    flattened : np.ndarray
        Плоская матрица размерности (S*k, N).
    n_subclasses : int
        Количество подклассов S.
    basis_size : int, default=2
        Количество векторов в базисе (k).

    Returns
    -------
    subspaces : list[np.ndarray]
        Список из S базисных матриц размерности (N, k).

    Raises
    ------
    ValueError
        Если размерность flattened не соответствует S*k строкам.

    Examples
    --------
    >>> flat = np.random.randn(16, 65536)  # 8 подклассов × 2 вектора
    >>> subspaces = unflatten_subspace_bases(flat, n_subclasses=8, basis_size=2)
    >>> len(subspaces)
    8
    >>> subspaces[0].shape
    (65536, 2)
    """
    expected_rows = n_subclasses * basis_size

    if flattened.shape[0] != expected_rows:
        raise ValueError(
            f"Ожидалось {expected_rows} строк (n_subclasses={n_subclasses}, "
            f"basis_size={basis_size}), получено {flattened.shape[0]}."
        )

    n_features = flattened.shape[1]
    subspaces = []

    for s in range(n_subclasses):
        start_row = s * basis_size
        end_row = start_row + basis_size
        Y_s_transposed = flattened[start_row:end_row]  # (k, N)
        Y_s = Y_s_transposed.T  # (N, k)
        subspaces.append(Y_s)

    logger.debug(
        "unflatten_subspace_bases: flattened.shape=%s -> %d базисов (N=%d, k=%d).",
        flattened.shape, n_subclasses, n_features, basis_size,
    )
    return subspaces


def equalize_subspace_bases(
    subspaces_by_class: Dict[Any, List[np.ndarray]],
) -> Tuple[Dict[Any, List[np.ndarray]], int]:
    """Усекает базисы ВСЕХ подклассов ВСЕХ классов до общего минимального k.

    Реализует буквальный рецепт статьи Korshikov & Fursov ("Description of
    the Clustering Method", theory/VI_Korshikov_VA_Fursov_..._Conjugacy.docx):
    "Since the correct operation of the algorithm requires that subspaces
    contain the same number of vectors, when the number of vectors in
    subspaces of different classes differs, only the first n elements
    corresponding to the number of vectors of the smallest space are taken
    for each vector." Без этого шага R(x, Y) для подпространств разного
    размера k несопоставимы напрямую — базис с большим k при прочих равных
    склонен давать больший R просто за счёт того, что охватывает больше
    измерений признакового пространства, а не за счёт реальной
    сопряжённости с классом.

    Типичный сценарий использования: подпространства получены через
    ConjugacyClusterGrowth(freeze_basis_at=None) — жадный рост без
    ограничения размера, из-за чего разные подклассы (и тем более разные
    классы) естественно вырастают до разных k. Этот шаг — единственный
    способ сравнить их между собой "по-честному" без искусственной
    заморозки размера заранее (freeze_basis_at=int), которая отбрасывает
    результат роста (см. SubspaceConjugacyClassifier, freeze_basis_at="auto",
    и refactoring_plan.txt, раздел 10, находка №2).

    Parameters
    ----------
    subspaces_by_class : Dict[Any, List[np.ndarray]]
        {class_label: [Y_0, Y_1, ...]}, где каждый Y_s — базисная матрица
        подкласса (N, k_s); k_s могут отличаться и между подклассами внутри
        одного класса, и между классами.

    Returns
    -------
    equalized : Dict[Any, List[np.ndarray]]
        Новый словарь той же формы, каждый базис усечён до Y[:, :min_size]
        (исходный subspaces_by_class не модифицируется).
    min_size : int
        Итоговый общий размер базиса — минимум среди ВСЕХ входных Y.shape[1].

    Raises
    ------
    ValueError
        Если subspaces_by_class пуст, или хотя бы один класс не содержит
        подпространств.

    Examples
    --------
    >>> import numpy as np
    >>> bases = {
    ...     "a": [np.random.randn(16, 5), np.random.randn(16, 3)],
    ...     "b": [np.random.randn(16, 7)],
    ... }
    >>> equalized, k = equalize_subspace_bases(bases)
    >>> k
    3
    >>> [Y.shape[1] for Y in equalized["a"]]
    [3, 3]
    """
    if not subspaces_by_class:
        logger.error("equalize_subspace_bases: subspaces_by_class пуст.")
        raise ValueError("subspaces_by_class пуст.")

    all_sizes: List[int] = []
    for cls, bases in subspaces_by_class.items():
        if not bases:
            logger.error("equalize_subspace_bases: класс '%s' без подпространств.", cls)
            raise ValueError(f"Класс '{cls}' не содержит подпространств.")
        all_sizes.extend(Y.shape[1] for Y in bases)

    min_size = min(all_sizes)
    max_size = max(all_sizes)
    logger.info(
        "equalize_subspace_bases: усечение до k=%d (наблюдаемый диапазон роста "
        "[%d, %d] среди %d классов).",
        min_size, min_size, max_size, len(subspaces_by_class),
    )

    equalized = {
        cls: [Y[:, :min_size] for Y in bases]
        for cls, bases in subspaces_by_class.items()
    }
    return equalized, min_size


def export_clusterer_bases(
    clusterer,
    output_path: Union[str, Path],
    expected_basis_size: int = 2,
) -> Path:
    """Экспортирует базисы подклассов из FursovClusterer в CSV файл.

    Parameters
    ----------
    clusterer : FursovClusterer or similar
        Обученный кластеризатор с атрибутом subspaces_ (list[np.ndarray]).
    output_path : str or Path
        Путь к выходному CSV файлу.
    expected_basis_size : int, default=2
        Ожидаемое количество векторов в базисе (для валидации).

    Returns
    -------
    filepath : Path
        Путь к сохранённому файлу.

    Raises
    ------
    RuntimeError
        Если кластеризатор не обучен (нет атрибута subspaces_).

    Examples
    --------
    >>> from subspace_conjugacy import FursovClusterer
    >>> import numpy as np
    >>> X = np.random.randn(100, 256)
    >>> clusterer = FursovClusterer(n_subclasses=8, freeze_basis_at=2)
    >>> clusterer.fit(X)
    >>>
    >>> path = export_clusterer_bases(clusterer, "8_glioma_subclasses_vectors.csv")
    >>> print(f"Базисы экспортированы в {path}")
    """
    if not hasattr(clusterer, "subspaces_") or clusterer.subspaces_ is None:
        logger.error("export_clusterer_bases: кластеризатор %r не обучен.", clusterer)
        raise RuntimeError(
            "Кластеризатор не обучен или не имеет атрибута subspaces_. "
            "Вызовите fit(X) перед экспортом."
        )

    subspaces = clusterer.subspaces_
    flattened = flatten_subspace_bases(subspaces, expected_basis_size=expected_basis_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохранение в CSV через numpy
    np.savetxt(output_path, flattened, delimiter=",", fmt="%.18e")
    logger.info(
        "export_clusterer_bases: сохранено %s (shape=%s).", output_path, flattened.shape,
    )

    return output_path


def export_all_classes(
    clusterers_dict: dict,
    output_dir: Union[str, Path],
    n_subclasses: int = 8,
) -> dict:
    """Экспортирует базисы для всех классов в заданную директорию.

    Parameters
    ----------
    clusterers_dict : dict
        Словарь {class_name: FursovClusterer}, где каждый кластеризатор обучен.
    output_dir : str or Path
        Директория для сохранения CSV файлов.
    n_subclasses : int, default=8
        Количество подклассов (используется в имени файла).

    Returns
    -------
    paths : dict
        Словарь {class_name: Path} с путями к экспортированным файлам.

    Examples
    --------
    >>> from subspace_conjugacy import FursovClusterer
    >>> import numpy as np
    >>>
    >>> # Обучаем кластеризаторы для каждого класса
    >>> X_glioma = np.random.randn(100, 256)
    >>> X_meningioma = np.random.randn(100, 256)
    >>> X_pituitary = np.random.randn(100, 256)
    >>>
    >>> clusterers = {
    ...     "glioma": FursovClusterer(n_subclasses=8, freeze_basis_at=2).fit(X_glioma),
    ...     "meningioma": FursovClusterer(n_subclasses=8, freeze_basis_at=2).fit(X_meningioma),
    ...     "pituitary": FursovClusterer(n_subclasses=8, freeze_basis_at=2).fit(X_pituitary),
    ... }
    >>>
    >>> paths = export_all_classes(clusterers, "data/subclass_bases")
    >>> for class_name, path in paths.items():
    ...     print(f"{class_name}: {path}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "export_all_classes: экспорт %d классов в %s.",
        len(clusterers_dict), output_dir,
    )

    paths = {}
    for class_name, clusterer in clusterers_dict.items():
        filename = f"{n_subclasses}_{class_name}_subclasses_vectors.csv"
        output_path = output_dir / filename

        paths[class_name] = export_clusterer_bases(
            clusterer,
            output_path,
            expected_basis_size=2,
        )

    logger.info("export_all_classes: готово, %d файлов записано.", len(paths))
    return paths
