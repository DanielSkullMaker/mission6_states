"""CSV IO для векторов признаков (совместимость с ноутбуками NB3-7).

Предоставляет функции для сохранения и загрузки векторов в формате CSV,
совместимом с существующими ноутбуками.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    from subspace_conjugacy.config import DatasetConfig
except ImportError:
    DatasetConfig = None


def save_vectors_csv(
    vectors: np.ndarray,
    filepath: Union[str, Path],
    delimiter: str = ",",
) -> None:
    """Сохраняет матрицу векторов в CSV файл.

    Parameters
    ----------
    vectors : np.ndarray
        Матрица векторов размерности (M, N).
    filepath : str or Path
        Путь к выходному CSV файлу.
    delimiter : str, default=","
        Разделитель CSV (по умолчанию запятая).

    Examples
    --------
    >>> X = np.random.randn(100, 65536)
    >>> save_vectors_csv(X, "glioma_horizontal_vector.csv")

    Notes
    -----
    Формат совместим с NB3:
    - Каждая строка CSV = один вектор признаков
    - Без заголовка (header)
    - Без индексов строк
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        csv.writer(f, delimiter=delimiter).writerows(vectors)
    logger.info("save_vectors_csv: %s сохранён (shape=%s).", filepath, np.shape(vectors))


def load_vectors_csv(
    filepath: Union[str, Path],
    delimiter: str = ",",
    dtype: type = np.float64,
) -> np.ndarray:
    """Загружает матрицу векторов из CSV файла.

    Parameters
    ----------
    filepath : str or Path
        Путь к CSV файлу.
    delimiter : str, default=","
        Разделитель CSV.
    dtype : type, default=np.float64
        Тип данных элементов массива.

    Returns
    -------
    vectors : np.ndarray
        Матрица векторов размерности (M, N).

    Raises
    ------
    FileNotFoundError
        Если файл не существует.

    Examples
    --------
    >>> X = load_vectors_csv("glioma_horizontal_vector.csv")
    >>> X.shape
    (100, 65536)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("load_vectors_csv: файл не найден '%s'.", filepath)
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # Быстрая загрузка через np.loadtxt
    vectors = np.loadtxt(filepath, delimiter=delimiter, dtype=dtype)

    # Обеспечиваем 2D массив (на случай одного вектора)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    logger.info("load_vectors_csv: %s загружен (shape=%s).", filepath, vectors.shape)
    return vectors


def save_class_vectors(
    vectors: np.ndarray,
    class_name: str,
    config: "DatasetConfig",
    vector_type: str = "horizontal",
) -> Path:
    """Сохраняет векторы класса в стандартное местоположение датасета.

    Parameters
    ----------
    vectors : np.ndarray
        Матрица векторов (M, N).
    class_name : str
        Название класса ("glioma", "meningioma", "pituitary", "test").
    config : DatasetConfig
        Конфигурация датасета.
    vector_type : str, default="horizontal"
        Тип векторизации: "horizontal" или "vertical".

    Returns
    -------
    filepath : Path
        Путь к сохранённому файлу.

    Examples
    --------
    >>> from subspace_conjugacy.config import DatasetConfig
    >>> config = DatasetConfig(root="data")
    >>> X_glioma = np.random.randn(100, 65536)
    >>> path = save_class_vectors(X_glioma, "glioma", config)
    >>> print(path)
    data/5_all_vectors/glioma/glioma_horizontal_vector.csv
    """
    csv_path = config.get_vector_csv_path(class_name, vector_type)
    logger.debug("save_class_vectors: класс=%s, vector_type=%s.", class_name, vector_type)
    save_vectors_csv(vectors, csv_path)
    return csv_path


def load_class_vectors(
    class_name: str,
    config: "DatasetConfig",
    vector_type: str = "horizontal",
) -> np.ndarray:
    """Загружает векторы класса из стандартного местоположения.

    Parameters
    ----------
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.
    vector_type : str, default="horizontal"
        Тип векторизации.

    Returns
    -------
    vectors : np.ndarray
        Матрица векторов (M, N).

    Examples
    --------
    >>> config = DatasetConfig(root="data")
    >>> X_glioma = load_class_vectors("glioma", config)
    >>> X_glioma.shape
    (100, 65536)
    """
    csv_path = config.get_vector_csv_path(class_name, vector_type)
    logger.debug("load_class_vectors: класс=%s, vector_type=%s.", class_name, vector_type)
    return load_vectors_csv(csv_path)


def _save_index_pairs_csv(
    pairs: List[List[int]], filepath: Union[str, Path]
) -> None:
    """Пишет список пар целых чисел построчно (общий формат NB4-6 CSV)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        csv.writer(f).writerows(pairs)
    logger.debug("_save_index_pairs_csv: %s сохранён (%d строк).", filepath, len(pairs))


def _load_index_pairs_csv(filepath: Union[str, Path]) -> List[List[int]]:
    """Читает список пар целых чисел построчно (общий формат NB4-6 CSV)."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("_load_index_pairs_csv: файл не найден '%s'.", filepath)
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(filepath, newline="") as f:
        rows = [[int(cell) for cell in row] for row in csv.reader(f)]
    logger.debug("_load_index_pairs_csv: %s загружен (%d строк).", filepath, len(rows))
    return rows


def save_initial_pair_indices(
    pair: "tuple[int, int]",
    class_name: str,
    config: "DatasetConfig",
) -> Path:
    """Сохраняет начальную пару опорных векторов в формате NB4-5 (Legacy).

    Формат совместим с ``{class}_class_first_and_second_core_vectors.csv``
    из ноутбуков: 2 строки вида ``[rank, vector_index]`` (rank = 1, 2).

    Parameters
    ----------
    pair : tuple[int, int]
        Индексы пары векторов (например, GlobalMinCosinePairFinder.pair_indices_).
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.

    Returns
    -------
    filepath : Path
        Путь к сохранённому файлу.
    """
    csv_path = config.get_initial_pair_path(class_name)
    rows = [[rank, int(idx)] for rank, idx in enumerate(pair, start=1)]
    _save_index_pairs_csv(rows, csv_path)
    return csv_path


def load_initial_pair_indices(
    class_name: str,
    config: "DatasetConfig",
) -> "tuple[int, int]":
    """Загружает начальную пару опорных векторов (Legacy NB4-5).

    Parameters
    ----------
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.

    Returns
    -------
    pair : tuple[int, int]
        Индексы пары векторов (idx1, idx2), как сохранены ноутбуком NB4/NB5.

    Examples
    --------
    >>> config = DatasetConfig(root="data")
    >>> idx1, idx2 = load_initial_pair_indices("glioma", config)
    """
    csv_path = config.get_initial_pair_path(class_name)
    rows = _load_index_pairs_csv(csv_path)
    return tuple(row[1] for row in rows)


def save_center_indices(
    centers: Union[List[int], np.ndarray],
    class_name: str,
    config: "DatasetConfig",
) -> Path:
    """Сохраняет индексы центров подклассов в формате NB5 (A.2-A.3, Legacy).

    Формат совместим с ``{class}_class_core_vectors.csv``: N строк вида
    ``[rank, vector_index]``, rank = 1..N.

    Parameters
    ----------
    centers : List[int] or np.ndarray
        Индексы центров (например, ReferenceCenterBuilder.center_indices_).
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.

    Returns
    -------
    filepath : Path
        Путь к сохранённому файлу.
    """
    csv_path = config.get_center_vectors_path(class_name)
    rows = [[rank, int(idx)] for rank, idx in enumerate(centers, start=1)]
    _save_index_pairs_csv(rows, csv_path)
    return csv_path


def load_center_indices(
    class_name: str,
    config: "DatasetConfig",
) -> np.ndarray:
    """Загружает индексы центров подклассов (Legacy NB5, теория A.2-A.3).

    Parameters
    ----------
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.

    Returns
    -------
    centers : np.ndarray
        Индексы центров подклассов (n_subclasses,), в порядке ноутбука.

    Examples
    --------
    >>> config = DatasetConfig(root="data")
    >>> centers = load_center_indices("glioma", config)
    >>> centers.shape
    (8,)
    """
    csv_path = config.get_center_vectors_path(class_name)
    rows = _load_index_pairs_csv(csv_path)
    return np.array([row[1] for row in rows], dtype=int)


def save_subclass_pairs(
    pairs: Union[List[List[int]], np.ndarray],
    class_name: str,
    config: "DatasetConfig",
) -> Path:
    """Сохраняет пары (центр, второй вектор) подклассов в формате NB6 (B.1).

    Формат совместим с ``{class}_new_classes.csv``: S строк вида
    ``[center_index, second_index]``.

    Parameters
    ----------
    pairs : List[List[int]] or np.ndarray
        Пары индексов размерности (S, 2) (например,
        CosineSecondVectorAttacher.pairs_).
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.

    Returns
    -------
    filepath : Path
        Путь к сохранённому файлу.
    """
    csv_path = config.get_subclass_pairs_path(class_name)
    rows = [[int(a), int(b)] for a, b in pairs]
    _save_index_pairs_csv(rows, csv_path)
    return csv_path


def load_subclass_pairs(
    class_name: str,
    config: "DatasetConfig",
) -> np.ndarray:
    """Загружает пары (центр, второй вектор) подклассов (Legacy NB6, B.1).

    Возвращаемый массив (S, 2) напрямую совместим с входом
    ``ConjugacyClusterGrowth.fit(X, pairs)`` — CSV ноутбука NB6 можно
    использовать как готовый вход канонической фазы B.2.

    Parameters
    ----------
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.

    Returns
    -------
    pairs : np.ndarray
        Пары индексов (n_subclasses, 2): [center_index, second_index].

    Examples
    --------
    >>> config = DatasetConfig(root="data")
    >>> pairs = load_subclass_pairs("glioma", config)
    >>> pairs.shape
    (8, 2)
    """
    csv_path = config.get_subclass_pairs_path(class_name)
    rows = _load_index_pairs_csv(csv_path)
    return np.array(rows, dtype=int)


def save_subclass_bases(
    bases: np.ndarray,
    class_name: str,
    config: "DatasetConfig",
) -> Path:
    """Сохраняет базисы подклассов в формате ноутбуков NB6-7.

    Parameters
    ----------
    bases : np.ndarray
        Матрица базисных векторов размерности (S*k, N), где S — количество
        подклассов, k — векторов на подкласс (обычно 2).
        Строки [2*s : 2*s+2] соответствуют базису подкласса s.
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.

    Returns
    -------
    filepath : Path
        Путь к сохранённому файлу (напр., 8_glioma_subclasses_vectors.csv).

    Examples
    --------
    >>> config = DatasetConfig(root="data", n_subclasses=8)
    >>> # 8 подклассов × 2 вектора = 16 строк
    >>> bases = np.random.randn(16, 65536)
    >>> path = save_subclass_bases(bases, "glioma", config)
    """
    csv_path = config.get_subclass_bases_path(class_name)
    save_vectors_csv(bases, csv_path)
    return csv_path


def load_subclass_bases(
    class_name: str,
    config: "DatasetConfig",
) -> np.ndarray:
    """Загружает базисы подклассов из файла ноутбуков NB6-7.

    Parameters
    ----------
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.

    Returns
    -------
    bases : np.ndarray
        Матрица базисных векторов (S*k, N).

    Examples
    --------
    >>> config = DatasetConfig(root="data", n_subclasses=8)
    >>> bases = load_subclass_bases("glioma", config)
    >>> bases.shape
    (16, 65536)
    >>> # Извлечь базис подкласса 0 (первые 2 строки)
    >>> Y_0 = bases[0:2].T  # (N, 2)
    """
    csv_path = config.get_subclass_bases_path(class_name)
    return load_vectors_csv(csv_path)


def load_subclass_bases_as_list(
    class_name: str,
    config: "DatasetConfig",
) -> List[np.ndarray]:
    """Загружает базисы подклассов как список матриц Y_s.

    Parameters
    ----------
    class_name : str
        Название класса.
    config : DatasetConfig
        Конфигурация датасета.

    Returns
    -------
    subspaces : List[np.ndarray]
        Список из S матриц базисов размерности (N, k).

    Examples
    --------
    >>> config = DatasetConfig(root="data", n_subclasses=8)
    >>> subspaces = load_subclass_bases_as_list("glioma", config)
    >>> len(subspaces)
    8
    >>> subspaces[0].shape
    (65536, 2)
    """
    bases = load_subclass_bases(class_name, config)
    k = config.subclass_factor  # Обычно 2

    # Разбиваем на подклассы: строки [k*s : k*(s+1)]
    subspaces = []
    for s in range(config.n_subclasses):
        start_idx = k * s
        end_idx = k * (s + 1)
        Y_s = bases[start_idx:end_idx].T  # Транспонируем: (k, N) → (N, k)
        subspaces.append(Y_s)

    return subspaces


def save_pipeline_artifact(
    classifier: "SubspaceConjugacyClassifier",
    config: "DatasetConfig",
) -> Dict[str, Path]:
    """Сохраняет обученный классификатор как набор CSV + JSON метаданные.

    В отличие от io.persistence.save_model (pickle, версионно хрупкий),
    здесь базисы каждого класса пишутся в CSV в формате NB6-7
    (совместимом с save_subclass_bases), а гиперпараметры — в отдельный
    JSON (config.get_pipeline_metadata_path()). Такой артефакт переносим
    между версиями библиотеки и читается load_pretrained_classifier.

    Parameters
    ----------
    classifier : SubspaceConjugacyClassifier
        Обученный классификатор (is_fitted_ должен быть True).
    config : DatasetConfig
        Конфигурация датасета — определяет, куда писать CSV каждого класса
        и файл метаданных.

    Returns
    -------
    written_paths : Dict[str, Path]
        Словарь {"metadata": path, class_name: path, ...} со всеми путями,
        куда что-либо было записано.

    Raises
    ------
    RuntimeError
        Если classifier не обучен.
    """
    if not getattr(classifier, "is_fitted_", False):
        logger.error("save_pipeline_artifact: классификатор %r не обучен.", classifier)
        raise RuntimeError(
            "Классификатор не обучен. Вызовите fit() или "
            "fit_from_subclass_bases() перед сохранением."
        )

    from subspace_conjugacy.algorithms.subclass_export import flatten_subspace_bases

    logger.info(
        "save_pipeline_artifact: сохранение %d классов в %s.",
        len(classifier.classes_), config.root,
    )
    written_paths: Dict[str, Path] = {}

    for cls in classifier.classes_:
        bases = flatten_subspace_bases(classifier.subspaces_[cls])
        written_paths[str(cls)] = save_subclass_bases(bases, str(cls), config)

    metadata = {
        "classes": [str(cls) for cls in classifier.classes_],
        "n_subclasses": classifier.n_subclasses,
        "freeze_basis_at": classifier.freeze_basis_at,
        "growth_strategy": classifier.growth_strategy,
        "reg_param": classifier.reg_param,
        "n_features_in": classifier.n_features_in_,
    }
    metadata_path = config.get_pipeline_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    written_paths["metadata"] = metadata_path

    logger.info(
        "save_pipeline_artifact: готово, %d файлов записано (включая метаданные %s).",
        len(written_paths), metadata_path,
    )
    return written_paths


def load_pretrained_classifier(
    config: "DatasetConfig",
) -> "SubspaceConjugacyClassifier":
    """Загружает классификатор, сохранённый через save_pipeline_artifact.

    Читает JSON метаданные (config.get_pipeline_metadata_path()) и CSV
    базисов каждого класса (load_subclass_bases_as_list), затем собирает
    SubspaceConjugacyClassifier через fit_from_subclass_bases — без
    повторной кластеризации.

    Parameters
    ----------
    config : DatasetConfig
        Конфигурация датасета, указывающая на сохранённый артефакт
        (n_subclasses и subclass_factor должны совпадать с теми, что были
        использованы при сохранении).

    Returns
    -------
    classifier : SubspaceConjugacyClassifier
        Готовый к предсказаниям классификатор.

    Raises
    ------
    FileNotFoundError
        Если файл метаданных не найден.

    Examples
    --------
    >>> config = DatasetConfig(root="data", n_subclasses=8)
    >>> clf = load_pretrained_classifier(config)
    >>> clf.predict(X_test)
    """
    from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

    metadata_path = config.get_pipeline_metadata_path()
    if not metadata_path.exists():
        logger.error("load_pretrained_classifier: метаданные не найдены '%s'.", metadata_path)
        raise FileNotFoundError(
            f"Метаданные пайплайна не найдены: '{metadata_path}'. "
            "Ожидался файл, созданный save_pipeline_artifact()."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(
        "load_pretrained_classifier: метаданные загружены из %s, классы=%s.",
        metadata_path, metadata["classes"],
    )

    subspaces_by_class = {
        cls: load_subclass_bases_as_list(cls, config)
        for cls in metadata["classes"]
    }

    classifier = SubspaceConjugacyClassifier(
        n_subclasses=metadata["n_subclasses"],
        freeze_basis_at=metadata["freeze_basis_at"],
        growth_strategy=metadata["growth_strategy"],
        reg_param=metadata["reg_param"],
    )
    classifier.fit_from_subclass_bases(subspaces_by_class)
    logger.info("load_pretrained_classifier: классификатор собран и готов к predict().")
    return classifier
