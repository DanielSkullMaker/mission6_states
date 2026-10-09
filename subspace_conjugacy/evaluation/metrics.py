"""Метрики оценки классификатора подпространственной сопряженности (Фаза C, NB8).

NB8 (8_Fursov_classification.ipynb, cell 8) считает точность вручную:
отдельно для первого/второго/третьего класса теста (по позиции в массиве),
затем общую accuracy. Здесь та же идея реализована через метки классов
(а не позицию в массиве, что избавляет от хрупкого предположения "первые
25 объектов — класс 1"), плюс переиспользуется sklearn для стандартных
метрик (accuracy_score, confusion_matrix уже являются зависимостью проекта).
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

ConfusionNormalize = Literal["true", "pred", "all"]


def per_class_accuracy(
    y_true: Sequence[Any], y_pred: Sequence[Any]
) -> Dict[Any, float]:
    """Считает accuracy отдельно для каждого класса, встречающегося в y_true.

    Эквивалент NB8 (first_class_score/first_class_test_images и т.д.), но
    по фактическим меткам классов, а не по позиции объекта в массиве.

    Parameters
    ----------
    y_true : Sequence[Any]
        Истинные метки классов (M,).
    y_pred : Sequence[Any]
        Предсказанные метки классов (M,).

    Returns
    -------
    accuracies : Dict[Any, float]
        Словарь {class_label: accuracy} для каждого класса из y_true.

    Raises
    ------
    ValueError
        Если размеры y_true и y_pred не совпадают.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        logger.error(
            "per_class_accuracy: несовпадение размеров y_true=%d, y_pred=%d.",
            y_true_arr.shape[0], y_pred_arr.shape[0],
        )
        raise ValueError(
            f"Несовпадение размеров: y_true содержит {y_true_arr.shape[0]} "
            f"элементов, y_pred — {y_pred_arr.shape[0]}."
        )

    accuracies = {}
    for cls in np.unique(y_true_arr):
        mask = y_true_arr == cls
        accuracies[cls] = float(np.mean(y_pred_arr[mask] == cls))

    logger.debug("per_class_accuracy: %s.", accuracies)
    return accuracies


def evaluate_classifier(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Формирует сводный отчёт о качестве классификации (аналог NB8, cell 8).

    Parameters
    ----------
    y_true : Sequence[Any]
        Истинные метки классов (M,).
    y_pred : Sequence[Any]
        Предсказанные метки классов (M,).
    labels : Sequence[Any], optional
        Порядок классов в confusion_matrix. Если None — сортированные
        уникальные значения y_true.

    Returns
    -------
    report : Dict[str, Any]
        Словарь с ключами:
        - "accuracy": общая точность (float)
        - "per_class_accuracy": Dict[class_label, float]
        - "confusion_matrix": np.ndarray (n_classes, n_classes)
        - "labels": np.ndarray меток в порядке confusion_matrix
        - "n_samples": количество объектов
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    labels_arr = np.unique(y_true_arr) if labels is None else np.asarray(labels)

    report = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "per_class_accuracy": per_class_accuracy(y_true_arr, y_pred_arr),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=labels_arr),
        "labels": labels_arr,
        "n_samples": int(y_true_arr.shape[0]),
    }
    logger.info(
        "evaluate_classifier: %d объектов, accuracy=%.4f, классы=%s.",
        report["n_samples"], report["accuracy"], list(labels_arr),
    )
    return report


def build_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None,
    normalize: Optional[ConfusionNormalize] = None,
) -> Dict[str, Any]:
    """Строит матрицу ошибок как самостоятельный, переиспользуемый отчёт.

    В отличие от ``evaluate_classifier`` (где матрица ошибок — один из
    нескольких показателей общего отчёта, всегда в сырых счётчиках), эта
    функция — лёгкий самостоятельный вызов для случаев, когда нужна только
    матрица, опционально нормализованная — например, для построения
    таблицы/графика в отчёте (см. ``confusion_matrix_to_table``,
    ``plot_confusion_matrix``).

    Parameters
    ----------
    y_true : Sequence[Any]
        Истинные метки классов (M,).
    y_pred : Sequence[Any]
        Предсказанные метки классов (M,).
    labels : Sequence[Any], optional
        Порядок классов в матрице. Если None — отсортированные уникальные
        значения y_true.
    normalize : {"true", "pred", "all"}, optional
        "true" — нормализация по строкам (полнота/recall на класс, сумма
        строки = 1); "pred" — нормализация по столбцам (точность/precision
        на класс, сумма столбца = 1); "all" — нормализация на общее число
        объектов (сумма всей матрицы = 1); None (по умолчанию) — сырые
        счётчики.

    Returns
    -------
    result : Dict[str, Any]
        Словарь с ключами:
        - "matrix": np.ndarray (n_classes, n_classes)
        - "labels": np.ndarray меток в порядке строк/столбцов matrix
        - "normalize": переданное значение normalize (для удобства downstream)

    Examples
    --------
    >>> y_true = ["glioma", "glioma", "meningioma"]
    >>> y_pred = ["glioma", "meningioma", "meningioma"]
    >>> result = build_confusion_matrix(y_true, y_pred)
    >>> result["matrix"].shape
    (2, 2)
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels_arr = np.unique(y_true_arr) if labels is None else np.asarray(labels)

    matrix = confusion_matrix(
        y_true_arr, y_pred_arr, labels=labels_arr, normalize=normalize,
    )
    logger.debug(
        "build_confusion_matrix: %d классов, normalize=%s.",
        len(labels_arr), normalize,
    )
    return {"matrix": matrix, "labels": labels_arr, "normalize": normalize}


def confusion_matrix_to_table(
    matrix: np.ndarray,
    labels: Sequence[Any],
    corner_label: str = "Истина \\ Предсказано",
    value_format: Optional[str] = None,
) -> Tuple[List[str], List[List[Any]]]:
    """Преобразует матрицу ошибок в заголовок + строки для табличного отчёта.

    Parameters
    ----------
    matrix : np.ndarray
        Матрица ошибок (n_classes, n_classes), как из
        ``build_confusion_matrix``/``evaluate_classifier``.
    labels : Sequence[Any]
        Метки классов в порядке строк/столбцов matrix.
    corner_label : str, default="Истина \\ Предсказано"
        Подпись левой верхней ячейки заголовка.
    value_format : str, optional
        Python format-спецификатор для значений ячеек (например, ``".1%"``
        для нормализованной матрицы). Если None — значения приводятся к
        ``int`` для сырых счётчиков (float без дробной части) или
        оставляются как есть иначе.

    Returns
    -------
    header : List[str]
        Первая строка таблицы: ``[corner_label, *labels]``.
    rows : List[List[Any]]
        По одной строке на класс: ``[label, *значения_строки]``.

    Examples
    --------
    >>> import numpy as np
    >>> header, rows = confusion_matrix_to_table(
    ...     np.array([[8, 2], [1, 9]]), ["glioma", "meningioma"],
    ... )
    >>> header
    ['Истина \\\\ Предсказано', 'glioma', 'meningioma']
    >>> rows[0]
    ['glioma', 8, 2]
    """
    matrix_arr = np.asarray(matrix)

    def _fmt(value: Any) -> Any:
        if value_format is not None:
            return format(value, value_format)
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return int(value)
        return value

    header = [corner_label] + [str(label) for label in labels]
    rows = [
        [str(label)] + [_fmt(matrix_arr[i, j]) for j in range(matrix_arr.shape[1])]
        for i, label in enumerate(labels)
    ]
    return header, rows


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[Any],
    ax: Optional["plt.Axes"] = None,
    cmap: str = "Blues",
    title: Optional[str] = None,
    value_format: Optional[str] = None,
) -> "plt.Axes":
    """Рисует матрицу ошибок в виде тепловой карты (требует matplotlib).

    Parameters
    ----------
    matrix : np.ndarray
        Матрица ошибок (n_classes, n_classes), как из
        ``build_confusion_matrix``/``evaluate_classifier``.
    labels : Sequence[Any]
        Метки классов в порядке строк/столбцов matrix.
    ax : matplotlib.axes.Axes, optional
        Оси для отрисовки. Если None — создаются новая фигура и оси.
    cmap : str, default="Blues"
        Имя colormap matplotlib (однотонная последовательная схема —
        рекомендуется для величин без знака, как счётчики/доли).
    title : str, optional
        Заголовок графика.
    value_format : str, optional
        Python format-спецификатор для подписей ячеек (например, ``".1%"``
        для нормализованной матрицы). Если None — значения приводятся к
        ``int`` для сырых счётчиков.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Оси с отрисованной матрицей (для дальнейшей настройки/сохранения).

    Raises
    ------
    ImportError
        Если matplotlib не установлен.
    """
    if plt is None:
        raise ImportError(
            "plot_confusion_matrix требует matplotlib (не входит в core-"
            "зависимости библиотеки): pip install matplotlib."
        )

    matrix_arr = np.asarray(matrix)
    _, rows = confusion_matrix_to_table(matrix_arr, labels, value_format=value_format)

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(matrix_arr, cmap=cmap, vmin=0)
    vmax = matrix_arr.max() if matrix_arr.size else 0
    for i, row in enumerate(rows):
        for j, value in enumerate(row[1:]):
            frac = matrix_arr[i, j] / vmax if vmax else 0.0
            ax.text(
                j, i, str(value), ha="center", va="center",
                color="white" if frac > 0.6 else "black", fontweight="bold",
            )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    if title is not None:
        ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def confidence_summary(confidence_ratio: np.ndarray) -> Dict[str, float]:
    """Сводная статистика по показателю уверенности (predict_confidence_ratio).

    Parameters
    ----------
    confidence_ratio : np.ndarray
        Результат SubspaceConjugacyClassifier.predict_confidence_ratio(X), (M,).

    Returns
    -------
    summary : Dict[str, float]
        Словарь с ключами "mean", "min", "max", "fraction_negative"
        (доля объектов, где лучший подкласс не выделяется на фоне
        остальных — proportion < 0).
    """
    confidence_arr = np.asarray(confidence_ratio, dtype=np.float64)

    summary = {
        "mean": float(np.mean(confidence_arr)),
        "min": float(np.min(confidence_arr)),
        "max": float(np.max(confidence_arr)),
        "fraction_negative": float(np.mean(confidence_arr < 0)),
    }
    logger.debug("confidence_summary: %s.", summary)
    return summary
