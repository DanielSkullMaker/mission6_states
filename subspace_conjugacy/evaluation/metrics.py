"""Метрики оценки классификатора подпространственной сопряженности (Фаза C, NB8).

NB8 (8_Fursov_classification.ipynb, cell 8) считает точность вручную:
отдельно для первого/второго/третьего класса теста (по позиции в массиве),
затем общую accuracy. Здесь та же идея реализована через метки классов
(а не позицию в массиве, что избавляет от хрупкого предположения "первые
25 объектов — класс 1"), плюс переиспользуется sklearn для стандартных
метрик (accuracy_score, confusion_matrix уже являются зависимостью проекта).
"""

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)


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
