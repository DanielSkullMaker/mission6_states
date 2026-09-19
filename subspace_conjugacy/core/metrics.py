"""Модуль математических вычислений и метрик показателя сопряженности.

Предоставляет векторизованные функции для расчета показателя сопряженности R(x, Y)
и косинусных расстояний

Логирование
-----------
conjugate_criterion — самая "горячая" функция библиотеки: алгоритмы
кластеризации (особенно B.2, ConjugacyClusterGrowth) вызывают её десятки
тысяч раз за один fit(). Поэтому здесь НЕТ лога на каждый вызов — это
раздуло бы вывод до бесполезности и заметно замедлило бы вычисления даже
при DEBUG-уровне. Вместо этого:
  - compute_gram_inverse логирует WARNING только в редком случае — когда
    матрица Грама вырождена и происходит откат на np.linalg.pinv;
  - алгоритмы верхнего уровня (algorithms/*.py) логируют осмысленные шаги
    ("центр найден", "вектор присоединён к подклассу"), а не отдельные
    вызовы conjugate_criterion.
"""

import logging
from typing import Union
import numpy as np

logger = logging.getLogger(__name__)


def compute_gram_inverse(
    Y: np.ndarray, reg_param: float = 1e-8
) -> np.ndarray:
    """Вычисляет устойчивую обратную матрицу Грама (Y^T * Y)^(-1).

    Parameters
    ----------
    Y : np.ndarray
        Матрица базисных векторов подкласса размерности (N, k),
        где N — размерность признаков, k — количество базисных векторов.
    reg_param : float, default=1e-8
        Коэффициент регуляризации Тихонова для предотвращения
        деления на ноль при линейно зависимых базисах.

    Returns
    -------
    inv_gram : np.ndarray
        Обратная матрица Грама размерности (k, k).
    """
    gram = Y.T @ Y
    k = gram.shape[0]
    reg_matrix = reg_param * np.eye(k, dtype=gram.dtype)

    try:
        inv_gram = np.linalg.inv(gram + reg_matrix)
    except np.linalg.LinAlgError:
        logger.warning(
            "compute_gram_inverse: матрица Грама (%dx%d) вырождена даже после "
            "регуляризации (reg_param=%s) — откат на np.linalg.pinv. "
            "Возможная причина: линейно зависимые/дублирующиеся базисные векторы.",
            k, k, reg_param,
        )
        inv_gram = np.linalg.pinv(gram)

    return inv_gram


def conjugate_criterion(
    X: np.ndarray, Y: np.ndarray, reg_param: float = 1e-8
) -> Union[float, np.ndarray]:
    """Рассчитывает показатель сопряженности R(x, Y) для вектора или батча.

    Математическая формула:
        R(x, Y) = (x^T * Y * (Y^T * Y)^(-1) * Y^T * x) / (x^T * x)

    Parameters
    ----------
    X : np.ndarray
        Входной вектор размерности (N,) или матрица векторов (M, N).
    Y : np.ndarray
        Базисная матрица подкласса размерности (N, k).
    reg_param : float, default=1e-8
        Параметр регуляризации для защиты от вырожденности.

    Returns
    -------
    R : Union[float, np.ndarray]
        Показатель сопряженности в диапазоне [0.0, 1.0].
        Возвращает float для одного вектора или np.ndarray (M,) для батча.

    Raises
    ------
    ValueError
        Если размерности признаков X и Y не совпадают.
    """
    X_arr = np.atleast_2d(X)
    M, N = X_arr.shape

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if Y.shape[0] != N:
        logger.error(
            "conjugate_criterion: несовпадение размерностей — X имеет %d "
            "признаков, а базис Y — %d.", N, Y.shape[0],
        )
        raise ValueError(
            f"Несовпадение размерностей признаков: X имеет размерность {N}, "
            f"а базис Y — {Y.shape[0]}."
        )

    # 1. Знаменатель: квадрат L2-нормы векторов (x^T x) -> shape (M,)
    x_norms_sq = np.sum(X_arr ** 2, axis=1)
    x_norms_sq = np.maximum(x_norms_sq, reg_param)

    # 2. Матрица проекций (X @ Y) -> shape (M, k)
    projections = X_arr @ Y

    # 3. Обратная матрица Грама (Y^T Y)^(-1) -> shape (k, k)
    inv_gram = compute_gram_inverse(Y, reg_param=reg_param)

    # 4. Числитель: поэлементное умножение и сумма по строкам -> shape (M,)
    # Эквивалентно скалярному произведению (x_i^T Y) @ inv_gram @ (Y^T x_i)
    proj_inv_gram = projections @ inv_gram
    numerator = np.sum(proj_inv_gram * projections, axis=1)

    # 5. Расчет критерия сопряженности с ограничением диапазона [0, 1]
    R = numerator / x_norms_sq
    R = np.clip(R, 0.0, 1.0)

    return float(R[0]) if X.ndim == 1 else R


def cosine_similarity_matrix(
    X: np.ndarray, Y: Union[np.ndarray, None] = None
) -> np.ndarray:
    """Вычисляет косинусное сходство между векторами.

    Используется на этапе поиска начальных центров подклассов.

    Parameters
    ----------
    X : np.ndarray
        Матрица векторов размерности (M, N).
    Y : np.ndarray, optional
        Матрица векторов размерности (K, N). Если None, вычисляется
        попарное сходство внутри X.

    Returns
    -------
    similarity : np.ndarray
        Матрица сходства размерности (M, K) или (M, M).
    """
    X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    if Y is None:
        logger.debug(
            "cosine_similarity_matrix: попарное сходство внутри X, shape=%s -> %s",
            X.shape, (X.shape[0], X.shape[0]),
        )
        return np.clip(X_norm @ X_norm.T, -1.0, 1.0)

    Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    logger.debug(
        "cosine_similarity_matrix: X.shape=%s, Y.shape=%s -> %s",
        X.shape, Y.shape, (X.shape[0], Y.shape[0]),
    )
    return np.clip(X_norm @ Y_norm.T, -1.0, 1.0)
