"""Legacy NB4: Per-Vector Pairs (trio_list) — NOT canonical algorithm.

⚠️ ВАЖНО: Этот модуль НЕ соответствует канонической теории A.1.
   Используется ТОЛЬКО для notebook parity tests с NB4.

NB4 Алгоритм (4_1_Fursov_vector_processing):
  Для КАЖДОГО вектора i находится argmin_j cos(v_i, v_j), где j ≠ i.
  Результат: trio_list[M × 3] = [(i, argmin_j, cos_value), ...].

Расхождение с теорией:
  Теория A.1: ОДНА глобальная пара argmin по ВСЕМ (i,j).
  NB4: M локальных пар (каждый вектор → свой argmin).

Используется только в:
  - tests/test_parity/test_nb4_legacy.py
  - algorithms/legacy/notebook_pipeline.py (staged CSV flow)

НЕ используется в:
  - FursovClusterer (канонический алгоритм)
  - Production code
"""

import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    from subspace_conjugacy.core.metrics import cosine_similarity_matrix
except ImportError:
    def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X_norm @ X_norm.T


def compute_per_vector_pairs_nb4(
    X: np.ndarray,
) -> List[Tuple[int, int, float]]:
    """Вычисляет trio_list в формате NB4 (per-vector argmin cosine).

    ⚠️ Legacy алгоритм — не соответствует теории A.1.

    Parameters
    ----------
    X : np.ndarray
        Матрица векторов (M, N).

    Returns
    -------
    trio_list : List[Tuple[int, int, float]]
        Список из M троек:
        [(i, argmin_j, min_cos_value), ...],
        где argmin_j — индекс наиболее непохожего вектора на v_i.

    Examples
    --------
    >>> X = np.random.randn(100, 512)
    >>> trio_list = compute_per_vector_pairs_nb4(X)
    >>> len(trio_list)
    100
    >>> trio_list[0]  # (vector_idx, most_different_idx, cos_value)
    (0, 42, -0.1523)

    Notes
    -----
    В NB5 вручную выбирается одна из этих пар (например, индекс 8).
    Канонический алгоритм (global_pair.py) делает это автоматически.
    """
    logger.debug(
        "compute_per_vector_pairs_nb4 [LEGACY NB4]: старт, X.shape=%s.", X.shape,
    )
    X_arr = np.asarray(X, dtype=np.float64)
    n_samples = X_arr.shape[0]

    # Вычисляем матрицу сходств
    sim_matrix = cosine_similarity_matrix(X_arr)
    np.fill_diagonal(sim_matrix, np.inf)  # Исключаем самосходство

    trio_list = []
    for i in range(n_samples):
        # Для вектора i находим argmin_j cos(i, j)
        argmin_j = int(np.argmin(sim_matrix[i]))
        min_cos_value = float(sim_matrix[i, argmin_j])
        trio_list.append((i, argmin_j, min_cos_value))

    logger.debug(
        "compute_per_vector_pairs_nb4 [LEGACY NB4]: готово, %d троек построено.",
        len(trio_list),
    )
    return trio_list


def extract_pair_from_trio_list(
    trio_list: List[Tuple[int, int, float]],
    index: int,
) -> Tuple[int, int, float]:
    """Извлекает пару из trio_list по индексу (эмуляция ручного выбора в NB5).

    Parameters
    ----------
    trio_list : List[Tuple[int, int, float]]
        Результат compute_per_vector_pairs_nb4().
    index : int
        Индекс элемента trio_list для выбора (в NB5 обычно 8).

    Returns
    -------
    idx1 : int
        Первый индекс пары.
    idx2 : int
        Второй индекс пары (argmin_j для idx1).
    cos_value : float
        Косинусное сходство пары.

    Examples
    --------
    >>> trio_list = compute_per_vector_pairs_nb4(X)
    >>> idx1, idx2, cos = extract_pair_from_trio_list(trio_list, index=8)
    >>> # Эмуляция: в NB5 вручную взяли 8-ю тройку
    """
    if index < 0 or index >= len(trio_list):
        logger.error(
            "extract_pair_from_trio_list [LEGACY NB4]: индекс %d вне диапазона "
            "[0, %d].", index, len(trio_list) - 1,
        )
        raise ValueError(
            f"Индекс {index} вне диапазона [0, {len(trio_list)-1}]."
        )

    logger.debug(
        "extract_pair_from_trio_list [LEGACY NB4]: ручной выбор индекса %d -> %s.",
        index, trio_list[index],
    )
    return trio_list[index]


def find_global_min_from_trio_list(
    trio_list: List[Tuple[int, int, float]],
) -> Tuple[int, int, float]:
    """Находит глобальный минимум косинуса среди всех элементов trio_list.

    Этот метод эквивалентен canonical global_pair.py, но работает
    на уже вычисленном trio_list.

    Parameters
    ----------
    trio_list : List[Tuple[int, int, float]]
        Результат compute_per_vector_pairs_nb4().

    Returns
    -------
    idx1 : int
        Первый индекс глобальной пары.
    idx2 : int
        Второй индекс глобальной пары.
    cos_value : float
        Минимальное косинусное сходство.

    Notes
    -----
    Этот метод показывает, что canonical алгоритм можно получить
    из trio_list, но прямое вычисление (global_pair.py) эффективнее.
    """
    if not trio_list:
        logger.error("find_global_min_from_trio_list [LEGACY NB4]: trio_list пуст.")
        raise ValueError("trio_list пуст.")

    # Находим тройку с минимальным cos_value
    min_trio = min(trio_list, key=lambda t: t[2])
    logger.debug(
        "find_global_min_from_trio_list [LEGACY NB4]: минимум среди %d троек -> %s.",
        len(trio_list), min_trio,
    )
    return min_trio
