"""Модуль математических вычислений и метрик показателя сопряженности.

Предоставляет векторизованные функции для расчета показателя сопряженности R(x, Y)
и косинусных расстояний
"""

from typing import Union
import numpy as np


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
        return np.clip(X_norm @ X_norm.T, -1.0, 1.0)

    Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return np.clip(X_norm @ Y_norm.T, -1.0, 1.0)


if __name__ == "__main__":
    # Фиксируем seed для воспроизводимости тестов
    np.random.seed(42)

    print("=== Запуск демонстрации и самотестирования metrics.py ===\n")

    # Конфигурация тестовых данных: M векторов, размерность N, k базисов
    n_samples = 5
    n_features = 128
    n_bases = 4

    # 1. Генерация синтетических данных
    X_batch = np.random.randn(n_samples, n_features)
    Y_basis = np.random.randn(n_features, n_bases)
    single_x = X_batch[0]

    # 2. Тестирование единичного вектора
    r_single = conjugate_criterion(single_x, Y_basis)
    print(f"1. Скалярный вектор x (N={n_features}):")
    print(f"   R(x, Y) = {r_single:.6f}")
    assert isinstance(r_single, float), "Результат должен быть float"
    assert 0.0 <= r_single <= 1.0, "Показатель R должен быть в диапазоне [0, 1]"

    # 3. Тестирование батча векторов
    r_batch = conjugate_criterion(X_batch, Y_basis)
    print(f"\n2. Батч векторов X (M={n_samples}, N={n_features}):")
    print(f"   R(X, Y) = {r_batch.round(6)}")
    assert isinstance(r_batch, np.ndarray), "Результат должен быть np.ndarray"
    assert r_batch.shape == (n_samples,), f"Ожидалась форма ({n_samples},)"
    assert np.all((r_batch >= 0.0) & (r_batch <= 1.0)), "Все R должны быть в [0, 1]"

    # 4. Проверка краевого случая: Вырожденность (линейная зависимость в Y)
    # Создаем вырожденный базис, добавив коллинеарный столбец
    Y_singular = np.column_stack([Y_basis, Y_basis[:, [0]] * 2.5])
    r_singular = conjugate_criterion(X_batch, Y_singular)
    print(f"\n3. Проверка регуляризации на вырожденном базисе Y (k={n_bases + 1}):")
    print(f"   R(X, Y_singular) = {r_singular.round(6)}")
    assert not np.isnan(r_singular).any(), "Результат не должен содержать NaN"

    # 5. Тестирование косинусного сходства
    cos_sim = cosine_similarity_matrix(X_batch)
    print(f"\n4. Попарное косинусное сходство (размерность {cos_sim.shape}):")
    print(f"   Диагональ (самосходство) = {np.diag(cos_sim).round(4)}")
    assert np.allclose(np.diag(cos_sim), 1.0), "Диагональ должна состоять из 1.0"

    print("\n Все проверки успешно пройдены!")