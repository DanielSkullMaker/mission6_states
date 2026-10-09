"""Сглаживание соседних векторов признаков (theory/code/Recogn_VSPU_S1.m).

Источник: Recogn_VSPU_S1.m, строки 21-49. Перед вычислением показателя
сопряжённости для каждого тестового вектора его окрестность (СОСЕДНИЕ ПО
ИНДЕКСУ строки матрицы признаков — т.е. соседние ИЗОБРАЖЕНИЯ в наборе, а не
соседние пиксели внутри одного изображения) усредняется скользящим окном
переменной ширины: 3 у самой границы набора, 5 чуть дальше, 7 — во всей
остальной "внутренней" части набора. Первый и последний вектор набора не
сглаживаются вовсе (нет полного окна с обеих сторон).

Эта операция ИМЕЕТ СМЫСЛ ТОЛЬКО если порядок строк X содержательный — т.е.
соседние по индексу векторы действительно похожи (например, последовательные
срезы одного пациента). Если порядок случаен/алфавитен по разным пациентам,
усреднение смешивает признаки несвязанных объектов и может ухудшить, а не
улучшить результат классификации. Вызывающий код отвечает за то, чтобы
порядок X был содержательным — эта функция такой порядок не проверяет.

НЕ часть опубликованной статьи Korshikov & Fursov — источник только в
theory/code/*.m.
"""

import logging
from typing import Sequence
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HALF_WIDTHS = (1, 2, 3)


def smooth_adjacent_vectors(
    X: np.ndarray,
    half_widths: Sequence[int] = DEFAULT_HALF_WIDTHS,
) -> np.ndarray:
    """Сглаживает каждую строку X скользящим средним по соседним строкам.

    Parameters
    ----------
    X : np.ndarray
        Матрица векторов размерности (M, N), строки — в содержательном
        порядке (например, последовательные срезы одного пациента).
    half_widths : Sequence[int], default=(1, 2, 3)
        Последовательность полуширин окна по мере удаления от ближайшей
        границы набора. По умолчанию воспроизводит Recogn_VSPU_S1.m:
        расстояние 1 от границы -> окно ±1 (размер 3), расстояние 2 -> ±2
        (размер 5), расстояние >= 3 -> ±3 (размер 7, для всех дальнейших
        internal-позиций — последнее значение последовательности
        используется как "потолок"). Первая и последняя строка (расстояние
        0 от границы) никогда не сглаживаются.

    Returns
    -------
    X_smoothed : np.ndarray
        Матрица той же формы (M, N). Копия X с усреднёнными internal-строками;
        строки, для которых окно не помещается целиком в границы X (M
        слишком мало относительно half_widths), остаются без изменений.

    Raises
    ------
    ValueError
        Если X не 2D, пуст, или half_widths пуст / содержит не строго
        положительные значения.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(35, dtype=float).reshape(7, 5)
    >>> X_smoothed = smooth_adjacent_vectors(X)
    >>> np.array_equal(X_smoothed[0], X[0])  # границы не тронуты
    True
    >>> np.array_equal(X_smoothed[-1], X[-1])
    True
    """
    X_arr = _validate_input(X)
    if len(half_widths) == 0 or any(hw <= 0 for hw in half_widths):
        raise ValueError(
            f"half_widths должен быть непустой последовательностью строго "
            f"положительных целых, получено {half_widths!r}."
        )

    n_samples = X_arr.shape[0]
    X_out = X_arr.copy()

    if n_samples < 3:
        logger.debug(
            "smooth_adjacent_vectors: M=%d < 3, границы всегда исключены — "
            "сглаживать нечего.", n_samples,
        )
        return X_out

    n_smoothed = 0
    for i in range(1, n_samples - 1):
        distance_to_boundary = min(i, n_samples - 1 - i)
        half_width = half_widths[min(distance_to_boundary - 1, len(half_widths) - 1)]

        lo, hi = i - half_width, i + half_width
        if lo < 0 or hi > n_samples - 1:
            # Окно не помещается целиком (M мало относительно half_widths) —
            # оставляем строку без изменений, а не обрезаем окно, чтобы не
            # смещать результат ассиметрично.
            continue

        X_out[i] = X_arr[lo : hi + 1].mean(axis=0)
        n_smoothed += 1

    logger.info(
        "smooth_adjacent_vectors: %d/%d строк сглажено (half_widths=%s).",
        n_smoothed, n_samples, tuple(half_widths),
    )
    return X_out


def _validate_input(X: np.ndarray) -> np.ndarray:
    """Валидирует входную матрицу X."""
    X_arr = np.asarray(X, dtype=np.float64)

    if X_arr.ndim != 2:
        raise ValueError(f"Ожидалась 2D матрица векторов, получена {X_arr.ndim}D.")

    if X_arr.shape[0] == 0:
        raise ValueError("Передана пустая матрица X.")

    return X_arr
