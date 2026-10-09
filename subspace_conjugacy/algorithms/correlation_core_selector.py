"""Correlation Core Selector — отбор "ядра" эталонных векторов класса.

Источник: theory/code/DATA_100.m (первая половина функции, строки 25-56).
Это НЕ часть опубликованной статьи Korshikov & Fursov (в отличие от
LinearDependencyFilter/LowInformativenessFilter, algorithms/reference_filter.py
и algorithms/informativeness_filter.py) — а отдельная, более ранняя MATLAB-
реализация метода, где 200 эталонных изображений класса сокращаются до 100
перед построением опорных центров (см. Run_200.m: DATA_200 -> DATA_100 -> ...).

Алгоритм (буквально по MATLAB, интерпретация как намеренная фича, а не баг):
  Пока не набрано n_target векторов:
    1. Среди ещё "активных" векторов найти пару (i, j), i < j, с максимальной
       корреляцией Пирсона (corrcoef).
    2. Исключить j из активного множества.
    3. Добавить вектор j в "ядро" (core) — в порядке исключения.

  То есть "ядро" — это векторы, которые на каждом шаге оказывались НАИБОЛЕЕ
  скоррелированы с чем-то ещё в наборе, т.е. наиболее типичные/прототипичные
  для класса представители (в отличие от LinearDependencyFilter, который,
  наоборот, ОТБРАСЫВАЕТ почти линейно зависимые векторы ради разнообразия).
  "Хвост" активного множества, оставшийся неисключённым (remaining_indices_),
  — это наиболее непохожие на остальные (потенциальные выбросы) векторы,
  тоже доступен для интроспекции, но не является основным выходом.

Оптимизация относительно MATLAB-прообраза:
  Оригинал на каждой из n_target итераций пересчитывает corrcoef для ВСЕХ
  ещё активных пар — O(n_target * M^2) вызовов corrcoef на класс (M — число
  векторов). Здесь корреляционная матрица (M, M) считается ОДИН раз в начале
  (np.corrcoef) — корреляция Пирсона между двумя векторами не зависит от
  того, что ещё присутствует в наборе, поэтому переиспользование полной
  матрицы с маскированием исключённых строк/столбцов математически
  эквивалентно пересчёту "с нуля" на каждом шаге, но на порядки быстрее.
"""

import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class CorrelationCoreSelector:
    """Отбирает n_target наиболее "типичных" (взаимно скоррелированных) векторов.

    Parameters
    ----------
    n_target : int
        Целевой размер "ядра". Должен удовлетворять 1 <= n_target <= M - 1,
        где M — число векторов, переданных в fit().

    Attributes
    ----------
    core_indices_ : np.ndarray or None
        Индексы (в исходном X) отобранных "ядерных" векторов, в порядке
        исключения из активного множества (длина n_target).
    remaining_indices_ : np.ndarray or None
        Индексы (в исходном X) векторов, ни разу не исключённых — "хвост"
        наименее похожих на остальные (длина M - n_target).
    r_values_ : Dict[int, float] or None
        {core_index: корреляция пары, на которой этот вектор был исключён}
        — для диагностики.
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms.correlation_core_selector import (
    ...     CorrelationCoreSelector,
    ... )
    >>> np.random.seed(0)
    >>> X = np.random.randn(20, 64)
    >>> selector = CorrelationCoreSelector(n_target=10)
    >>> selector.fit(X)
    >>> X_core = selector.transform(X)
    >>> X_core.shape
    (10, 64)
    """

    def __init__(self, n_target: int) -> None:
        self.n_target = n_target

        self.core_indices_: Optional[np.ndarray] = None
        self.remaining_indices_: Optional[np.ndarray] = None
        self.r_values_: Optional[Dict[int, float]] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray) -> "CorrelationCoreSelector":
        """Итеративно отбирает n_target "ядерных" векторов из X.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N).

        Returns
        -------
        self : CorrelationCoreSelector
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если X не 2D, пуст, или n_target вне диапазона [1, M-1].
        """
        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]

        if not (1 <= self.n_target <= n_samples - 1):
            logger.error(
                "CorrelationCoreSelector.fit: n_target=%d вне диапазона [1, %d].",
                self.n_target, n_samples - 1,
            )
            raise ValueError(
                f"n_target должен быть в диапазоне [1, {n_samples - 1}] "
                f"(получено {self.n_target} при M={n_samples})."
            )

        logger.info(
            "CorrelationCoreSelector.fit: старт, %d векторов, n_target=%d.",
            n_samples, self.n_target,
        )

        # Полная матрица корреляций Пирсона (M, M), считается один раз —
        # корреляция пары не зависит от остальных векторов набора.
        # errstate подавляет RuntimeWarning от деления на нулевую дисперсию
        # (константные векторы) — результат (NaN) ожидаемо обрабатывается
        # ниже через nan_to_num, а не является признаком ошибки.
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(X_arr)
        # np.corrcoef даёт NaN для векторов с нулевой дисперсией (константных)
        # — такие пары никогда не должны быть выбраны как "максимально
        # скоррелированные".
        corr = np.nan_to_num(corr, nan=-np.inf)
        np.fill_diagonal(corr, -np.inf)

        active = np.ones(n_samples, dtype=bool)
        core_order = []
        r_values: Dict[int, float] = {}

        for step in range(self.n_target):
            active_idx = np.where(active)[0]
            sub = corr[np.ix_(active_idx, active_idx)]
            # Верхний треугольник (i < j в терминах active_idx, который
            # отсортирован по возрастанию) — исключаем нижний треугольник и
            # диагональ, чтобы однозначно определить (i, j) с i < j, как в
            # оригинальном MATLAB-цикле (i=1..end, j=i+1..end).
            triu_mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
            sub_masked = np.where(triu_mask, sub, -np.inf)

            flat_idx = np.argmax(sub_masked)
            li, lj = np.unravel_index(flat_idx, sub_masked.shape)
            r_max = float(sub_masked[li, lj])
            j_out = int(active_idx[lj])

            active[j_out] = False
            core_order.append(j_out)
            r_values[j_out] = r_max

            logger.debug(
                "CorrelationCoreSelector.fit: шаг %d/%d — вектор %d исключён "
                "в ядро (R=%.6f, партнёр по паре=%d).",
                step + 1, self.n_target, j_out, r_max, int(active_idx[li]),
            )

        self.core_indices_ = np.array(core_order, dtype=int)
        self.remaining_indices_ = np.where(active)[0]
        self.r_values_ = r_values
        self.is_fitted_ = True

        logger.info(
            "CorrelationCoreSelector.fit: готово, ядро=%d векторов, "
            "остаток=%d векторов.", len(self.core_indices_), len(self.remaining_indices_),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Возвращает подматрицу "ядерных" векторов, в порядке отбора.

        Parameters
        ----------
        X : np.ndarray
            Матрица (M, N), на которой был выполнен fit() (или с тем же
            порядком строк).

        Returns
        -------
        X_core : np.ndarray
            Подматрица (n_target, N).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr[self.core_indices_]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Эквивалент fit(X).transform(X) за один проход.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов (M, N).

        Returns
        -------
        X_core : np.ndarray
            Подматрица "ядерных" векторов (n_target, N).
        """
        self.fit(X)
        return self.transform(X)

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Валидирует входную матрицу X."""
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim != 2:
            raise ValueError(
                f"Ожидалась 2D матрица векторов, получена {X_arr.ndim}D."
            )

        if X_arr.shape[0] == 0:
            raise ValueError("Передана пустая матрица X.")

        return X_arr

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            logger.error("CorrelationCoreSelector: обращение к результатам до fit().")
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X) перед использованием."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"CorrelationCoreSelector(n_target={self.n_target}, "
                f"core={len(self.core_indices_)}, remaining={len(self.remaining_indices_)})"
            )
        return f"CorrelationCoreSelector(n_target={self.n_target}, not fitted)"
