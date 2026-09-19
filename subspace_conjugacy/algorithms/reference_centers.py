"""Canonical Algorithm A.2-A.3: Reference Center Builder.

Теория (секция 1.2, шаги A.2-A.3 из refactoring_plan.txt):
  После выбора первой пары центров (A.1), остальные центры находятся
  последовательно через минимум показателя сопряжённости R(x, Y),
  где Y — растущая матрица из уже найденных центров.

  ШАГ A.2 — Третий центр:
    Y = matrix(center_1, center_2) размерности N×2.
    Для каждого оставшегося x: вычислить R(x, Y).
    Выбрать x с МИНИМАЛЬНЫМ R → третий центр.

  ШАГ A.3 — Четвёртый и последующие центры (до n_subclasses):
    Y = matrix(все найденные центры) размерности N×k, k = текущее число центров.
    Для каждого оставшегося x: R(x, Y).
    Выбрать x с МИНИМАЛЬНЫМ R → следующий центр.
    Повторять, пока len(centers) < n_subclasses.

  Инвариант: при поиске k-го центра Y строится из (k-1) уже найденных центров
  (растущая матрица), а не из фиксированной пары.

Соответствие SubspaceClusterer:
  clusterer.py:109-123 → ✅ совпадает с этим модулем (argmin R, растущий Y).

Связь с GlobalMinCosinePairFinder:
  Требует initial_pair из фазы A.1 для инициализации centers = [idx1, idx2].
"""

import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    from subspace_conjugacy.core.metrics import conjugate_criterion
except ImportError:
    # Fallback для автономного запуска
    def conjugate_criterion(
        X: np.ndarray, Y: np.ndarray, reg_param: float = 1e-8
    ) -> np.ndarray:
        X_arr = np.atleast_2d(X)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        x_norms_sq = np.maximum(np.sum(X_arr ** 2, axis=1), reg_param)
        projections = X_arr @ Y
        gram = Y.T @ Y
        reg = reg_param * np.eye(gram.shape[0])
        try:
            inv_gram = np.linalg.inv(gram + reg)
        except np.linalg.LinAlgError:
            inv_gram = np.linalg.pinv(gram)
        numerator = np.sum((projections @ inv_gram) * projections, axis=1)
        R = np.clip(numerator / x_norms_sq, 0.0, 1.0)
        return float(R[0]) if X.ndim == 1 else R


class ReferenceCenterBuilder:
    """Находит центры подклассов через минимум R с растущим базисом (A.2-A.3).

    Этот алгоритм последовательно выбирает n_subclasses опорных векторов,
    максимизируя их «непохожесть» на уже выбранные через показатель
    сопряжённости R(x, Y).

    Parameters
    ----------
    n_subclasses : int, default=8
        Количество центров (подклассов) для формирования.
    reg_param : float, default=1e-8
        Параметр регуляризации Тихонова при обращении матрицы Грама.

    Attributes
    ----------
    center_indices_ : np.ndarray or None
        Массив индексов найденных центров размерности (n_subclasses,).
    r_values_history_ : list[np.ndarray] or None
        История значений R на каждом шаге (для анализа).
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms import (
    ...     GlobalMinCosinePairFinder,
    ...     ReferenceCenterBuilder,
    ... )
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 512)
    >>>
    >>> # Фаза A.1: глобальная пара
    >>> pair_finder = GlobalMinCosinePairFinder()
    >>> pair_finder.fit(X)
    >>> initial_pair = pair_finder.pair_indices_
    >>>
    >>> # Фаза A.2-A.3: остальные центры
    >>> builder = ReferenceCenterBuilder(n_subclasses=8)
    >>> builder.fit(X, initial_pair)
    >>> centers = builder.center_indices_
    >>> print(f"Найдено центров: {len(centers)}")
    """

    def __init__(
        self,
        n_subclasses: int = 8,
        reg_param: float = 1e-8,
    ) -> None:
        self.n_subclasses = n_subclasses
        self.reg_param = reg_param
        self.center_indices_: Optional[np.ndarray] = None
        self.r_values_history_: Optional[List[np.ndarray]] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: np.ndarray,
        initial_pair: Tuple[int, int],
        store_history: bool = False,
    ) -> "ReferenceCenterBuilder":
        """Находит n_subclasses центров через последовательный argmin R.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N), где M ≥ n_subclasses.
        initial_pair : tuple[int, int]
            Первые два центра из фазы A.1 (GlobalMinCosinePairFinder).
        store_history : bool, default=False
            Сохранять ли историю значений R на каждом шаге.

        Returns
        -------
        self : ReferenceCenterBuilder
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если M < n_subclasses или initial_pair некорректен.
        """
        X_arr = self._validate_input(X)
        n_samples, n_features = X_arr.shape
        logger.info(
            "ReferenceCenterBuilder.fit: A.2-A.3 старт, initial_pair=%s, "
            "n_subclasses=%d, %d векторов доступно.",
            initial_pair, self.n_subclasses, n_samples,
        )

        if n_samples < self.n_subclasses:
            logger.error(
                "ReferenceCenterBuilder.fit: недостаточно векторов (%d < %d).",
                n_samples, self.n_subclasses,
            )
            raise ValueError(
                f"Недостаточно векторов: требуется минимум {self.n_subclasses}, "
                f"получено {n_samples}."
            )

        self._validate_initial_pair(initial_pair, n_samples)

        # Инициализация: первые два центра из A.1
        centers = list(initial_pair)
        r_history = [] if store_history else None

        # Итеративный поиск остальных центров (A.2 для 3-го, A.3 для остальных)
        while len(centers) < self.n_subclasses:
            # 1. Строим растущую базисную матрицу Y из найденных центров
            Y = X_arr[centers].T  # (N, k), k = len(centers)

            # 2. Находим кандидатов (векторы, которые ещё не центры)
            remaining = [i for i in range(n_samples) if i not in centers]

            if not remaining:
                logger.error(
                    "ReferenceCenterBuilder.fit: кандидаты закончились на %d/%d центрах.",
                    len(centers), self.n_subclasses,
                )
                raise RuntimeError(
                    f"Недостаточно уникальных векторов для {self.n_subclasses} центров."
                )

            X_candidates = X_arr[remaining]

            # 3. Вычисляем R(x, Y) для всех кандидатов
            r_values = conjugate_criterion(
                X_candidates, Y, reg_param=self.reg_param
            )

            if store_history:
                r_history.append(r_values)

            # 4. Выбираем вектор с МИНИМАЛЬНЫМ R (наименее сопряжённый)
            min_r_idx = np.argmin(r_values)
            next_center = remaining[min_r_idx]
            logger.debug(
                "ReferenceCenterBuilder.fit: центр %d/%d -> вектор %d "
                "(R=%.6f, базис Y k=%d, кандидатов было %d).",
                len(centers) + 1, self.n_subclasses, next_center,
                float(r_values[min_r_idx]), Y.shape[1], len(remaining),
            )

            # 5. Добавляем найденный центр
            centers.append(next_center)

        self.center_indices_ = np.array(centers, dtype=int)
        self.r_values_history_ = r_history
        self.is_fitted_ = True
        logger.info(
            "ReferenceCenterBuilder.fit: A.2-A.3 готово, центры=%s.",
            list(self.center_indices_),
        )

        return self

    def get_center_vectors(self, X: np.ndarray) -> np.ndarray:
        """Возвращает матрицу векторов-центров.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов (M, N), на которой был выполнен fit().

        Returns
        -------
        centers : np.ndarray
            Матрица центров размерности (n_subclasses, N).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        return X[self.center_indices_]

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Валидирует входную матрицу X."""
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim != 2:
            raise ValueError(
                f"Ожидалась 2D матрица векторов, получена {X_arr.ndim}D."
            )

        if X_arr.size == 0:
            raise ValueError("Передана пустая матрица X.")

        return X_arr

    def _validate_initial_pair(
        self, initial_pair: Tuple[int, int], n_samples: int
    ) -> None:
        """Валидирует initial_pair."""
        if len(initial_pair) != 2:
            raise ValueError(
                f"initial_pair должен содержать 2 индекса, получено {len(initial_pair)}."
            )

        idx1, idx2 = initial_pair

        if idx1 == idx2:
            raise ValueError(
                f"Индексы initial_pair должны быть различными, получено ({idx1}, {idx2})."
            )

        if not (0 <= idx1 < n_samples and 0 <= idx2 < n_samples):
            raise ValueError(
                f"Индексы initial_pair вне диапазона [0, {n_samples-1}]: ({idx1}, {idx2})."
            )

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            logger.error("ReferenceCenterBuilder: обращение к результатам до fit().")
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X, initial_pair) перед использованием."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"ReferenceCenterBuilder(n_subclasses={self.n_subclasses}, "
                f"centers={list(self.center_indices_)})"
            )
        return f"ReferenceCenterBuilder(n_subclasses={self.n_subclasses}, not fitted)"
