"""Canonical Algorithm B.1: Cosine Second Vector Attacher.

Теория (секция 1.3, шаг B.1 из refactoring_plan.txt):
  После нахождения центров подклассов (фаза A), для каждого центра
  необходимо найти второй вектор для формирования базиса подпространства (N×2).

  ШАГ B.1 — Cosine seed (второй вектор в каждый подкласс):
    Для КАЖДОГО центра c отдельно:
      из remaining выбрать вектор v с МИНИМАЛЬНЫМ cos(v, center_c)
      (наиболее «непохожий» на центр по косинусу — второй элемент пары).
    Результат: n_subclasses пар (center_c, v_c), в каждом подклассе ровно 2 вектора.
    Использованные v удалить из remaining.

  Примечание: в каждом подклассе формируется подпространство Y_s (N×2).

Связь с ноутбуками:
  NB6 (6_Fursov_new subclasses): схожая логика, но как отдельный CSV-этап.
  Здесь реализуем in-memory без промежуточных файлов.

Связь с другими модулями:
  Требует center_indices из ReferenceCenterBuilder (фаза A.2-A.3).
  Выход используется в ConjugacyClusterGrowth (фаза B.2).
"""

import logging
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    from subspace_conjugacy.core.metrics import cosine_similarity_matrix
except ImportError:
    # Fallback для автономного запуска
    def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X_normalized = X / norms
        return X_normalized @ X_normalized.T


class CosineSecondVectorAttacher:
    """Присоединяет второй вектор к каждому центру через минимум косинуса (B.1).

    Этот алгоритм формирует начальные пары векторов для каждого подкласса,
    выбирая для каждого центра наиболее «непохожий» вектор по косинусному сходству.

    Parameters
    ----------
    None

    Attributes
    ----------
    pairs_ : np.ndarray or None
        Массив пар индексов размерности (n_subclasses, 2).
        pairs_[i] = [center_index, second_vector_index] для подкласса i.
    cosine_values_ : np.ndarray or None
        Значения косинусного сходства для каждой пары (для анализа).
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms import (
    ...     GlobalMinCosinePairFinder,
    ...     ReferenceCenterBuilder,
    ...     CosineSecondVectorAttacher,
    ... )
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 512)
    >>>
    >>> # Фаза A.1: глобальная пара
    >>> pair_finder = GlobalMinCosinePairFinder()
    >>> pair_finder.fit(X)
    >>> initial_pair = pair_finder.pair_indices_
    >>>
    >>> # Фаза A.2-A.3: центры
    >>> builder = ReferenceCenterBuilder(n_subclasses=8)
    >>> builder.fit(X, initial_pair)
    >>> centers = builder.center_indices_
    >>>
    >>> # Фаза B.1: второй вектор к каждому центру
    >>> attacher = CosineSecondVectorAttacher()
    >>> attacher.fit(X, centers)
    >>> pairs = attacher.pairs_
    >>> print(f"Сформировано пар: {len(pairs)}")
    """

    def __init__(self) -> None:
        self.pairs_: Optional[np.ndarray] = None
        self.cosine_values_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: np.ndarray,
        center_indices: np.ndarray,
        store_cosine_values: bool = False,
    ) -> "CosineSecondVectorAttacher":
        """Находит второй вектор для каждого центра через минимум cos.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N).
        center_indices : np.ndarray
            Индексы центров подклассов размерности (n_subclasses,).
            Результат из ReferenceCenterBuilder.
        store_cosine_values : bool, default=False
            Сохранять ли значения косинусного сходства для каждой пары.

        Returns
        -------
        self : CosineSecondVectorAttacher
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если недостаточно векторов для формирования пар.
        """
        X_arr = self._validate_input(X)
        center_indices_arr = self._validate_center_indices(center_indices, X_arr.shape[0])

        n_samples = X_arr.shape[0]
        n_subclasses = len(center_indices_arr)
        logger.info(
            "CosineSecondVectorAttacher.fit: B.1 старт, %d центров, %d векторов доступно.",
            n_subclasses, n_samples,
        )

        # Проверка: нужно минимум n_subclasses * 2 векторов
        if n_samples < n_subclasses * 2:
            logger.error(
                "CosineSecondVectorAttacher.fit: недостаточно векторов (%d < %d).",
                n_samples, n_subclasses * 2,
            )
            raise ValueError(
                f"Недостаточно векторов для формирования пар: "
                f"требуется минимум {n_subclasses * 2}, получено {n_samples}."
            )

        # Вычисляем косинусную матрицу один раз
        cos_matrix = cosine_similarity_matrix(X_arr)

        # Инициализация
        pairs = []
        cosine_vals = []
        remaining = set(range(n_samples)) - set(center_indices_arr)

        # Для каждого центра находим второй вектор с минимальным cos
        for center_idx in center_indices_arr:
            if not remaining:
                logger.error(
                    "CosineSecondVectorAttacher.fit: remaining пуст на центре %d "
                    "(сформировано %d/%d пар).", center_idx, len(pairs), n_subclasses,
                )
                raise RuntimeError(
                    f"Не осталось векторов для подкласса с центром {center_idx}."
                )

            # Косинусные сходства с текущим центром для оставшихся векторов
            remaining_list = list(remaining)
            cos_with_center = cos_matrix[center_idx, remaining_list]

            # Находим индекс с минимальным cos (наиболее непохожий)
            min_cos_idx_local = np.argmin(cos_with_center)
            second_vector_idx = remaining_list[min_cos_idx_local]
            min_cos_value = cos_with_center[min_cos_idx_local]
            logger.debug(
                "CosineSecondVectorAttacher.fit: подкласс %d — центр=%d, "
                "второй вектор=%d, cos=%.6f (кандидатов было %d).",
                len(pairs), center_idx, second_vector_idx, min_cos_value,
                len(remaining_list),
            )

            # Сохраняем пару
            pairs.append([center_idx, second_vector_idx])
            if store_cosine_values:
                cosine_vals.append(min_cos_value)

            # Удаляем использованный вектор из remaining
            remaining.remove(second_vector_idx)

        self.pairs_ = np.array(pairs, dtype=int)
        self.cosine_values_ = np.array(cosine_vals) if store_cosine_values else None
        self.is_fitted_ = True
        logger.info(
            "CosineSecondVectorAttacher.fit: B.1 готово, %d пар сформировано.",
            len(pairs),
        )

        return self

    def get_subspace_bases(self, X: np.ndarray) -> List[np.ndarray]:
        """Возвращает список базисов подпространств Y_s (N×2) для каждого подкласса.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов (M, N), на которой был выполнен fit().

        Returns
        -------
        bases : list[np.ndarray]
            Список из n_subclasses матриц размерности (N, 2).
            bases[i] — базис подпространства для подкласса i.

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()

        bases = []
        for pair in self.pairs_:
            center_idx, second_idx = pair
            Y = X[[center_idx, second_idx]].T  # (N, 2)
            bases.append(Y)

        return bases

    def get_pair_vectors(self, X: np.ndarray, subclass_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает пару векторов для указанного подкласса.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов (M, N).
        subclass_index : int
            Индекс подкласса (0 <= i < n_subclasses).

        Returns
        -------
        center_vector : np.ndarray
            Вектор центра подкласса.
        second_vector : np.ndarray
            Второй вектор подкласса.

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        IndexError
            Если subclass_index вне диапазона.
        """
        self._check_is_fitted()

        if not (0 <= subclass_index < len(self.pairs_)):
            raise IndexError(
                f"subclass_index {subclass_index} вне диапазона [0, {len(self.pairs_)})"
            )

        center_idx, second_idx = self.pairs_[subclass_index]
        return X[center_idx], X[second_idx]

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

    def _validate_center_indices(
        self, center_indices: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """Валидирует массив индексов центров."""
        center_indices_arr = np.asarray(center_indices, dtype=int)

        if center_indices_arr.ndim != 1:
            raise ValueError(
                f"center_indices должен быть 1D массивом, получен {center_indices_arr.ndim}D."
            )

        if len(center_indices_arr) == 0:
            raise ValueError("center_indices не может быть пустым.")

        # Проверка уникальности
        if len(set(center_indices_arr)) != len(center_indices_arr):
            raise ValueError("center_indices содержит дубликаты.")

        # Проверка диапазона
        if not all(0 <= idx < n_samples for idx in center_indices_arr):
            raise ValueError(
                f"Некоторые индексы в center_indices вне диапазона [0, {n_samples})."
            )

        return center_indices_arr

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            logger.error("CosineSecondVectorAttacher: обращение к результатам до fit().")
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X, center_indices) перед использованием."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"CosineSecondVectorAttacher(n_pairs={len(self.pairs_)}, "
                f"fitted=True)"
            )
        return "CosineSecondVectorAttacher(fitted=False)"
