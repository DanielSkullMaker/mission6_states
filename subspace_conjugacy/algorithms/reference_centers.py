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

from typing import List, Optional, Tuple
import numpy as np

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

        if n_samples < self.n_subclasses:
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

            # 5. Добавляем найденный центр
            centers.append(next_center)

        self.center_indices_ = np.array(centers, dtype=int)
        self.r_values_history_ = r_history
        self.is_fitted_ = True

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


if __name__ == "__main__":
    print("=== Демонстрация ReferenceCenterBuilder (теория A.2-A.3) ===\n")
    np.random.seed(42)

    # 1. Базовый пример с фазами A.1 + A.2-A.3
    print("1. Полный цикл: A.1 (пара) → A.2-A.3 (центры):")
    X_small = np.random.randn(50, 64)

    # Фаза A.1: глобальная пара
    try:
        from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X_small)
        initial_pair = pair_finder.pair_indices_
        print(f"   A.1: Найдена начальная пара: {initial_pair}")
    except ImportError:
        # Fallback: вручную выбираем пару
        initial_pair = (0, 1)
        print(f"   A.1 [Fallback]: Используем пару {initial_pair}")

    # Фаза A.2-A.3: остальные центры
    builder = ReferenceCenterBuilder(n_subclasses=8, reg_param=1e-8)
    builder.fit(X_small, initial_pair, store_history=True)

    centers = builder.center_indices_
    print(f"   A.2-A.3: Найдено центров: {len(centers)}")
    print(f"   Индексы центров: {list(centers)}")

    # 2. Проверка растущего базиса
    print("\n2. Проверка инварианта «растущий базис Y»:")
    print(f"   Начало: Y (N, 2) из пары {initial_pair}")
    if builder.r_values_history_:
        for step, r_vals in enumerate(builder.r_values_history_, start=3):
            print(f"   Шаг {step}: Y (N, {step-1}) → {len(r_vals)} кандидатов, "
                  f"min(R)={r_vals.min():.4f}, выбран центр {centers[step-1]}")

    # 3. Извлечение матрицы центров
    print("\n3. Извлечение векторов-центров:")
    centers_matrix = builder.get_center_vectors(X_small)
    print(f"   Матрица центров: shape={centers_matrix.shape}")
    print(f"   Первый центр (норма): {np.linalg.norm(centers_matrix[0]):.2f}")

    # 4. Проверка уникальности центров
    print("\n4. Проверка уникальности:")
    assert len(set(centers)) == len(centers), "Центры должны быть уникальными"
    print(f"   ✓ Все {len(centers)} центров уникальны")

    # 5. Большой батч
    print("\n5. Производительность на большом батче:")
    import time
    X_large = np.random.randn(1000, 512)
    pair_large = (10, 65)  # Предвычисленная пара

    start = time.perf_counter()
    builder_large = ReferenceCenterBuilder(n_subclasses=16)
    builder_large.fit(X_large, pair_large)
    elapsed = time.perf_counter() - start

    print(f"   Векторов: {X_large.shape[0]}, размерность: {X_large.shape[1]}")
    print(f"   Подклассов: {builder_large.n_subclasses}")
    print(f"   Время выполнения: {elapsed:.3f}s")
    print(f"   Центры: {list(builder_large.center_indices_[:5])}...")

    # 6. Краевой случай: n_subclasses = 2
    print("\n6. Краевой случай — n_subclasses=2 (только initial_pair):")
    builder_two = ReferenceCenterBuilder(n_subclasses=2)
    builder_two.fit(X_small, initial_pair)
    print(f"   Центры: {list(builder_two.center_indices_)}")
    assert list(builder_two.center_indices_) == list(initial_pair), "Должна быть только пара"
    print(f"   ✓ Возвращена только initial_pair (цикл while не выполнился)")

    # 7. Проверка ошибок
    print("\n7. Проверка валидации:")
    try:
        builder_bad = ReferenceCenterBuilder(n_subclasses=100)
        builder_bad.fit(X_small, initial_pair)  # 50 < 100
    except ValueError as err:
        print(f"   [Перехвачена ошибка M < n_subclasses]: {err}")

    try:
        builder.fit(X_small, initial_pair=(0, 0))  # Одинаковые индексы
    except ValueError as err:
        print(f"   [Перехвачена ошибка idx1==idx2]: {err}")

    print("\n✓ Все демонстрационные проверки завершены!")
    print(f"\n{builder}")
