"""Canonical Algorithm A.1: Global Minimum Cosine Pair Finder.

Теория (секция 1.2, шаг A.1 из refactoring_plan.txt):
  Первая пара центров подклассов выбирается как глобальный argmin косинусного
  сходства по всем возможным парам векторов (i, j), i ≠ j.

  Перебрать ВСЕ возможные пары (i, j), i ≠ j.
  Для каждой пары вычислить cos(v_i, v_j).
  Сохранить пару с МИНИМАЛЬНЫМ значением cos — наиболее «непохожие» изображения.
  Результат: center_indices = [idx1, idx2].

Расхождение с NB4:
  NB4 использует per-vector подход: для КАЖДОГО i ищется argmin_j cos(i,j) → trio_list[M×3].
  Канон — ОДНА глобальная пара.
  NB4 → algorithms/legacy/per_vector_pairs.py (parity only).

Соответствие SubspaceClusterer:
  clusterer.py:102-106 — ✅ совпадает с этим модулем (глобальный argmin).
"""

from typing import Optional, Tuple
import numpy as np

try:
    from subspace_conjugacy.core.metrics import cosine_similarity_matrix
except ImportError:
    # Fallback для автономного запуска
    def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X_norm @ X_norm.T


class GlobalMinCosinePairFinder:
    """Находит пару векторов с минимальным косинусным сходством (теория A.1).

    Этот шаг инициализирует кластеризацию выбором двух наиболее различающихся
    векторов из обучающей выборки одного класса.

    Parameters
    ----------
    None

    Attributes
    ----------
    pair_indices_ : tuple[int, int] or None
        Индексы найденной пары (idx1, idx2), где idx1 < idx2.
    similarity_value_ : float or None
        Минимальное значение косинусного сходства найденной пары.
    similarity_matrix_ : np.ndarray or None
        Полная матрица попарных косинусных сходств (M, M).
        Сохраняется при store_matrix=True.
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 512)
    >>> finder = GlobalMinCosinePairFinder()
    >>> finder.fit(X)
    >>> idx1, idx2 = finder.pair_indices_
    >>> print(f"Найдена пара: ({idx1}, {idx2}), cos={finder.similarity_value_:.4f}")
    """

    def __init__(self) -> None:
        self.pair_indices_: Optional[Tuple[int, int]] = None
        self.similarity_value_: Optional[float] = None
        self.similarity_matrix_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: np.ndarray,
        store_matrix: bool = False,
    ) -> "GlobalMinCosinePairFinder":
        """Находит пару векторов с минимальным косинусным сходством.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N), где M ≥ 2.
        store_matrix : bool, default=False
            Сохранять ли полную матрицу сходств в similarity_matrix_.
            Для больших M (>1000) может потреблять много памяти.

        Returns
        -------
        self : GlobalMinCosinePairFinder
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если M < 2 (недостаточно векторов для пары).
        """
        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]

        if n_samples < 2:
            raise ValueError(
                f"Требуется минимум 2 вектора для поиска пары, получено {n_samples}."
            )

        # 1. Вычисляем матрицу косинусных сходств (M, M)
        sim_matrix = cosine_similarity_matrix(X_arr)

        # 2. Маскируем диагональ (самосходство = 1.0 → inf, чтобы не выбирать i=j)
        np.fill_diagonal(sim_matrix, np.inf)

        # 3. Находим глобальный argmin среди всех пар (i, j)
        min_idx_flat = np.argmin(sim_matrix)
        idx1, idx2 = np.unravel_index(min_idx_flat, sim_matrix.shape)

        # 4. Сохраняем результаты (idx1 < idx2 для каноничности)
        self.pair_indices_ = tuple(sorted([int(idx1), int(idx2)]))
        self.similarity_value_ = float(sim_matrix[idx1, idx2])

        if store_matrix:
            # Восстанавливаем диагональ для корректного отображения
            np.fill_diagonal(sim_matrix, 1.0)
            self.similarity_matrix_ = sim_matrix

        self.is_fitted_ = True
        return self

    def get_pair_vectors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Возвращает сами векторы найденной пары.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов (M, N), на которой был выполнен fit().

        Returns
        -------
        v1 : np.ndarray
            Первый вектор пары, размерность (N,).
        v2 : np.ndarray
            Второй вектор пары, размерность (N,).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        idx1, idx2 = self.pair_indices_
        return X[idx1], X[idx2]

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

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X) перед использованием."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            idx1, idx2 = self.pair_indices_
            return (
                f"GlobalMinCosinePairFinder(pair=({idx1}, {idx2}), "
                f"cos={self.similarity_value_:.4f})"
            )
        return "GlobalMinCosinePairFinder(not fitted)"


if __name__ == "__main__":
    print("=== Демонстрация GlobalMinCosinePairFinder (теория A.1) ===\n")
    np.random.seed(42)

    # 1. Базовый пример
    print("1. Поиск глобальной пары на синтетических данных:")
    X_small = np.random.randn(20, 64)
    finder = GlobalMinCosinePairFinder()
    finder.fit(X_small)

    idx1, idx2 = finder.pair_indices_
    print(f"   Найдена пара: ({idx1}, {idx2})")
    print(f"   Косинусное сходство: {finder.similarity_value_:.6f}")
    print(f"   Это минимальное значение среди {20*19//2} возможных пар")

    # 2. Получение векторов пары
    print("\n2. Извлечение векторов пары:")
    v1, v2 = finder.get_pair_vectors(X_small)
    print(f"   Вектор 1: shape={v1.shape}, norm={np.linalg.norm(v1):.2f}")
    print(f"   Вектор 2: shape={v2.shape}, norm={np.linalg.norm(v2):.2f}")

    # Проверка косинусного сходства вручную
    manual_cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"   Проверка: ручной расчёт cos = {manual_cos:.6f}")
    assert np.isclose(manual_cos, finder.similarity_value_), "Косинусы не совпадают!"

    # 3. Сравнение с диагональю
    print("\n3. Проверка инварианта: cos(v_i, v_i) = 1.0 > min_cos:")
    finder_with_matrix = GlobalMinCosinePairFinder()
    finder_with_matrix.fit(X_small, store_matrix=True)
    diag_values = np.diag(finder_with_matrix.similarity_matrix_)
    print(f"   Самосходство (диагональ): {diag_values[:5].round(4)}")
    print(f"   Все значения ≈ 1.0: {np.allclose(diag_values, 1.0)}")
    print(f"   Минимальное сходство пары: {finder.similarity_value_:.4f} < 1.0")

    # 4. Большой батч
    print("\n4. Производительность на большом батче:")
    import time
    X_large = np.random.randn(1000, 512)

    start = time.perf_counter()
    finder_large = GlobalMinCosinePairFinder()
    finder_large.fit(X_large)
    elapsed = time.perf_counter() - start

    print(f"   Векторов: {X_large.shape[0]}, размерность: {X_large.shape[1]}")
    print(f"   Возможных пар: {1000*999//2} = 499,500")
    print(f"   Время выполнения: {elapsed:.3f}s")
    print(f"   Найдена пара: {finder_large.pair_indices_}")

    # 5. Краевой случай: два вектора
    print("\n5. Краевой случай — всего 2 вектора:")
    X_two = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    finder_two = GlobalMinCosinePairFinder()
    finder_two.fit(X_two)
    print(f"   Единственная возможная пара: {finder_two.pair_indices_}")
    print(f"   Косинус (ортогональные векторы): {finder_two.similarity_value_:.6f}")

    # 6. Проверка ошибки при M < 2
    print("\n6. Проверка валидации (M < 2):")
    X_one = np.random.randn(1, 64)
    try:
        finder.fit(X_one)
    except ValueError as err:
        print(f"   [Перехвачена ожидаемая ошибка]: {err}")

    print("\n✓ Все демонстрационные проверки завершены!")
    print(f"\n{finder_large}")
