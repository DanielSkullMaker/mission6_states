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

from typing import List, Optional, Tuple
import numpy as np

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

        # Проверка: нужно минимум n_subclasses * 2 векторов
        if n_samples < n_subclasses * 2:
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

            # Сохраняем пару
            pairs.append([center_idx, second_vector_idx])
            if store_cosine_values:
                cosine_vals.append(min_cos_value)

            # Удаляем использованный вектор из remaining
            remaining.remove(second_vector_idx)

        self.pairs_ = np.array(pairs, dtype=int)
        self.cosine_values_ = np.array(cosine_vals) if store_cosine_values else None
        self.is_fitted_ = True

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


if __name__ == "__main__":
    print("=== Demonstracija CosineSecondVectorAttacher (teorija B.1) ===\n")
    np.random.seed(42)

    # 1. Polnyj cikl: A.1 + A.2-A.3 + B.1
    print("1. Polnyj pipeline: A.1 -> A.2-A.3 -> B.1:")
    X_small = np.random.randn(50, 64)

    # Faza A.1: globalnaja para
    try:
        from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X_small)
        initial_pair = pair_finder.pair_indices_
        print(f"   A.1: Najdena para: {initial_pair}")
    except ImportError:
        initial_pair = (0, 1)
        print(f"   A.1 [Fallback]: Para {initial_pair}")

    # Faza A.2-A.3: centry
    try:
        from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
        builder = ReferenceCenterBuilder(n_subclasses=8)
        builder.fit(X_small, initial_pair)
        centers = builder.center_indices_
        print(f"   A.2-A.3: Najdeno centrov: {len(centers)}")
        print(f"   Indeksy centrov: {list(centers)}")
    except ImportError:
        centers = np.array([0, 1, 5, 10, 15, 20, 25, 30])
        print(f"   A.2-A.3 [Fallback]: Centry {list(centers)}")

    # Faza B.1: vtoroj vektor k kazhdomu centru
    attacher = CosineSecondVectorAttacher()
    attacher.fit(X_small, centers, store_cosine_values=True)

    pairs = attacher.pairs_
    print(f"   B.1: Sformirovano par: {len(pairs)}")
    print(f"   Pervaja para: center={pairs[0][0]}, second={pairs[0][1]}")

    # 2. Proverka unikalnosti
    print("\n2. Proverka unikalnosti indeksov:")
    all_indices = pairs.flatten()
    unique_indices = set(all_indices)
    print(f"   Vsego indeksov v parah: {len(all_indices)}")
    print(f"   Unikalnyh indeksov: {len(unique_indices)}")
    assert len(unique_indices) == len(all_indices), "Dolzhny byt unikalnymi"
    print(f"   OK: Vse indeksy unikalnye")

    # 3. Proverka, chto pervye elementy par = centry
    print("\n3. Proverka sootvetstvija centrov:")
    for i, (center_in_pair, _) in enumerate(pairs):
        assert center_in_pair == centers[i], f"Nesootvetstvie v podklasse {i}"
    print(f"   OK: Vse pervye elementy par sovpadajut s centrami")

    # 4. Proverka minimalnyh kosinusov
    print("\n4. Proverka minimal'nyh kosinusov:")
    if attacher.cosine_values_ is not None:
        print(f"   Sredneje kosinus: {attacher.cosine_values_.mean():.4f}")
        print(f"   Min kosinus: {attacher.cosine_values_.min():.4f}")
        print(f"   Max kosinus: {attacher.cosine_values_.max():.4f}")

    # 5. Izvlechenie bazisov podprostranstv
    print("\n5. Izvlechenie bazisov Y_s (N×2):")
    bases = attacher.get_subspace_bases(X_small)
    print(f"   Kolichestvo bazisov: {len(bases)}")
    print(f"   Forma pervogo bazisa: {bases[0].shape}")
    assert all(Y.shape == (64, 2) for Y in bases), "Vse bazisy dolzhny byt' (N, 2)"
    print(f"   OK: Vse bazisy imejut formu (64, 2)")

    # 6. Proverka get_pair_vectors
    print("\n6. Proverka get_pair_vectors:")
    v1, v2 = attacher.get_pair_vectors(X_small, subclass_index=0)
    print(f"   Vektor centra (norma): {np.linalg.norm(v1):.2f}")
    print(f"   Vtoroj vektor (norma): {np.linalg.norm(v2):.2f}")
    manual_cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"   Kosinus mezhdu nimi: {manual_cos:.4f}")

    # 7. Bolshoj batch
    print("\n7. Proizvoditelnost na bolshom batche:")
    import time
    X_large = np.random.randn(1000, 512)
    centers_large = np.arange(0, 160, 10)  # 16 centrov

    start = time.perf_counter()
    attacher_large = CosineSecondVectorAttacher()
    attacher_large.fit(X_large, centers_large)
    elapsed = time.perf_counter() - start

    print(f"   Vektorov: {X_large.shape[0]}, razmernost: {X_large.shape[1]}")
    print(f"   Podklassov: {len(centers_large)}")
    print(f"   Vremja vypolnenija: {elapsed:.3f}s")
    print(f"   Pary: {attacher_large.pairs_.shape}")

    # 8. Kraevoj sluchaj: 2 centra
    print("\n8. Kraevoj sluchaj - 2 centra:")
    X_tiny = np.random.randn(10, 16)
    centers_tiny = np.array([0, 5])
    attacher_tiny = CosineSecondVectorAttacher()
    attacher_tiny.fit(X_tiny, centers_tiny)
    print(f"   Pary: {attacher_tiny.pairs_}")
    assert len(attacher_tiny.pairs_) == 2, "Dolzhno byt 2 pary"
    print(f"   OK: Sformirovano 2 pary")

    # 9. Proverka oshibok
    print("\n9. Proverka validacii:")
    try:
        attacher_bad = CosineSecondVectorAttacher()
        attacher_bad.fit(X_small, centers=np.array([0, 0, 1]))  # Dublikaty
    except ValueError as err:
        print(f"   [Oshibka dublikaty]: {err}")

    try:
        attacher_bad2 = CosineSecondVectorAttacher()
        attacher_bad2.fit(X_small[:5], centers=np.arange(8))  # 5 < 16
    except ValueError as err:
        print(f"   [Oshibka nedostatochno vektorov]: {err}")

    print("\n OK Vse demonstracionnye proverki zaversheny!")
    print(f"\n{attacher}")
