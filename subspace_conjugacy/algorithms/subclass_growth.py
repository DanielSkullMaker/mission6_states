"""Canonical Algorithm B.2: Conjugacy Cluster Growth.

Теория (секция 1.3, шаг B.2 из refactoring_plan.txt):
  После формирования начальных пар (фаза B.1), оставшиеся векторы
  последовательно присоединяются к подклассам на основе максимума
  показателя сопряжённости R(x, Y_s).

  ШАГ B.2 — Conjugacy growth (наполнение кластеров):
    Условие входа: в каждом подклассе уже ≥ 2 вектора (после B.1 выполнено).
    Пока remaining не пуст:
      1. Для каждого x ∈ remaining и каждого подкласса s:
           вычислить R(x, Y_s), где Y_s — текущая базисная матрица (N×k_s).
      2. Найти пару (x*, s*) с МАКСИМАЛЬНЫМ R(x*, Y_{s*}).
      3. Присоединить x* к подклассу s*, обновить Y_{s*}.
      4. Удалить x* из remaining.

    Критично: присоединять СТРОГО ПО ОДНОМУ вектору за итерацию (глобальный argmax
    по всем remaining × всем подклассам). Нельзя bulk-assign множество в один кластер.

Связь с ноутбуками:
  NB7 (7_2_Fursov_fulfilling_new_subclasses): схожая логика, но с ratio к среднему.
  strategy="master" воспроизводит NB7 поведение.

Связь с другими модулями:
  Требует pairs из CosineSecondVectorAttacher (фаза B.1).
  Выход используется в FursovClusterer и экспортируется для classifier.
"""

from typing import List, Literal, Optional, Tuple
import numpy as np

try:
    from subspace_conjugacy.core.metrics import conjugate_criterion
except ImportError:
    # Fallback для автономного запуска
    def conjugate_criterion(x: np.ndarray, Y: np.ndarray, reg_param: float = 1e-8) -> float:
        """Вычисляет R(x, Y) = (x^T Y (Y^T Y)^{-1} Y^T x) / (x^T x)."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        gram = Y.T @ Y
        gram_reg = gram + reg_param * np.eye(gram.shape[0])
        gram_inv = np.linalg.inv(gram_reg)

        projection = Y @ gram_inv @ Y.T @ x.T
        R = np.sum(x * projection.T, axis=1) / np.sum(x * x, axis=1)
        return np.clip(R, 0.0, 1.0)


class ConjugacyClusterGrowth:
    """Последовательное наполнение кластеров через максимум R (B.2).

    Этот алгоритм распределяет оставшиеся векторы по подклассам,
    присоединяя на каждой итерации ровно один вектор к подклассу
    с максимальным показателем сопряжённости.

    Parameters
    ----------
    freeze_basis_at : int or None, default=2
        Ограничение размерности базисов для финального экспорта.
        Если 2 — после fit базисы будут содержать только первые 2 столбца.
        Если None — базисы растут без ограничений.
    strategy : {"default", "master"}, default="default"
        Стратегия выбора вектора:
        - "default": простой argmax R(x, Y_s)
        - "master": ratio R(x, s*) / mean(R(x, others)) из NB7
    reg_param : float, default=1e-8
        Параметр регуляризации Тихонова для (Y^T Y)^{-1}.

    Attributes
    ----------
    labels_ : np.ndarray or None
        Метки подклассов для каждого вектора (M,).
    subspace_bases_ : list[np.ndarray] or None
        Финальные базисы подпространств (N, k).
        Если freeze_basis_at задан, k = freeze_basis_at.
    n_subclasses_ : int or None
        Количество подклассов.
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms import (
    ...     GlobalMinCosinePairFinder,
    ...     ReferenceCenterBuilder,
    ...     CosineSecondVectorAttacher,
    ...     ConjugacyClusterGrowth,
    ... )
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 256)
    >>>
    >>> # Фазы A.1 → A.3 → B.1
    >>> pair_finder = GlobalMinCosinePairFinder()
    >>> pair_finder.fit(X)
    >>>
    >>> builder = ReferenceCenterBuilder(n_subclasses=8)
    >>> builder.fit(X, pair_finder.pair_indices_)
    >>>
    >>> attacher = CosineSecondVectorAttacher()
    >>> attacher.fit(X, builder.center_indices_)
    >>> pairs = attacher.pairs_
    >>>
    >>> # Фаза B.2: наполнение кластеров
    >>> growth = ConjugacyClusterGrowth(freeze_basis_at=2)
    >>> growth.fit(X, pairs)
    >>> labels = growth.labels_
    >>> bases = growth.subspace_bases_
    >>> print(f"Все векторы распределены: {len(set(labels)) == 8}")
    >>> print(f"Базисы для классификации: {all(Y.shape[1] == 2 for Y in bases)}")
    """

    def __init__(
        self,
        freeze_basis_at: Optional[int] = 2,
        strategy: Literal["default", "master"] = "default",
        reg_param: float = 1e-8,
    ) -> None:
        if freeze_basis_at is not None and freeze_basis_at < 2:
            raise ValueError(f"freeze_basis_at должен быть >= 2, получено {freeze_basis_at}")

        if strategy not in ("default", "master"):
            raise ValueError(f"strategy должен быть 'default' или 'master', получено {strategy}")

        self.freeze_basis_at = freeze_basis_at
        self.strategy = strategy
        self.reg_param = reg_param

        self.labels_: Optional[np.ndarray] = None
        self.subspace_bases_: Optional[List[np.ndarray]] = None
        self.n_subclasses_: Optional[int] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        X: np.ndarray,
        pairs: np.ndarray,
    ) -> "ConjugacyClusterGrowth":
        """Распределяет векторы по подклассам через последовательный рост.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N).
        pairs : np.ndarray
            Начальные пары индексов размерности (S, 2) из CosineSecondVectorAttacher.
            pairs[s] = [center_index, second_vector_index] для подкласса s.

        Returns
        -------
        self : ConjugacyClusterGrowth
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если недостаточно векторов или некорректные пары.
        """
        X_arr = self._validate_input(X)
        pairs_arr = self._validate_pairs(pairs, X_arr.shape[0])

        n_samples, n_features = X_arr.shape
        n_subclasses = len(pairs_arr)

        # Инициализация меток и базисов
        labels = np.full(n_samples, -1, dtype=int)
        Y_bases = []

        # Заполняем начальные базисы из пар и помечаем векторы
        for s, (idx1, idx2) in enumerate(pairs_arr):
            Y_bases.append(X_arr[[idx1, idx2]].T)  # (N, 2)
            labels[idx1] = s
            labels[idx2] = s

        # Оставшиеся векторы для распределения
        remaining = np.where(labels == -1)[0].tolist()

        # Основной цикл: присоединяем по одному вектору
        iteration = 0
        while remaining:
            # 1. Вычисляем R_matrix (len(remaining), n_subclasses)
            R_matrix = self._compute_conjugacy_matrix(
                X_arr, remaining, Y_bases
            )

            # 2. Находим глобальный максимум
            if self.strategy == "default":
                r_idx, s_idx = self._find_argmax_default(R_matrix)
            else:  # master
                r_idx, s_idx = self._find_argmax_master(R_matrix)

            # 3. Присоединяем вектор к подклассу
            vector_idx = remaining[r_idx]
            labels[vector_idx] = s_idx

            # 4. Обновляем базис подкласса
            Y_bases[s_idx] = self._append_to_basis(
                Y_bases[s_idx], X_arr[vector_idx]
            )

            # 5. Удаляем вектор из remaining
            remaining.pop(r_idx)
            iteration += 1

        # Финализация: freeze базисов если требуется
        if self.freeze_basis_at is not None:
            Y_bases = [Y[:, :self.freeze_basis_at] for Y in Y_bases]

        self.labels_ = labels
        self.subspace_bases_ = Y_bases
        self.n_subclasses_ = n_subclasses
        self.is_fitted_ = True

        return self

    def _compute_conjugacy_matrix(
        self,
        X: np.ndarray,
        remaining: List[int],
        Y_bases: List[np.ndarray],
    ) -> np.ndarray:
        """Вычисляет матрицу R для remaining × subclasses."""
        n_remaining = len(remaining)
        n_subclasses = len(Y_bases)
        R_matrix = np.zeros((n_remaining, n_subclasses))

        for r_idx, vector_idx in enumerate(remaining):
            x = X[vector_idx]
            for s_idx, Y in enumerate(Y_bases):
                R_matrix[r_idx, s_idx] = conjugate_criterion(
                    x.reshape(1, -1), Y, self.reg_param
                )[0]

        return R_matrix

    def _find_argmax_default(self, R_matrix: np.ndarray) -> Tuple[int, int]:
        """Находит глобальный argmax в R_matrix (default strategy)."""
        flat_idx = np.argmax(R_matrix)
        r_idx, s_idx = np.unravel_index(flat_idx, R_matrix.shape)
        return int(r_idx), int(s_idx)

    def _find_argmax_master(self, R_matrix: np.ndarray) -> Tuple[int, int]:
        """Находит argmax через ratio к среднему (master strategy, NB7)."""
        n_remaining, n_subclasses = R_matrix.shape

        best_ratio = -np.inf
        best_r_idx = 0
        best_s_idx = 0

        for r_idx in range(n_remaining):
            R_row = R_matrix[r_idx]

            # Для каждого подкласса: ratio = R_s / mean(R_others)
            for s_idx in range(n_subclasses):
                R_s = R_row[s_idx]
                R_others = np.concatenate([R_row[:s_idx], R_row[s_idx+1:]])
                mean_others = R_others.mean() if len(R_others) > 0 else 1.0

                ratio = R_s / mean_others if mean_others > 1e-10 else R_s

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_r_idx = r_idx
                    best_s_idx = s_idx

        return best_r_idx, best_s_idx

    def _append_to_basis(self, Y: np.ndarray, new_vector: np.ndarray) -> np.ndarray:
        """Добавляет новый вектор как столбец к базису."""
        return np.column_stack([Y, new_vector])

    def get_subclass_sizes(self) -> np.ndarray:
        """Возвращает количество векторов в каждом подклассе.

        Returns
        -------
        sizes : np.ndarray
            Массив размера (n_subclasses,) с количеством векторов.

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        return np.bincount(self.labels_, minlength=self.n_subclasses_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает подкласс для новых векторов.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N).

        Returns
        -------
        labels : np.ndarray
            Предсказанные метки подклассов (M,).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        X_arr = self._validate_input(X)

        n_samples = X_arr.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            x = X_arr[i]
            R_values = np.zeros(self.n_subclasses_)

            for s, Y in enumerate(self.subspace_bases_):
                R_values[s] = conjugate_criterion(
                    x.reshape(1, -1), Y, self.reg_param
                )[0]

            labels[i] = np.argmax(R_values)

        return labels

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

    def _validate_pairs(self, pairs: np.ndarray, n_samples: int) -> np.ndarray:
        """Валидирует массив пар."""
        pairs_arr = np.asarray(pairs, dtype=int)

        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError(
                f"pairs должен иметь форму (n_subclasses, 2), получено {pairs_arr.shape}."
            )

        if len(pairs_arr) == 0:
            raise ValueError("pairs не может быть пустым.")

        # Проверка диапазона индексов
        all_indices = pairs_arr.flatten()
        if not np.all((all_indices >= 0) & (all_indices < n_samples)):
            raise ValueError(
                f"Некоторые индексы в pairs вне диапазона [0, {n_samples})."
            )

        # Проверка уникальности
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("pairs содержит дублирующиеся индексы.")

        return pairs_arr

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X, pairs) перед использованием."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"ConjugacyClusterGrowth(n_subclasses={self.n_subclasses_}, "
                f"freeze_basis_at={self.freeze_basis_at}, strategy='{self.strategy}', "
                f"fitted=True)"
            )
        return (
            f"ConjugacyClusterGrowth(freeze_basis_at={self.freeze_basis_at}, "
            f"strategy='{self.strategy}', fitted=False)"
        )


if __name__ == "__main__":
    print("=== Demonstracija ConjugacyClusterGrowth (teorija B.2) ===\n")
    np.random.seed(42)

    # 1. Polnyj cikl: A.1 + A.2-A.3 + B.1 + B.2
    print("1. Polnyj pipeline: A.1 -> A.2-A.3 -> B.1 -> B.2:")
    X_demo = np.random.randn(100, 128)

    # Fazy A.1, A.2-A.3, B.1
    try:
        from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
        from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
        from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher

        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X_demo)
        print(f"   A.1: Para {pair_finder.pair_indices_}")

        builder = ReferenceCenterBuilder(n_subclasses=8)
        builder.fit(X_demo, pair_finder.pair_indices_)
        print(f"   A.2-A.3: Centrov {len(builder.center_indices_)}")

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X_demo, builder.center_indices_)
        pairs = attacher.pairs_
        print(f"   B.1: Par {len(pairs)}")
    except ImportError:
        pairs = np.array([[0, 1], [5, 6], [10, 11], [15, 16],
                          [20, 21], [25, 26], [30, 31], [35, 36]])
        print(f"   A.1-B.1 [Fallback]: Par {len(pairs)}")

    # Faza B.2: napolnenie
    growth = ConjugacyClusterGrowth(freeze_basis_at=2, strategy="default")
    growth.fit(X_demo, pairs)

    labels = growth.labels_
    bases = growth.subspace_bases_

    print(f"   B.2 (default): Vse vektory raspredeleny")
    print(f"       Unikalnyh metok: {len(set(labels))}")
    print(f"       Bazisov: {len(bases)}, forma pervogo: {bases[0].shape}")

    # 2. Proverka raspredelenija
    print("\n2. Raspredelenie vektorov po podklassam:")
    sizes = growth.get_subclass_sizes()
    print(f"   Razmery podklassov: {sizes}")
    print(f"   Min razmer: {sizes.min()}, Max razmer: {sizes.max()}")
    assert sizes.sum() == len(X_demo), "Ne vse vektory raspredeleny"
    print(f"   OK: Vse {len(X_demo)} vektorov raspredeleny")

    # 3. Proverka freeze_basis_at
    print("\n3. Proverka freeze_basis_at=2:")
    for i, Y in enumerate(bases):
        assert Y.shape[1] == 2, f"Bazis {i} ne zamorozhen"
    print(f"   OK: Vse bazisy imejut razmer (N, 2) dlja klassifikatora")

    # 4. Master strategy (NB7)
    print("\n4. Strategy='master' (NB7 replica):")
    growth_master = ConjugacyClusterGrowth(freeze_basis_at=2, strategy="master")
    growth_master.fit(X_demo, pairs)

    sizes_master = growth_master.get_subclass_sizes()
    print(f"   Razmery (master): {sizes_master}")
    print(f"   Sravnenie s default: {'identichno' if np.array_equal(sizes, sizes_master) else 'razlichaetsja'}")

    # 5. Predict na novyh vektorah
    print("\n5. Predikcija na novyh vektorah:")
    X_new = np.random.randn(10, 128)
    pred_labels = growth.predict(X_new)
    print(f"   Prediktov: {len(pred_labels)}")
    print(f"   Unikalnye metki: {sorted(set(pred_labels))}")
    assert all(0 <= l < 8 for l in pred_labels), "Metki vne diapazona"
    print(f"   OK: Vse metki v diapazone [0, 7]")

    # 6. Bez freeze (rastushhij bazis)
    print("\n6. Bez ogranichenia bazisa (freeze_basis_at=None):")
    growth_no_freeze = ConjugacyClusterGrowth(freeze_basis_at=None)
    growth_no_freeze.fit(X_demo, pairs)
    bases_full = growth_no_freeze.subspace_bases_

    print(f"   Razmery bazisov po podklassam:")
    for i, Y in enumerate(bases_full):
        print(f"       Podklass {i}: {Y.shape}")

    # 7. Malenkie dannye
    print("\n7. Kraevoj sluchaj - 20 vektorov, 4 podklassa:")
    X_small = np.random.randn(20, 32)
    pairs_small = np.array([[0, 1], [5, 6], [10, 11], [15, 16]])

    growth_small = ConjugacyClusterGrowth(freeze_basis_at=2)
    growth_small.fit(X_small, pairs_small)

    sizes_small = growth_small.get_subclass_sizes()
    print(f"   Razmery: {sizes_small}")
    assert sizes_small.sum() == 20, "Ne vse vektory"
    print(f"   OK: Vse 20 vektorov raspredeleny")

    # 8. Proizvoditelnost
    print("\n8. Proizvoditelnost na bolshom batche:")
    import time
    X_large = np.random.randn(500, 256)
    pairs_large = np.array([[i*30, i*30+1] for i in range(8)])

    start = time.perf_counter()
    growth_large = ConjugacyClusterGrowth(freeze_basis_at=2)
    growth_large.fit(X_large, pairs_large)
    elapsed = time.perf_counter() - start

    print(f"   Vektorov: {X_large.shape[0]}, razmernost: {X_large.shape[1]}")
    print(f"   Podklassov: {len(pairs_large)}")
    print(f"   Vremja: {elapsed:.3f}s")
    print(f"   Iteracij: {X_large.shape[0] - 2*len(pairs_large)}")

    print("\n OK Vse demonstracionnye proverki zaversheny!")
    print(f"\n{growth}")
