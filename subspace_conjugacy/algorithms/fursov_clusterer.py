"""Canonical Fursov Clusterer: Facade for complete pipeline A.1 → A.3 → B.1 → B.2.

Теория (секция 1.2-1.3 из refactoring_plan.txt):
  Этот модуль объединяет все этапы канонического алгоритма кластеризации:

  ФАЗА A — Поиск центров кластеров:
    A.1: GlobalMinCosinePairFinder — глобальная пара с min косинусом
    A.2-A.3: ReferenceCenterBuilder — последовательное добавление центров через min R

  ФАЗА B — Формирование кластеров:
    B.1: CosineSecondVectorAttacher — второй вектор для каждого центра через min cos
    B.2: ConjugacyClusterGrowth — последовательное наполнение через max R

  Результат: n_subclasses базисов Y_s (N×freeze_basis_at) для классификатора.

Связь с ноутбуками:
  Объединяет NB4 (legacy), NB5, NB6, NB7 в единый in-memory pipeline.

Связь с другими модулями:
  - Использует algorithms/global_pair, reference_centers, subclass_seed, subclass_growth
  - Результат используется в SubspaceConjugacyClassifier (фаза C, NB8)
"""

import logging
import time
from typing import List, Literal, Optional
import numpy as np

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth

logger = logging.getLogger(__name__)


class FursovClusterer:
    """Фасад для канонического алгоритма кластеризации Fursov (A+B).

    Объединяет все этапы построения подклассов в единый метод fit(),
    последовательно выполняя фазы A.1, A.2-A.3, B.1, B.2.

    Parameters
    ----------
    n_subclasses : int, default=8
        Количество формируемых подклассов (подпространств).
    freeze_basis_at : int or None, default=2
        Ограничение размерности базисов для экспорта в классификатор.
        Если 2 — финальные базисы содержат только первые 2 столбца.
        Если None — базисы растут без ограничений.
    growth_strategy : {"default", "master"}, default="default"
        Стратегия выбора векторов в фазе B.2:
        - "default": простой argmax R(x, Y_s)
        - "master": ratio R(x, s*) / mean(R(x, others)) — канон-совместимая
          аппроксимация идеи NB7, НЕ побитовая реплика
          (см. algorithms/subclass_growth.py, docstring модуля)
    reg_param : float, default=1e-8
        Параметр регуляризации Тихонова для (Y^T Y)^{-1}.

    Attributes
    ----------
    subspaces_ : list[np.ndarray] or None
        Финальные базисы подпространств (N, k) для каждого подкласса.
        Если freeze_basis_at задан, k = freeze_basis_at.
    labels_ : np.ndarray or None
        Метки подклассов для каждого вектора (M,).
    center_indices_ : np.ndarray or None
        Индексы центров подклассов из фазы A.2-A.3.
    initial_pairs_ : np.ndarray or None
        Начальные пары векторов (n_subclasses, 2) из фазы B.1.
    n_subclasses_ : int or None
        Количество подклассов (равно n_subclasses).
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms import FursovClusterer
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 256)
    >>>
    >>> # Единый вызов для полного pipeline
    >>> clusterer = FursovClusterer(n_subclasses=8, freeze_basis_at=2)
    >>> clusterer.fit(X)
    >>>
    >>> # Результаты готовы для классификатора
    >>> bases = clusterer.subspaces_  # 8 матриц (256, 2)
    >>> labels = clusterer.labels_     # (100,) метки подклассов
    >>> print(f"Подклассов: {len(bases)}")
    >>> print(f"Размеры подклассов: {clusterer.get_subclass_sizes()}")
    """

    def __init__(
        self,
        n_subclasses: int = 8,
        freeze_basis_at: Optional[int] = 2,
        growth_strategy: Literal["default", "master"] = "default",
        reg_param: float = 1e-8,
    ) -> None:
        if n_subclasses < 2:
            raise ValueError(f"n_subclasses должен быть >= 2, получено {n_subclasses}")

        if freeze_basis_at is not None and freeze_basis_at < 2:
            raise ValueError(
                f"freeze_basis_at должен быть >= 2 или None, получено {freeze_basis_at}"
            )

        if growth_strategy not in ("default", "master"):
            raise ValueError(
                f"growth_strategy должен быть 'default' или 'master', получено {growth_strategy}"
            )

        self.n_subclasses = n_subclasses
        self.freeze_basis_at = freeze_basis_at
        self.growth_strategy = growth_strategy
        self.reg_param = reg_param

        # Результаты fit()
        self.subspaces_: Optional[List[np.ndarray]] = None
        self.labels_: Optional[np.ndarray] = None
        self.center_indices_: Optional[np.ndarray] = None
        self.initial_pairs_: Optional[np.ndarray] = None
        self.n_subclasses_: Optional[int] = None
        self.is_fitted_: bool = False

        # Внутренние компоненты (для отладки/анализа)
        self._pair_finder: Optional[GlobalMinCosinePairFinder] = None
        self._center_builder: Optional[ReferenceCenterBuilder] = None
        self._seed_attacher: Optional[CosineSecondVectorAttacher] = None
        self._cluster_growth: Optional[ConjugacyClusterGrowth] = None

    def fit(self, X: np.ndarray) -> "FursovClusterer":
        """Выполняет полный канонический алгоритм кластеризации A.1→A.3→B.1→B.2.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N).

        Returns
        -------
        self : FursovClusterer
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если недостаточно векторов или некорректные параметры.
        """
        X_arr = self._validate_input(X)
        fit_start = time.perf_counter()
        logger.info(
            "FursovClusterer.fit: старт, %d векторов, N=%d, n_subclasses=%d, "
            "freeze_basis_at=%s, growth_strategy=%s.",
            X_arr.shape[0], X_arr.shape[1], self.n_subclasses,
            self.freeze_basis_at, self.growth_strategy,
        )

        # ФАЗА A.1: Глобальная пара с минимальным косинусом
        phase_start = time.perf_counter()
        self._pair_finder = GlobalMinCosinePairFinder()
        self._pair_finder.fit(X_arr)
        initial_pair = self._pair_finder.pair_indices_
        logger.debug("FursovClusterer.fit: A.1 заняла %.3fs.", time.perf_counter() - phase_start)

        # ФАЗА A.2-A.3: Последовательное добавление центров через min R
        phase_start = time.perf_counter()
        self._center_builder = ReferenceCenterBuilder(
            n_subclasses=self.n_subclasses,
            reg_param=self.reg_param,
        )
        self._center_builder.fit(X_arr, initial_pair)
        center_indices = self._center_builder.center_indices_
        logger.debug("FursovClusterer.fit: A.2-A.3 заняла %.3fs.", time.perf_counter() - phase_start)

        # ФАЗА B.1: Второй вектор для каждого центра через min cos
        phase_start = time.perf_counter()
        self._seed_attacher = CosineSecondVectorAttacher()
        self._seed_attacher.fit(X_arr, center_indices)
        pairs = self._seed_attacher.pairs_
        logger.debug("FursovClusterer.fit: B.1 заняла %.3fs.", time.perf_counter() - phase_start)

        # ФАЗА B.2: Последовательное наполнение кластеров через max R
        phase_start = time.perf_counter()
        self._cluster_growth = ConjugacyClusterGrowth(
            freeze_basis_at=self.freeze_basis_at,
            strategy=self.growth_strategy,
            reg_param=self.reg_param,
        )
        self._cluster_growth.fit(X_arr, pairs)
        logger.debug("FursovClusterer.fit: B.2 заняла %.3fs.", time.perf_counter() - phase_start)

        # Сохраняем результаты
        self.subspaces_ = self._cluster_growth.subspace_bases_
        self.labels_ = self._cluster_growth.labels_
        self.center_indices_ = center_indices
        self.initial_pairs_ = pairs
        self.n_subclasses_ = self.n_subclasses
        self.is_fitted_ = True

        logger.info(
            "FursovClusterer.fit: готово за %.3fs, размеры подклассов=%s.",
            time.perf_counter() - fit_start, self.get_subclass_sizes().tolist(),
        )

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Выполняет fit и возвращает метки подклассов.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N).

        Returns
        -------
        labels : np.ndarray
            Метки подклассов (M,).
        """
        self.fit(X)
        return self.labels_

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
        logger.debug("FursovClusterer.predict: делегируем ConjugacyClusterGrowth.predict.")
        return self._cluster_growth.predict(X)

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
        return self._cluster_growth.get_subclass_sizes()

    def get_initial_pair(self) -> tuple[int, int]:
        """Возвращает индексы глобальной пары из фазы A.1.

        Returns
        -------
        pair : tuple[int, int]
            Индексы двух векторов с минимальным косинусом.

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        return self._pair_finder.pair_indices_

    def get_center_indices(self) -> np.ndarray:
        """Возвращает индексы центров подклассов из фазы A.2-A.3.

        Returns
        -------
        centers : np.ndarray
            Массив индексов центров (n_subclasses,).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        return self.center_indices_

    def get_initial_pairs(self) -> np.ndarray:
        """Возвращает начальные пары векторов из фазы B.1.

        Returns
        -------
        pairs : np.ndarray
            Массив пар (n_subclasses, 2).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        return self.initial_pairs_

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Валидирует входную матрицу X."""
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim != 2:
            raise ValueError(
                f"Ожидалась 2D матрица векторов, получена {X_arr.ndim}D."
            )

        if X_arr.size == 0:
            raise ValueError("Передана пустая матрица X.")

        n_samples = X_arr.shape[0]
        min_required = self.n_subclasses * 2  # Минимум 2 вектора на подкласс

        if n_samples < min_required:
            raise ValueError(
                f"Недостаточно векторов: требуется минимум {min_required} "
                f"для {self.n_subclasses} подклассов, получено {n_samples}."
            )

        return X_arr

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            logger.error("FursovClusterer: обращение к результатам до fit().")
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X) перед использованием."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"FursovClusterer(n_subclasses={self.n_subclasses_}, "
                f"freeze_basis_at={self.freeze_basis_at}, "
                f"strategy='{self.growth_strategy}', fitted=True)"
            )
        return (
            f"FursovClusterer(n_subclasses={self.n_subclasses}, "
            f"freeze_basis_at={self.freeze_basis_at}, "
            f"strategy='{self.growth_strategy}', fitted=False)"
        )


# Alias для обратной совместимости с legacy SubspaceClusterer
SubspaceClusterer = FursovClusterer
