"""ParallelFursovClusterer — фасад A.1->A.3->B.1->B.2(параллельный) для бенчмарка.

Самостоятельная альтернатива `algorithms/fursov_clusterer.py::FursovClusterer`
(канон НЕ изменён — см. docstring `subclass_growth_parallel.py`). Фазы A.1,
A.2-A.3, B.1 — ТЕ ЖЕ самые модули канона (GlobalMinCosinePairFinder,
ReferenceCenterBuilder, CosineSecondVectorAttacher), импортированы без
изменений; отличается ТОЛЬКО фаза B.2 —
`subclass_growth_parallel.ParallelConjugacyClusterGrowth` вместо
`subclass_growth.ConjugacyClusterGrowth`. Это даёт честное сравнение "один
к одному": любая разница в скорости/результате объясняется исключительно
différence в B.2, а не случайными различиями в остальном пайплайне.

growth_strategy и center_strategy="windowed" канона здесь не поддержаны —
не нужны для цели сравнения (эффект векторизации/параллелизма B.2).
"""

import logging
import time
from typing import List, Optional
import numpy as np

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.informativeness_filter import (
    DEFAULT_BRIGHTNESS_THRESHOLD,
    DEFAULT_MIN_FRACTION_OF_MEAN,
    LowInformativenessFilter,
)
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.reference_filter import LinearDependencyFilter
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.algorithms.subclass_growth_parallel import ParallelConjugacyClusterGrowth

logger = logging.getLogger(__name__)


class ParallelFursovClusterer:
    """FursovClusterer-совместимый фасад с векторизованным/параллельным B.2.

    Parameters
    ----------
    n_subclasses : int, default=8
    freeze_basis_at : int or None, default=2
    n_jobs : int, default=1
        Прокидывается в ParallelConjugacyClusterGrowth (B.2). 1 — только
        векторизация (без joblib); >1 или -1 — векторизация + параллелизм
        по подклассам внутри каждой итерации роста.
    reg_param : float, default=1e-8
    filter_dependent, dependency_threshold, filter_low_informativeness,
    informativeness_threshold, informativeness_min_fraction :
        См. FursovClusterer — та же семантика, те же (неизменённые) классы
        фильтров, для паритета сравнения по полному пространству параметров.

    Attributes
    ----------
    subspaces_ : list[np.ndarray] or None
    labels_ : np.ndarray or None
    center_indices_ : np.ndarray or None
    initial_pairs_ : np.ndarray or None
    kept_indices_ : np.ndarray or None
    excluded_indices_ : np.ndarray or None
    n_subclasses_ : int or None
    is_fitted_ : bool
    fit_time_s_ : float or None
        Полное время fit() — для сравнения с FursovClusterer на тех же данных.
    """

    def __init__(
        self,
        n_subclasses: int = 8,
        freeze_basis_at: Optional[int] = 2,
        n_jobs: int = 1,
        reg_param: float = 1e-8,
        filter_dependent: bool = False,
        dependency_threshold: float = 0.999,
        filter_low_informativeness: bool = False,
        informativeness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
        informativeness_min_fraction: float = DEFAULT_MIN_FRACTION_OF_MEAN,
    ) -> None:
        if n_subclasses < 2:
            raise ValueError(f"n_subclasses должен быть >= 2, получено {n_subclasses}")
        if freeze_basis_at is not None and freeze_basis_at < 2:
            raise ValueError(f"freeze_basis_at должен быть >= 2 или None, получено {freeze_basis_at}")

        self.n_subclasses = n_subclasses
        self.freeze_basis_at = freeze_basis_at
        self.n_jobs = n_jobs
        self.reg_param = reg_param
        self.filter_dependent = filter_dependent
        self.dependency_threshold = dependency_threshold
        self.filter_low_informativeness = filter_low_informativeness
        self.informativeness_threshold = informativeness_threshold
        self.informativeness_min_fraction = informativeness_min_fraction

        self.subspaces_: Optional[List[np.ndarray]] = None
        self.labels_: Optional[np.ndarray] = None
        self.center_indices_: Optional[np.ndarray] = None
        self.initial_pairs_: Optional[np.ndarray] = None
        self.kept_indices_: Optional[np.ndarray] = None
        self.excluded_indices_: Optional[np.ndarray] = None
        self.n_subclasses_: Optional[int] = None
        self.is_fitted_: bool = False
        self.fit_time_s_: Optional[float] = None

        self._growth: Optional[ParallelConjugacyClusterGrowth] = None

    def fit(self, X: np.ndarray) -> "ParallelFursovClusterer":
        t0 = time.perf_counter()
        X_arr = self._validate_input(X)

        if self.filter_low_informativeness:
            informativeness_filter = LowInformativenessFilter(
                brightness_threshold=self.informativeness_threshold,
                min_fraction_of_mean=self.informativeness_min_fraction,
            )
            informativeness_filter.fit(X_arr)
            kept_idx = informativeness_filter.kept_indices_
            excluded_by_informativeness = informativeness_filter.excluded_indices_
        else:
            kept_idx = np.arange(X_arr.shape[0])
            excluded_by_informativeness = np.array([], dtype=int)

        if self.filter_dependent:
            dependency_filter = LinearDependencyFilter(
                threshold=self.dependency_threshold, reg_param=self.reg_param,
            )
            dependency_filter.fit(X_arr[kept_idx])
            excluded_by_dependency = kept_idx[dependency_filter.excluded_indices_]
            kept_idx = kept_idx[dependency_filter.kept_indices_]
        else:
            excluded_by_dependency = np.array([], dtype=int)

        excluded_idx = np.union1d(excluded_by_informativeness, excluded_by_dependency)
        X_for_clustering = X_arr[kept_idx]

        min_required = self.n_subclasses * 2
        if X_for_clustering.shape[0] < min_required:
            raise ValueError(
                f"После фильтрации осталось {X_for_clustering.shape[0]} векторов, "
                f"а требуется минимум {min_required} для {self.n_subclasses} подклассов."
            )

        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X_for_clustering)
        initial_pair_local = pair_finder.pair_indices_

        center_builder = ReferenceCenterBuilder(
            n_subclasses=self.n_subclasses, reg_param=self.reg_param,
        )
        center_builder.fit(X_for_clustering, initial_pair_local)
        center_indices_local = center_builder.center_indices_

        seed_attacher = CosineSecondVectorAttacher()
        seed_attacher.fit(X_for_clustering, center_indices_local)
        pairs_local = seed_attacher.pairs_

        self._growth = ParallelConjugacyClusterGrowth(
            freeze_basis_at=self.freeze_basis_at,
            reg_param=self.reg_param,
            n_jobs=self.n_jobs,
        )
        self._growth.fit(X_for_clustering, pairs_local)

        self.subspaces_ = self._growth.subspace_bases_
        full_labels = np.full(X_arr.shape[0], -1, dtype=int)
        full_labels[kept_idx] = self._growth.labels_
        self.labels_ = full_labels
        self.center_indices_ = kept_idx[center_indices_local]
        self.initial_pairs_ = kept_idx[pairs_local]
        self.kept_indices_ = kept_idx
        self.excluded_indices_ = excluded_idx
        self.n_subclasses_ = self.n_subclasses
        self.is_fitted_ = True
        self.fit_time_s_ = time.perf_counter() - t0

        logger.info(
            "ParallelFursovClusterer.fit: готово за %.3fs (n_jobs=%s), размеры подклассов=%s.",
            self.fit_time_s_, self.n_jobs, self.get_subclass_sizes().tolist(),
        )
        return self

    def get_subclass_sizes(self) -> np.ndarray:
        self._check_is_fitted()
        return self._growth.get_subclass_sizes()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        return self._growth.predict(X)

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Ожидалась 2D матрица векторов, получена {X_arr.ndim}D.")
        if X_arr.size == 0:
            raise ValueError("Передана пустая матрица X.")
        n_samples = X_arr.shape[0]
        min_required = self.n_subclasses * 2
        if n_samples < min_required:
            raise ValueError(
                f"Недостаточно векторов: требуется минимум {min_required} "
                f"для {self.n_subclasses} подклассов, получено {n_samples}."
            )
        return X_arr

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("Модель не обучена. Вызовите fit(X) перед использованием.")

    def __repr__(self) -> str:
        status = "fitted=True" if self.is_fitted_ else "fitted=False"
        return (
            f"ParallelFursovClusterer(n_subclasses={self.n_subclasses}, "
            f"freeze_basis_at={self.freeze_basis_at}, n_jobs={self.n_jobs}, {status})"
        )
