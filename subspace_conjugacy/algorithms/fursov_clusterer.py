"""Canonical Fursov Clusterer: Facade for complete pipeline A.1 → A.3 → B.1 → B.2.

Теория (секция 1.2-1.3 из refactoring_plan.txt):
  Этот модуль объединяет все этапы канонического алгоритма кластеризации:

  ФАЗА 0 (опционально, по порядку — сначала качество, потом избыточность,
  потом разбиение на похожие пары):
    0a. LowInformativenessFilter (algorithms/informativeness_filter.py) —
        статья, 3-й эксперимент: "images with the number of white pixels
        less than 50% of the average... are cut off" (находка №5).
    0b. LinearDependencyFilter (algorithms/reference_filter.py) — статья,
        "Problem Definition": "almost linearly dependent vectors are
        excluded" (находка №3).
    0c. CorrelatedPairSplitter (algorithms/correlated_pair_splitter.py) —
        ЧЕРНОВИК другой (неопубликованной) статьи (theory/Макет новой
        статьи.docx, "Первый этап"): парное разбиение векторов на два
        подмножества похожих (по косинусному сходству) пар, из которых
        для кластеризации берётся только одно. В отличие от 0a/0b, это НЕ
        экспорт из проверенной публикации — экспериментальная возможность.
    Все три выключены по умолчанию (filter_low_informativeness=False,
    filter_dependent=False, split_correlated_pairs=False) — обратная
    совместимость. Порядок при включении нескольких: 0a применяется К
    ИСХОДНОМУ X, 0b — к тому, что ОСТАЛОСЬ после 0a, 0c — к тому, что
    ОСТАЛОСЬ после 0a и 0b (сначала отбрасываем заведомо плохие по качеству
    образы, затем ищем дубликаты среди того, что осталось содержательным,
    затем — при желании — вдвое сокращаем оставшееся через парное
    разбиение).

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
  - Использует algorithms/global_pair, reference_centers, subclass_seed,
    subclass_growth, reference_filter (опционально)
  - Результат используется в SubspaceConjugacyClassifier (фаза C, NB8)
"""

import logging
import time
from typing import List, Literal, Optional
import numpy as np

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.informativeness_filter import (
    DEFAULT_BRIGHTNESS_THRESHOLD,
    DEFAULT_MIN_FRACTION_OF_MEAN,
    LowInformativenessFilter,
)
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.reference_filter import LinearDependencyFilter
from subspace_conjugacy.algorithms.correlated_pair_splitter import CorrelatedPairSplitter
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
    filter_dependent : bool, default=False
        Если True — перед фазой A.1 прогоняет LinearDependencyFilter
        (algorithms/reference_filter.py) на входных векторах: почти линейно
        зависимые (близкие дубликаты/комбинации уже принятых) векторы
        исключаются из кластеризации и получают label -1 в self.labels_ —
        это статья Korshikov & Fursov, "Problem Definition": "almost
        linearly dependent vectors are excluded from the set of reference
        vectors" (refactoring_plan.txt, раздел 10, находка №3). По
        умолчанию выключено — обратная совместимость, поведение не меняется.
    dependency_threshold : float, default=0.999
        Порог показателя сопряжённости для LinearDependencyFilter (см. его
        docstring). Используется только если filter_dependent=True.
    filter_low_informativeness : bool, default=False
        Если True — перед фильтром зависимости (и перед фазой A.1)
        прогоняет LowInformativenessFilter
        (algorithms/informativeness_filter.py): векторы с числом "белых"
        (не фоновых) элементов меньше informativeness_min_fraction от
        среднего по выборке исключаются — статья Korshikov & Fursov, 3-й
        эксперимент: "images with the number of white pixels less than 50%
        of the average... are cut off" (refactoring_plan.txt, раздел 10,
        находка №5). По умолчанию выключено — обратная совместимость.
    informativeness_threshold : float, default=10
        Порог яркости "белого"/полезного элемента для LowInformativenessFilter
        (см. его docstring). Используется только если
        filter_low_informativeness=True.
    informativeness_min_fraction : float, default=0.5
        Минимальная допустимая доля от среднего числа "белых" элементов по
        выборке (статья: 0.5 = 50%). Используется только если
        filter_low_informativeness=True.
    split_correlated_pairs : bool, default=False
        Если True — после фильтров 0a/0b, перед фазой A.1, прогоняет
        CorrelatedPairSplitter (algorithms/correlated_pair_splitter.py):
        итеративно разбивает оставшиеся векторы на пары наиболее похожих
        (максимум косинусного сходства) и делит каждую пару между двумя
        подмножествами — для кластеризации используется только одно из них
        (см. correlated_pairs_subset). Источник — ЧЕРНОВИК другой,
        неопубликованной статьи (theory/Макет новой статьи.docx, "Первый
        этап"), НЕ проверенная публикация Korshikov & Fursov про МРТ мозга
        (в отличие от filter_dependent/filter_low_informativeness). По
        умолчанию выключено — обратная совместимость.
    correlated_pairs_subset : {"a", "b"}, default="a"
        Какое из двух подмножеств CorrelatedPairSplitter использовать для
        кластеризации. Черновик утверждает, что оба равноценны ("любое из
        этих подмножеств может использоваться"). Используется только если
        split_correlated_pairs=True.

    Attributes
    ----------
    subspaces_ : list[np.ndarray] or None
        Финальные базисы подпространств (N, k) для каждого подкласса.
        Если freeze_basis_at задан, k = freeze_basis_at.
    labels_ : np.ndarray or None
        Метки подклассов для каждого вектора (M,) — ИНДЕКСАЦИЯ
        соответствует исходному X, переданному в fit(). Векторы,
        исключённые ЛЮБЫМ из включённых фильтров (informativeness и/или
        dependency), получают label -1 (не участвуют ни в одном подпространстве).
    center_indices_ : np.ndarray or None
        Индексы центров подклассов из фазы A.2-A.3 (в исходном X).
    initial_pairs_ : np.ndarray or None
        Начальные пары векторов (n_subclasses, 2) из фазы B.1 (в исходном X).
    kept_indices_ : np.ndarray or None
        Индексы (в исходном X) векторов, реально участвовавших в
        кластеризации. Совпадает с np.arange(M), если оба фильтра выключены.
    excluded_indices_ : np.ndarray or None
        Индексы (в исходном X) векторов, исключённых ЛЮБЫМ из фильтров
        (объединение LowInformativenessFilter и LinearDependencyFilter, если
        включены оба). Пустой массив, если оба фильтра выключены.
    excluded_by_informativeness_ : np.ndarray or None
        Индексы (в исходном X), исключённые именно LowInformativenessFilter
        (подмножество excluded_indices_). Пустой массив, если
        filter_low_informativeness=False.
    excluded_by_dependency_ : np.ndarray or None
        Индексы (в исходном X), исключённые именно LinearDependencyFilter
        (подмножество excluded_indices_). Пустой массив, если
        filter_dependent=False.
    excluded_by_correlation_split_ : np.ndarray or None
        Индексы (в исходном X) векторов из НЕиспользованного подмножества
        CorrelatedPairSplitter (плюс непарный вектор при нечётном числе
        входных векторов), подмножество excluded_indices_. Пустой массив,
        если split_correlated_pairs=False.
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
        filter_dependent: bool = False,
        dependency_threshold: float = 0.999,
        filter_low_informativeness: bool = False,
        informativeness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
        informativeness_min_fraction: float = DEFAULT_MIN_FRACTION_OF_MEAN,
        split_correlated_pairs: bool = False,
        correlated_pairs_subset: Literal["a", "b"] = "a",
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

        if not (0.0 < dependency_threshold <= 1.0):
            raise ValueError(
                f"dependency_threshold должен быть в (0, 1], получено {dependency_threshold}"
            )

        if not (0.0 < informativeness_min_fraction <= 1.0):
            raise ValueError(
                f"informativeness_min_fraction должен быть в (0, 1], получено "
                f"{informativeness_min_fraction}"
            )

        if correlated_pairs_subset not in ("a", "b"):
            raise ValueError(
                f"correlated_pairs_subset должен быть 'a' или 'b', получено "
                f"{correlated_pairs_subset!r}"
            )

        self.n_subclasses = n_subclasses
        self.freeze_basis_at = freeze_basis_at
        self.growth_strategy = growth_strategy
        self.reg_param = reg_param
        self.filter_dependent = filter_dependent
        self.dependency_threshold = dependency_threshold
        self.filter_low_informativeness = filter_low_informativeness
        self.informativeness_threshold = informativeness_threshold
        self.informativeness_min_fraction = informativeness_min_fraction
        self.split_correlated_pairs = split_correlated_pairs
        self.correlated_pairs_subset = correlated_pairs_subset

        # Результаты fit()
        self.subspaces_: Optional[List[np.ndarray]] = None
        self.labels_: Optional[np.ndarray] = None
        self.center_indices_: Optional[np.ndarray] = None
        self.initial_pairs_: Optional[np.ndarray] = None
        self.kept_indices_: Optional[np.ndarray] = None
        self.excluded_indices_: Optional[np.ndarray] = None
        self.excluded_by_informativeness_: Optional[np.ndarray] = None
        self.excluded_by_dependency_: Optional[np.ndarray] = None
        self.excluded_by_correlation_split_: Optional[np.ndarray] = None
        self.n_subclasses_: Optional[int] = None
        self.is_fitted_: bool = False

        # Внутренние компоненты (для отладки/анализа)
        self._informativeness_filter: Optional[LowInformativenessFilter] = None
        self._dependency_filter: Optional[LinearDependencyFilter] = None
        self._pair_splitter: Optional[CorrelatedPairSplitter] = None
        self._pair_finder: Optional[GlobalMinCosinePairFinder] = None
        self._center_builder: Optional[ReferenceCenterBuilder] = None
        self._seed_attacher: Optional[CosineSecondVectorAttacher] = None
        self._cluster_growth: Optional[ConjugacyClusterGrowth] = None
        self._initial_pair: Optional[tuple] = None

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
            "freeze_basis_at=%s, growth_strategy=%s, filter_low_informativeness=%s, "
            "filter_dependent=%s, split_correlated_pairs=%s.",
            X_arr.shape[0], X_arr.shape[1], self.n_subclasses,
            self.freeze_basis_at, self.growth_strategy,
            self.filter_low_informativeness, self.filter_dependent,
            self.split_correlated_pairs,
        )

        # ФАЗА 0a (опционально): фильтр малоинформативных векторов (статья,
        # 3-й эксперимент) — применяется К ИСХОДНОМУ X, до фильтра зависимости.
        if self.filter_low_informativeness:
            phase_start = time.perf_counter()
            self._informativeness_filter = LowInformativenessFilter(
                brightness_threshold=self.informativeness_threshold,
                min_fraction_of_mean=self.informativeness_min_fraction,
            )
            self._informativeness_filter.fit(X_arr)
            kept_idx = self._informativeness_filter.kept_indices_
            excluded_by_informativeness = self._informativeness_filter.excluded_indices_
            logger.debug(
                "FursovClusterer.fit: фильтр малоинформативности занял %.3fs, "
                "исключено %d/%d.", time.perf_counter() - phase_start,
                len(excluded_by_informativeness), X_arr.shape[0],
            )
        else:
            self._informativeness_filter = None
            kept_idx = np.arange(X_arr.shape[0])
            excluded_by_informativeness = np.array([], dtype=int)

        # ФАЗА 0b (опционально): фильтр почти линейно зависимых векторов
        # (Korshikov & Fursov, "Problem Definition") — применяется к тому,
        # что ОСТАЛОСЬ после 0a (kept_idx — уже в пространстве индексов
        # исходного X_arr, поэтому X_arr[kept_idx] корректно даже если 0a
        # была выключена, когда kept_idx = arange(M)).
        if self.filter_dependent:
            phase_start = time.perf_counter()
            self._dependency_filter = LinearDependencyFilter(
                threshold=self.dependency_threshold,
                reg_param=self.reg_param,
            )
            self._dependency_filter.fit(X_arr[kept_idx])
            kept_idx_local = self._dependency_filter.kept_indices_
            excluded_by_dependency = kept_idx[self._dependency_filter.excluded_indices_]
            kept_idx = kept_idx[kept_idx_local]
            logger.debug(
                "FursovClusterer.fit: фильтр зависимости занял %.3fs, "
                "исключено %d/%d (среди оставшихся после 0a).",
                time.perf_counter() - phase_start,
                len(excluded_by_dependency), X_arr.shape[0] - len(excluded_by_informativeness),
            )
        else:
            self._dependency_filter = None
            excluded_by_dependency = np.array([], dtype=int)

        # ФАЗА 0c (опционально): разбиение на пары похожих векторов
        # (черновик другой статьи, theory/Макет новой статьи.docx, "Первый
        # этап") — применяется к тому, что ОСТАЛОСЬ после 0a и 0b. Для
        # кластеризации берётся только одно из двух построенных подмножеств
        # (correlated_pairs_subset); другое (и непарный вектор, если M
        # нечётно) исключается точно так же, как и 0a/0b.
        if self.split_correlated_pairs:
            phase_start = time.perf_counter()
            self._pair_splitter = CorrelatedPairSplitter()
            self._pair_splitter.fit(X_arr[kept_idx])
            if self.correlated_pairs_subset == "a":
                used_subset_local = self._pair_splitter.subset_a_indices_
                other_subset_local = self._pair_splitter.subset_b_indices_
            else:
                used_subset_local = self._pair_splitter.subset_b_indices_
                other_subset_local = self._pair_splitter.subset_a_indices_
            dropped_local = other_subset_local
            if self._pair_splitter.unpaired_index_ is not None:
                dropped_local = np.concatenate(
                    [dropped_local, [self._pair_splitter.unpaired_index_]]
                )
            excluded_by_correlation_split = kept_idx[dropped_local]
            kept_idx = kept_idx[used_subset_local]
            logger.debug(
                "FursovClusterer.fit: разбиение на похожие пары заняло %.3fs, "
                "оставлено %d/%d (подмножество '%s').",
                time.perf_counter() - phase_start, len(used_subset_local),
                len(used_subset_local) + len(dropped_local), self.correlated_pairs_subset,
            )
        else:
            self._pair_splitter = None
            excluded_by_correlation_split = np.array([], dtype=int)

        excluded_idx = np.union1d(
            np.union1d(excluded_by_informativeness, excluded_by_dependency),
            excluded_by_correlation_split,
        )
        X_for_clustering = X_arr[kept_idx]

        min_required = self.n_subclasses * 2
        if X_for_clustering.shape[0] < min_required:
            logger.error(
                "FursovClusterer.fit: после фильтрации осталось %d векторов "
                "(было %d) < %d требуемых.", X_for_clustering.shape[0],
                X_arr.shape[0], min_required,
            )
            raise ValueError(
                f"После фильтрации осталось {X_for_clustering.shape[0]} векторов "
                f"(было {X_arr.shape[0]}), а требуется минимум {min_required} "
                f"для {self.n_subclasses} подклассов. Ослабьте фильтры "
                f"(informativeness_min_fraction ближе к 0, dependency_threshold "
                f"ближе к 1.0) или отключите filter_low_informativeness/"
                f"filter_dependent."
            )

        # ФАЗА A.1: Глобальная пара с минимальным косинусом
        phase_start = time.perf_counter()
        self._pair_finder = GlobalMinCosinePairFinder()
        self._pair_finder.fit(X_for_clustering)
        initial_pair_local = self._pair_finder.pair_indices_
        logger.debug("FursovClusterer.fit: A.1 заняла %.3fs.", time.perf_counter() - phase_start)

        # ФАЗА A.2-A.3: Последовательное добавление центров через min R
        phase_start = time.perf_counter()
        self._center_builder = ReferenceCenterBuilder(
            n_subclasses=self.n_subclasses,
            reg_param=self.reg_param,
        )
        self._center_builder.fit(X_for_clustering, initial_pair_local)
        center_indices_local = self._center_builder.center_indices_
        logger.debug("FursovClusterer.fit: A.2-A.3 заняла %.3fs.", time.perf_counter() - phase_start)

        # ФАЗА B.1: Второй вектор для каждого центра через min cos
        phase_start = time.perf_counter()
        self._seed_attacher = CosineSecondVectorAttacher()
        self._seed_attacher.fit(X_for_clustering, center_indices_local)
        pairs_local = self._seed_attacher.pairs_
        logger.debug("FursovClusterer.fit: B.1 заняла %.3fs.", time.perf_counter() - phase_start)

        # ФАЗА B.2: Последовательное наполнение кластеров через max R
        phase_start = time.perf_counter()
        self._cluster_growth = ConjugacyClusterGrowth(
            freeze_basis_at=self.freeze_basis_at,
            strategy=self.growth_strategy,
            reg_param=self.reg_param,
        )
        self._cluster_growth.fit(X_for_clustering, pairs_local)
        logger.debug("FursovClusterer.fit: B.2 заняла %.3fs.", time.perf_counter() - phase_start)

        # Сохраняем результаты, переводя индексы из локального пространства
        # X_for_clustering обратно в индексы исходного X (kept_idx[local]) —
        # если фильтр не применялся, kept_idx = arange(M) и это тождественно.
        self.subspaces_ = self._cluster_growth.subspace_bases_
        full_labels = np.full(X_arr.shape[0], -1, dtype=int)
        full_labels[kept_idx] = self._cluster_growth.labels_
        self.labels_ = full_labels
        self.center_indices_ = kept_idx[center_indices_local]
        self.initial_pairs_ = kept_idx[pairs_local]
        self._initial_pair = (
            int(kept_idx[initial_pair_local[0]]),
            int(kept_idx[initial_pair_local[1]]),
        )
        self.kept_indices_ = kept_idx
        self.excluded_indices_ = excluded_idx
        self.excluded_by_informativeness_ = excluded_by_informativeness
        self.excluded_by_dependency_ = excluded_by_dependency
        self.excluded_by_correlation_split_ = excluded_by_correlation_split
        self.n_subclasses_ = self.n_subclasses
        self.is_fitted_ = True

        logger.info(
            "FursovClusterer.fit: готово за %.3fs, размеры подклассов=%s "
            "(исключено всего: %d; малоинформативных: %d; почти линейно "
            "зависимых: %d; отсеяно разбиением на похожие пары: %d).",
            time.perf_counter() - fit_start, self.get_subclass_sizes().tolist(),
            len(excluded_idx), len(excluded_by_informativeness),
            len(excluded_by_dependency), len(excluded_by_correlation_split),
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
            Индексы двух векторов с минимальным косинусом, в индексном
            пространстве исходного X, переданного в fit() (даже если
            filter_dependent=True и A.1 фактически считалась на
            отфильтрованном подмножестве).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        return self._initial_pair

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
