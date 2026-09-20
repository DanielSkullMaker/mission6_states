"""Reference Vector Linear Dependency Filter (статья, секция "Problem Definition").

Теория (Korshikov & Fursov, "Pathology Recognition Based on Conjugacy
Criteria with Subspaces of Reference Images", theory/VI_Korshikov_VA_
Fursov_..._Conjugacy.docx):
  "A vector of dimension N×1 is assigned to each image (by row sweep), and
  almost linearly dependent vectors are excluded from the set of reference
  vectors."

Статья не даёт явной формулы порога "почти линейной зависимости". Здесь
используется тот же показатель сопряжённости R(x, Y), который уже
центральный для всего метода (core/metrics.py::conjugate_criterion):
R(x, Y) -> 1 означает, что x почти целиком лежит в подпространстве,
натянутом на столбцы Y — это и есть формальное определение "почти линейной
зависимости" вектора x от множества векторов Y. Вектор считается почти
линейно зависимым от уже принятых эталонных векторов своего класса, если
его сопряжённость с пространством, натянутым на них, не меньше threshold
(близко к 1 — почти точный дубликат/линейная комбинация уже принятых).

Связь с canonical pipeline (refactoring_plan.txt, раздел 10, находка №3):
  Применяется (опционально, по умолчанию выключено) ПЕРЕД фазой A.1, на
  исходном множестве эталонных векторов класса — как и в статье, отдельно
  от кластеризации A.1-A.3/B.1-B.2. FursovClusterer(filter_dependent=True)
  прогоняет этот фильтр внутри своего fit() перед запуском канона;
  исключённые векторы никогда не участвуют в кластеризации и получают
  label -1 в FursovClusterer.labels_ — это ровно то "some number of
  reference vectors will remain unallocated", о котором пишет статья в
  разделе "Description of the Clustering Method". Канонический growth-цикл
  (ConjugacyClusterGrowth) при этом остаётся исчерпывающим по своим
  собственным правилам — он ничего не знает о фильтрации, "неприсоединённые"
  векторы в смысле статьи возникают на этапе ДО кластеризации, а не как
  особое поведение growth-цикла.
"""

import logging
from typing import Dict, Optional
import numpy as np

from subspace_conjugacy.core.metrics import conjugate_criterion

logger = logging.getLogger(__name__)


class LinearDependencyFilter:
    """Исключает почти линейно зависимые векторы из эталонного множества.

    Проходит по векторам X в порядке следования и последовательно набирает
    "принятое" подпространство Y_accepted: первый вектор принимается
    безусловно, каждый следующий — если его показатель сопряжённости с
    ТЕКУЩИМ Y_accepted строго меньше threshold (иначе он почти целиком
    объясняется уже принятыми векторами и исключается, не расширяя
    Y_accepted).

    Parameters
    ----------
    threshold : float, default=0.999
        Порог показателя сопряжённости R(x, Y_accepted) из диапазона (0, 1].
        Вектор исключается, если R >= threshold. Чем ближе к 1, тем
        консервативнее фильтр (исключаются только почти точные дубликаты
        или векторы, лежащие практически точно в уже накопленном
        подпространстве); слишком маленький threshold рискует исключить
        содержательно разные векторы.
    reg_param : float, default=1e-8
        Регуляризация Тихонова для (Y^T Y)^{-1}
        (см. core.metrics.conjugate_criterion).

    Notes
    -----
    Как только число ПРИНЯТЫХ векторов достигает размерности признаков N,
    Y_accepted становится базисом полного ранга всего N-мерного пространства
    — и ЛЮБОЙ следующий вектор математически неизбежно лежит в его span
    (R -> 1), то есть будет исключён, независимо от того, насколько
    "непохож" он содержательно. Для реальных МРТ-векторов (N=65536, M порядка
    сотен эталонных изображений на класс) M << N, и это никогда не
    проявляется — но на низкоразмерных синтетических данных (M > N, типично
    для юнит-тестов) фильтр может начать исключать почти всё подряд после
    N-го принятого вектора. Это ожидаемое математическое свойство критерия
    сопряжённости, а не ошибка фильтра.

    Attributes
    ----------
    kept_indices_ : np.ndarray or None
        Индексы (в исходном X, в порядке обработки) принятых векторов.
    excluded_indices_ : np.ndarray or None
        Индексы (в исходном X) исключённых как почти линейно зависимые.
    r_values_ : Dict[int, float] or None
        {excluded_index: R(x, Y_accepted на момент проверки)} — для
        диагностики, насколько именно вектор был близок к зависимости.
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms.reference_filter import LinearDependencyFilter
    >>> np.random.seed(0)
    >>> X = np.random.randn(50, 64)
    >>> X[10] = X[0] * 2.0 + 1e-6  # почти точный дубликат (с точностью до масштаба) X[0]
    >>> filt = LinearDependencyFilter(threshold=0.999)
    >>> filt.fit(X)
    >>> 10 in filt.excluded_indices_
    True
    >>> X_filtered = filt.transform(X)
    >>> X_filtered.shape[0] < X.shape[0]
    True
    """

    def __init__(
        self,
        threshold: float = 0.999,
        reg_param: float = 1e-8,
    ) -> None:
        if not (0.0 < threshold <= 1.0):
            raise ValueError(
                f"threshold должен быть в диапазоне (0, 1], получено {threshold}."
            )

        self.threshold = threshold
        self.reg_param = reg_param

        self.kept_indices_: Optional[np.ndarray] = None
        self.excluded_indices_: Optional[np.ndarray] = None
        self.r_values_: Optional[Dict[int, float]] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray) -> "LinearDependencyFilter":
        """Находит и исключает почти линейно зависимые векторы из X.

        Parameters
        ----------
        X : np.ndarray
            Матрица эталонных векторов размерности (M, N), M >= 1.

        Returns
        -------
        self : LinearDependencyFilter
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если X не 2D или пуст.
        """
        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]
        logger.info(
            "LinearDependencyFilter.fit: старт, %d векторов, threshold=%.4f.",
            n_samples, self.threshold,
        )

        kept = [0]
        excluded: list = []
        r_values: Dict[int, float] = {}
        Y_accepted = X_arr[[0]].T  # (N, 1)

        for i in range(1, n_samples):
            x = X_arr[i]
            r = conjugate_criterion(x, Y_accepted, reg_param=self.reg_param)

            if r >= self.threshold:
                excluded.append(i)
                r_values[i] = float(r)
                logger.debug(
                    "LinearDependencyFilter.fit: вектор %d исключён "
                    "(R=%.6f >= threshold=%.4f, k=%d).",
                    i, r, self.threshold, Y_accepted.shape[1],
                )
            else:
                kept.append(i)
                Y_accepted = np.column_stack([Y_accepted, x])

        self.kept_indices_ = np.array(kept, dtype=int)
        self.excluded_indices_ = np.array(excluded, dtype=int)
        self.r_values_ = r_values
        self.is_fitted_ = True

        logger.info(
            "LinearDependencyFilter.fit: готово, принято %d/%d, исключено %d.",
            len(kept), n_samples, len(excluded),
        )
        if excluded:
            logger.warning(
                "LinearDependencyFilter.fit: %d вектор(ов) исключены как почти "
                "линейно зависимые (R >= %.4f) — индексы (в исходном X): %s.",
                len(excluded), self.threshold, excluded,
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Возвращает только принятые (не почти линейно зависимые) векторы.

        Parameters
        ----------
        X : np.ndarray
            Матрица (M, N), на которой был выполнен fit() (или с тем же
            порядком строк).

        Returns
        -------
        X_kept : np.ndarray
            Подматрица (len(kept_indices_), N).

        Raises
        ------
        RuntimeError
            Если fit() ещё не был вызван.
        """
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr[self.kept_indices_]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Эквивалент fit(X).transform(X) за один проход.

        Parameters
        ----------
        X : np.ndarray
            Матрица эталонных векторов (M, N).

        Returns
        -------
        X_kept : np.ndarray
            Подматрица принятых векторов.
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
            logger.error("LinearDependencyFilter: обращение к результатам до fit().")
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X) перед использованием."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"LinearDependencyFilter(threshold={self.threshold}, "
                f"kept={len(self.kept_indices_)}, excluded={len(self.excluded_indices_)})"
            )
        return f"LinearDependencyFilter(threshold={self.threshold}, not fitted)"
