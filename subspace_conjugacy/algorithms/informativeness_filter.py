"""Low Informativeness Filter (статья, 3-й эксперимент, находка №5).

Korshikov & Fursov, "Pathology Recognition Based on Conjugacy Criteria with
Subspaces of Reference Images" (theory/VI_Korshikov_VA_Fursov_..._Conjugacy.docx):
  "When analyzing the results of the second experiment, the images
  affecting only a small area of the brain and having low informativeness
  show low recognition rate, therefore, it is necessary to filter these
  images. For this purpose, in the third experiment, an additional step is
  added during preprocessing, during which images with the number of white
  pixels less than 50% of the average number of white pixels in the images
  of the set are cut off. The results in Table 2 show that the
  classification accuracy has increased."

"Белые пиксели" в статье — это пиксели, не относящиеся к фону (та же идея,
что и в preprocessing.normalization.suppress_background: маска "не фон"
строится по порогу яркости). Здесь фильтр применяется к уже
ВЕКТОРИЗОВАННЫМ данным (эквивалентно применению к изображению — подсчёт
"белых" элементов не зависит от порядка обхода при векторизации), поэтому
он умещается в тот же слой, что и LinearDependencyFilter (reference_filter.py):
работает с матрицей X (M, N), а не с изображениями напрямую — согласовано с
тем, как FursovClusterer/SubspaceConjugacyClassifier принимают только
векторы, не изображения.

Связь с canonical pipeline (refactoring_plan.txt, раздел 10, находка №5):
  Применяется (опционально, по умолчанию выключено) ПЕРЕД фазой A.1, вместе
  с (и, по порядку, раньше) LinearDependencyFilter — сначала отбрасываем
  малоинформативные образы (качество данных), затем — почти дублирующие
  среди оставшихся (избыточность). FursovClusterer(filter_low_informativeness=True)
  прогоняет оба фильтра последовательно; исключённые обоими векторы
  получают label -1 в FursovClusterer.labels_.
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

#: Порог яркости "белого"/полезного (не фонового) элемента — согласован с
#: preprocessing.normalization.DEFAULT_THRESHOLD (та же идея "фон/не фон").
DEFAULT_BRIGHTNESS_THRESHOLD = 10

#: Статья: "less than 50% of the average" — используется буквально.
DEFAULT_MIN_FRACTION_OF_MEAN = 0.5


class LowInformativenessFilter:
    """Исключает малоинформативные векторы (статья, 3-й эксперимент).

    Для каждого вектора считает число "белых" (не фоновых) элементов —
    тех, что >= brightness_threshold. Вектор исключается, если это число
    меньше ``min_fraction_of_mean`` от СРЕДНЕГО числа "белых" элементов по
    всей выборке (статья использует 0.5, т.е. 50%).

    Parameters
    ----------
    brightness_threshold : float, default=10
        Порог яркости. Элемент вектора считается "белым"/полезным, если
        его значение >= brightness_threshold (та же идея, что и в
        preprocessing.normalization.suppress_background).
    min_fraction_of_mean : float, default=0.5
        Минимальная допустимая доля от среднего числа "белых" элементов по
        выборке — статья использует 0.5 (50%). Должен быть в (0, 1].

    Attributes
    ----------
    white_counts_ : np.ndarray or None
        Число "белых" элементов для каждого вектора выборки (M,).
    mean_white_count_ : float or None
        Среднее число "белых" элементов по выборке.
    cutoff_ : float or None
        Порог отсечения = min_fraction_of_mean * mean_white_count_.
    kept_indices_ : np.ndarray or None
        Индексы (в исходном X) принятых векторов.
    excluded_indices_ : np.ndarray or None
        Индексы (в исходном X) исключённых как малоинформативные.
    is_fitted_ : bool

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.algorithms.informativeness_filter import LowInformativenessFilter
    >>> X = np.full((10, 100), 200.0)  # почти всё "белое"
    >>> X[5] = 0.0  # один почти полностью "чёрный" (фоновый) вектор
    >>> filt = LowInformativenessFilter()
    >>> filt.fit(X)
    >>> 5 in filt.excluded_indices_
    True
    """

    def __init__(
        self,
        brightness_threshold: float = DEFAULT_BRIGHTNESS_THRESHOLD,
        min_fraction_of_mean: float = DEFAULT_MIN_FRACTION_OF_MEAN,
    ) -> None:
        if not (0.0 < min_fraction_of_mean <= 1.0):
            raise ValueError(
                f"min_fraction_of_mean должен быть в (0, 1], получено "
                f"{min_fraction_of_mean}."
            )

        self.brightness_threshold = brightness_threshold
        self.min_fraction_of_mean = min_fraction_of_mean

        self.white_counts_: Optional[np.ndarray] = None
        self.mean_white_count_: Optional[float] = None
        self.cutoff_: Optional[float] = None
        self.kept_indices_: Optional[np.ndarray] = None
        self.excluded_indices_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray) -> "LowInformativenessFilter":
        """Находит и исключает малоинформативные векторы из X.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N), M >= 1.

        Returns
        -------
        self : LowInformativenessFilter
            Возвращает экземпляр самого себя.

        Raises
        ------
        ValueError
            Если X не 2D или пуст.
        """
        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]

        white_counts = np.sum(X_arr >= self.brightness_threshold, axis=1)
        mean_white_count = float(white_counts.mean())
        cutoff = self.min_fraction_of_mean * mean_white_count

        logger.info(
            "LowInformativenessFilter.fit: старт, %d векторов, "
            "brightness_threshold=%s, mean_white_count=%.2f, cutoff=%.2f "
            "(min_fraction_of_mean=%.2f).",
            n_samples, self.brightness_threshold, mean_white_count, cutoff,
            self.min_fraction_of_mean,
        )

        kept_mask = white_counts >= cutoff
        kept = np.where(kept_mask)[0]
        excluded = np.where(~kept_mask)[0]

        self.white_counts_ = white_counts
        self.mean_white_count_ = mean_white_count
        self.cutoff_ = cutoff
        self.kept_indices_ = kept
        self.excluded_indices_ = excluded
        self.is_fitted_ = True

        logger.info(
            "LowInformativenessFilter.fit: готово, принято %d/%d, исключено %d.",
            len(kept), n_samples, len(excluded),
        )
        if len(excluded) > 0:
            logger.warning(
                "LowInformativenessFilter.fit: %d вектор(ов) исключены как "
                "малоинформативные (белых элементов < %.2f) — индексы (в "
                "исходном X): %s.", len(excluded), cutoff, excluded.tolist(),
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Возвращает только принятые (не малоинформативные) векторы.

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
            Матрица векторов (M, N).

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
            logger.error("LowInformativenessFilter: обращение к результатам до fit().")
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(X) перед использованием."
            )

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"LowInformativenessFilter(min_fraction_of_mean="
                f"{self.min_fraction_of_mean}, kept={len(self.kept_indices_)}, "
                f"excluded={len(self.excluded_indices_)})"
            )
        return (
            f"LowInformativenessFilter(min_fraction_of_mean="
            f"{self.min_fraction_of_mean}, not fitted)"
        )
