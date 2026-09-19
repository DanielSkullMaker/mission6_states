"""Базовый абстрактный модуль для моделей подпространственной сопряженности.

Предоставляет базовый класс BaseSubspaceEstimator, определяющий общий
интерфейс, валидацию входных данных и управление состоянием обученности.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class BaseSubspaceEstimator(BaseEstimator, ABC):
    """Абстрактный базовый класс для моделей на основе подпространств.

    Parameters
    ----------
    n_subclasses : int, default=8
        Количество подклассов (подпространств), формируемых для каждого класса.
    n_centers_init : int, default=2
        Количество начальных опорных векторов (центров) для каждого подкласса.
    reg_param : float, default=1e-8
        Параметр регуляризации для защиты матрицы Грама от вырожденности.

    Attributes
    ----------
    n_features_in_ : int or None
        Количество признаков, переданных во время вызова метода `fit`.
    is_fitted_ : bool
        Флаг, указывающий, обучена ли модель.
    """

    def __init__(
        self,
        n_subclasses: int = 8,
        n_centers_init: int = 2,
        reg_param: float = 1e-8,
    ) -> None:
        self.n_subclasses = n_subclasses
        self.n_centers_init = n_centers_init
        self.reg_param = reg_param
        self.n_features_in_: Optional[int] = None
        self.is_fitted_: bool = False

    def _validate_input_params(self) -> None:
        """Проверяет гиперпараметры инициализации модели.

        Raises
        ------
        ValueError
            Если параметры выходят за допустимые границы.
        """
        if self.n_subclasses <= 0:
            logger.error(
                "%s._validate_input_params: n_subclasses=%s <= 0.",
                type(self).__name__, self.n_subclasses,
            )
            raise ValueError(
                f"Параметр n_subclasses должен быть > 0, получено {self.n_subclasses}."
            )
        if self.n_centers_init <= 0:
            logger.error(
                "%s._validate_input_params: n_centers_init=%s <= 0.",
                type(self).__name__, self.n_centers_init,
            )
            raise ValueError(
                f"Параметр n_centers_init должен быть > 0, получено {self.n_centers_init}."
            )
        if self.reg_param <= 0:
            logger.error(
                "%s._validate_input_params: reg_param=%s <= 0.",
                type(self).__name__, self.reg_param,
            )
            raise ValueError(
                f"Параметр reg_param должен быть > 0, получено {self.reg_param}."
            )

    def _validate_data(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Проверяет и приводить тип и размерность входных данных.

        Parameters
        ----------
        X : np.ndarray
            Входная матрица признаков.
        y : np.ndarray, optional
            Вектор меток классов.

        Returns
        -------
        X_validated : np.ndarray
            Двумерный массив numpy с типом float64.
        y_validated : np.ndarray or None
            Массив меток классов, если был передан.

        Raises
        ------
        ValueError
            При некорректных размерностях или пустых данных.
        """
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        elif X_arr.ndim != 2:
            logger.error(
                "%s._validate_data: неверная размерность X.ndim=%d.",
                type(self).__name__, X_arr.ndim,
            )
            raise ValueError(
                f"Ожидалась 1D или 2D матрица признаков, получена размерность {X_arr.ndim}D."
            )

        if X_arr.size == 0:
            logger.error("%s._validate_data: передана пустая матрица X.", type(self).__name__)
            raise ValueError("Передана пустая матрица признаков X.")

        if self.is_fitted_ and self.n_features_in_ is not None:
            if X_arr.shape[1] != self.n_features_in_:
                logger.error(
                    "%s._validate_data: несовпадение числа признаков (%d != %d).",
                    type(self).__name__, X_arr.shape[1], self.n_features_in_,
                )
                raise ValueError(
                    f"Количество признаков X ({X_arr.shape[1]}) не совпадает "
                    f"с количеством признаков при обучении ({self.n_features_in_})."
                )

        y_arr = None
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.shape[0] != X_arr.shape[0]:
                logger.error(
                    "%s._validate_data: несовпадение размера X (%d) и y (%d).",
                    type(self).__name__, X_arr.shape[0], y_arr.shape[0],
                )
                raise ValueError(
                    f"Несовпадение размера: X содержит {X_arr.shape[0]} объектов, "
                    f"а y содержит {y_arr.shape[0]} элементов."
                )

        logger.debug(
            "%s._validate_data: X.shape=%s%s.",
            type(self).__name__, X_arr.shape,
            f", y.shape={y_arr.shape}" if y_arr is not None else "",
        )
        return X_arr, y_arr

    def _check_is_fitted(self) -> None:
        """Проверяет, обучена ли модель перед выполнением предсказаний.

        Raises
        ------
        RuntimeError
            Если вызов выполняется до проведения метода `fit`.
        """
        if not self.is_fitted_:
            logger.error(
                "%s._check_is_fitted: обращение к предсказаниям до fit().",
                type(self).__name__,
            )
            raise RuntimeError(
                "Экземпляр модели еще не обучен. Вызовите метод 'fit' "
                "перед запуском вычислений предсказаний."
            )

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "BaseSubspaceEstimator":
        """Обучает модель на входных данных.

        Parameters
        ----------
        X : np.ndarray
            Матрица признаков обучающей выборки.
        y : np.ndarray, optional
            Целевые метки классов.

        Returns
        -------
        self : BaseSubspaceEstimator
            Возвращает экземпляр самого себя.
        """
        pass

    @abstractmethod
    def predict_r_matrix(self, X: np.ndarray) -> np.ndarray:
        """Рассчитывает матрицу показателей сопряженности R(x, Y).

        Parameters
        ----------
        X : np.ndarray
            Матрица признаков размерности (M, N).

        Returns
        -------
        R_matrix : np.ndarray
            Матрица сопряженностей.
        """
        pass
