"""Базовый абстрактный модуль для моделей подпространственной сопряженности.

Предоставляет базовый класс BaseSubspaceEstimator, определяющий общий
интерфейс, валидацию входных данных и управление состоянием обученности.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator


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
            raise ValueError(
                f"Параметр n_subclasses должен быть > 0, получено {self.n_subclasses}."
            )
        if self.n_centers_init <= 0:
            raise ValueError(
                f"Параметр n_centers_init должен быть > 0, получено {self.n_centers_init}."
            )
        if self.reg_param <= 0:
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
            raise ValueError(
                f"Ожидалась 1D или 2D матрица признаков, получена размерность {X_arr.ndim}D."
            )

        if X_arr.size == 0:
            raise ValueError("Передана пустая матрица признаков X.")

        if self.is_fitted_ and self.n_features_in_ is not None:
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Количество признаков X ({X_arr.shape[1]}) не совпадает "
                    f"с количеством признаков при обучении ({self.n_features_in_})."
                )

        y_arr = None
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.shape[0] != X_arr.shape[0]:
                raise ValueError(
                    f"Несовпадение размера: X содержит {X_arr.shape[0]} объектов, "
                    f"а y содержит {y_arr.shape[0]} элементов."
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


if __name__ == "__main__":
    print("=== Запуск самотестирования и проверки base.py ===\n")

    # Вспомогательный класс-заглушка для проверки абстрактного класса
    class DummySubspaceEstimator(BaseSubspaceEstimator):
        """Тестовая реализация базового класса."""

        def fit(
            self, X: np.ndarray, y: Optional[np.ndarray] = None
        ) -> "DummySubspaceEstimator":
            self._validate_input_params()
            X_clean, y_clean = self._validate_data(X, y)
            self.n_features_in_ = X_clean.shape[1]
            self.is_fitted_ = True
            print(f"   [fit] Успешно обработано объектов: {X_clean.shape[0]}, "
                  f"признаков: {self.n_features_in_}")
            return self

        def predict_r_matrix(self, X: np.ndarray) -> np.ndarray:
            self._check_is_fitted()
            X_clean, _ = self._validate_data(X)
            # Возвращаем заглушку матрицы показателей R
            return np.ones((X_clean.shape[0], self.n_subclasses))

    # 1. Проверка валидации параметров инициализации
    print("1. Тестирование проверки валидности гиперпараметров:")
    try:
        invalid_estimator = DummySubspaceEstimator(n_subclasses=-1)
        invalid_estimator.fit(np.random.randn(10, 5))
    except ValueError as err:
        print(f"   [Перехвачена ошибка]: {err}")

    # 2. Инициализация нормального эстиматора
    estimator = DummySubspaceEstimator(n_subclasses=4, n_centers_init=2)

    # 3. Проверка метода _check_is_fitted до обучения
    print("\n2. Проверка защиты от предсказаний без предварительного fit:")
    try:
        estimator.predict_r_matrix(np.random.randn(5, 10))
    except RuntimeError as err:
        print(f"   [Перехвачена ошибка]: {err}")

    # 4. Нормальное обучение и валидация
    print("\n3. Тестирование валидации данных во время обучения (fit):")
    X_train = np.random.randn(20, 16)
    y_train = np.random.randint(0, 2, size=20)
    estimator.fit(X_train, y_train)
    assert estimator.is_fitted_ is True
    assert estimator.n_features_in_ == 16

    # 5. Проверка ошибочной размерности на шаге predict
    print("\n4. Тестирование несоответствия размерностей признаков:")
    X_test_invalid = np.random.randn(5, 8)  # 8 признаков вместо 16
    try:
        estimator.predict_r_matrix(X_test_invalid)
    except ValueError as err:
        print(f"   [Перехвачена ошибка]: {err}")

    # 6. Успешное выполнение предсказания
    X_test_valid = np.random.randn(5, 16)
    r_res = estimator.predict_r_matrix(X_test_valid)
    print("\n5. Успешное получение матрицы результатов предсказания:")
    print(f"   Форма матрицы R: {r_res.shape}")
    assert r_res.shape == (5, 4)

    print("\n Все базовые проверки и валидации пройдены!")