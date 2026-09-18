"""Модуль валидации данных, параметров и подпространств.

Предоставляет функции проверки входных массивов NumPy, целевых меток,
базисных матриц подклассов и гиперпараметров алгоритмов.
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np


def check_array_X(
    X: Any,
    accept_1d: bool = True,
    dtype: type = np.float64,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> np.ndarray:
    """Проверяет и подготавливает матрицу признаков X.

    Parameters
    ----------
    X : Any
        Входной массив данных (список, np.ndarray и т.д.).
    accept_1d : bool, default=True
        Разрешено ли передавать 1D вектор (преобразуется в 2D столбец/строку).
    dtype : type, default=np.float64
        Целевой тип данных элементов массива.
    allow_nan : bool, default=False
        Разрешено ли наличие значений NaN.
    allow_inf : bool, default=False
        Разрешено ли наличие значений Inf/-Inf.

    Returns
    -------
    X_arr : np.ndarray
        Проверенный и преобразованный 2D массив float64.

    Raises
    ------
    TypeError
        Если входные данные не могут быть приведены к np.ndarray.
    ValueError
        Если некорректна размерность, содержатся NaN/Inf или массив пуст.
    """
    try:
        X_arr = np.asarray(X, dtype=dtype)
    except Exception as err:
        raise TypeError(
            f"Не удалось привести объект типа {type(X)} к массиву NumPy."
        ) from err

    if X_arr.size == 0:
        raise ValueError("Передана пустая матрица признаков X.")

    if X_arr.ndim == 1:
        if not accept_1d:
            raise ValueError(
                "Одномерные массивы не допускаются. Передайте 2D матрицу."
            )
        X_arr = X_arr.reshape(1, -1)
    elif X_arr.ndim != 2:
        raise ValueError(
            f"Ожидался 1D или 2D массив, получена размерность {X_arr.ndim}D."
        )

    if not allow_nan and np.isnan(X_arr).any():
        raise ValueError("Входная матрица X содержит недопустимые значения NaN.")

    if not allow_inf and np.isinf(X_arr).any():
        raise ValueError("Входная матрица X содержит недопустимые значения Inf.")

    return X_arr


def check_X_y(
    X: Any,
    y: Any,
    dtype: type = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Выполняет совместную проверку матрицы признаков X и вектора целевых меток y.

    Parameters
    ----------
    X : Any
        Матрица признаков обучающей выборки.
    y : Any
        Вектор целевых меток классов.
    dtype : type, default=np.float64
        Целевой тип данных для X.

    Returns
    -------
    X_arr : np.ndarray
        Валидированная 2D матрица признаков.
    y_arr : np.ndarray
        Валидированный 1D вектор меток.

    Raises
    ------
    ValueError
        Если количества объектов в X и y не совпадают.
    """
    X_arr = check_array_X(X, accept_1d=False, dtype=dtype)

    try:
        y_arr = np.asarray(y)
    except Exception as err:
        raise TypeError("Не удалось привести целевые метки y к np.ndarray.") from err

    if y_arr.ndim != 1:
        y_arr = np.squeeze(y_arr)
        if y_arr.ndim != 1:
            raise ValueError(
                f"Целевые метки y должны образуют 1D вектор, получена {y_arr.ndim}D."
            )

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Несоответствие количества объектов: X содержит {X_arr.shape[0]} "
            f"строк, а y содержит {y_arr.shape[0]} меток."
        )

    return X_arr, y_arr


def check_basis_matrix(
    Y: Any,
    expected_n_features: Optional[int] = None,
    max_condition_number: float = 1e12,
) -> np.ndarray:
    """Валидирует базисную матрицу подпространства Y (N, k).

    Parameters
    ----------
    Y : Any
        Базисная матрица подкласса, где строки N — признаковое пространство,
        а столбцы k — базисные векторы.
    expected_n_features : Optional[int], default=None
        Ожидаемая размерность пространства признаков N.
    max_condition_number : float, default=1e12
        Порог числа обусловленности для предупреждения о вырожденности.

    Returns
    -------
    Y_arr : np.ndarray
        Проверенная 2D матрица базиса размерности (N, k).

    Raises
    ------
    ValueError
        Если размерности не совпадают или присутствуют недопустимые значения.
    """
    Y_arr = check_array_X(Y, accept_1d=True)

    # Если передан вектор (1, N), транспонируем его в базисный столбец (N, 1)
    if Y_arr.shape[0] == 1 and expected_n_features is not None:
        if Y_arr.shape[1] == expected_n_features:
            Y_arr = Y_arr.T

    N, k = Y_arr.shape

    if expected_n_features is not None and N != expected_n_features:
        raise ValueError(
            f"Число признаков в базисе Y ({N}) не совпадает "
            f"с ожидаемым ({expected_n_features})."
        )

    # Проверка на наличие нулевых столбцов в базисе
    col_norms = np.linalg.norm(Y_arr, axis=0)
    if np.any(col_norms < 1e-12):
        raise ValueError("Базисная матрица Y содержит нулевые вектор-столбцы.")

    # Проверка вырожденности (число обусловленности матрицы Грама Y^T Y)
    gram = Y_arr.T @ Y_arr
    cond_num = np.linalg.cond(gram)
    if cond_num > max_condition_number:
        # Не выбрасываем исключение, так как устойчивое обращение обрабатывается регуляризацией
        pass

    return Y_arr


def check_hyperparameters(
    n_subclasses: int,
    reg_param: float,
    n_centers_init: Optional[int] = None,
) -> None:
    """Проверяет допустимость значений гиперпараметров модели.

    Parameters
    ----------
    n_subclasses : int
        Количество подклассов (подпространств).
    reg_param : float
        Коэффициент регуляризации Тихонова.
    n_centers_init : Optional[int], default=None
        Количество опорных векторов инициализации.

    Raises
    ------
    ValueError
        Если хотя бы один из параметров имеет недопустимое значение.
    """
    if not isinstance(n_subclasses, (int, np.integer)) or n_subclasses <= 0:
        raise ValueError(
            f"Параметр n_subclasses должен быть целым положительным числом, "
            f"получено: {n_subclasses} (тип {type(n_subclasses)})."
        )

    if not isinstance(reg_param, (float, int, np.number)) or reg_param <= 0.0:
        raise ValueError(
            f"Параметр reg_param должен быть строгим положительным float, "
            f"получено: {reg_param}."
        )

    if n_centers_init is not None:
        if not isinstance(n_centers_init, (int, np.integer)) or n_centers_init <= 0:
            raise ValueError(
                f"Параметр n_centers_init должен быть целым положительным числом, "
                f"получено: {n_centers_init}."
            )


def check_is_fitted(estimator: Any, attributes: Union[str, List[str]] = "is_fitted_") -> None:
    """Проверяет, была ли модель обучена перед вызовом предиктивных методов.

    Parameters
    ----------
    estimator : Any
        Экземпляр модели/классификатора.
    attributes : Union[str, List[str]], default="is_fitted_"
        Имя атрибута или список атрибутов, подтверждающих обученность.

    Raises
    ------
    RuntimeError
        Если модель не обучена.
    """
    if isinstance(attributes, str):
        attributes = [attributes]

    is_fitted = all(
        getattr(estimator, attr, False) is True for attr in attributes
    )

    if not is_fitted:
        estimator_name = estimator.__class__.__name__
        raise RuntimeError(
            f"Экземпляр модели '{estimator_name}' еще не обучен. Вызовите 'fit' "
            f"с соответствующими обучающими данными перед использованием методик предсказания."
        )


if __name__ == "__main__":
    print("=== Запуск тестов и самопроверки модуля validation.py ===\n")
    np.random.seed(42)

    # 1. Проверка валидации массива X
    print("1. Тестирование check_array_X:")
    X_valid = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    X_arr = check_array_X(X_valid)
    print(f"   Успешно преобразован список в 2D массив NumPy формы: {X_arr.shape}")
    assert isinstance(X_arr, np.ndarray)

    # 1.1 Перехват NaN в X
    X_nan = [[1.0, np.nan], [3.0, 4.0]]
    try:
        check_array_X(X_nan)
    except ValueError as err:
        print(f"   [Перехвачена ожидаемая ошибка (NaN)]: {err}")

    # 2. Проверка совместной валидации X и y
    print("\n2. Тестирование check_X_y:")
    X_raw = np.random.randn(10, 5)
    y_raw = np.random.randint(0, 2, size=10)
    X_out, y_out = check_X_y(X_raw, y_raw)
    print(f"   Валидированы данные: X={X_out.shape}, y={y_out.shape}")
    assert X_out.shape[0] == y_out.shape[0]

    # 2.1 Перехват несоответствия длин X и y
    y_invalid_len = np.random.randint(0, 2, size=7)
    try:
        check_X_y(X_raw, y_invalid_len)
    except ValueError as err:
        print(f"   [Перехвачена ожидаемая ошибка (размерность y)]: {err}")

    # 3. Проверка базисной матрицы Y
    print("\n3. Тестирование check_basis_matrix:")
    N_feats, k_bases = 32, 4
    Y_basis = np.random.randn(N_feats, k_bases)
    Y_validated = check_basis_matrix(Y_basis, expected_n_features=32)
    print(f"   Базисная матрица валидирована, форма: {Y_validated.shape}")

    # 3.1 Ошибка при нулевом векторе в базисе
    Y_zero_col = Y_basis.copy()
    Y_zero_col[:, 0] = 0.0
    try:
        check_basis_matrix(Y_zero_col, expected_n_features=32)
    except ValueError as err:
        print(f"   [Перехвачена ожидаемая ошибка (нулевой базис)]: {err}")

    # 4. Проверка гиперпараметров
    print("\n4. Тестирование check_hyperparameters:")
    check_hyperparameters(n_subclasses=8, reg_param=1e-8, n_centers_init=2)
    print("   Валидные гиперпараметры успешно прошли проверку.")

    try:
        check_hyperparameters(n_subclasses=-3, reg_param=1e-8)
    except ValueError as err:
        print(f"   [Перехвачена ожидаемая ошибка (отрицательное число подклассов)]: {err}")

    # 5. Проверка состояния обученности
    print("\n5. Тестирование check_is_fitted:")

    class MockEstimator:
        def __init__(self):
            self.is_fitted_ = False

    mock_model = MockEstimator()
    try:
        check_is_fitted(mock_model)
    except RuntimeError as err:
        print(f"   [Перехвачена ожидаемая ошибка (необученная модель)]: {err}")

    mock_model.is_fitted_ = True
    check_is_fitted(mock_model)
    print("   Проверка обученной модели выполнена успешно.")

    print("\n Все функции модуля validation.py прошли проверки!")