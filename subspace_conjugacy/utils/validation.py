"""Модуль валидации данных, параметров и подпространств.

Предоставляет функции проверки входных массивов NumPy, целевых меток,
базисных матриц подклассов и гиперпараметров алгоритмов.
"""

import logging
from typing import Any, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


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
        logger.error("check_array_X: не удалось привести %s к np.ndarray: %s.", type(X), err)
        raise TypeError(
            f"Не удалось привести объект типа {type(X)} к массиву NumPy."
        ) from err

    if X_arr.size == 0:
        logger.error("check_array_X: передана пустая матрица.")
        raise ValueError("Передана пустая матрица признаков X.")

    if X_arr.ndim == 1:
        if not accept_1d:
            logger.error("check_array_X: 1D массив запрещён (accept_1d=False).")
            raise ValueError(
                "Одномерные массивы не допускаются. Передайте 2D матрицу."
            )
        X_arr = X_arr.reshape(1, -1)
    elif X_arr.ndim != 2:
        logger.error("check_array_X: неверная размерность X.ndim=%d.", X_arr.ndim)
        raise ValueError(
            f"Ожидался 1D или 2D массив, получена размерность {X_arr.ndim}D."
        )

    if not allow_nan and np.isnan(X_arr).any():
        logger.error("check_array_X: обнаружены NaN (allow_nan=False).")
        raise ValueError("Входная матрица X содержит недопустимые значения NaN.")

    if not allow_inf and np.isinf(X_arr).any():
        logger.error("check_array_X: обнаружены Inf (allow_inf=False).")
        raise ValueError("Входная матрица X содержит недопустимые значения Inf.")

    logger.debug("check_array_X: OK, shape=%s, dtype=%s.", X_arr.shape, X_arr.dtype)
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
        logger.error("check_X_y: не удалось привести y к np.ndarray: %s.", err)
        raise TypeError("Не удалось привести целевые метки y к np.ndarray.") from err

    if y_arr.ndim != 1:
        y_arr = np.squeeze(y_arr)
        if y_arr.ndim != 1:
            logger.error("check_X_y: y.ndim=%d после squeeze, ожидался 1D.", y_arr.ndim)
            raise ValueError(
                f"Целевые метки y должны образуют 1D вектор, получена {y_arr.ndim}D."
            )

    if X_arr.shape[0] != y_arr.shape[0]:
        logger.error(
            "check_X_y: несовпадение количества объектов X=%d, y=%d.",
            X_arr.shape[0], y_arr.shape[0],
        )
        raise ValueError(
            f"Несоответствие количества объектов: X содержит {X_arr.shape[0]} "
            f"строк, а y содержит {y_arr.shape[0]} меток."
        )

    logger.debug("check_X_y: OK, X.shape=%s, y.shape=%s.", X_arr.shape, y_arr.shape)
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
        logger.error(
            "check_basis_matrix: N=%d не совпадает с expected_n_features=%d.",
            N, expected_n_features,
        )
        raise ValueError(
            f"Число признаков в базисе Y ({N}) не совпадает "
            f"с ожидаемым ({expected_n_features})."
        )

    # Проверка на наличие нулевых столбцов в базисе
    col_norms = np.linalg.norm(Y_arr, axis=0)
    if np.any(col_norms < 1e-12):
        logger.error(
            "check_basis_matrix: найдены нулевые столбцы (col_norms=%s).",
            col_norms.tolist(),
        )
        raise ValueError("Базисная матрица Y содержит нулевые вектор-столбцы.")

    # Проверка вырожденности (число обусловленности матрицы Грама Y^T Y)
    gram = Y_arr.T @ Y_arr
    cond_num = np.linalg.cond(gram)
    if cond_num > max_condition_number:
        # Не выбрасываем исключение — устойчивое обращение обеспечивает
        # регуляризация в core.metrics.compute_gram_inverse, но пользователю
        # стоит знать, что базис близок к линейно зависимому.
        logger.warning(
            "check_basis_matrix: число обусловленности Y^T Y = %.3e превышает "
            "порог %.3e — базис почти вырожден (близкие/коллинеарные "
            "базисные векторы).", cond_num, max_condition_number,
        )

    logger.debug("check_basis_matrix: OK, N=%d, k=%d, cond=%.3e.", N, k, cond_num)
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
        logger.error("check_hyperparameters: n_subclasses некорректен: %r.", n_subclasses)
        raise ValueError(
            f"Параметр n_subclasses должен быть целым положительным числом, "
            f"получено: {n_subclasses} (тип {type(n_subclasses)})."
        )

    if not isinstance(reg_param, (float, int, np.number)) or reg_param <= 0.0:
        logger.error("check_hyperparameters: reg_param некорректен: %r.", reg_param)
        raise ValueError(
            f"Параметр reg_param должен быть строгим положительным float, "
            f"получено: {reg_param}."
        )

    if n_centers_init is not None:
        if not isinstance(n_centers_init, (int, np.integer)) or n_centers_init <= 0:
            logger.error(
                "check_hyperparameters: n_centers_init некорректен: %r.", n_centers_init,
            )
            raise ValueError(
                f"Параметр n_centers_init должен быть целым положительным числом, "
                f"получено: {n_centers_init}."
            )

    logger.debug(
        "check_hyperparameters: OK, n_subclasses=%s, reg_param=%s, n_centers_init=%s.",
        n_subclasses, reg_param, n_centers_init,
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
        logger.error("check_is_fitted: '%s' не обучен (attributes=%s).", estimator_name, attributes)
        raise RuntimeError(
            f"Экземпляр модели '{estimator_name}' еще не обучен. Вызовите 'fit' "
            f"с соответствующими обучающими данными перед использованием методик предсказания."
        )
