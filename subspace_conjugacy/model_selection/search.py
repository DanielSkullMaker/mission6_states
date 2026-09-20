"""Поиск гиперпараметров SubspaceConjugacyClassifier: grid search и random search.

Тонкая обёртка над sklearn.model_selection.{GridSearchCV,RandomizedSearchCV}:
SubspaceConjugacyClassifier — обычный sklearn-эстиматор (ClassifierMixin +
BaseEstimator, get_params/set_params/clone работают из коробки), поэтому
никакой собственной логики перебора/кросс-валидации здесь не реализуется —
только сборка эстиматора, разумные дефолтные пространства параметров
(param_space.py) и единообразное логирование результата.

⚠ Порядок классов ClassifierMixin/BaseSubspaceEstimator в самом
classifier.py важен для корректной работы этого модуля: is_classifier()
должен возвращать True, иначе sklearn с целочисленным cv использует
обычный KFold вместо StratifiedKFold и на данных, сгруппированных по
классам, даёт полностью неверный score (см. комментарий над классом в
models/classifier.py). Здесь этот инвариант дополнительно проверяется явно
(_ensure_stratifiable), чтобы поломка не проявлялась молча.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from subspace_conjugacy.model_selection.param_space import (
    DEFAULT_PARAM_DISTRIBUTIONS,
    DEFAULT_PARAM_GRID,
)
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

logger = logging.getLogger(__name__)

ParamGrid = Dict[str, Union[List[Any], Any]]


def _ensure_stratifiable(estimator: Any) -> None:
    """Проверяет, что sklearn корректно распознаёт эстиматор как классификатор.

    Raises
    ------
    RuntimeError
        Если is_classifier(estimator) == False — тогда GridSearchCV/
        RandomizedSearchCV с целочисленным ``cv`` тихо используют KFold
        вместо StratifiedKFold, что для этой задачи (кросс-валидация по
        классам опухолей) даёт неверные, но не вызывающие исключений
        результаты. Лучше упасть явно, чем вернуть тихо неверный score.
    """
    if not is_classifier(estimator):
        raise RuntimeError(
            f"is_classifier({type(estimator).__name__}) вернул False — "
            "sklearn не распознает эстиматор как классификатор, поэтому "
            "стратификация по классам (StratifiedKFold) не сработает. "
            "Если estimator — не SubspaceConjugacyClassifier, убедитесь, "
            "что ClassifierMixin указан первым в списке базовых классов."
        )


def _log_cv_results(search: Union["GridSearchCV", "RandomizedSearchCV"], label: str) -> None:
    """Логирует итог поиска и предупреждает о провалившихся комбинациях."""
    n_failed = int(np.sum(np.isnan(search.cv_results_["mean_test_score"])))
    if n_failed > 0:
        logger.warning(
            "%s: %d/%d комбинаций гиперпараметров провалились (mean_test_score=NaN) — "
            "вероятно, n_subclasses слишком велико относительно размера класса "
            "в одном из cv-фолдов. Смотрите search.cv_results_['params'] и "
            "search.cv_results_['mean_test_score'] для деталей.",
            label, n_failed, len(search.cv_results_["mean_test_score"]),
        )
    logger.info(
        "%s: готово, best_params_=%s, best_score_=%.4f (scoring=%s).",
        label, search.best_params_, search.best_score_, search.scoring,
    )


def grid_search_classifier(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Optional[ParamGrid] = None,
    cv: Union[int, Any] = 5,
    scoring: Optional[str] = "accuracy",
    n_jobs: Optional[int] = None,
    refit: bool = True,
    verbose: int = 0,
    base_estimator: Optional[SubspaceConjugacyClassifier] = None,
) -> GridSearchCV:
    """Полный (исчерпывающий) поиск гиперпараметров по сетке значений.

    Перебирает ВСЕ комбинации значений из ``param_grid`` (декартово
    произведение), для каждой считает cv-fold кросс-валидацию через
    ``SubspaceConjugacyClassifier`` (независимая кластеризация каждого
    класса FursovClusterer + классификация по теории C).

    Parameters
    ----------
    X : np.ndarray
        Матрица признаков (M, N).
    y : np.ndarray
        Метки классов (M,) — например, "glioma"/"meningioma"/"pituitary".
    param_grid : dict, optional
        {"n_subclasses": [...], "growth_strategy": [...], "reg_param": [...]}.
        Если None — используется model_selection.param_space.DEFAULT_PARAM_GRID.
        Можно передать и списки значений для freeze_basis_at/любых других
        параметров конструктора SubspaceConjugacyClassifier.
    cv : int or cross-validation generator, default=5
        Число фолдов (или готовый splitter, например StratifiedKFold(...)).
        При целом cv sklearn сам выбирает StratifiedKFold, т.к.
        SubspaceConjugacyClassifier korректно распознаётся как классификатор
        (см. предупреждение в docstring модуля) — не нужно передавать
        StratifiedKFold явно, но можно, если нужен свой random_state/shuffle.
    scoring : str, optional, default="accuracy"
        Метрика sklearn (см. sklearn.metrics.get_scorer_names()).
    n_jobs : int, optional
        Число параллельных процессов (joblib). None — последовательно;
        -1 — все ядра. Внимание: каждая комбинация обучает
        ConjugacyClusterGrowth (O(M^2 * S) на класс) — на реальных МРТ
        (N=65536) это не бесплатно, параллелизм имеет смысл при cv*|grid|>1.
    refit : bool, default=True
        Переобучить лучшую комбинацию на всей выборке X, y
        (``search.best_estimator_`` тогда готов к predict()).
    verbose : int, default=0
        Уровень детализации вывода sklearn (не связан с logging модуля).
    base_estimator : SubspaceConjugacyClassifier, optional
        Шаблон эстиматора (для фиксации параметров, которые не участвуют
        в переборе). Если None — используется SubspaceConjugacyClassifier()
        с параметрами по умолчанию.

    Returns
    -------
    search : sklearn.model_selection.GridSearchCV
        Обученный (после .fit) объект поиска. ``search.best_estimator_``,
        ``search.best_params_``, ``search.best_score_``, ``search.cv_results_``.

    Examples
    --------
    >>> from subspace_conjugacy.model_selection import grid_search_classifier
    >>> search = grid_search_classifier(X_train, y_train, cv=3)
    >>> search.best_params_
    {'growth_strategy': 'default', 'n_subclasses': 8, 'reg_param': 1e-08}
    >>> search.best_estimator_.predict(X_test)
    """
    estimator = base_estimator if base_estimator is not None else SubspaceConjugacyClassifier()
    _ensure_stratifiable(estimator)
    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    n_combinations = 1
    for values in grid.values():
        n_combinations *= len(values)
    logger.info(
        "grid_search_classifier: старт, %d объектов, %d комбинаций x cv=%s, scoring=%s.",
        len(y), n_combinations, cv, scoring,
    )

    search = GridSearchCV(
        estimator,
        param_grid=grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
        error_score=np.nan,
    )
    search.fit(X, y)
    _log_cv_results(search, "grid_search_classifier")
    return search


def random_search_classifier(
    X: np.ndarray,
    y: np.ndarray,
    param_distributions: Optional[ParamGrid] = None,
    n_iter: int = 20,
    cv: Union[int, Any] = 5,
    scoring: Optional[str] = "accuracy",
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    refit: bool = True,
    verbose: int = 0,
    base_estimator: Optional[SubspaceConjugacyClassifier] = None,
) -> RandomizedSearchCV:
    """Случайный поиск гиперпараметров с фиксированным бюджетом попыток.

    В отличие от grid_search_classifier не перебирает все комбинации, а
    делает ``n_iter`` случайных выборок из ``param_distributions`` — не
    страдает от комбинаторного взрыва при добавлении параметров/значений и
    позволяет использовать непрерывные распределения (например,
    ``scipy.stats.loguniform`` для reg_param) вместо дискретной сетки.

    Parameters
    ----------
    X : np.ndarray
        Матрица признаков (M, N).
    y : np.ndarray
        Метки классов (M,).
    param_distributions : dict, optional
        Значения — либо списки (сэмплируются равновероятно), либо объекты
        с методом ``.rvs()`` (scipy.stats distributions). Если None —
        используется model_selection.param_space.DEFAULT_PARAM_DISTRIBUTIONS.
    n_iter : int, default=20
        Количество случайных комбинаций гиперпараметров для проверки.
    cv : int or cross-validation generator, default=5
        См. grid_search_classifier — тот же механизм выбора StratifiedKFold.
    scoring : str, optional, default="accuracy"
        Метрика sklearn.
    random_state : int, optional
        Seed для воспроизводимости выборки комбинаций.
    n_jobs : int, optional
        См. grid_search_classifier.
    refit : bool, default=True
        См. grid_search_classifier.
    verbose : int, default=0
        Уровень детализации вывода sklearn.
    base_estimator : SubspaceConjugacyClassifier, optional
        См. grid_search_classifier.

    Returns
    -------
    search : sklearn.model_selection.RandomizedSearchCV
        Обученный (после .fit) объект поиска — тот же интерфейс результатов,
        что и у GridSearchCV (best_estimator_/best_params_/best_score_/cv_results_).

    Examples
    --------
    >>> from subspace_conjugacy.model_selection import random_search_classifier
    >>> search = random_search_classifier(X_train, y_train, n_iter=15, cv=3, random_state=42)
    >>> search.best_params_
    """
    estimator = base_estimator if base_estimator is not None else SubspaceConjugacyClassifier()
    _ensure_stratifiable(estimator)
    distributions = (
        param_distributions if param_distributions is not None else DEFAULT_PARAM_DISTRIBUTIONS
    )

    logger.info(
        "random_search_classifier: старт, %d объектов, n_iter=%d x cv=%s, scoring=%s.",
        len(y), n_iter, cv, scoring,
    )

    search = RandomizedSearchCV(
        estimator,
        param_distributions=distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=refit,
        verbose=verbose,
        error_score=np.nan,
    )
    search.fit(X, y)
    _log_cv_results(search, "random_search_classifier")
    return search


def search_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "grid",
    **kwargs: Any,
) -> Union[GridSearchCV, RandomizedSearchCV]:
    """Единая точка входа: диспетчер к grid_search_classifier/random_search_classifier.

    Удобно, когда метод поиска сам является параметром (например, приходит
    из конфига эксперимента), и не нужно писать if/else на стороне вызывающего.

    Parameters
    ----------
    X : np.ndarray
        Матрица признаков (M, N).
    y : np.ndarray
        Метки классов (M,).
    method : {"grid", "random"}, default="grid"
        Какой поиск запустить.
    **kwargs
        Прокидываются в grid_search_classifier (method="grid") или
        random_search_classifier (method="random").

    Returns
    -------
    search : GridSearchCV or RandomizedSearchCV
        См. grid_search_classifier/random_search_classifier.

    Raises
    ------
    ValueError
        Если method не "grid" и не "random".
    """
    if method == "grid":
        return grid_search_classifier(X, y, **kwargs)
    elif method == "random":
        return random_search_classifier(X, y, **kwargs)
    else:
        logger.error("search_hyperparameters: неизвестный method='%s'.", method)
        raise ValueError(f"method должен быть 'grid' или 'random', получено '{method}'.")


def summarize_search_results(
    search: Union[GridSearchCV, RandomizedSearchCV], top_n: int = 10
) -> List[Dict[str, Any]]:
    """Компактная сводка результатов поиска без зависимости от pandas.

    Parameters
    ----------
    search : GridSearchCV or RandomizedSearchCV
        Обученный (после .fit) объект поиска.
    top_n : int, default=10
        Сколько лучших комбинаций вернуть (по mean_test_score, убывание;
        NaN — провалившиеся комбинации — идут последними).

    Returns
    -------
    rows : list[dict]
        Список словарей {"rank", "mean_test_score", "std_test_score", "params"},
        отсортированный по возрастанию rank_test_score, длиной min(top_n, n).

    Examples
    --------
    >>> search = grid_search_classifier(X_train, y_train)
    >>> for row in summarize_search_results(search, top_n=3):
    ...     print(row["rank"], row["mean_test_score"], row["params"])
    """
    results = search.cv_results_
    order = np.argsort(results["rank_test_score"])[:top_n]

    return [
        {
            "rank": int(results["rank_test_score"][i]),
            "mean_test_score": float(results["mean_test_score"][i]),
            "std_test_score": float(results["std_test_score"][i]),
            "params": results["params"][i],
        }
        for i in order
    ]
