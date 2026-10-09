"""Параллельное обучение SubspaceConjugacyClassifier по классам.

`SubspaceConjugacyClassifier.fit()` (models/classifier.py) кластеризует
КАЖДЫЙ класс НЕЗАВИСИМО последовательным циклом (`for cls in self.classes_`)
— классы не взаимодействуют друг с другом на этапе кластеризации, поэтому
это "внешний" уровень параллелизма, независимый от "внутреннего" уровня
(параллелизация B.2 внутри одного класса, см. subclass_growth_parallel.py).

По просьбе не менять уже написанный код — `SubspaceConjugacyClassifier`
не изменён. Эта функция — самостоятельная обёртка НАД публичным API
(`fit_from_subclass_bases`), собирающая тот же результат, что и обычный
`.fit()`, но распределяя кластеризацию классов по потокам joblib.

Принимает clusterer_cls — любой класс с интерфейсом
`fit(X) -> self` + `.subspaces_` (и `FursovClusterer`, и
`fursov_clusterer_parallel.ParallelFursovClusterer` подходят) — так можно
независимо комбинировать "внешний" (по классам) и "внутренний" (внутри
B.2) уровни параллелизма для бенчмарка 2x2 (оба выключены / только внешний
/ только внутренний / оба).
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
from joblib import Parallel, delayed

from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

logger = logging.getLogger(__name__)


def _fit_one_class(
    cls: Any,
    X_cls: np.ndarray,
    clusterer_cls: Type,
    clusterer_kwargs: Dict[str, Any],
) -> Tuple[Any, list]:
    clusterer = clusterer_cls(**clusterer_kwargs)
    clusterer.fit(X_cls)
    return cls, clusterer.subspaces_


def fit_classifier_parallel(
    X: np.ndarray,
    y: np.ndarray,
    n_subclasses: int = 8,
    freeze_basis_at: Optional[int] = 2,
    clusterer_cls: Type = FursovClusterer,
    clusterer_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
) -> SubspaceConjugacyClassifier:
    """Обучает SubspaceConjugacyClassifier, распараллеливая кластеризацию по классам.

    Эквивалентно (по результату — с точностью до детерминизма самих
    алгоритмов, см. Notes) ``SubspaceConjugacyClassifier(n_subclasses=...,
    freeze_basis_at=...).fit(X, y)``, но независимые классы кластеризуются
    параллельно (joblib, backend="threading" — каждая задача CPU-bound
    через numpy/BLAS, GIL отпускается).

    Parameters
    ----------
    X : np.ndarray
        Матрица признаков (M, N).
    y : np.ndarray
        Метки классов (M,).
    n_subclasses : int, default=8
        Передаётся в конструктор каждого clusterer_cls.
    freeze_basis_at : int or None, default=2
        Передаётся в конструктор каждого clusterer_cls.
    clusterer_cls : type, default=FursovClusterer
        Класс кластеризации с интерфейсом ``fit(X) -> self`` + ``.subspaces_``.
        И канонический ``FursovClusterer``, и
        ``algorithms.fursov_clusterer_parallel.ParallelFursovClusterer``
        подходят — второй добавляет ВТОРОЙ, независимый уровень
        параллелизма (внутри B.2 одного класса).
    clusterer_kwargs : dict, optional
        Дополнительные параметры конструктора clusterer_cls (например,
        ``{"n_jobs": 4}`` для ParallelFursovClusterer). n_subclasses и
        freeze_basis_at добавляются автоматически — не указывайте их здесь.
    n_jobs : int, default=1
        Число потоков joblib для параллельных классов. 1 — последовательно
        (эквивалентно оригинальному SubspaceConjugacyClassifier.fit(), но
        собранному через fit_from_subclass_bases — честная база для
        сравнения "накладные расходы joblib при n_jobs=1 отсутствуют").

    Returns
    -------
    classifier : SubspaceConjugacyClassifier
        Обученный (через fit_from_subclass_bases) классификатор, готовый к
        predict()/predict_proba().

    Notes
    -----
    Результат идентичен последовательному ``.fit()`` с точностью до
    порядка floating-point операций (кластеризация каждого класса —
    детерминированная функция ТОЛЬКО своих данных, порядок обработки
    классов не влияет на результат) — расхождения, если появятся, должны
    быть на уровне 1e-10 и ниже (ассоциативность сложения float), не
    структурные.

    Examples
    --------
    >>> clf = fit_classifier_parallel(X_train, y_train, n_jobs=3)
    >>> clf.predict(X_test)
    """
    t0 = time.perf_counter()
    classes = np.unique(y)
    kwargs = dict(clusterer_kwargs or {})
    kwargs["n_subclasses"] = n_subclasses
    kwargs["freeze_basis_at"] = freeze_basis_at

    logger.info(
        "fit_classifier_parallel: старт, %d класс(ов), clusterer_cls=%s, n_jobs=%s.",
        len(classes), clusterer_cls.__name__, n_jobs,
    )

    tasks = [(cls, X[y == cls]) for cls in classes]

    if n_jobs == 1:
        results = [_fit_one_class(cls, X_cls, clusterer_cls, kwargs) for cls, X_cls in tasks]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_one_class)(cls, X_cls, clusterer_cls, kwargs) for cls, X_cls in tasks
        )

    bases_by_class = dict(results)
    classifier = SubspaceConjugacyClassifier().fit_from_subclass_bases(bases_by_class)

    elapsed = time.perf_counter() - t0
    logger.info("fit_classifier_parallel: готово за %.3fs.", elapsed)
    return classifier
