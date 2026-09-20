"""Пространства гиперпараметров для поиска по SubspaceConjugacyClassifier.

Гиперпараметры классификатора (см. models/classifier.py, models/base.py):
  - n_subclasses    — число подпространств на класс (теория A.2-A.3/B.2).
  - growth_strategy — "default" (argmax R) или "master" (ratio к среднему,
                       канон-совместимая аппроксимация NB7).
  - reg_param       — регуляризация Тихонова (Y^T Y)^{-1} (core/metrics.py).

freeze_basis_at сознательно НЕ входит в дефолтные пространства поиска:
теория Фазы C (refactoring_plan.txt, раздел 1.4) требует ровно k=2 для
базисов, участвующих в классификации, — это не тюнинг-параметр метода, а
условие корректности алгоритма. Пользователь может переопределить
param_grid/param_distributions и включить freeze_basis_at сам, если это
осознанное решение (например, экспериментирует с самим кластеризатором, а
не только с классификатором).
"""

from typing import Any, Dict

from scipy.stats import loguniform, randint

#: Сетка для GridSearchCV — небольшая и быстрая по умолчанию, т.к.
#: количество комбинаций растёт как произведение длин списков, а каждая
#: комбинация обучает FursovClusterer отдельно для каждого класса и фолда.
DEFAULT_PARAM_GRID: Dict[str, Any] = {
    "n_subclasses": [4, 8, 12],
    "growth_strategy": ["default", "master"],
    "reg_param": [1e-10, 1e-8, 1e-6],
}

#: Распределения для RandomizedSearchCV — шире, чем DEFAULT_PARAM_GRID,
#: т.к. случайный поиск тратит фиксированный бюджет (n_iter) независимо от
#: размера пространства и не страдает от комбинаторного взрыва как grid search.
DEFAULT_PARAM_DISTRIBUTIONS: Dict[str, Any] = {
    "n_subclasses": randint(2, 17),  # целые из [2, 16]
    "growth_strategy": ["default", "master"],
    "reg_param": loguniform(1e-10, 1e-3),
}
