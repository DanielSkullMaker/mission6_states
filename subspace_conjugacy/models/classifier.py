"""Модуль классификатора на основе подпространственной сопряженности.

Реализует Фазу C теории (refactoring_plan.txt, раздел 1.4) и NB8
(8_Fursov_classification.ipynb): для каждого класса патологии строится
n_subclasses подпространств Y_{c,s} (N x k, k = freeze_basis_at = 2),
итого C x n_subclasses подпространств. Классификация — плоский argmax
показателя сопряженности R(x, Y) по ВСЕМ подпространствам сразу, после
чего подкласс сопоставляется владеющему им классу.

Кластеризация внутри каждого класса делегирована каноническому
FursovClusterer (фазы A.1->A.3->B.1->B.2, algorithms/fursov_clusterer.py) —
единственная реализация метода в библиотеке (refactoring_plan.txt, раздел 6,
п.1-2: канон как единственный путь, без дублирования conjugate_criterion).
"""

import logging
from typing import Dict, List, Optional, Union
import numpy as np
from sklearn.base import ClassifierMixin

from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer
from subspace_conjugacy.core.metrics import conjugate_criterion
from subspace_conjugacy.models.base import BaseSubspaceEstimator

logger = logging.getLogger(__name__)

ClassLabel = Union[int, str, float]


class SubspaceConjugacyClassifier(BaseSubspaceEstimator, ClassifierMixin):
    """Классификатор многомерных данных на основе подпространств (теория C).

    Parameters
    ----------
    n_subclasses : int, default=8
        Количество подклассов (подпространств), формируемых для каждого класса.
    freeze_basis_at : int, default=2
        Размер базиса каждого подкласса после кластеризации (k). Теория
        Фазы C требует фиксированного k=2 для всех подпространств
        классификатора (refactoring_plan.txt, раздел 6, п.9).
    growth_strategy : {"default", "master"}, default="default"
        Стратегия наполнения кластеров в FursovClusterer (Фаза B.2):
        "default" — argmax R(x, Y_s); "master" — ratio к среднему,
        канон-совместимая аппроксимация идеи NB7, не побитовая реплика
        (см. algorithms/subclass_growth.py, docstring модуля).
    reg_param : float, default=1e-8
        Коэффициент регуляризации при обращении матрицы Грама.

    Attributes
    ----------
    classes_ : np.ndarray
        Массив уникальных меток классов.
    subspaces_ : Dict[ClassLabel, List[np.ndarray]]
        Словарь, содержащий список базисных матриц Y_s (N, k) для каждого класса.
    flat_subclass_labels_ : np.ndarray or None
        Метка класса для каждого "плоского" подпространства (используется
        predict_subclass/predict_r_matrix_flat); длина = сумма n_subclasses
        по всем классам, порядок соответствует classes_ и порядку в subspaces_.
    """

    def __init__(
        self,
        n_subclasses: int = 8,
        freeze_basis_at: int = 2,
        growth_strategy: str = "default",
        reg_param: float = 1e-8,
    ) -> None:
        super().__init__(
            n_subclasses=n_subclasses,
            n_centers_init=2,
            reg_param=reg_param,
        )
        self.freeze_basis_at = freeze_basis_at
        self.growth_strategy = growth_strategy
        self.classes_: Optional[np.ndarray] = None
        self.subspaces_: Dict[ClassLabel, List[np.ndarray]] = {}
        self.flat_subclass_labels_: Optional[np.ndarray] = None

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "SubspaceConjugacyClassifier":
        """Обучает модель: строит подпространства подклассов для всех классов.

        Для каждого класса запускает независимый FursovClusterer (канон
        A.1->A.3->B.1->B.2) на подвыборке этого класса.

        Parameters
        ----------
        X : np.ndarray
            Обучающая матрица признаков размерности (M, N).
        y : np.ndarray
            Вектор меток классов размерности (M,).

        Returns
        -------
        self : SubspaceConjugacyClassifier
            Возвращает обученный экземпляр модели.
        """
        self._validate_input_params()
        X_clean, y_clean = self._validate_data(X, y)

        if y_clean is None:
            logger.error("SubspaceConjugacyClassifier.fit: метки y не переданы.")
            raise ValueError("Для обучения классификатора необходимы метки y.")

        self.classes_ = np.unique(y_clean)
        logger.info(
            "SubspaceConjugacyClassifier.fit: старт, %d объектов, классы=%s, "
            "n_subclasses=%d, growth_strategy=%s.",
            X_clean.shape[0], list(self.classes_), self.n_subclasses,
            self.growth_strategy,
        )
        if len(self.classes_) < 2:
            logger.error(
                "SubspaceConjugacyClassifier.fit: найдено %d класс(ов), нужно минимум 2.",
                len(self.classes_),
            )
            raise ValueError(
                "Для классификации требуется как минимум 2 класса."
            )

        self.n_features_in_ = X_clean.shape[1]
        self.subspaces_ = {}

        for cls in self.classes_:
            X_cls = X_clean[y_clean == cls]
            logger.info(
                "SubspaceConjugacyClassifier.fit: класс '%s' — кластеризация %d объектов.",
                cls, X_cls.shape[0],
            )
            if X_cls.shape[0] < self.n_subclasses:
                logger.error(
                    "SubspaceConjugacyClassifier.fit: класс '%s' содержит %d объектов "
                    "< n_subclasses=%d.", cls, X_cls.shape[0], self.n_subclasses,
                )
                raise ValueError(
                    f"Класс '{cls}' содержит {X_cls.shape[0]} объектов, "
                    f"что меньше числа подклассов ({self.n_subclasses})."
                )

            clusterer = FursovClusterer(
                n_subclasses=self.n_subclasses,
                freeze_basis_at=self.freeze_basis_at,
                growth_strategy=self.growth_strategy,
                reg_param=self.reg_param,
            )
            clusterer.fit(X_cls)
            self.subspaces_[cls] = clusterer.subspaces_
            logger.debug(
                "SubspaceConjugacyClassifier.fit: класс '%s' готов, %d подпространств.",
                cls, len(clusterer.subspaces_),
            )

        self.flat_subclass_labels_ = self._build_flat_subclass_labels()
        self.is_fitted_ = True
        logger.info(
            "SubspaceConjugacyClassifier.fit: готово, %d классов x %d подклассов = "
            "%d подпространств всего.",
            len(self.classes_), self.n_subclasses, len(self.flat_subclass_labels_),
        )
        return self

    def fit_from_subclass_bases(
        self, subspaces_by_class: Dict[ClassLabel, List[np.ndarray]]
    ) -> "SubspaceConjugacyClassifier":
        """Собирает классификатор из уже готовых базисов подклассов.

        Позволяет пропустить повторную кластеризацию, если базисы Y_{c,s}
        уже получены отдельно — например, кластеризацией через
        FursovClusterer + algorithms/subclass_export.py, или загружены из
        CSV ноутбуков NB6-7 (io.vectors.load_subclass_bases_as_list).

        Parameters
        ----------
        subspaces_by_class : Dict[ClassLabel, List[np.ndarray]]
            Словарь {class_label: [Y_0, Y_1, ..., Y_{S-1}]}, где каждый
            Y_s — базисная матрица подкласса размерности (N, k). Размерность
            N должна совпадать для всех базисов всех классов.

        Returns
        -------
        self : SubspaceConjugacyClassifier
            Возвращает готовый к предсказаниям экземпляр модели.

        Raises
        ------
        ValueError
            Если словарь пуст или базисы имеют разную размерность N.
        """
        if not subspaces_by_class:
            logger.error("SubspaceConjugacyClassifier.fit_from_subclass_bases: словарь пуст.")
            raise ValueError("Словарь subspaces_by_class пуст.")

        n_features_set = {
            Y.shape[0] for bases in subspaces_by_class.values() for Y in bases
        }
        if len(n_features_set) != 1:
            logger.error(
                "SubspaceConjugacyClassifier.fit_from_subclass_bases: "
                "разные N среди базисов: %s.", sorted(n_features_set),
            )
            raise ValueError(
                "Все базисы всех классов должны иметь одинаковую размерность "
                f"признаков N. Получено значений N: {sorted(n_features_set)}."
            )

        self.classes_ = np.array(list(subspaces_by_class.keys()))
        self.subspaces_ = {
            cls: list(bases) for cls, bases in subspaces_by_class.items()
        }
        self.n_features_in_ = n_features_set.pop()
        self.flat_subclass_labels_ = self._build_flat_subclass_labels()
        self.is_fitted_ = True
        logger.info(
            "SubspaceConjugacyClassifier.fit_from_subclass_bases: загружено %d классов "
            "(без повторной кластеризации), N=%d.",
            len(self.classes_), self.n_features_in_,
        )
        return self

    def _build_flat_subclass_labels(self) -> np.ndarray:
        """Строит массив меток классов для "плоского" перечня подпространств."""
        labels = []
        for cls in self.classes_:
            labels.extend([cls] * len(self.subspaces_[cls]))
        return np.array(labels)

    def predict_r_matrix(self, X: np.ndarray) -> np.ndarray:
        """Вычисляет матрицу максимальных показателей сопряженности R по классам.

        Parameters
        ----------
        X : np.ndarray
            Матрица объектов размерности (M, N).

        Returns
        -------
        R_matrix : np.ndarray
            Матрица размерности (M, n_classes), где элемент (i, c) равен
            max_s R(x_i, Y_{c, s}).
        """
        self._check_is_fitted()
        X_clean, _ = self._validate_data(X)

        M = X_clean.shape[0]
        n_classes = len(self.classes_)
        R_matrix = np.zeros((M, n_classes), dtype=np.float64)

        for cls_idx, cls in enumerate(self.classes_):
            subspace_bases = self.subspaces_[cls]

            # Вычисляем R для всех подклассов данного класса
            r_subclasses = np.column_stack([
                conjugate_criterion(
                    X_clean, Y_s, reg_param=self.reg_param
                )
                for Y_s in subspace_bases
            ])

            # Выбираем максимальную сопряженность среди всех подклассов
            R_matrix[:, cls_idx] = np.max(r_subclasses, axis=1)

        logger.debug(
            "SubspaceConjugacyClassifier.predict_r_matrix: %d объектов x %d классов.",
            M, n_classes,
        )
        return R_matrix

    def predict_r_matrix_flat(self, X: np.ndarray) -> np.ndarray:
        """Вычисляет R(x, Y_{c,s}) для КАЖДОГО подпространства отдельно (Фаза C, NB8).

        В отличие от predict_r_matrix (максимум по подклассам внутри класса),
        здесь каждый столбец соответствует ровно одному подклассу одного
        класса — как 24 базиса в NB8 (3 класса x 8 подклассов). Порядок
        столбцов соответствует flat_subclass_labels_.

        Parameters
        ----------
        X : np.ndarray
            Матрица объектов размерности (M, N).

        Returns
        -------
        R_flat : np.ndarray
            Матрица размерности (M, total_subclasses).
        """
        self._check_is_fitted()
        X_clean, _ = self._validate_data(X)

        columns = [
            conjugate_criterion(X_clean, Y_s, reg_param=self.reg_param)
            for cls in self.classes_
            for Y_s in self.subspaces_[cls]
        ]
        R_flat = np.column_stack(columns)
        logger.debug(
            "SubspaceConjugacyClassifier.predict_r_matrix_flat: %d объектов x "
            "%d подпространств.", R_flat.shape[0], R_flat.shape[1],
        )
        return R_flat

    def predict_subclass(self, X: np.ndarray) -> np.ndarray:
        """Определяет глобальный индекс подкласса (Фаза C: flat argmax).

        Эквивалент NB8: ``subclass* = argmax_{c,s} R_{c,s}``. Для 3 классов
        по 8 подклассов результат лежит в диапазоне [0, 23]; метку класса,
        которому принадлежит подкласс, можно получить как
        ``self.flat_subclass_labels_[predict_subclass(X)]``.

        Parameters
        ----------
        X : np.ndarray
            Матрица объектов размерности (M, N).

        Returns
        -------
        subclass_indices : np.ndarray
            Индексы подклассов (M,) в диапазоне [0, total_subclasses).
        """
        R_flat = self.predict_r_matrix_flat(X)
        subclass_indices = np.argmax(R_flat, axis=1)
        logger.debug(
            "SubspaceConjugacyClassifier.predict_subclass: %d объектов -> подклассы %s.",
            len(subclass_indices),
            np.bincount(subclass_indices, minlength=R_flat.shape[1]).tolist(),
        )
        return subclass_indices

    def predict_confidence_ratio(self, X: np.ndarray) -> np.ndarray:
        """Показатель уверенности предсказания (NB8: proportion).

        Формула из NB8 (8_Fursov_classification.ipynb, cell 7)::

            proportion = best_R / mean(R_others) - 1

        где ``best_R`` — максимальный показатель сопряженности среди ВСЕХ
        подпространств, а ``R_others`` — все остальные значения. Чем больше
        proportion, тем увереннее объект отнесён к выбранному подклассу.

        Parameters
        ----------
        X : np.ndarray
            Матрица объектов размерности (M, N).

        Returns
        -------
        proportion : np.ndarray
            Показатель уверенности (M,). Может быть отрицательным, если
            лучший показатель ниже среднего по остальным (вырожденный случай).
        """
        R_flat = self.predict_r_matrix_flat(X)
        n_subspaces = R_flat.shape[1]

        if n_subspaces < 2:
            logger.error(
                "SubspaceConjugacyClassifier.predict_confidence_ratio: "
                "недостаточно подпространств (%d < 2).", n_subspaces,
            )
            raise ValueError(
                "predict_confidence_ratio требует минимум 2 подпространства "
                f"для сравнения, получено {n_subspaces}."
            )

        best = np.max(R_flat, axis=1)
        mean_others = (np.sum(R_flat, axis=1) - best) / (n_subspaces - 1)

        # Защита от деления на ноль, если все "остальные" R равны нулю.
        n_degenerate = int(np.sum(mean_others <= 0))
        if n_degenerate > 0:
            logger.warning(
                "SubspaceConjugacyClassifier.predict_confidence_ratio: у %d/%d "
                "объектов mean_others<=0 — используется eps вместо деления на ноль.",
                n_degenerate, len(mean_others),
            )
        mean_others_safe = np.where(
            mean_others > 0, mean_others, np.finfo(np.float64).eps
        )
        proportion = best / mean_others_safe - 1
        logger.debug(
            "SubspaceConjugacyClassifier.predict_confidence_ratio: mean=%.4f, "
            "min=%.4f, max=%.4f.", proportion.mean(), proportion.min(), proportion.max(),
        )
        return proportion

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Нормализует показатели сопряженности в вероятности через Softmax.

        Parameters
        ----------
        X : np.ndarray
            Матрица объектов размерности (M, N).

        Returns
        -------
        probabilities : np.ndarray
            Матрица вероятностей размерности (M, n_classes).
        """
        R_matrix = self.predict_r_matrix(X)

        # Стабильный Softmax
        exp_r = np.exp(R_matrix - np.max(R_matrix, axis=1, keepdims=True))
        probabilities = exp_r / np.sum(exp_r, axis=1, keepdims=True)

        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает метку класса с наивысшим показателем сопряженности.

        Parameters
        ----------
        X : np.ndarray
            Матрица объектов размерности (M, N).

        Returns
        -------
        y_pred : np.ndarray
            Предсказанные метки классов размерности (M,).
        """
        R_matrix = self.predict_r_matrix(X)
        best_indices = np.argmax(R_matrix, axis=1)
        y_pred = self.classes_[best_indices]
        logger.info(
            "SubspaceConjugacyClassifier.predict: %d объектов -> распределение по классам %s.",
            len(y_pred), dict(zip(*np.unique(y_pred, return_counts=True))),
        )
        return y_pred
