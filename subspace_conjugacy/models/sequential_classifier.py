"""Двухэтапная последовательная классификация (статья, находка №4).

Korshikov & Fursov, "Pathology Recognition Based on Conjugacy Criteria with
Subspaces of Reference Images":
  "When considering this program, it is important to note that it is
  capable of performing sequential classification. The first step involves
  the identification of the essential feature, and the second step
  involves the classification of objects for which this feature is
  matched. In other words, it allows you to first determine the projection
  of the image, and then to determine the presence of a tumor and its
  specific type. At the same time, the result obtained at the previous
  stage of classification becomes a set of data for the next stage."

Мотивирующий пример статьи — этап 1: определение проекции снимка
(axial/sagittal/coronal) по Otsu-бинаризованным векторам
(preprocessing.binarization.otsu_binarize — статья явно требует
бинаризацию ТОЛЬКО на этом этапе); этап 2: определение типа опухоли по
НЕ бинаризованным векторам, с помощью классификатора, обученного отдельно
для КАЖДОЙ проекции (статья: "For the task of tumor type detection in
projection distributed images, binarization is not performed").

SequentialClassifier ниже — универсальная композиция ЛЮБЫХ двух
sklearn-совместимых классификаторов (не привязана к
SubspaceConjugacyClassifier конкретно, хотя это основной сценарий) в
двухступенчатый пайплайн "определить группу -> классифицировать внутри
группы своим классификатором". Сам Otsu/векторизация — ответственность
пользователя (preprocessing/features), сюда передаются уже готовые
матрицы признаков — так же, как FursovClusterer/SubspaceConjugacyClassifier
работают с векторами, а не с изображениями напрямую.
"""

import logging
from typing import Any, Dict, Mapping, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SequentialClassifier:
    """Композиция классификатора этапа 1 и словаря классификаторов этапа 2.

    Parameters
    ----------
    stage1_classifier : object
        Классификатор первого этапа (например, определение проекции) —
        любой объект с методами ``fit(X, y)`` и ``predict(X)`` (sklearn API,
        включая SubspaceConjugacyClassifier).
    stage2_classifiers : Dict[Any, object]
        {label: classifier} — классификатор второго этапа для КАЖДОГО
        значения, которое может предсказать stage1_classifier (например,
        {"axial": clf_axial, "sagittal": clf_sagittal, "coronal": clf_coronal}).
        Каждый classifier — тот же sklearn API (fit/predict).

    Attributes
    ----------
    is_fitted_ : bool
        Вычисляемое свойство: True, если stage1_classifier и ВСЕ
        stage2_classifiers считаются обученными — проверяется через их
        собственный атрибут ``is_fitted_``, если он есть (как у
        SubspaceConjugacyClassifier); для сторонних sklearn-эстиматоров без
        такого атрибута считается обученным всегда (ответственность за
        вызов fit() — на пользователе, как и для самого sklearn). Поэтому
        уже обученные классификаторы, переданные напрямую в конструктор,
        сразу дают is_fitted_=True без вызова SequentialClassifier.fit().

    Examples
    --------
    >>> import numpy as np
    >>> from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier
    >>> from subspace_conjugacy.models.sequential_classifier import SequentialClassifier
    >>>
    >>> # Этап 1: определение проекции по бинаризованным векторам.
    >>> stage1 = SubspaceConjugacyClassifier(n_subclasses=4)
    >>> # Этап 2: по классификатору типа опухоли на каждую проекцию.
    >>> stage2 = {
    ...     "axial": SubspaceConjugacyClassifier(n_subclasses=8),
    ...     "sagittal": SubspaceConjugacyClassifier(n_subclasses=8),
    ...     "coronal": SubspaceConjugacyClassifier(n_subclasses=8),
    ... }
    >>> seq = SequentialClassifier(stage1, stage2)
    >>> seq.fit(X1_train, y1_train, X2_by_projection, y2_by_projection)
    >>> projection_pred, tumor_pred = seq.predict(X1_test, X2_test)
    """

    def __init__(
        self,
        stage1_classifier: Any,
        stage2_classifiers: Mapping[Any, Any],
    ) -> None:
        if not stage2_classifiers:
            raise ValueError("stage2_classifiers не может быть пустым.")

        self.stage1_classifier = stage1_classifier
        self.stage2_classifiers = dict(stage2_classifiers)

    @property
    def is_fitted_(self) -> bool:
        """True, если stage1_classifier и все stage2_classifiers обучены.

        Не хранится как обычный атрибут: вычисляется на лету из состояния
        вложенных классификаторов (у сторонних sklearn-эстиматоров без
        атрибута ``is_fitted_`` считается True — как и в самом sklearn,
        ответственность за вызов fit() лежит на пользователе).
        """
        classifiers = [self.stage1_classifier, *self.stage2_classifiers.values()]
        return all(getattr(clf, "is_fitted_", True) for clf in classifiers)

    def fit(
        self,
        X1: np.ndarray,
        y1: np.ndarray,
        X2_by_group: Mapping[Any, np.ndarray],
        y2_by_group: Mapping[Any, np.ndarray],
    ) -> "SequentialClassifier":
        """Обучает классификатор этапа 1 и каждый классификатор этапа 2.

        Классификаторы этапов обучаются на СВОИХ, необязательно связанных
        по объектам, датасетах — как в статье (этап 1 — 100 изображений на
        проекцию, этап 2 — отдельные наборы по 300-450 изображений на
        проекцию), а не на одном общем наборе с двумя параллельными метками.

        Parameters
        ----------
        X1 : np.ndarray
            Матрица признаков (M1, N1) для обучения stage1_classifier
            (например, векторы Otsu-бинаризованных изображений).
        y1 : np.ndarray
            Метки этапа 1 (M1,) (например, "axial"/"sagittal"/"coronal").
        X2_by_group : Mapping[Any, np.ndarray]
            {label: X2_label} — для КАЖДОГО значения из y1 (и из
            stage2_classifiers) отдельная матрица признаков (M2_label, N2)
            для обучения stage2_classifiers[label] (например, векторы НЕ
            бинаризованных изображений именно этой проекции).
        y2_by_group : Mapping[Any, np.ndarray]
            {label: y2_label} — соответствующие метки этапа 2 (тип опухоли)
            для каждой группы.

        Returns
        -------
        self : SequentialClassifier
            Возвращает обученный экземпляр модели.

        Raises
        ------
        ValueError
            Если для какого-то класса из y1 не задан stage2_classifiers/
            X2_by_group/y2_by_group.
        """
        stage1_labels = np.unique(y1)
        logger.info(
            "SequentialClassifier.fit: этап 1 — %d объектов, классы=%s.",
            len(y1), list(stage1_labels),
        )

        missing_stage2 = [
            label for label in stage1_labels if label not in self.stage2_classifiers
        ]
        if missing_stage2:
            logger.error(
                "SequentialClassifier.fit: нет stage2_classifiers для классов %s.",
                missing_stage2,
            )
            raise ValueError(
                f"stage2_classifiers не содержит классификатор для классов "
                f"этапа 1: {missing_stage2}. Есть ключи: "
                f"{list(self.stage2_classifiers.keys())}."
            )

        missing_data = [
            label for label in stage1_labels
            if label not in X2_by_group or label not in y2_by_group
        ]
        if missing_data:
            logger.error(
                "SequentialClassifier.fit: нет данных этапа 2 для классов %s.",
                missing_data,
            )
            raise ValueError(
                f"X2_by_group/y2_by_group не содержат данных для классов "
                f"этапа 1: {missing_data}."
            )

        self.stage1_classifier.fit(X1, y1)

        for label in stage1_labels:
            X2_label = X2_by_group[label]
            y2_label = y2_by_group[label]
            logger.info(
                "SequentialClassifier.fit: этап 2, группа '%s' — %d объектов.",
                label, len(y2_label),
            )
            self.stage2_classifiers[label].fit(X2_label, y2_label)

        logger.info(
            "SequentialClassifier.fit: готово, %d групп этапа 2 обучено.",
            len(stage1_labels),
        )
        return self

    def predict_stage1(self, X1: np.ndarray) -> np.ndarray:
        """Предсказывает метки этапа 1 (например, проекцию снимка).

        Parameters
        ----------
        X1 : np.ndarray
            Матрица признаков (M, N1), в том же представлении, что и при
            обучении stage1_classifier (например, Otsu-бинаризованные
            векторы).

        Returns
        -------
        stage1_pred : np.ndarray
            Предсказанные метки этапа 1 (M,).
        """
        self._check_is_fitted()
        return self.stage1_classifier.predict(X1)

    def predict_stage2(
        self, X2: np.ndarray, stage1_labels: np.ndarray
    ) -> np.ndarray:
        """Предсказывает метки этапа 2, используя уже известные метки этапа 1.

        Каждый объект классифицируется классификатором
        ``stage2_classifiers[stage1_labels[i]]`` — то есть "результат
        предыдущего этапа становится данными для следующего" буквально:
        для группировки объектов по классификатору используется РЕЗУЛЬТАТ
        этапа 1 (предсказанный или истинный — оба сценария допустимы,
        см. Parameters), а не повторное вычисление.

        Parameters
        ----------
        X2 : np.ndarray
            Матрица признаков (M, N2) для этапа 2 (например, НЕ
            бинаризованные векторы), в том же порядке объектов, что и
            ``stage1_labels``.
        stage1_labels : np.ndarray
            Метки этапа 1 (M,) — обычно результат predict_stage1(X1) на тех
            же объектах, но можно передать и истинные метки (например, для
            изолированной проверки качества этапа 2 без ошибок этапа 1).

        Returns
        -------
        stage2_pred : np.ndarray
            Предсказанные метки этапа 2 (M,), в исходном порядке объектов.

        Raises
        ------
        ValueError
            Если среди stage1_labels встретилась метка, для которой нет
            классификатора в stage2_classifiers.
        """
        self._check_is_fitted()
        stage1_labels = np.asarray(stage1_labels)

        unknown = set(np.unique(stage1_labels)) - set(self.stage2_classifiers.keys())
        if unknown:
            logger.error(
                "SequentialClassifier.predict_stage2: нет классификатора для %s.",
                unknown,
            )
            raise ValueError(
                f"stage2_classifiers не содержит классификатор для меток "
                f"{unknown}, встретившихся в stage1_labels."
            )

        stage2_pred = np.empty(len(stage1_labels), dtype=object)
        for label in np.unique(stage1_labels):
            mask = stage1_labels == label
            stage2_pred[mask] = self.stage2_classifiers[label].predict(X2[mask])

        return stage2_pred

    def predict(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Полное последовательное предсказание: этап 1, затем этап 2.

        Parameters
        ----------
        X1 : np.ndarray
            Матрица признаков (M, N1) для этапа 1 (например, Otsu-
            бинаризованные векторы).
        X2 : np.ndarray
            Матрица признаков (M, N2) для этапа 2 (например, НЕ
            бинаризованные векторы), для ТЕХ ЖЕ M объектов в том же порядке.

        Returns
        -------
        stage1_pred : np.ndarray
            Предсказанные метки этапа 1 (M,) (например, проекция).
        stage2_pred : np.ndarray
            Предсказанные метки этапа 2 (M,) (например, тип опухоли),
            каждая — результат классификатора, соответствующего
            ПРЕДСКАЗАННОЙ (а не истинной) метке этапа 1.
        """
        stage1_pred = self.predict_stage1(X1)
        stage2_pred = self.predict_stage2(X2, stage1_pred)
        logger.info(
            "SequentialClassifier.predict: %d объектов, этап1=%s.",
            len(stage1_pred),
            dict(zip(*np.unique(stage1_pred, return_counts=True))),
        )
        return stage1_pred, stage2_pred

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            logger.error("SequentialClassifier: обращение к предсказаниям до fit().")
            raise RuntimeError(
                "Модель не обучена. Вызовите fit(...) перед использованием, "
                "либо передайте в конструктор уже обученные stage1_classifier/"
                "stage2_classifiers и установите is_fitted_=True вручную."
            )

    def __repr__(self) -> str:
        return (
            f"SequentialClassifier(stage1={type(self.stage1_classifier).__name__}, "
            f"stage2_groups={list(self.stage2_classifiers.keys())}, "
            f"fitted={self.is_fitted_})"
        )
