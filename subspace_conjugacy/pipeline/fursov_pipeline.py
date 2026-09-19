"""FursovPipeline — оркестратор end-to-end пайплайна (Фаза 6 плана).

Собирает стадии из pipeline/stages.py в высокоуровневые операции:
``run_stage()`` — точечный вызов одной стадии (см. таблицу в stages.py),
``run_class()``/``run_all_classes()`` — vectorize -> cluster (канон) ->
export_subspaces для одного/всех классов, ``build_classifier()`` — сборка
SubspaceConjugacyClassifier из обученных clusterers_, ``classify_test()`` —
предсказание + accuracy на тестовой выборке (аналог NB8).

Канон — единственный путь в run_class()/run_all_classes() (refactoring_plan.txt,
раздел 6, п.1): стадия "cluster" всегда использует FursovClusterer. Стадия
"legacy_notebook" доступна только через run_stage() явно, для parity-тестов.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from subspace_conjugacy.pipeline.stages import STAGE_REGISTRY

logger = logging.getLogger(__name__)


class FursovPipeline:
    """Оркестратор пайплайна Fursov поверх DatasetConfig.

    Parameters
    ----------
    config : DatasetConfig
        Конфигурация путей и гиперпараметров по умолчанию (n_subclasses,
        subclass_factor, classes, image_size).

    Attributes
    ----------
    clusterers_ : Dict[str, FursovClusterer]
        Обученные кластеризаторы по классам, заполняется run_class()/
        run_all_classes().
    classifier_ : SubspaceConjugacyClassifier or None
        Классификатор, собранный build_classifier() (или неявно —
        classify_test()).

    Examples
    --------
    >>> from subspace_conjugacy.config import DatasetConfig
    >>> from subspace_conjugacy.pipeline import FursovPipeline
    >>> config = DatasetConfig(root="data", n_subclasses=8)
    >>> pipeline = FursovPipeline(config)
    >>> pipeline.run_all_classes()
    >>> report = pipeline.classify_test(y_test=y_true)
    >>> report["report"]["accuracy"]
    """

    def __init__(self, config: "DatasetConfig") -> None:
        self.config = config
        self.clusterers_: Dict[str, Any] = {}
        self.classifier_: Optional[Any] = None

    def run_stage(self, name: str, **kwargs: Any) -> Any:
        """Вызывает одну именованную стадию из STAGE_REGISTRY.

        Если сигнатура стадии принимает параметр ``config`` и он не передан
        явно в ``kwargs``, автоматически подставляется ``self.config`` —
        так вызовы вида ``run_stage("vectorize", class_name="glioma")`` не
        требуют повторять конфигурацию на каждой стадии.

        Parameters
        ----------
        name : str
            Имя стадии (см. таблицу в pipeline/stages.py).
        **kwargs
            Аргументы, специфичные для стадии.

        Returns
        -------
        result : Any
            Результат стадии (тип зависит от стадии — см. stages.py).

        Raises
        ------
        ValueError
            Если ``name`` не найдено в STAGE_REGISTRY.
        """
        if name not in STAGE_REGISTRY:
            available = ", ".join(sorted(STAGE_REGISTRY))
            logger.error("FursovPipeline.run_stage: неизвестная стадия '%s'.", name)
            raise ValueError(f"Неизвестная стадия '{name}'. Доступны: {available}.")

        func = STAGE_REGISTRY[name]
        signature = inspect.signature(func)
        if "config" in signature.parameters and "config" not in kwargs:
            kwargs["config"] = self.config

        logger.debug("FursovPipeline.run_stage: '%s', kwargs=%s.", name, list(kwargs))
        return func(**kwargs)

    def run_preprocessing(
        self, class_name: str, raw_pattern: str = "*.jpg", **kwargs: Any
    ) -> List:
        """NB1+NB2 для одного класса: raw -> resized -> centered.

        Parameters
        ----------
        class_name : str
            Название класса.
        raw_pattern : str, default="*.jpg"
            Glob-паттерн исходных файлов в config.paths[class_name]["raw"].
        **kwargs
            Дополнительные параметры, переданные в стадии "resize"/"center"
            (например, background_threshold, min_shift).

        Returns
        -------
        centered_paths : list[Path]
            Пути к центрированным PNG (результат стадии "center").
        """
        self.run_stage("resize", class_name=class_name, pattern=raw_pattern)
        return self.run_stage("center", class_name=class_name, **kwargs)

    def run_class(
        self,
        class_name: str,
        n_subclasses: Optional[int] = None,
        freeze_basis_at: int = 2,
        growth_strategy: str = "default",
        reg_param: float = 1e-8,
        vector_stage: str = "centered",
        vector_method: str = "horizontal",
        save_vectors: bool = True,
        export: bool = True,
    ) -> Any:
        """vectorize -> cluster (канон) -> export_subspaces для одного класса.

        Parameters
        ----------
        class_name : str
            Название класса.
        n_subclasses : int, optional
            Количество подклассов. По умолчанию — ``self.config.n_subclasses``.
        freeze_basis_at : int, default=2
            Размер базиса подкласса (теория C требует k=2 для классификатора).
        growth_strategy : {"default", "master"}, default="default"
            Стратегия наполнения кластеров (Фаза B.2).
        reg_param : float, default=1e-8
            Параметр регуляризации.
        vector_stage : str, default="centered"
            Стадия препроцессинга, откуда берутся изображения для векторизации.
        vector_method : str, default="horizontal"
            Метод векторизации ("horizontal" или "vertical").
        save_vectors : bool, default=True
            Сохранять ли CSV с векторами (формат NB3).
        export : bool, default=True
            Экспортировать ли базисы обученного кластеризатора в CSV (NB6-7).

        Returns
        -------
        clusterer : FursovClusterer
            Обученный кластеризатор для класса (также сохранён в
            ``self.clusterers_[class_name]``).
        """
        if n_subclasses is None:
            n_subclasses = self.config.n_subclasses

        logger.info(
            "FursovPipeline.run_class: класс='%s', n_subclasses=%d, "
            "growth_strategy=%s.", class_name, n_subclasses, growth_strategy,
        )
        X = self.run_stage(
            "vectorize",
            class_name=class_name,
            stage=vector_stage,
            method=vector_method,
            save=save_vectors,
        )
        clusterer = self.run_stage(
            "cluster",
            X=X,
            n_subclasses=n_subclasses,
            freeze_basis_at=freeze_basis_at,
            growth_strategy=growth_strategy,
            reg_param=reg_param,
        )
        self.clusterers_[class_name] = clusterer

        if export:
            self.run_stage(
                "export_subspaces", clusterer=clusterer, class_name=class_name
            )

        logger.info("FursovPipeline.run_class: класс='%s' готов.", class_name)
        return clusterer

    def run_all_classes(self, **kwargs: Any) -> Dict[str, Any]:
        """Вызывает run_class() для каждого класса из self.config.classes.

        Parameters
        ----------
        **kwargs
            Прокидываются в run_class() (n_subclasses, freeze_basis_at, ...).

        Returns
        -------
        clusterers : Dict[str, FursovClusterer]
            То же, что и self.clusterers_, после обучения всех классов.
        """
        logger.info("FursovPipeline.run_all_classes: %d классов -> %s.", len(self.config.classes), self.config.classes)
        for class_name in self.config.classes:
            self.run_class(class_name, **kwargs)
        logger.info("FursovPipeline.run_all_classes: все классы обучены.")
        return self.clusterers_

    def build_classifier(self) -> Any:
        """Собирает SubspaceConjugacyClassifier из self.clusterers_.

        Требует, чтобы run_class()/run_all_classes() уже были вызваны для
        всех классов, базисы которых нужно включить в классификатор.

        Returns
        -------
        classifier : SubspaceConjugacyClassifier
            Классификатор, собранный через fit_from_subclass_bases() —
            без повторной кластеризации (также сохранён в self.classifier_).

        Raises
        ------
        RuntimeError
            Если ни один класс ещё не обучен.
        """
        if not self.clusterers_:
            logger.error("FursovPipeline.build_classifier: нет обученных кластеризаторов.")
            raise RuntimeError(
                "Нет обученных кластеризаторов. Вызовите run_class() или "
                "run_all_classes() перед build_classifier()."
            )

        from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

        reference = next(iter(self.clusterers_.values()))
        subspaces_by_class = {
            cls: clusterer.subspaces_ for cls, clusterer in self.clusterers_.items()
        }

        classifier = SubspaceConjugacyClassifier(
            n_subclasses=reference.n_subclasses,
            freeze_basis_at=reference.freeze_basis_at or 2,
            reg_param=reference.reg_param,
        )
        classifier.fit_from_subclass_bases(subspaces_by_class)
        self.classifier_ = classifier
        logger.info(
            "FursovPipeline.build_classifier: собран из %d классов.",
            len(subspaces_by_class),
        )
        return classifier

    def classify_test(
        self,
        test_class_name: str = "test",
        vector_method: str = "horizontal",
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Классифицирует тестовую выборку и считает accuracy (аналог NB8).

        Parameters
        ----------
        test_class_name : str, default="test"
            Ключ тестовой выборки в self.config.paths (обычно "test").
        vector_method : str, default="horizontal"
            Метод векторизации тестовых изображений.
        y_test : np.ndarray, optional
            Истинные метки классов тестовых объектов. Если не заданы,
            используется ПОЗИЦИОННОЕ разбиение по config.classes и
            config.test_samples_per_class (буквальное поведение NB8: первые
            test_samples_per_class объектов — config.classes[0], следующие —
            config.classes[1], и т.д.). ⚠ Такое разбиение хрупкое — требует,
            чтобы тестовая выборка была физически упорядочена ровно такими
            блоками. Явно передавайте y_test, если порядок файлов не
            гарантирован.

        Returns
        -------
        result : Dict[str, Any]
            {"predictions": ..., "report": evaluate_classifier(...)}.

        Raises
        ------
        RuntimeError
            Если классификатор не собран и не может быть собран (нет
            обученных кластеризаторов).
        """
        logger.info("FursovPipeline.classify_test: старт (test_class_name='%s').", test_class_name)
        if self.classifier_ is None:
            self.build_classifier()

        X_test = self.run_stage(
            "vectorize",
            class_name=test_class_name,
            stage="centered",
            method=vector_method,
            save=False,
        )

        if y_test is None:
            logger.warning(
                "FursovPipeline.classify_test: y_test не задан, используется "
                "хрупкая позиционная разметка NB8 (первые N объектов = classes[0], "
                "следующие N = classes[1], ...)."
            )
            y_test = self._positional_test_labels(X_test.shape[0])

        result = self.run_stage(
            "classify", classifier=self.classifier_, X_test=X_test, y_test=y_test
        )
        if "report" in result:
            logger.info(
                "FursovPipeline.classify_test: accuracy=%.4f.",
                result["report"]["accuracy"],
            )
        return result

    def _positional_test_labels(self, n_samples: int) -> np.ndarray:
        """Буквальная эмуляция позиционной разметки теста из NB8.

        NB8 (8_Fursov_classification.ipynb, cell 8) не хранит метки теста
        отдельно — предполагает, что первые N объектов относятся к первому
        классу, следующие N — ко второму, и т.д., N = test_samples_per_class.
        """
        per_class = self.config.test_samples_per_class
        expected_total = per_class * len(self.config.classes)
        if n_samples != expected_total:
            logger.error(
                "FursovPipeline._positional_test_labels: n_samples=%d != expected=%d.",
                n_samples, expected_total,
            )
            raise ValueError(
                f"Позиционная разметка теста требует ровно {expected_total} "
                f"объектов ({per_class} x {len(self.config.classes)} классов), "
                f"получено {n_samples}. Передайте y_test явно."
            )

        return np.concatenate(
            [np.full(per_class, cls) for cls in self.config.classes]
        )

    def __repr__(self) -> str:
        return (
            f"FursovPipeline(root={self.config.root}, "
            f"classes={self.config.classes}, "
            f"trained={sorted(self.clusterers_.keys())})"
        )
