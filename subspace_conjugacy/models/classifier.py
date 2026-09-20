"""Модуль классификатора на основе подпространственной сопряженности.

Реализует Фазу C теории (refactoring_plan.txt, раздел 1.4) и NB8
(8_Fursov_classification.ipynb): для каждого класса патологии строится
n_subclasses подпространств Y_{c,s} (N x k), итого C x n_subclasses
подпространств. Классификация — плоский argmax показателя сопряженности
R(x, Y) по ВСЕМ подпространствам сразу, после чего подкласс сопоставляется
владеющему им классу.

Кластеризация внутри каждого класса делегирована каноническому
FursovClusterer (фазы A.1->A.3->B.1->B.2, algorithms/fursov_clusterer.py) —
единственная реализация метода в библиотеке (refactoring_plan.txt, раздел 6,
п.1-2: канон как единственный путь, без дублирования conjugate_criterion).

Размер базиса k каждого подпространства управляется freeze_basis_at:
  - int (по умолчанию 2) — фиксированный k для всех подпространств, как в
    NB6-8: ConjugacyClusterGrowth растит подпространство ВСЕГДА до конца
    (независимо от freeze_basis_at — рост векторов на итерации и заморозка
    размера на выходе не связаны по стоимости), но на выходе остаются
    только первые k столбцов — т.е. фактически исходная пара из фазы B.1,
    а всё, что подпространство "набрало" в фазе B.2, отбрасывается.
  - "auto" — НЕ отбрасывает рост: каждый класс кластеризуется с
    freeze_basis_at=None (полный рост), а после того как ВСЕ классы
    обучены, все подпространства усекаются до общего МИНИМАЛЬНОГО
    наблюдённого k (algorithms.subclass_export.equalize_subspace_bases) —
    это буквальный рецепт статьи Korshikov & Fursov ("Description of the
    Clustering Method": "...only the first n elements corresponding to the
    number of vectors of the smallest space are taken for each vector"),
    см. refactoring_plan.txt, раздел 10, находка №2.
"""

import logging
from typing import Dict, List, Mapping, Optional, Union
import numpy as np
from sklearn.base import ClassifierMixin

from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer
from subspace_conjugacy.algorithms.subclass_export import equalize_subspace_bases
from subspace_conjugacy.core.metrics import conjugate_criterion
from subspace_conjugacy.models.base import BaseSubspaceEstimator

logger = logging.getLogger(__name__)

ClassLabel = Union[int, str, float]


# ClassifierMixin ДОЛЖЕН идти первым в списке баз ниже: в современном
# sklearn is_classifier(estimator) резолвит __sklearn_tags__ через MRO, и
# при обратном порядке (BaseSubspaceEstimator, ClassifierMixin) тег
# "classifier" перекрывается дефолтным из BaseEstimator. Итог —
# is_classifier() == False, и sklearn (GridSearchCV/RandomizedSearchCV/
# cross_val_score с целочисленным cv) молча использует обычный KFold вместо
# StratifiedKFold, что на данных, сгруппированных по классам, даёт
# полностью неверный (например, accuracy≡0) score. См. model_selection/.
class SubspaceConjugacyClassifier(ClassifierMixin, BaseSubspaceEstimator):
    """Классификатор многомерных данных на основе подпространств (теория C).

    Parameters
    ----------
    n_subclasses : int or Mapping[ClassLabel, int], default=8
        Количество подклассов (подпространств), формируемых для каждого
        класса. Целое число — одинаковое число подклассов для всех классов
        (поведение по умолчанию, как раньше). Словарь {class_label: int} —
        независимая настройка на класс: статья Korshikov & Fursov
        ("Description of the Clustering Method") явно отмечает, что
        оптимальное число подклассов для разных патологий, как правило,
        различается (см. также Table I в статье — glioma/meningioma/pituitary
        достигают максимума accuracy при разном числе подклассов). При
        словаре ключи ДОЛЖНЫ покрывать все классы, встречающиеся в y при
        fit() — иначе ValueError с списком недостающих классов; лишние
        ключи (классов, которых нет в y) допускаются и игнорируются с
        предупреждением в лог (например, если словарь общий для нескольких
        похожих датасетов). Фактически использованные значения после fit()
        доступны в self.n_subclasses_by_class_.
    freeze_basis_at : int or "auto", default=2
        Размер базиса каждого подпространства после кластеризации (k).
        - int (>= 2) — классическое поведение (совпадает с NB6-8 и всеми
          более ранними версиями библиотеки): каждый подкласс получает
          ровно этот k, независимо от того, сколько векторов реально
          "выиграли" argmax R на фазе B.2 — берутся только первые k
          столбцов (исходная пара из B.1 при k=2).
        - "auto" — НЕ отбрасывает результат роста B.2: каждый класс растится
          без ограничения (freeze_basis_at=None внутри FursovClusterer), а
          после кластеризации ВСЕХ классов подпространства усекаются до
          общего минимального фактически достигнутого k
          (equalize_subspace_bases) — так делает статья Korshikov & Fursov,
          чтобы R(x, Y) оставался сопоставим между подпространствами разного
          размера, не жертвуя всем, что подпространство накопило при росте.
          Итоговое k — в self.equalized_basis_size_ после fit().
        ⚠ "auto" — только для fit(); fit_from_subclass_bases() использует
        уже готовые базисы "как есть", без усечения.
    growth_strategy : {"default", "master"}, default="default"
        Стратегия наполнения кластеров в FursovClusterer (Фаза B.2):
        "default" — argmax R(x, Y_s); "master" — ratio к среднему,
        канон-совместимая аппроксимация идеи NB7, не побитовая реплика
        (см. algorithms/subclass_growth.py, docstring модуля).
    reg_param : float, default=1e-8
        Коэффициент регуляризации при обращении матрицы Грама.
    filter_dependent : bool, default=False
        Если True — перед кластеризацией КАЖДОГО класса исключает почти
        линейно зависимые эталонные векторы (LinearDependencyFilter,
        algorithms/reference_filter.py) — статья Korshikov & Fursov,
        "Problem Definition": "almost linearly dependent vectors are
        excluded from the set of reference vectors" (refactoring_plan.txt,
        раздел 10, находка №3). Исключённые объекты никогда не участвуют в
        кластеризации своего класса; их индексы (в исходном X, переданном в
        fit()) — в self.excluded_indices_by_class_. По умолчанию выключено —
        обратная совместимость.
    dependency_threshold : float, default=0.999
        Порог показателя сопряжённости для LinearDependencyFilter. Действует
        только если filter_dependent=True.
    filter_low_informativeness : bool, default=False
        Если True — перед кластеризацией КАЖДОГО класса (и перед фильтром
        зависимости, если он тоже включён) исключает малоинформативные
        эталонные векторы (LowInformativenessFilter,
        algorithms/informativeness_filter.py) — статья Korshikov & Fursov,
        3-й эксперимент: "images with the number of white pixels less than
        50% of the average... are cut off" (refactoring_plan.txt, раздел
        10, находка №5). Порог "белых" пикселей считается НЕЗАВИСИМО в
        каждом классе (среднее по объектам ЭТОГО класса, не по всей
        выборке) — так же, как n_subclasses/filter_dependent применяются
        независимо на класс. По умолчанию выключено — обратная совместимость.
    informativeness_threshold : float, default=10
        Порог яркости "белого"/полезного элемента для LowInformativenessFilter.
        Действует только если filter_low_informativeness=True.
    informativeness_min_fraction : float, default=0.5
        Минимальная допустимая доля от среднего числа "белых" элементов по
        выборке класса (статья: 0.5 = 50%). Действует только если
        filter_low_informativeness=True.
    split_correlated_pairs : bool, default=False
        Если True — перед кластеризацией КАЖДОГО класса (после фильтров
        filter_low_informativeness/filter_dependent, если они тоже
        включены) прогоняет CorrelatedPairSplitter
        (algorithms/correlated_pair_splitter.py): делит оставшиеся эталонные
        векторы класса на два подмножества похожих пар и использует для
        кластеризации только одно (correlated_pairs_subset). В отличие от
        filter_dependent/filter_low_informativeness, источник этого метода —
        ЧЕРНОВИК другой, неопубликованной статьи (theory/Макет новой
        статьи.docx, "Первый этап"), а не проверенная публикация Korshikov &
        Fursov про МРТ мозга. По умолчанию выключено — обратная
        совместимость.
    correlated_pairs_subset : {"a", "b"}, default="a"
        Какое из двух подмножеств CorrelatedPairSplitter использовать.
        Действует только если split_correlated_pairs=True.

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
    n_subclasses_by_class_ : Dict[ClassLabel, int] or None
        Фактически использованное число подклассов для каждого класса после
        fit() — всегда словарь, даже если n_subclasses был передан как int
        (тогда все значения одинаковы). Удобно для интроспекции при
        n_subclasses=dict.
    equalized_basis_size_ : int or None
        Заполняется только когда freeze_basis_at="auto": итоговый общий
        размер базиса k, до которого были усечены ВСЕ подпространства всех
        классов (минимум среди фактически выросших размеров). None, если
        freeze_basis_at был int (усечение делает FursovClusterer напрямую,
        равнять между классами не требуется — все и так одного размера) или
        модель заполнена через fit_from_subclass_bases().
    excluded_indices_by_class_ : Dict[ClassLabel, np.ndarray] or None
        Заполняется только когда filter_dependent=True и/или
        filter_low_informativeness=True: {class_label: индексы (в исходном
        X, переданном в fit()) векторов ЭТОГО класса, исключённых хотя бы
        одним из включённых фильтров — объединение, тем же способом, что и
        FursovClusterer.excluded_indices_}. None, если оба фильтра выключены.
    """

    def __init__(
        self,
        n_subclasses: Union[int, Mapping[ClassLabel, int]] = 8,
        freeze_basis_at: Union[int, str, None] = 2,
        growth_strategy: str = "default",
        reg_param: float = 1e-8,
        filter_dependent: bool = False,
        dependency_threshold: float = 0.999,
        filter_low_informativeness: bool = False,
        informativeness_threshold: float = 10,
        informativeness_min_fraction: float = 0.5,
        split_correlated_pairs: bool = False,
        correlated_pairs_subset: str = "a",
    ) -> None:
        super().__init__(
            n_subclasses=n_subclasses,
            n_centers_init=2,
            reg_param=reg_param,
        )
        self.freeze_basis_at = freeze_basis_at
        self.growth_strategy = growth_strategy
        self.filter_dependent = filter_dependent
        self.dependency_threshold = dependency_threshold
        self.filter_low_informativeness = filter_low_informativeness
        self.informativeness_threshold = informativeness_threshold
        self.informativeness_min_fraction = informativeness_min_fraction
        self.split_correlated_pairs = split_correlated_pairs
        self.correlated_pairs_subset = correlated_pairs_subset
        self.classes_: Optional[np.ndarray] = None
        self.subspaces_: Dict[ClassLabel, List[np.ndarray]] = {}
        self.flat_subclass_labels_: Optional[np.ndarray] = None
        self.n_subclasses_by_class_: Optional[Dict[ClassLabel, int]] = None
        self.equalized_basis_size_: Optional[int] = None
        self.excluded_indices_by_class_: Optional[Dict[ClassLabel, np.ndarray]] = None

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
        n_subclasses_by_class = self._resolve_n_subclasses(self.classes_)
        use_auto_equalization = self.freeze_basis_at == "auto"
        effective_freeze_basis_at = self._resolve_freeze_basis_at()
        logger.info(
            "SubspaceConjugacyClassifier.fit: старт, %d объектов, классы=%s, "
            "n_subclasses=%s, freeze_basis_at=%r, growth_strategy=%s.",
            X_clean.shape[0], list(self.classes_), n_subclasses_by_class,
            self.freeze_basis_at, self.growth_strategy,
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
        any_filter_enabled = (
            self.filter_dependent
            or self.filter_low_informativeness
            or self.split_correlated_pairs
        )
        excluded_indices_by_class = {} if any_filter_enabled else None

        for cls in self.classes_:
            cls_global_indices = np.where(y_clean == cls)[0]
            X_cls = X_clean[cls_global_indices]
            n_subclasses_cls = n_subclasses_by_class[cls]
            logger.info(
                "SubspaceConjugacyClassifier.fit: класс '%s' — кластеризация %d объектов "
                "на %d подклассов.", cls, X_cls.shape[0], n_subclasses_cls,
            )
            if X_cls.shape[0] < n_subclasses_cls:
                logger.error(
                    "SubspaceConjugacyClassifier.fit: класс '%s' содержит %d объектов "
                    "< n_subclasses=%d.", cls, X_cls.shape[0], n_subclasses_cls,
                )
                raise ValueError(
                    f"Класс '{cls}' содержит {X_cls.shape[0]} объектов, "
                    f"что меньше числа подклассов ({n_subclasses_cls})."
                )

            clusterer = FursovClusterer(
                n_subclasses=n_subclasses_cls,
                freeze_basis_at=effective_freeze_basis_at,
                growth_strategy=self.growth_strategy,
                reg_param=self.reg_param,
                filter_dependent=self.filter_dependent,
                dependency_threshold=self.dependency_threshold,
                filter_low_informativeness=self.filter_low_informativeness,
                informativeness_threshold=self.informativeness_threshold,
                informativeness_min_fraction=self.informativeness_min_fraction,
                split_correlated_pairs=self.split_correlated_pairs,
                correlated_pairs_subset=self.correlated_pairs_subset,
            )
            clusterer.fit(X_cls)
            self.subspaces_[cls] = clusterer.subspaces_
            logger.debug(
                "SubspaceConjugacyClassifier.fit: класс '%s' готов, %d подпространств "
                "(размеры базисов: %s).", cls, len(clusterer.subspaces_),
                [Y.shape[1] for Y in clusterer.subspaces_],
            )

            if any_filter_enabled:
                # clusterer.excluded_indices_ — индексы внутри X_cls (подвыборки
                # этого класса, объединение обоих фильтров), переводим в
                # индексы исходного X, переданного в fit(), чтобы пользователь
                # мог писать X[excluded] без необходимости самостоятельно
                # восстанавливать маску по y.
                excluded_indices_by_class[cls] = cls_global_indices[
                    clusterer.excluded_indices_
                ]
                if len(clusterer.excluded_indices_) > 0:
                    logger.warning(
                        "SubspaceConjugacyClassifier.fit: класс '%s' — %d "
                        "объект(ов) исключены фильтрами (малоинформативные: %d, "
                        "почти линейно зависимые: %d).",
                        cls, len(clusterer.excluded_indices_),
                        len(clusterer.excluded_by_informativeness_),
                        len(clusterer.excluded_by_dependency_),
                    )

        self.excluded_indices_by_class_ = excluded_indices_by_class

        if use_auto_equalization:
            self.subspaces_, self.equalized_basis_size_ = equalize_subspace_bases(
                self.subspaces_
            )
        else:
            self.equalized_basis_size_ = None

        self.n_subclasses_by_class_ = n_subclasses_by_class
        self.flat_subclass_labels_ = self._build_flat_subclass_labels()
        self.is_fitted_ = True
        logger.info(
            "SubspaceConjugacyClassifier.fit: готово, %d подпространств всего "
            "(по классам: %s)%s.",
            len(self.flat_subclass_labels_), n_subclasses_by_class,
            f", equalized_basis_size_={self.equalized_basis_size_}" if use_auto_equalization else "",
        )
        return self

    def _resolve_freeze_basis_at(self) -> Optional[int]:
        """Транслирует self.freeze_basis_at в значение для FursovClusterer.

        "auto" -> None (полный рост без ограничения; равнение между классами
        применяется отдельно, после кластеризации ВСЕХ классов — см. fit(),
        FursovClusterer ничего не знает о других классах). None -> None без
        изменений (полный рост, БЕЗ автоматического равнения — на свой риск,
        R(x, Y) между подпространствами разного размера тогда не гарантированно
        сопоставим). int -> проверяется и возвращается как есть.

        Raises
        ------
        ValueError
            Если freeze_basis_at — не None, не "auto" и не целое >= 2.
        """
        if self.freeze_basis_at is None or self.freeze_basis_at == "auto":
            return None
        if not isinstance(self.freeze_basis_at, (int, np.integer)) or self.freeze_basis_at < 2:
            logger.error(
                "SubspaceConjugacyClassifier._resolve_freeze_basis_at: "
                "freeze_basis_at=%r недопустим.", self.freeze_basis_at,
            )
            raise ValueError(
                "freeze_basis_at должен быть целым числом >= 2, None или "
                f"'auto', получено {self.freeze_basis_at!r}."
            )
        return int(self.freeze_basis_at)

    def _resolve_n_subclasses(self, classes: np.ndarray) -> Dict[ClassLabel, int]:
        """Разворачивает self.n_subclasses (int или dict) в словарь по классам.

        Parameters
        ----------
        classes : np.ndarray
            Уникальные метки классов обучающей выборки (self.classes_).

        Returns
        -------
        n_subclasses_by_class : Dict[ClassLabel, int]
            {class_label: n_subclasses} для каждого класса из ``classes``.

        Raises
        ------
        ValueError
            Если self.n_subclasses — словарь, но не содержит значения хотя
            бы для одного класса из ``classes``.
        """
        if not isinstance(self.n_subclasses, Mapping):
            return {cls: int(self.n_subclasses) for cls in classes}

        missing = [cls for cls in classes if cls not in self.n_subclasses]
        if missing:
            logger.error(
                "SubspaceConjugacyClassifier._resolve_n_subclasses: n_subclasses "
                "не содержит значения для классов %s (есть ключи: %s).",
                missing, list(self.n_subclasses.keys()),
            )
            raise ValueError(
                "n_subclasses задан словарём, но не содержит значения для "
                f"классов: {missing}. Есть ключи: {list(self.n_subclasses.keys())}."
            )

        extra = [key for key in self.n_subclasses if key not in set(classes)]
        if extra:
            logger.warning(
                "SubspaceConjugacyClassifier._resolve_n_subclasses: ключи n_subclasses "
                "%s отсутствуют среди классов обучающей выборки %s — игнорируются.",
                extra, list(classes),
            )

        return {cls: int(self.n_subclasses[cls]) for cls in classes}

    def fit_from_subclass_bases(
        self,
        subspaces_by_class: Dict[ClassLabel, List[np.ndarray]],
        equalize: bool = False,
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
            N должна совпадать для всех базисов всех классов; k может
            отличаться между подклассами/классами, если equalize=True
            (иначе см. предупреждение в predict_r_matrix_flat про
            несопоставимость R(x, Y) для базисов разного размера).
        equalize : bool, default=False
            Если True — перед использованием усекает все базисы всех
            классов до общего минимального k (equalize_subspace_bases, тот
            же механизм, что и SubspaceConjugacyClassifier(freeze_basis_at=
            "auto") в fit()). Нужно, например, когда базисы получены через
            FursovPipeline.run_class(..., freeze_basis_at=None) независимо
            для каждого класса и могли вырасти до разных k.

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

        if equalize:
            subspaces_by_class, self.equalized_basis_size_ = equalize_subspace_bases(
                subspaces_by_class
            )
        else:
            self.equalized_basis_size_ = None

        self.classes_ = np.array(list(subspaces_by_class.keys()))
        self.subspaces_ = {
            cls: list(bases) for cls, bases in subspaces_by_class.items()
        }
        self.n_features_in_ = n_features_set.pop()
        self.n_subclasses_by_class_ = {
            cls: len(bases) for cls, bases in self.subspaces_.items()
        }
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

        Notes
        -----
        R(x, Y) не масштабируется по k (числу столбцов Y) — базис большего
        размера при прочих равных склонен давать больший R просто за счёт
        того, что охватывает больше измерений признакового пространства.
        Сравнение между классами честно только если все Y_{c,s} имеют
        одинаковый k — это гарантируется по умолчанию (freeze_basis_at=2)
        и явным равнением при freeze_basis_at="auto"/fit_from_subclass_bases
        (equalize=True). Если подпространства собраны вручную с разным k
        (например, fit_from_subclass_bases(..., equalize=False) на базисах
        неравного размера), результат predict()/predict_r_matrix будет
        смещён в пользу классов с более крупными подпространствами.
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
