"""Theory Algorithm: Recursive Bisection Clusterer.

Источник: theory/Теоретические основы.docx (В.А. Фурсов, IAPR,
"Применение показателя сопряженности как меры близости в задачах
распознавания изображений"), раздел 5 "Кластеризация изображений с помощью
показателей сопряженности".

Статья буквально описывает алгоритм, который строит РОВНО ДВА подкласса за
один проход:
  1. Два наиболее "удалённых" (по минимуму косинуса — формула (8)) вектора
     класса назначаются центрами — эквивалент A.1 (GlobalMinCosinePairFinder).
  2. К каждому из двух центров присоединяется по одному вектору, для
     которых величины (9), (10) принимают МАКСИМАЛЬНЫЕ значения — формулы
     встроены в документ как WMF-картинки (Equation Editor) и не читаются
     как текст, поэтому точная величина не подтверждена побитово, но
     направление экстремума ("максимальные") в тексте однозначно. См.
     algorithms/subclass_seed.py::CosineSecondVectorAttacher(criterion="max").
  3. Оставшиеся векторы класса присоединяются ПО ОДНОМУ к ближайшему (по
     максимуму R) из двух подпространств, пока не распределены ВСЕ векторы
     класса — эквивалент B.2 (ConjugacyClusterGrowth), но исчерпывающий
     (freeze_basis_at=None) и строго между двумя кластерами, не n_subclasses.
  4. Чтобы получить БОЛЬШЕ двух подклассов, статья прямо предлагает
     "продолжить разбиение каждого из этих подклассов по той же схеме" —
     т.е. РЕКУРСИВНУЮ бисекцию (2 -> 4 -> 8 -> ...), а НЕ единый плоский
     проход, который строит все n_subclasses центров сразу (так делает
     канонический ReferenceCenterBuilder, algorithms/reference_centers.py).

Это и есть ключевое отличие от канона библиотеки (FursovClusterer):
  - Канон: ОДНА растущая базисная матрица из уже найденных центров, все
    n_subclasses центров находятся за один проход (A.2-A.3), затем ВСЕ
    n_subclasses подпространств растут СОВМЕСТНО в одном конкурентном
    цикле B.2 (глобальный argmax по remaining x n_subclasses).
  - Эта реализация: РЕКУРСИВНАЯ бисекция. На каждом шаге выбирается самый
    крупный ещё не финализированный "лист" и делится РОВНО НА ДВА (A.1 +
    B.1 + исчерпывающий B.2 для ДВУХ кластеров), пока не набрано
    n_subclasses листьев. n_subclasses не обязан быть степенью двойки —
    на каждой итерации делится самый крупный лист (аналог bisecting
    k-means), что естественно обобщает идею статьи (буквально описанную
    только для 2 -> 4 -> 8 -> ...) на произвольное n_subclasses.

Финальный базис каждого листа строится ОТДЕЛЬНО, заново на его собственном
подмножестве векторов (A.1 + исчерпывающий рост с ОДНОЙ парой, truncated до
freeze_basis_at) — а не "наследуется" от промежуточного шага бисекции,
который его породил. Это проще и эквивалентно: при freeze_basis_at=2
результат — буквально пара из A.1 листа, как и в каноне.

НЕ часть опубликованной статьи Korshikov & Fursov (theory/VI_Korshikov_...);
источник — более ранняя статья В.А. Фурсова, другой документ.
"""

import logging
from typing import List, Literal, Optional
import numpy as np

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth
from subspace_conjugacy.core.metrics import conjugate_criterion

logger = logging.getLogger(__name__)


class RecursiveBisectionClusterer:
    """Строит n_subclasses подпространств рекурсивной бисекцией (теория, раздел 5).

    См. docstring модуля для полного разбора соответствия и отличий от
    канонического FursovClusterer.

    Parameters
    ----------
    n_subclasses : int, default=8
        Количество формируемых подклассов (листьев дерева бисекции).
    freeze_basis_at : int or None, default=2
        Размер финального базиса каждого листа. Если None — базис листа
        растёт без ограничения (до фактического размера листа).
    b1_criterion : {"max", "min"}, default="max"
        Критерий присоединения второго вектора к каждому из двух центров
        при делении узла (см. CosineSecondVectorAttacher). "max" — буквальное
        текстуальное указание статьи ("максимальные значения"); "min" —
        тот же критерий, что и в каноне B.1, но внутри рекурсивной, а не
        плоской, схемы (удобно для изолированной проверки эффекта именно
        бисекции, отдельно от эффекта направления критерия).
    reg_param : float, default=1e-8
        Параметр регуляризации Тихонова для (Y^T Y)^{-1}.

    Attributes
    ----------
    subspaces_ : list[np.ndarray] or None
        Финальные базисы листьев (N, k).
    labels_ : np.ndarray or None
        Метка листа для каждого входного вектора (M,).
    n_subclasses_ : int or None
        Фактическое количество листьев (всегда равно n_subclasses при
        успешном fit() — иначе fit() завершается ValueError).
    leaf_sizes_history_ : list[int] or None
        Размеры финальных листьев, в порядке self.subspaces_/метки 0..n-1.
    is_fitted_ : bool
        Флаг, указывающий, был ли выполнен fit().
    """

    def __init__(
        self,
        n_subclasses: int = 8,
        freeze_basis_at: Optional[int] = 2,
        b1_criterion: Literal["max", "min"] = "max",
        reg_param: float = 1e-8,
    ) -> None:
        if n_subclasses < 2:
            raise ValueError(f"n_subclasses должен быть >= 2, получено {n_subclasses}")

        if freeze_basis_at is not None and freeze_basis_at < 2:
            raise ValueError(
                f"freeze_basis_at должен быть >= 2 или None, получено {freeze_basis_at}"
            )

        if b1_criterion not in ("max", "min"):
            raise ValueError(
                f"b1_criterion должен быть 'max' или 'min', получено {b1_criterion!r}"
            )

        self.n_subclasses = n_subclasses
        self.freeze_basis_at = freeze_basis_at
        self.b1_criterion = b1_criterion
        self.reg_param = reg_param

        self.subspaces_: Optional[List[np.ndarray]] = None
        self.labels_: Optional[np.ndarray] = None
        self.n_subclasses_: Optional[int] = None
        self.leaf_sizes_history_: Optional[List[int]] = None
        self.is_fitted_: bool = False

    def fit(self, X: np.ndarray) -> "RecursiveBisectionClusterer":
        """Строит n_subclasses подпространств рекурсивной бисекцией.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов одного класса, размерности (M, N).

        Returns
        -------
        self : RecursiveBisectionClusterer

        Raises
        ------
        ValueError
            Если M < n_subclasses * 2, либо бисекция не может продолжиться
            (крупнейший оставшийся лист содержит < 4 векторов, а нужно
            больше листьев), либо лист меньше freeze_basis_at.
        """
        X_arr = self._validate_input(X)
        n_samples = X_arr.shape[0]
        min_required = self.n_subclasses * 2

        if n_samples < min_required:
            raise ValueError(
                f"Недостаточно векторов: требуется минимум {min_required} "
                f"для {self.n_subclasses} подклассов (>=2 на лист), "
                f"получено {n_samples}."
            )

        logger.info(
            "RecursiveBisectionClusterer.fit: старт, %d векторов, "
            "n_subclasses=%d, b1_criterion=%s.",
            n_samples, self.n_subclasses, self.b1_criterion,
        )

        leaves: List[np.ndarray] = [np.arange(n_samples)]

        while len(leaves) < self.n_subclasses:
            sizes = [len(leaf) for leaf in leaves]
            largest_idx = int(np.argmax(sizes))

            if sizes[largest_idx] < 4:
                raise ValueError(
                    f"Невозможно получить {self.n_subclasses} подклассов "
                    f"рекурсивной бисекцией: крупнейший оставшийся лист "
                    f"содержит {sizes[largest_idx]} < 4 векторов (нужно >= 2 "
                    f"на каждую из двух половин при делении). Сформировано "
                    f"{len(leaves)}/{self.n_subclasses} листьев из {n_samples} "
                    f"векторов."
                )

            leaf = leaves.pop(largest_idx)
            child0, child1 = self._split_leaf(X_arr, leaf)
            leaves.append(child0)
            leaves.append(child1)
            logger.debug(
                "RecursiveBisectionClusterer.fit: разбит лист размера %d -> "
                "%d + %d (листьев: %d/%d).",
                len(leaf), len(child0), len(child1), len(leaves), self.n_subclasses,
            )

        labels = np.full(n_samples, -1, dtype=int)
        subspaces = []
        for leaf_id, leaf in enumerate(leaves):
            labels[leaf] = leaf_id
            subspaces.append(self._build_leaf_basis(X_arr, leaf))

        self.labels_ = labels
        self.subspaces_ = subspaces
        self.n_subclasses_ = len(leaves)
        self.leaf_sizes_history_ = [len(leaf) for leaf in leaves]
        self.is_fitted_ = True
        logger.info(
            "RecursiveBisectionClusterer.fit: готово, размеры листьев=%s.",
            self.leaf_sizes_history_,
        )

        return self

    def _split_leaf(self, X_arr: np.ndarray, leaf: np.ndarray):
        """Делит один лист ровно на два подмножества (A.1 + B.1 + исчерпывающий B.2)."""
        X_leaf = X_arr[leaf]

        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X_leaf)
        idx1_local, idx2_local = pair_finder.pair_indices_

        attacher = CosineSecondVectorAttacher(criterion=self.b1_criterion)
        attacher.fit(X_leaf, np.array([idx1_local, idx2_local]))

        growth = ConjugacyClusterGrowth(freeze_basis_at=None, reg_param=self.reg_param)
        growth.fit(X_leaf, attacher.pairs_)

        child0 = leaf[growth.labels_ == 0]
        child1 = leaf[growth.labels_ == 1]
        return child0, child1

    def _build_leaf_basis(self, X_arr: np.ndarray, leaf: np.ndarray) -> np.ndarray:
        """Строит финальный базис (N, k) листа — заново, на его собственных векторах."""
        X_leaf = X_arr[leaf]

        pair_finder = GlobalMinCosinePairFinder()
        pair_finder.fit(X_leaf)
        idx1_local, idx2_local = pair_finder.pair_indices_

        growth = ConjugacyClusterGrowth(
            freeze_basis_at=self.freeze_basis_at, reg_param=self.reg_param,
        )
        growth.fit(X_leaf, np.array([[idx1_local, idx2_local]]))
        basis = growth.subspace_bases_[0]

        if self.freeze_basis_at is not None and basis.shape[1] < self.freeze_basis_at:
            raise ValueError(
                f"Лист из {len(leaf)} векторов недостаточен для "
                f"freeze_basis_at={self.freeze_basis_at} (доступно только "
                f"{basis.shape[1]} столбцов)."
            )
        return basis

    def get_subclass_sizes(self) -> np.ndarray:
        """Возвращает количество векторов в каждом финальном листе.

        Returns
        -------
        sizes : np.ndarray
            Массив размера (n_subclasses,).
        """
        self._check_is_fitted()
        return np.bincount(self.labels_, minlength=self.n_subclasses_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает лист (подкласс) для новых векторов через argmax R.

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов размерности (M, N).

        Returns
        -------
        labels : np.ndarray
            Предсказанные метки листьев (M,).
        """
        self._check_is_fitted()
        X_arr = self._validate_input(X)

        R_matrix = np.column_stack(
            [conjugate_criterion(X_arr, Y, reg_param=self.reg_param) for Y in self.subspaces_]
        )
        return np.argmax(R_matrix, axis=1)

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Валидирует входную матрицу X."""
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim != 2:
            raise ValueError(f"Ожидалась 2D матрица векторов, получена {X_arr.ndim}D.")

        if X_arr.size == 0:
            raise ValueError("Передана пустая матрица X.")

        return X_arr

    def _check_is_fitted(self) -> None:
        """Проверяет, был ли вызван fit()."""
        if not self.is_fitted_:
            raise RuntimeError("Модель не обучена. Вызовите fit(X) перед использованием.")

    def __repr__(self) -> str:
        if self.is_fitted_:
            return (
                f"RecursiveBisectionClusterer(n_subclasses={self.n_subclasses_}, "
                f"freeze_basis_at={self.freeze_basis_at}, "
                f"b1_criterion={self.b1_criterion!r}, fitted=True)"
            )
        return (
            f"RecursiveBisectionClusterer(n_subclasses={self.n_subclasses}, "
            f"freeze_basis_at={self.freeze_basis_at}, "
            f"b1_criterion={self.b1_criterion!r}, fitted=False)"
        )
