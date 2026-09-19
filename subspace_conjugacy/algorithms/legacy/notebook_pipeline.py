"""Legacy Notebook Pipeline: staged CSV flow (NB4→NB5→NB6→NB7).

⚠️ ВАЖНО: Этот модуль воспроизводит STAGED подход ноутбуков с промежуточными CSV.
   Для production используйте FursovClusterer (канонический in-memory алгоритм).

Цель: parity тесты с оригинальными ноутбуками.

Staged flow:
  NB4 (legacy) → {class}_class_first_and_second_*.csv
  NB5 → {class}_class_core_vectors.csv
  NB6 → {class}_new_classes.csv
  NB7 → 8_{class}_subclasses_vectors.csv

Используется только в:
  - tests/test_parity/ для проверки совместимости с ноутбуками
  - Воспроизведение экспериментов из оригинальных ноутбуков

НЕ используется в production (используйте FursovClusterer).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from subspace_conjugacy.algorithms.legacy.per_vector_pairs import (
    compute_per_vector_pairs_nb4,
)
from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder
from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth
from subspace_conjugacy.algorithms.subclass_export import flatten_subspace_bases

logger = logging.getLogger(__name__)


class NotebookStagedPipeline:
    """Staged CSV pipeline для воспроизведения ноутбуков NB4-7.

    ⚠️ Legacy — используйте FursovClusterer для production.

    Parameters
    ----------
    n_subclasses : int, default=8
        Количество подклассов.
    use_canonical_pair : bool, default=True
        Если True: использует глобальную пару из GlobalMinCosinePairFinder (теория A.1).
        Если False: использует NB4 trio_list (legacy, для parity).
    manual_pair_index : int or None, default=None
        Если задан: вручную выбирает пару из trio_list по индексу (как в NB5).
        Используется только если use_canonical_pair=False.
    reg_param : float, default=1e-8
        Параметр регуляризации.

    Examples
    --------
    >>> # Канонический подход (recommended)
    >>> pipeline = NotebookStagedPipeline(n_subclasses=8, use_canonical_pair=True)
    >>> centers = pipeline.run_nb5_reference_centers(X, initial_pair=(8, 36))
    >>>
    >>> # Legacy NB4-5 подход (parity only)
    >>> pipeline = NotebookStagedPipeline(use_canonical_pair=False, manual_pair_index=8)
    >>> trio_list = pipeline.run_nb4_per_vector_pairs(X)
    >>> pair = trio_list[8][:2]  # manual selection like NB5
    """

    def __init__(
        self,
        n_subclasses: int = 8,
        use_canonical_pair: bool = True,
        manual_pair_index: Optional[int] = None,
        reg_param: float = 1e-8,
    ):
        self.n_subclasses = n_subclasses
        self.use_canonical_pair = use_canonical_pair
        self.manual_pair_index = manual_pair_index
        self.reg_param = reg_param

    def run_nb4_per_vector_pairs(self, X: np.ndarray):
        """NB4: Per-vector trio_list (legacy, не канон)."""
        logger.info("NotebookStagedPipeline.run_nb4_per_vector_pairs [LEGACY]: старт.")
        return compute_per_vector_pairs_nb4(X)

    def run_nb5_reference_centers(
        self,
        X: np.ndarray,
        initial_pair: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """NB5: Reference centers через min R (теория A.2-A.3).

        Parameters
        ----------
        X : np.ndarray
            Матрица векторов (M, N).
        initial_pair : tuple[int, int] or None
            Начальная пара индексов. Если None, используется автоопределение
            на основе use_canonical_pair.

        Returns
        -------
        center_indices : np.ndarray
            Индексы центров (n_subclasses,).
        """
        if initial_pair is None:
            if self.use_canonical_pair:
                # Канонический: глобальная пара
                logger.info(
                    "NotebookStagedPipeline.run_nb5_reference_centers: "
                    "initial_pair не задан, используем канон A.1 (GlobalMinCosinePairFinder)."
                )
                finder = GlobalMinCosinePairFinder()
                finder.fit(X)
                initial_pair = finder.pair_indices_
            else:
                # Legacy NB4: вручную из trio_list
                logger.info(
                    "NotebookStagedPipeline.run_nb5_reference_centers [LEGACY]: "
                    "initial_pair не задан, берём из NB4 trio_list по индексу %s.",
                    self.manual_pair_index,
                )
                trio_list = self.run_nb4_per_vector_pairs(X)
                if self.manual_pair_index is None:
                    logger.error(
                        "NotebookStagedPipeline.run_nb5_reference_centers [LEGACY]: "
                        "manual_pair_index не задан при use_canonical_pair=False."
                    )
                    raise ValueError(
                        "manual_pair_index required when use_canonical_pair=False"
                    )
                i, j, _ = trio_list[self.manual_pair_index]
                initial_pair = (i, j)

        builder = ReferenceCenterBuilder(
            n_subclasses=self.n_subclasses,
            reg_param=self.reg_param,
        )
        builder.fit(X, initial_pair)
        return builder.center_indices_

    def run_nb6_subclass_pairs(
        self,
        X: np.ndarray,
        center_indices: np.ndarray,
    ) -> np.ndarray:
        """NB6: Формирование пар для подклассов (теория B.1)."""
        logger.info("NotebookStagedPipeline.run_nb6_subclass_pairs: старт (B.1).")
        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, center_indices)
        return attacher.pairs_

    def run_nb7_cluster_growth(
        self,
        X: np.ndarray,
        pairs: np.ndarray,
        strategy: str = "default",
    ):
        """NB7: Наполнение кластеров (теория B.2)."""
        logger.info(
            "NotebookStagedPipeline.run_nb7_cluster_growth: старт (B.2, strategy=%s).",
            strategy,
        )
        growth = ConjugacyClusterGrowth(
            freeze_basis_at=2,
            strategy=strategy,
            reg_param=self.reg_param,
        )
        growth.fit(X, pairs)
        return growth.subspace_bases_, growth.labels_

    def run_full_pipeline(
        self,
        X: np.ndarray,
        strategy: str = "default",
    ):
        """Выполняет полный staged pipeline NB5→NB6→NB7.

        Returns
        -------
        result : dict
            {
                'center_indices': np.ndarray,
                'pairs': np.ndarray,
                'subspaces': list[np.ndarray],
                'labels': np.ndarray,
                'flattened_bases': np.ndarray,  # для CSV экспорта
            }
        """
        logger.info(
            "NotebookStagedPipeline.run_full_pipeline [LEGACY NB5->NB6->NB7]: "
            "старт, X.shape=%s, use_canonical_pair=%s, strategy=%s.",
            X.shape, self.use_canonical_pair, strategy,
        )

        # NB5
        centers = self.run_nb5_reference_centers(X)

        # NB6
        pairs = self.run_nb6_subclass_pairs(X, centers)

        # NB7
        subspaces, labels = self.run_nb7_cluster_growth(X, pairs, strategy=strategy)

        # Flatten для CSV (формат 8_{class}_subclasses_vectors.csv)
        flattened = flatten_subspace_bases(subspaces, expected_basis_size=2)

        logger.info(
            "NotebookStagedPipeline.run_full_pipeline [LEGACY]: готово, "
            "flattened_bases.shape=%s.", flattened.shape,
        )
        return {
            'center_indices': centers,
            'pairs': pairs,
            'subspaces': subspaces,
            'labels': labels,
            'flattened_bases': flattened,
        }
