"""Deprecated location for SubspaceClusterer.

Раньше этот модуль содержал самостоятельную (не каноническую) реализацию
кластеризации: начальные центры выбирались так же, как в теории A.1-A.3,
но фаза наполнения кластеров сразу росла через max R, без промежуточного
B.1 (cosine-seed второго вектора для каждого центра) — см.
refactoring_plan.txt, раздел 1.6 ("⚠ нет фазы B.1"). Модуль также
дублировал резервные реализации conjugate_criterion/cosine_similarity_matrix
из core/metrics.py.

По плану рефакторинга (раздел 6, п.1-2: "Канон — единственный путь...",
"Убрать duplicate conjugate_criterion в clusterer.py/classifier.py") этот
модуль сведён к алиасу канонического FursovClusterer (фасад A.1→A.3→B.1→B.2
из algorithms/fursov_clusterer.py). Импортируйте FursovClusterer напрямую
из subspace_conjugacy или subspace_conjugacy.algorithms — этот модуль
сохранён только ради обратной совместимости старых импортов вида
``from subspace_conjugacy.models.clusterer import SubspaceClusterer``.
"""

from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer

SubspaceClusterer = FursovClusterer

__all__ = ["SubspaceClusterer"]
