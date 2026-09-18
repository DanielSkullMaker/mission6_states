"""Parity-тесты для legacy NB4 (per-vector trio_list) на реальных данных.

NB4 (4_1_Fursov_vector_processing) для каждого вектора i ищет
argmin_j cos(v_i, v_j) и складывает результат в trio_list[M x 3].
Это НЕ канонический шаг A.1 (одна глобальная пара) — расхождение
задокументировано в algorithms/legacy/per_vector_pairs.py и в
refactoring_plan.txt (раздел 1.7, 2.1).

Тесты ниже проверяют:
  1. compute_per_vector_pairs_nb4 корректен и не воспроизводит баг
     NB4 (vector_norm(first_reference)) на реальных МРТ-векторах.
  2. Глобальный минимум, извлечённый из trio_list, математически обязан
     совпадать с каноническим GlobalMinCosinePairFinder (симметричная
     матрица косинусов — доказательство see docstring теста).
  3. NB4 расходится с каноном на уровне отдельных строк trio_list —
     это ожидаемое поведение, а не регрессия.

Требуют реального датасета (datasets/{class}_centered/*.png) —
skip, если недоступен (см. tests/test_parity/conftest.py).
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
from subspace_conjugacy.algorithms.legacy.per_vector_pairs import (
    compute_per_vector_pairs_nb4,
    extract_pair_from_trio_list,
    find_global_min_from_trio_list,
)

pytestmark = pytest.mark.notebook_parity


class TestNB4TrioListStructure:
    """Структурные инварианты trio_list на реальных МРТ-векторах."""

    def test_trio_list_has_one_entry_per_vector(self, glioma_vectors_full):
        X = glioma_vectors_full
        trio_list = compute_per_vector_pairs_nb4(X)

        assert len(trio_list) == len(X)

    def test_trio_list_entries_are_valid(self, glioma_vectors_full):
        X = glioma_vectors_full
        trio_list = compute_per_vector_pairs_nb4(X)

        for i, (idx, argmin_j, cos_value) in enumerate(trio_list):
            assert idx == i
            assert argmin_j != i  # диагональ исключена (np.inf на диагонали)
            assert 0 <= argmin_j < len(X)
            assert -1.0 <= cos_value <= 1.0

    def test_extract_pair_from_trio_list_matches_row(self, glioma_vectors_full):
        X = glioma_vectors_full
        trio_list = compute_per_vector_pairs_nb4(X)

        idx1, idx2, cos_value = extract_pair_from_trio_list(trio_list, index=8)
        assert (idx1, idx2, cos_value) == trio_list[8]


class TestNB4VsCanonicalGlobalPair:
    """Мат. связь между NB4 (per-vector) и каноническим A.1 (global argmin)."""

    def test_global_min_from_trio_list_matches_canonical(self, glioma_vectors_full):
        """Глобальный минимум по trio_list обязан совпасть с canonical A.1.

        Обоснование: матрица косинусных сходств симметрична
        (cos(a, b) == cos(b, a)). trio_list[i] = argmin_j cos(i, j) — это
        минимум i-й строки. Глобальный минимум всей матрицы лежит в какой-то
        строке i* и является минимумом этой строки по определению argmin,
        поэтому min_i(trio_list[i].cos) обязан равняться глобальному
        минимуму всей матрицы — тому же значению, которое находит
        GlobalMinCosinePairFinder. Это не совпадение реализаций, а свойство
        симметричной матрицы, которое должно выполняться на любых реальных
        данных.
        """
        X = glioma_vectors_full

        trio_list = compute_per_vector_pairs_nb4(X)
        legacy_i, legacy_j, legacy_cos = find_global_min_from_trio_list(trio_list)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        canon_i, canon_j = finder.pair_indices_
        canon_cos = finder.similarity_value_

        assert sorted([legacy_i, legacy_j]) == sorted([canon_i, canon_j])
        assert legacy_cos == pytest.approx(canon_cos, abs=1e-9)

    @pytest.mark.parametrize("class_name", ["glioma", "meningioma", "pituitary"])
    def test_global_min_matches_canonical_for_all_classes(
        self, load_real_class_vectors, class_name
    ):
        """Тот же инвариант — на всех трёх реальных классах датасета."""
        X = load_real_class_vectors(class_name, count=100)

        trio_list = compute_per_vector_pairs_nb4(X)
        legacy_i, legacy_j, legacy_cos = find_global_min_from_trio_list(trio_list)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        canon_i, canon_j = finder.pair_indices_

        assert sorted([legacy_i, legacy_j]) == sorted([canon_i, canon_j])

    def test_nb4_local_pairs_can_diverge_from_global_pair(self, glioma_vectors_full):
        """Документирует РАСХОЖДЕНИЕ NB4 с каноном (не баг, а факт метода).

        Для большинства векторов i их "локальная" наиболее непохожая пара
        (trio_list[i]) НЕ совпадает с глобально самой непохожей парой —
        именно поэтому NB5 требует ручного выбора индекса из trio_list
        (см. refactoring_plan.txt, раздел 1.7), а канонический алгоритм
        вычисляет глобальную пару напрямую вместо этого.
        """
        X = glioma_vectors_full
        trio_list = compute_per_vector_pairs_nb4(X)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        canon_pair = set(finder.pair_indices_)

        local_pairs_matching_global = sum(
            1 for (i, j, _) in trio_list if {i, j} == canon_pair
        )

        # Глобальная пара встречается как "своя лучшая пара" самое большее
        # у двух векторов (i* и j* друг для друга) — не у всех M векторов.
        assert local_pairs_matching_global <= 2
        assert local_pairs_matching_global < len(trio_list)
