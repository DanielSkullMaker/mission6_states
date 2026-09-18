"""Parity/integration-тесты staged pipeline (NB5->NB6->NB7) на реальных данных.

В отличие от tests/test_theory (синтетические i.i.d. векторы), здесь
пайплайн прогоняется на настоящих отцентрированных МРТ-изображениях
(datasets/{class}_centered/*.png), чтобы:
  - проверить, что весь путь image -> vectorize -> cluster -> export
    действительно работает на данных с реальной статистикой (сильно
    коррелированные соседние пиксели, крупные по модулю суммы квадратов
    -- совсем не то же самое, что np.random.randn);
  - зафиксировать формат экспортируемого CSV (S*2, N), совместимый с
    NB6-7/NB8, как того требует раздел 3.6 плана рефакторинга;
  - убедиться, что NotebookStagedPipeline (staged, per-cell как в
    ноутбуках) и FursovClusterer (единый fit()) дают идентичный результат
    на одних и тех же данных -- т.к. оба вызывают одни и те же
    algorithms/* компоненты в одном порядке.

Настоящих CSV, сгенерированных оригинальными (баговыми, macOS-путёвыми)
ноутбуками, в репозитории нет и получить их неоткуда -- поэтому здесь
"parity" проверяется структурно (формы, отсутствие дублей, диапазоны R),
а не побайтовым сравнением с чужим CSV.

Требуют реального датасета -- skip, если недоступен.
"""

import numpy as np
import pytest

from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer
from subspace_conjugacy.algorithms.legacy.notebook_pipeline import (
    NotebookStagedPipeline,
)
from subspace_conjugacy.algorithms.subclass_export import (
    export_clusterer_bases,
    flatten_subspace_bases,
    unflatten_subspace_bases,
)
from subspace_conjugacy.core.metrics import conjugate_criterion

pytestmark = [pytest.mark.notebook_parity, pytest.mark.slow]

N_SUBCLASSES = 8
BASIS_SIZE = 2


class TestStagedPipelineOnRealGlioma:
    """NB5->NB6->NB7 (canonical pair) на реальных glioma-векторах."""

    def test_full_pipeline_produces_valid_structure(self, glioma_vectors_subset):
        X = glioma_vectors_subset
        M = X.shape[0]

        pipeline = NotebookStagedPipeline(
            n_subclasses=N_SUBCLASSES, use_canonical_pair=True
        )
        result = pipeline.run_full_pipeline(X)

        centers = result["center_indices"]
        pairs = result["pairs"]
        subspaces = result["subspaces"]
        labels = result["labels"]
        flattened = result["flattened_bases"]

        # A.2-A.3: n_subclasses уникальных центров
        assert len(centers) == N_SUBCLASSES
        assert len(set(centers)) == N_SUBCLASSES

        # B.1: n_subclasses пар, все 2*n_subclasses индексов уникальны
        assert pairs.shape == (N_SUBCLASSES, 2)
        assert len(set(pairs.flatten())) == 2 * N_SUBCLASSES

        # B.2: все M векторов размечены, ровно n_subclasses базисов k=2
        assert len(labels) == M
        assert np.all(labels >= 0) and np.all(labels < N_SUBCLASSES)
        assert len(subspaces) == N_SUBCLASSES
        for Y in subspaces:
            assert Y.shape == (X.shape[1], BASIS_SIZE)

        # Формат экспорта NB6-7: (S*k, N)
        assert flattened.shape == (N_SUBCLASSES * BASIS_SIZE, X.shape[1])

    def test_r_values_within_valid_range_on_real_pixels(self, glioma_vectors_subset):
        """R(x, Y) должен остаться в [0, 1] на реальных (не i.i.d.) данных.

        Пиксельные векторы сильно коррелированы и имеют большие нормы
        (суммы квадратов uint8 по 65536 признакам) -- это другой числовой
        режим по сравнению с synthetic np.random.randn из test_theory,
        и стоит убедиться, что регуляризация (Y^T Y)^-1 остаётся
        устойчивой именно на таких данных.
        """
        X = glioma_vectors_subset

        clusterer = FursovClusterer(n_subclasses=N_SUBCLASSES, freeze_basis_at=2)
        clusterer.fit(X)

        for Y in clusterer.subspaces_:
            R = conjugate_criterion(X, Y)
            assert np.all(np.isfinite(R))
            assert np.all(R >= 0.0)
            assert np.all(R <= 1.0)

    def test_staged_pipeline_matches_fursov_clusterer_facade(
        self, glioma_vectors_subset
    ):
        """NotebookStagedPipeline(canonical) и FursovClusterer.fit() совпадают.

        Оба вызывают GlobalMinCosinePairFinder -> ReferenceCenterBuilder ->
        CosineSecondVectorAttacher -> ConjugacyClusterGrowth в одном и том
        же порядке с одинаковыми гиперпараметрами, поэтому на одинаковом
        X результат должен быть идентичен -- это гарантия того, что фасад
        (фаза 3.5) не разошёлся со staged-версией (фаза 3.6) на реальных
        данных.
        """
        X = glioma_vectors_subset

        pipeline = NotebookStagedPipeline(
            n_subclasses=N_SUBCLASSES, use_canonical_pair=True
        )
        staged_result = pipeline.run_full_pipeline(X)

        clusterer = FursovClusterer(n_subclasses=N_SUBCLASSES, freeze_basis_at=2)
        clusterer.fit(X)

        assert list(clusterer.center_indices_) == list(staged_result["center_indices"])
        np.testing.assert_array_equal(
            clusterer.initial_pairs_, staged_result["pairs"]
        )
        np.testing.assert_array_equal(clusterer.labels_, staged_result["labels"])
        for Y_facade, Y_staged in zip(clusterer.subspaces_, staged_result["subspaces"]):
            np.testing.assert_array_equal(Y_facade, Y_staged)

    def test_export_roundtrip_with_real_bases(self, glioma_vectors_subset, tmp_path):
        """flatten -> CSV -> unflatten не теряет точность на реальных базисах."""
        X = glioma_vectors_subset

        clusterer = FursovClusterer(n_subclasses=N_SUBCLASSES, freeze_basis_at=2)
        clusterer.fit(X)

        csv_path = export_clusterer_bases(
            clusterer, tmp_path / "8_glioma_subclasses_vectors.csv"
        )
        loaded = np.loadtxt(csv_path, delimiter=",")

        expected = flatten_subspace_bases(clusterer.subspaces_, expected_basis_size=2)
        assert loaded.shape == expected.shape
        np.testing.assert_allclose(loaded, expected, rtol=1e-12)

        restored = unflatten_subspace_bases(
            loaded, n_subclasses=N_SUBCLASSES, basis_size=BASIS_SIZE
        )
        for Y_orig, Y_restored in zip(clusterer.subspaces_, restored):
            np.testing.assert_allclose(Y_orig, Y_restored, rtol=1e-12)


class TestStagedPipelineAllRealClasses:
    """Тот же структурный контракт -- на всех трёх классах датасета."""

    @pytest.mark.parametrize("class_name", ["glioma", "meningioma", "pituitary"])
    def test_pipeline_runs_on_each_real_class(self, load_real_class_vectors, class_name):
        X = load_real_class_vectors(class_name, count=60)

        clusterer = FursovClusterer(n_subclasses=N_SUBCLASSES, freeze_basis_at=2)
        clusterer.fit(X)

        assert len(clusterer.subspaces_) == N_SUBCLASSES
        assert len(clusterer.labels_) == len(X)
        assert set(clusterer.labels_) == set(range(N_SUBCLASSES))
        sizes = clusterer.get_subclass_sizes()
        assert sizes.sum() == len(X)


class TestLegacyNB4PairVsCanonicalOnRealData:
    """use_canonical_pair=False (NB4 trio_list) не должен ломать staged pipeline."""

    def test_legacy_manual_pair_produces_valid_pipeline(self, glioma_vectors_subset):
        X = glioma_vectors_subset

        pipeline = NotebookStagedPipeline(
            n_subclasses=N_SUBCLASSES,
            use_canonical_pair=False,
            manual_pair_index=8,
        )
        result = pipeline.run_full_pipeline(X)

        assert len(result["center_indices"]) == N_SUBCLASSES
        assert len(set(result["center_indices"])) == N_SUBCLASSES
        assert len(result["labels"]) == len(X)
        assert result["flattened_bases"].shape == (
            N_SUBCLASSES * BASIS_SIZE,
            X.shape[1],
        )

    def test_legacy_pair_can_differ_from_canonical_pair(self, glioma_vectors_subset):
        """Документирует известное расхождение (план, раздел 1.7 / 2.1)."""
        X = glioma_vectors_subset

        pipeline_canon = NotebookStagedPipeline(
            n_subclasses=N_SUBCLASSES, use_canonical_pair=True
        )
        canon_centers = pipeline_canon.run_nb5_reference_centers(X)

        pipeline_legacy = NotebookStagedPipeline(
            n_subclasses=N_SUBCLASSES,
            use_canonical_pair=False,
            manual_pair_index=8,
        )
        legacy_centers = pipeline_legacy.run_nb5_reference_centers(X)

        # Оба валидны структурно, но не обязаны совпадать -- ручной выбор
        # пары (как в NB5) -- это другая начальная точка алгоритма A.2-A.3.
        assert len(canon_centers) == len(legacy_centers) == N_SUBCLASSES
        assert len(set(canon_centers)) == N_SUBCLASSES
        assert len(set(legacy_centers)) == N_SUBCLASSES
