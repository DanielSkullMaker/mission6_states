"""Unit tests для CSV IO (io/vectors.py)."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from subspace_conjugacy.io.vectors import (
    save_vectors_csv,
    load_vectors_csv,
    save_class_vectors,
    load_class_vectors,
    save_subclass_bases,
    load_subclass_bases,
    load_subclass_bases_as_list,
    save_initial_pair_indices,
    load_initial_pair_indices,
    save_center_indices,
    load_center_indices,
    save_subclass_pairs,
    load_subclass_pairs,
    save_pipeline_artifact,
    load_pretrained_classifier,
)
from subspace_conjugacy.config import DatasetConfig
from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier


class TestSaveLoadVectorsCSV:
    """Тесты базовых функций save/load CSV."""

    def test_save_and_load_roundtrip(self):
        """Сохранение и загрузка должны восстанавливать данные."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            X_original = np.random.randn(10, 64)

            save_vectors_csv(X_original, csv_path)
            X_loaded = load_vectors_csv(csv_path)

            assert np.allclose(X_original, X_loaded)
            assert X_loaded.shape == (10, 64)

    def test_load_nonexistent_file_raises_error(self):
        """Загрузка несуществующего файла должна вызывать FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_vectors_csv("/nonexistent/file.csv")

    def test_single_vector_roundtrip(self):
        """Один вектор должен загружаться как 2D массив (1, N)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "single.csv"
            x_original = np.random.randn(64)

            # Сохраняем как (1, 64)
            save_vectors_csv(x_original.reshape(1, -1), csv_path)
            x_loaded = load_vectors_csv(csv_path)

            assert x_loaded.ndim == 2
            assert x_loaded.shape == (1, 64)
            assert np.allclose(x_original, x_loaded[0])

    def test_large_matrix_roundtrip(self):
        """Проверка работы с большой матрицей (256×256 → 65536)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "large.csv"
            X_large = np.random.randn(100, 65536)

            save_vectors_csv(X_large, csv_path)
            X_loaded = load_vectors_csv(csv_path)

            assert X_loaded.shape == (100, 65536)
            assert np.allclose(X_large, X_loaded)

    def test_custom_delimiter(self):
        """Проверка работы с нестандартным разделителем."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "semicolon.csv"
            X = np.random.randn(5, 10)

            save_vectors_csv(X, csv_path, delimiter=";")
            X_loaded = load_vectors_csv(csv_path, delimiter=";")

            assert np.allclose(X, X_loaded)

    def test_integer_dtype(self):
        """Проверка сохранения и загрузки целочисленных данных."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "integers.csv"
            X_int = np.random.randint(0, 256, (10, 64), dtype=np.int32)

            save_vectors_csv(X_int, csv_path)
            X_loaded = load_vectors_csv(csv_path, dtype=np.int32)

            assert np.array_equal(X_int, X_loaded)
            assert X_loaded.dtype == np.int32

    def test_directory_creation(self):
        """save_vectors_csv должен создавать родительские директории."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "test.csv"
            X = np.random.randn(5, 10)

            save_vectors_csv(X, nested_path)

            assert nested_path.exists()
            X_loaded = load_vectors_csv(nested_path)
            assert np.allclose(X, X_loaded)


class TestDatasetConfigIntegration:
    """Тесты интеграции с DatasetConfig."""

    def test_save_and_load_class_vectors(self):
        """Сохранение и загрузка векторов класса через DatasetConfig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir)
            config.create_directories(stages=["vectors"])

            X_glioma = np.random.randn(100, 65536)

            # Сохранение
            path = save_class_vectors(X_glioma, "glioma", config, "horizontal")
            assert path.exists()
            assert path.name == "glioma_horizontal_vector.csv"

            # Загрузка
            X_loaded = load_class_vectors("glioma", config, "horizontal")
            assert np.allclose(X_glioma, X_loaded)

    def test_vertical_vector_type(self):
        """Проверка работы с вертикальной векторизацией."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir)
            config.create_directories(stages=["vectors"])

            X_vertical = np.random.randn(100, 65536)

            save_class_vectors(X_vertical, "meningioma", config, "vertical")
            X_loaded = load_class_vectors("meningioma", config, "vertical")

            assert np.allclose(X_vertical, X_loaded)

    def test_test_class_vectors(self):
        """Проверка работы с тестовой выборкой."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir)
            config.create_directories(stages=["vectors"])

            X_test = np.random.randn(75, 65536)

            save_class_vectors(X_test, "test", config)
            X_loaded = load_class_vectors("test", config)

            assert X_loaded.shape == (75, 65536)
            assert np.allclose(X_test, X_loaded)


class TestSubclassBases:
    """Тесты для сохранения/загрузки базисов подклассов."""

    def test_save_and_load_subclass_bases(self):
        """Сохранение и загрузка базисов подклассов."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir, n_subclasses=8, subclass_factor=2)
            config.create_directories(stages=["subclasses"])

            # 8 подклассов × 2 вектора = 16 строк
            bases = np.random.randn(16, 65536)

            # Сохранение
            path = save_subclass_bases(bases, "glioma", config)
            assert path.exists()
            assert path.name == "8_glioma_subclasses_vectors.csv"

            # Загрузка
            bases_loaded = load_subclass_bases("glioma", config)
            assert np.allclose(bases, bases_loaded)
            assert bases_loaded.shape == (16, 65536)

    def test_load_subclass_bases_as_list(self):
        """Загрузка базисов как списка подпространств Y_s."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir, n_subclasses=4, subclass_factor=2)
            config.create_directories(stages=["subclasses"])

            # 4 подкласса × 2 вектора = 8 строк
            bases = np.random.randn(8, 65536)
            save_subclass_bases(bases, "pituitary", config)

            # Загрузка как списка
            subspaces = load_subclass_bases_as_list("pituitary", config)

            assert len(subspaces) == 4
            assert all(Y.shape == (65536, 2) for Y in subspaces)

            # Проверка, что Y_s соответствует строкам [2*s : 2*s+2].T
            for s in range(4):
                Y_s_expected = bases[2*s : 2*(s+1)].T  # (2, 65536) → (65536, 2)
                assert np.allclose(subspaces[s], Y_s_expected)

    def test_different_subclass_factor(self):
        """Проверка работы с нестандартным subclass_factor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir, n_subclasses=4, subclass_factor=3)
            config.create_directories(stages=["subclasses"])

            # 4 подкласса × 3 вектора = 12 строк
            bases = np.random.randn(12, 65536)
            save_subclass_bases(bases, "meningioma", config)

            subspaces = load_subclass_bases_as_list("meningioma", config)

            assert len(subspaces) == 4
            assert all(Y.shape == (65536, 3) for Y in subspaces)


class TestInitialPairAndCenters:
    """Тесты IO для Legacy NB4-5 (начальная пара, центры подклассов)."""

    def test_save_and_load_initial_pair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir)
            config.create_directories(stages=["vectors"])

            path = save_initial_pair_indices((27, 29), "glioma", config)
            assert path.exists()
            assert path.name == "glioma_class_first_and_second_core_vectors.csv"

            pair = load_initial_pair_indices("glioma", config)
            assert pair == (27, 29)

    def test_load_initial_pair_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir)
            with pytest.raises(FileNotFoundError):
                load_initial_pair_indices("glioma", config)

    def test_save_and_load_center_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir, n_subclasses=8)
            config.create_directories(stages=["vectors"])

            centers = np.array([27, 29, 32, 5, 71, 12, 88, 3])
            path = save_center_indices(centers, "glioma", config)
            assert path.name == "glioma_class_core_vectors.csv"

            loaded = load_center_indices("glioma", config)
            np.testing.assert_array_equal(loaded, centers)

    def test_center_indices_roundtrip_with_algorithms(self):
        """CSV должен быть совместим с ReferenceCenterBuilder.center_indices_."""
        from subspace_conjugacy.algorithms.global_pair import GlobalMinCosinePairFinder
        from subspace_conjugacy.algorithms.reference_centers import ReferenceCenterBuilder

        np.random.seed(0)
        X = np.random.randn(40, 32)

        finder = GlobalMinCosinePairFinder()
        finder.fit(X)
        builder = ReferenceCenterBuilder(n_subclasses=4)
        builder.fit(X, finder.pair_indices_)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir, n_subclasses=4)
            config.create_directories(stages=["vectors"])

            save_initial_pair_indices(finder.pair_indices_, "glioma", config)
            save_center_indices(builder.center_indices_, "glioma", config)

            assert load_initial_pair_indices("glioma", config) == finder.pair_indices_
            np.testing.assert_array_equal(
                load_center_indices("glioma", config), builder.center_indices_
            )


class TestSubclassPairsIO:
    """Тесты IO для Legacy NB6 (пары подклассов, теория B.1)."""

    def test_save_and_load_subclass_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir, n_subclasses=3)
            config.create_directories(stages=["vectors"])

            pairs = np.array([[1, 9], [4, 17], [8, 2]])
            path = save_subclass_pairs(pairs, "meningioma", config)
            assert path.name == "meningioma_new_classes.csv"

            loaded = load_subclass_pairs("meningioma", config)
            np.testing.assert_array_equal(loaded, pairs)
            assert loaded.shape == (3, 2)

    def test_loaded_pairs_feed_directly_into_cluster_growth(self):
        """load_subclass_pairs() -> ConjugacyClusterGrowth.fit() без адаптации."""
        from subspace_conjugacy.algorithms.subclass_seed import CosineSecondVectorAttacher
        from subspace_conjugacy.algorithms.subclass_growth import ConjugacyClusterGrowth

        np.random.seed(1)
        X = np.random.randn(30, 16)
        centers = np.array([0, 5, 10])

        attacher = CosineSecondVectorAttacher()
        attacher.fit(X, centers)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir, n_subclasses=3)
            config.create_directories(stages=["vectors"])
            save_subclass_pairs(attacher.pairs_, "glioma", config)

            pairs_from_csv = load_subclass_pairs("glioma", config)

        growth = ConjugacyClusterGrowth(freeze_basis_at=2)
        growth.fit(X, pairs_from_csv)

        assert len(growth.labels_) == len(X)
        assert len(growth.subspace_bases_) == 3


class TestPipelineArtifact:
    """Тесты save_pipeline_artifact / load_pretrained_classifier."""

    def test_roundtrip_predictions_match(self):
        np.random.seed(7)
        X = np.vstack([
            np.random.randn(20, 32) + 0.0,
            np.random.randn(20, 32) + 10.0,
        ])
        y = np.array(["glioma"] * 20 + ["meningioma"] * 20)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir, n_subclasses=4, subclass_factor=2)
            config.create_directories()

            clf = SubspaceConjugacyClassifier(n_subclasses=4, freeze_basis_at=2)
            clf.fit(X, y)

            paths = save_pipeline_artifact(clf, config)
            assert "metadata" in paths
            assert paths["metadata"].exists()
            assert set(paths.keys()) == {"metadata", "glioma", "meningioma"}

            clf_loaded = load_pretrained_classifier(config)

            assert clf_loaded.is_fitted_
            assert set(clf_loaded.classes_) == set(clf.classes_)
            np.testing.assert_array_equal(clf.predict(X), clf_loaded.predict(X))
            np.testing.assert_allclose(
                clf.predict_r_matrix(X), clf_loaded.predict_r_matrix(X)
            )

    def test_save_unfitted_classifier_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir)
            clf = SubspaceConjugacyClassifier()
            with pytest.raises(RuntimeError):
                save_pipeline_artifact(clf, config)

    def test_load_without_saved_artifact_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DatasetConfig(root=tmpdir)
            with pytest.raises(FileNotFoundError):
                load_pretrained_classifier(config)


@pytest.mark.notebook_parity
class TestNotebookParity:
    """Тесты совместимости с CSV из ноутбуков (требуется датасет)."""

    def test_load_glioma_horizontal_vectors(self, skip_if_no_dataset, default_dataset_config):
        """Загрузка glioma_horizontal_vector.csv из NB3."""
        skip_if_no_dataset

        X_glioma = load_class_vectors("glioma", default_dataset_config, "horizontal")

        assert X_glioma.shape[0] == 100, "Должно быть 100 изображений glioma"
        assert X_glioma.shape[1] == 65536, "Размерность вектора должна быть 65536"

    def test_load_test_horizontal_vectors(self, skip_if_no_dataset, default_dataset_config):
        """Загрузка test_horizontal_vector.csv из NB3."""
        skip_if_no_dataset

        X_test = load_class_vectors("test", default_dataset_config, "horizontal")

        assert X_test.shape[0] == 75, "Должно быть 75 тестовых изображений"
        assert X_test.shape[1] == 65536

    def test_load_glioma_subclass_bases(self, skip_if_no_dataset, default_dataset_config):
        """Загрузка 8_glioma_subclasses_vectors.csv из NB6-7."""
        skip_if_no_dataset

        bases = load_subclass_bases("glioma", default_dataset_config)

        # 8 подклассов × 2 вектора = 16 строк
        assert bases.shape == (16, 65536)

        # Проверка как списка подпространств
        subspaces = load_subclass_bases_as_list("glioma", default_dataset_config)
        assert len(subspaces) == 8
        assert all(Y.shape == (65536, 2) for Y in subspaces)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
