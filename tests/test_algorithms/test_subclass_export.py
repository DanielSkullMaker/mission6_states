"""Theory tests для algorithms/subclass_export.py.

Проверяет корректность экспорта базисов подклассов в формат CSV.
"""

import pytest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from subspace_conjugacy.algorithms.subclass_export import (
    flatten_subspace_bases,
    unflatten_subspace_bases,
    export_clusterer_bases,
    export_all_classes,
)
from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer


@pytest.mark.theory
class TestFlattenUnflattenBases:
    """Тесты flatten/unflatten преобразований."""

    def test_flatten_basic(self):
        """flatten должен преобразовывать список (N,k) → (S*k, N)."""
        Y_0 = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
        Y_1 = np.array([[7, 8], [9, 10], [11, 12]])  # (3, 2)
        subspaces = [Y_0, Y_1]

        flattened = flatten_subspace_bases(subspaces)

        assert flattened.shape == (4, 3)  # 2*2 строк, 3 признака
        # Первые 2 строки = Y_0.T
        assert np.allclose(flattened[0:2], Y_0.T)
        assert np.allclose(flattened[2:4], Y_1.T)

    def test_unflatten_basic(self):
        """unflatten должен восстанавливать список базисов."""
        flattened = np.array([
            [1, 3, 5],
            [2, 4, 6],
            [7, 9, 11],
            [8, 10, 12],
        ])  # (4, 3)

        subspaces = unflatten_subspace_bases(flattened, n_subclasses=2, basis_size=2)

        assert len(subspaces) == 2
        assert subspaces[0].shape == (3, 2)
        assert subspaces[1].shape == (3, 2)

        expected_Y_0 = np.array([[1, 2], [3, 4], [5, 6]])
        expected_Y_1 = np.array([[7, 8], [9, 10], [11, 12]])
        assert np.allclose(subspaces[0], expected_Y_0)
        assert np.allclose(subspaces[1], expected_Y_1)

    def test_roundtrip_consistency(self):
        """flatten → unflatten должен восстанавливать оригинал."""
        np.random.seed(42)
        subspaces = [np.random.randn(256, 2) for _ in range(8)]

        flattened = flatten_subspace_bases(subspaces, expected_basis_size=2)
        restored = unflatten_subspace_bases(flattened, n_subclasses=8, basis_size=2)

        assert len(restored) == 8
        for orig, rest in zip(subspaces, restored):
            assert np.allclose(orig, rest)

    def test_flatten_with_expected_basis_size(self):
        """expected_basis_size должен валидировать размер базисов."""
        Y_0 = np.random.randn(100, 2)
        Y_1 = np.random.randn(100, 2)
        subspaces = [Y_0, Y_1]

        # Корректный размер
        flattened = flatten_subspace_bases(subspaces, expected_basis_size=2)
        assert flattened.shape == (4, 100)

    def test_flatten_raises_on_mismatched_basis_size(self):
        """Должна быть ошибка при несовпадении размера базисов."""
        Y_0 = np.random.randn(100, 2)
        Y_1 = np.random.randn(100, 3)  # ← неверный размер
        subspaces = [Y_0, Y_1]

        with pytest.raises(ValueError, match="Ожидалось 2 векторов в каждом базисе"):
            flatten_subspace_bases(subspaces, expected_basis_size=2)

    def test_flatten_raises_on_mismatched_features(self):
        """Должна быть ошибка при разной размерности признаков."""
        Y_0 = np.random.randn(256, 2)
        Y_1 = np.random.randn(128, 2)  # ← неверная размерность N
        subspaces = [Y_0, Y_1]

        with pytest.raises(ValueError, match="одинаковую размерность признаков"):
            flatten_subspace_bases(subspaces)

    def test_flatten_raises_on_empty_list(self):
        """Должна быть ошибка при пустом списке."""
        with pytest.raises(ValueError, match="Список подпространств пуст"):
            flatten_subspace_bases([])

    def test_unflatten_raises_on_incorrect_rows(self):
        """Должна быть ошибка при неверном количестве строк."""
        flattened = np.random.randn(15, 256)  # 15 строк, не кратно 8*2=16

        with pytest.raises(ValueError, match="Ожидалось 16 строк"):
            unflatten_subspace_bases(flattened, n_subclasses=8, basis_size=2)


@pytest.mark.theory
class TestExportClustererBases:
    """Тесты экспорта базисов из FursovClusterer."""

    def test_export_from_fitted_clusterer(self):
        """export_clusterer_bases должен сохранять CSV из FursovClusterer."""
        np.random.seed(42)
        X = np.random.randn(60, 128)

        clusterer = FursovClusterer(n_subclasses=6, freeze_basis_at=2)
        clusterer.fit(X)

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_bases.csv"
            saved_path = export_clusterer_bases(clusterer, output_path)

            assert saved_path.exists()
            assert saved_path == output_path

            # Проверка содержимого
            loaded = np.loadtxt(saved_path, delimiter=",")
            assert loaded.shape == (12, 128)  # 6*2 строк, 128 признаков

    def test_export_raises_on_unfitted_clusterer(self):
        """Должна быть ошибка при попытке экспорта до fit()."""
        clusterer = FursovClusterer(n_subclasses=4)

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            with pytest.raises(RuntimeError, match="не обучен"):
                export_clusterer_bases(clusterer, output_path)

    def test_export_creates_parent_directories(self):
        """export_clusterer_bases должен создавать родительские директории."""
        np.random.seed(42)
        X = np.random.randn(40, 64)
        clusterer = FursovClusterer(n_subclasses=4, freeze_basis_at=2).fit(X)

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "bases.csv"
            saved_path = export_clusterer_bases(clusterer, output_path)

            assert saved_path.exists()
            assert saved_path.parent.exists()

    def test_exported_csv_is_readable(self):
        """Экспортированный CSV должен корректно загружаться."""
        np.random.seed(42)
        X = np.random.randn(50, 100)
        clusterer = FursovClusterer(n_subclasses=5, freeze_basis_at=2).fit(X)

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            export_clusterer_bases(clusterer, output_path)

            # Загрузка через numpy
            loaded = np.loadtxt(output_path, delimiter=",")
            assert loaded.shape == (10, 100)  # 5*2 строк

            # Загрузка через unflatten
            restored = unflatten_subspace_bases(loaded, n_subclasses=5, basis_size=2)
            assert len(restored) == 5
            assert restored[0].shape == (100, 2)


@pytest.mark.theory
class TestExportAllClasses:
    """Тесты массового экспорта для всех классов."""

    def test_export_all_classes_basic(self):
        """export_all_classes должен сохранять CSV для всех классов."""
        np.random.seed(42)

        # Обучаем кластеризаторы для каждого класса
        clusterers = {
            "glioma": FursovClusterer(n_subclasses=4, freeze_basis_at=2).fit(
                np.random.randn(50, 64)
            ),
            "meningioma": FursovClusterer(n_subclasses=4, freeze_basis_at=2).fit(
                np.random.randn(50, 64)
            ),
            "pituitary": FursovClusterer(n_subclasses=4, freeze_basis_at=2).fit(
                np.random.randn(50, 64)
            ),
        }

        with TemporaryDirectory() as tmpdir:
            paths = export_all_classes(clusterers, tmpdir, n_subclasses=4)

            assert len(paths) == 3
            assert "glioma" in paths
            assert "meningioma" in paths
            assert "pituitary" in paths

            # Проверка файлов
            for class_name, path in paths.items():
                assert path.exists()
                assert path.name == f"4_{class_name}_subclasses_vectors.csv"

                # Проверка содержимого
                loaded = np.loadtxt(path, delimiter=",")
                assert loaded.shape == (8, 64)  # 4*2 строк

    def test_export_all_classes_with_8_subclasses(self):
        """Должен работать с n_subclasses=8 (стандартное значение)."""
        np.random.seed(42)
        clusterers = {
            "test_class": FursovClusterer(n_subclasses=8, freeze_basis_at=2).fit(
                np.random.randn(80, 128)
            ),
        }

        with TemporaryDirectory() as tmpdir:
            paths = export_all_classes(clusterers, tmpdir, n_subclasses=8)

            path = paths["test_class"]
            assert path.name == "8_test_class_subclasses_vectors.csv"

            loaded = np.loadtxt(path, delimiter=",")
            assert loaded.shape == (16, 128)  # 8*2 строк


@pytest.mark.theory
class TestIntegrationWithFursovClusterer:
    """Интеграционные тесты: FursovClusterer → export → load → classifier."""

    def test_full_pipeline_export_and_load(self):
        """Полный цикл: fit → export → load → unflatten."""
        np.random.seed(42)
        X = np.random.randn(100, 256)

        # 1. Кластеризация
        clusterer = FursovClusterer(n_subclasses=8, freeze_basis_at=2)
        clusterer.fit(X)

        original_subspaces = clusterer.subspaces_

        # 2. Экспорт
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "8_test_subclasses_vectors.csv"
            export_clusterer_bases(clusterer, output_path)

            # 3. Загрузка
            loaded = np.loadtxt(output_path, delimiter=",")

            # 4. Unflatten
            restored_subspaces = unflatten_subspace_bases(
                loaded, n_subclasses=8, basis_size=2
            )

            # 5. Проверка идентичности
            assert len(restored_subspaces) == len(original_subspaces)
            for orig, rest in zip(original_subspaces, restored_subspaces):
                assert np.allclose(orig, rest, atol=1e-10)

    def test_bases_ready_for_classifier(self):
        """Экспортированные базисы должны быть готовы для classifier."""
        np.random.seed(42)
        X = np.random.randn(80, 200)

        clusterer = FursovClusterer(n_subclasses=6, freeze_basis_at=2).fit(X)

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "bases.csv"
            export_clusterer_bases(clusterer, output_path)

            # Загрузка для classifier
            loaded = np.loadtxt(output_path, delimiter=",")
            bases_list = unflatten_subspace_bases(loaded, n_subclasses=6, basis_size=2)

            # Проверка: каждый базис (N, 2)
            for Y in bases_list:
                assert Y.shape == (200, 2)

            # Можно использовать в conjugate_criterion
            from subspace_conjugacy.core.metrics import conjugate_criterion

            x_test = np.random.randn(200)
            for Y in bases_list:
                R = conjugate_criterion(x_test, Y)
                assert 0.0 <= R <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
