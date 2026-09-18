"""Smoke tests для проверки корректности установки и импортов пакета."""

import pytest
import numpy as np


def test_import_package():
    """Проверяет, что пакет subspace_conjugacy импортируется."""
    import subspace_conjugacy
    assert subspace_conjugacy.__version__ == "0.1.0"


def test_import_core_metrics():
    """Проверяет импорт метрик из core."""
    from subspace_conjugacy import conjugate_criterion, cosine_similarity_matrix
    assert callable(conjugate_criterion)
    assert callable(cosine_similarity_matrix)


def test_import_models():
    """Проверяет импорт моделей."""
    from subspace_conjugacy import SubspaceClusterer, SubspaceConjugacyClassifier
    assert SubspaceClusterer is not None
    assert SubspaceConjugacyClassifier is not None


def test_import_config():
    """Проверяет импорт конфигурации датасета."""
    from subspace_conjugacy.config import DatasetConfig
    config = DatasetConfig(root="data")
    assert config.n_subclasses == 8
    assert config.n_features == 65536
    assert len(config.classes) == 3


def test_conjugate_criterion_basic():
    """Базовый тест вычисления показателя сопряжённости."""
    from subspace_conjugacy import conjugate_criterion

    np.random.seed(42)
    x = np.random.randn(64)
    Y = np.random.randn(64, 2)

    r = conjugate_criterion(x, Y)

    assert isinstance(r, float)
    assert 0.0 <= r <= 1.0


def test_cosine_similarity_basic():
    """Базовый тест вычисления косинусного сходства."""
    from subspace_conjugacy import cosine_similarity_matrix

    np.random.seed(42)
    X = np.random.randn(5, 32)

    sim = cosine_similarity_matrix(X)

    assert sim.shape == (5, 5)
    assert np.allclose(np.diag(sim), 1.0), "Диагональ должна содержать 1.0"


def test_dataset_config_paths():
    """Проверяет генерацию путей в DatasetConfig."""
    from subspace_conjugacy.config import DatasetConfig

    config = DatasetConfig(root="/tmp/test_data")

    # Проверка путей для glioma (используем относительные части пути)
    assert config.paths["glioma"]["raw"].name == "glioma_raw"
    assert config.paths["glioma"]["vectors"].parts[-2:] == ("5_all_vectors", "glioma")

    # Проверка CSV путей
    csv_path = config.get_vector_csv_path("glioma", "horizontal")
    assert csv_path.name == "glioma_horizontal_vector.csv"

    # Проверка базисов подклассов
    subclass_path = config.get_subclass_bases_path("meningioma")
    assert subclass_path.name == "8_meningioma_subclasses_vectors.csv"


def test_validation_check_array_x():
    """Проверяет валидацию массива X."""
    from subspace_conjugacy.utils.validation import check_array_X

    # Корректный массив
    X_valid = [[1.0, 2.0], [3.0, 4.0]]
    X_arr = check_array_X(X_valid)
    assert X_arr.shape == (2, 2)
    assert X_arr.dtype == np.float64

    # NaN должен вызвать ошибку
    X_nan = [[1.0, np.nan], [3.0, 4.0]]
    with pytest.raises(ValueError, match="NaN"):
        check_array_X(X_nan)


def test_clusterer_initialization():
    """Проверяет инициализацию кластеризатора."""
    from subspace_conjugacy import SubspaceClusterer

    clusterer = SubspaceClusterer(n_subclasses=4, reg_param=1e-8)
    assert clusterer.n_subclasses == 4
    assert clusterer.reg_param == 1e-8


def test_classifier_initialization():
    """Проверяет инициализацию классификатора."""
    from subspace_conjugacy import SubspaceConjugacyClassifier

    clf = SubspaceConjugacyClassifier(n_subclasses=8, reg_param=1e-8)
    assert clf.n_subclasses == 8
    assert clf.reg_param == 1e-8
    assert clf.is_fitted_ is False


@pytest.mark.theory
def test_conjugate_criterion_batch(small_random_vectors):
    """Тест батчевого вычисления R(X, Y) на синтетических данных."""
    from subspace_conjugacy import conjugate_criterion

    X = small_random_vectors  # (20, 64)
    Y = small_random_vectors[:4].T  # (64, 4)

    R = conjugate_criterion(X, Y)

    assert R.shape == (20,)
    assert np.all((R >= 0.0) & (R <= 1.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
