"""Тесты features/extraction.py — высокоуровневое извлечение признаков.

Раньше эти проверки жили только в блоке ``if __name__ == "__main__"`` самого
модуля (Фаза 0 плана рефакторинга требует перенести такие блоки в tests/) и
были ограничены выводом структуры API без реального датасета. Здесь
extract_class_vectors/extract_all_classes/extract_training_data проверяются
на синтетических PNG в tmp_path — не требует ни реального MRI-архива, ни
датасета в data/.
"""

import cv2
import numpy as np
import pytest

from subspace_conjugacy.config import DatasetConfig
from subspace_conjugacy.features.extraction import (
    extract_all_classes,
    extract_class_vectors,
    extract_training_data,
)

IMAGE_SIZE = 32  # маленький размер -> быстрые тесты


def _write_synthetic_images(config: DatasetConfig, class_name: str, stage: str, count: int) -> None:
    paths = config.get_image_paths(class_name, stage, count=count)
    for i, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        img = np.full((IMAGE_SIZE, IMAGE_SIZE), fill_value=i, dtype=np.uint8)
        cv2.imwrite(str(path), img)


@pytest.fixture
def small_config(tmp_path) -> DatasetConfig:
    return DatasetConfig(
        root=tmp_path,
        classes=["glioma", "meningioma", "pituitary"],
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
    )


class TestExtractClassVectors:
    def test_returns_matrix_with_expected_shape(self, small_config):
        _write_synthetic_images(small_config, "glioma", "centered", count=5)

        X = extract_class_vectors(small_config, "glioma", stage="centered", count=5)

        assert X.shape == (5, IMAGE_SIZE * IMAGE_SIZE)

    def test_vertical_method_produces_same_shape(self, small_config):
        _write_synthetic_images(small_config, "glioma", "centered", count=3)

        X = extract_class_vectors(
            small_config, "glioma", stage="centered", method="vertical", count=3
        )

        assert X.shape == (3, IMAGE_SIZE * IMAGE_SIZE)


class TestExtractAllClasses:
    """extract_all_classes/extract_training_data не принимают count и всегда
    запрашивают у DatasetConfig стандартное число изображений (100 на класс,
    75 для test) — поэтому фикстуры здесь создают ровно столько же PNG."""

    def test_extracts_every_configured_class(self, small_config):
        for cls in small_config.classes:
            _write_synthetic_images(small_config, cls, "centered", count=100)

        result = extract_all_classes(small_config, stage="centered")

        assert set(result.keys()) == set(small_config.classes)
        for cls in small_config.classes:
            assert result[cls].shape == (100, IMAGE_SIZE * IMAGE_SIZE)

    def test_include_test_adds_test_class(self, small_config):
        for cls in small_config.classes:
            _write_synthetic_images(small_config, cls, "centered", count=100)
        _write_synthetic_images(small_config, "test", "centered", count=75)

        result = extract_all_classes(small_config, stage="centered", include_test=True)

        assert "test" in result
        assert result["test"].shape == (75, IMAGE_SIZE * IMAGE_SIZE)

    def test_does_not_include_test_by_default(self, small_config):
        for cls in small_config.classes:
            _write_synthetic_images(small_config, cls, "centered", count=100)

        result = extract_all_classes(small_config, stage="centered")

        assert "test" not in result


class TestExtractTrainingData:
    def test_returns_stacked_x_and_matching_labels(self, small_config):
        for cls in small_config.classes:
            _write_synthetic_images(small_config, cls, "centered", count=100)

        X, y = extract_training_data(small_config, stage="centered")

        assert X.shape == (300, IMAGE_SIZE * IMAGE_SIZE)  # 3 класса * 100 изображений
        assert y.shape == (300,)
        assert set(np.unique(y)) == set(small_config.classes)

    def test_label_counts_match_per_class_image_counts(self, small_config):
        _write_synthetic_images(small_config, "glioma", "centered", count=100)
        _write_synthetic_images(small_config, "meningioma", "centered", count=100)
        _write_synthetic_images(small_config, "pituitary", "centered", count=100)

        _, y = extract_training_data(small_config, stage="centered")

        assert int(np.sum(y == "glioma")) == 100
        assert int(np.sum(y == "meningioma")) == 100
        assert int(np.sum(y == "pituitary")) == 100
