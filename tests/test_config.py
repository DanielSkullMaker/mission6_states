"""Тесты config/dataset.py — DatasetConfig (пути к датасету и гиперпараметры).

Раньше часть проверок жила только в блоке ``if __name__ == "__main__"``
самого модуля (Фаза 0 плана рефакторинга требует перенести такие блоки в
tests/). DatasetConfig и так использовался во многих fixtures/тестах, но
не имел собственного файла с прямыми проверками своего публичного API.
"""

from pathlib import Path

from subspace_conjugacy.config.dataset import DatasetConfig


class TestDatasetConfigPaths:
    def test_default_classes_and_root_is_resolved(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        assert config.classes == ["glioma", "meningioma", "pituitary"]
        assert config.root == tmp_path.resolve()

    def test_paths_built_for_every_class_and_test(self, tmp_path):
        config = DatasetConfig(root=tmp_path)

        for cls in ["glioma", "meningioma", "pituitary", "test"]:
            assert cls in config.paths
            assert "raw" in config.paths[cls]
            assert "resized" in config.paths[cls]
            assert "centered" in config.paths[cls]
            assert "vectors" in config.paths[cls]

        assert config.paths["glioma"]["raw"] == tmp_path.resolve() / "glioma_raw"
        assert config.paths["test"]["raw"] == tmp_path.resolve() / "test_raw"

    def test_custom_classes_list(self, tmp_path):
        config = DatasetConfig(root=tmp_path, classes=["glioma"])
        assert config.classes == ["glioma"]
        assert "glioma" in config.paths
        assert "meningioma" not in config.paths
        # "test" всегда присутствует независимо от classes.
        assert "test" in config.paths


class TestDatasetConfigCsvPaths:
    def test_get_vector_csv_path(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        path = config.get_vector_csv_path("glioma", "horizontal")
        assert path.name == "glioma_horizontal_vector.csv"
        assert path.parent == config.paths["glioma"]["vectors"]

    def test_get_vector_csv_path_vertical(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        path = config.get_vector_csv_path("test", "vertical")
        assert path.name == "test_vertical_vector.csv"

    def test_get_initial_pair_path(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        path = config.get_initial_pair_path("meningioma")
        assert path.name == "meningioma_class_first_and_second_core_vectors.csv"

    def test_get_center_vectors_path(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        path = config.get_center_vectors_path("pituitary")
        assert path.name == "pituitary_class_core_vectors.csv"

    def test_get_subclass_pairs_path(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        path = config.get_subclass_pairs_path("glioma")
        assert path.name == "glioma_new_classes.csv"

    def test_get_subclass_bases_path_uses_n_subclasses(self, tmp_path):
        config = DatasetConfig(root=tmp_path, n_subclasses=12)
        path = config.get_subclass_bases_path("glioma")
        assert path.name == "12_glioma_subclasses_vectors.csv"

    def test_get_pipeline_metadata_path(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        path = config.get_pipeline_metadata_path()
        assert path == config.root / "pipeline_metadata.json"


class TestDatasetConfigImagePaths:
    def test_default_count_for_class(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        paths = config.get_image_paths("glioma", "centered")
        assert len(paths) == 100
        assert paths[0].name == "glioma1.png"

    def test_default_count_for_test_uses_test_samples_per_class(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        paths = config.get_image_paths("test", "centered")
        # 25 (test_samples_per_class) * 3 (len(classes)) = 75.
        assert len(paths) == 75

    def test_explicit_count_overrides_default(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        paths = config.get_image_paths("glioma", "resized", count=5)
        assert len(paths) == 5

    def test_raw_stage_uses_jpg_extension_for_classes(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        paths = config.get_image_paths("glioma", "raw", count=2)
        assert all(p.suffix == ".jpg" for p in paths)

    def test_raw_stage_uses_png_extension_for_test(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        paths = config.get_image_paths("test", "raw", count=2)
        assert all(p.suffix == ".png" for p in paths)

    def test_non_raw_stage_always_uses_png(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        paths = config.get_image_paths("glioma", "centered", count=2)
        assert all(p.suffix == ".png" for p in paths)


class TestDatasetConfigDirectories:
    def test_create_directories_creates_all_stages_by_default(self, tmp_path):
        config = DatasetConfig(root=tmp_path, classes=["glioma"])
        config.create_directories()

        assert config.paths["glioma"]["raw"].is_dir()
        assert config.paths["glioma"]["resized"].is_dir()
        assert config.paths["glioma"]["centered"].is_dir()
        assert config.paths["test"]["raw"].is_dir()

    def test_create_directories_respects_explicit_stages(self, tmp_path):
        config = DatasetConfig(root=tmp_path, classes=["glioma"])
        config.create_directories(stages=["raw"])

        assert config.paths["glioma"]["raw"].is_dir()
        assert not config.paths["glioma"]["resized"].exists()


class TestDatasetConfigProperties:
    def test_n_features_matches_image_size(self, tmp_path):
        config = DatasetConfig(root=tmp_path, image_size=(256, 256))
        assert config.n_features == 65536

    def test_total_subclasses_multiplies_classes_by_n_subclasses(self, tmp_path):
        config = DatasetConfig(root=tmp_path, n_subclasses=8)
        assert config.total_subclasses == 24  # 3 класса * 8

    def test_repr_contains_key_fields(self, tmp_path):
        config = DatasetConfig(root=tmp_path)
        text = repr(config)
        assert "DatasetConfig" in text
        assert "n_subclasses" in text
        assert str(config.n_features) in text
