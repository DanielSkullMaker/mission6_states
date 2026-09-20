"""Тесты pipeline/ (Фаза 6: FursovPipeline + STAGE_REGISTRY)."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from subspace_conjugacy.config import DatasetConfig
from subspace_conjugacy.pipeline import STAGE_REGISTRY, FursovPipeline

ARCHIVE_ROOT = Path(
    os.environ.get("BRAIN_MRI_ARCHIVE_ROOT", r"C:\Users\nazar\Downloads\archive")
)


def _write_synthetic_centered_images(
    config: DatasetConfig, class_name: str, count: int, seed: int = 0
) -> None:
    """Пишет count синтетических PNG прямо в стадию 'centered' (без raw/resize)."""
    rng = np.random.default_rng(seed)
    centered_dir = config.paths[class_name]["centered"]
    centered_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, count + 1):
        img = rng.integers(0, 256, size=(*config.image_size, 3), dtype=np.uint8)
        cv2.imwrite(str(centered_dir / f"{class_name}{i}.png"), img)


@pytest.fixture
def small_config(tmp_path) -> DatasetConfig:
    config = DatasetConfig(
        root=tmp_path,
        classes=["glioma", "meningioma", "pituitary"],
        n_subclasses=4,
        subclass_factor=2,
        test_samples_per_class=5,
    )
    for i, cls in enumerate(config.classes):
        _write_synthetic_centered_images(config, cls, count=20, seed=i)
    return config


class TestStageRegistry:
    def test_all_planned_stages_registered(self):
        expected = {
            "resize", "center", "binarize", "vectorize", "global_pair",
            "reference_centers", "subclass_seed", "subclass_growth", "cluster",
            "export_subspaces", "classify", "legacy_notebook",
        }
        assert expected == set(STAGE_REGISTRY.keys())


class TestRunStageDispatch:
    def test_unknown_stage_raises(self, small_config):
        pipeline = FursovPipeline(small_config)
        with pytest.raises(ValueError):
            pipeline.run_stage("not_a_real_stage")

    def test_config_auto_injected(self, small_config):
        pipeline = FursovPipeline(small_config)
        X = pipeline.run_stage("vectorize", class_name="glioma", save=False)
        assert X.shape == (20, small_config.n_features)

    def test_explicit_config_not_overridden(self, small_config, tmp_path):
        other_config = DatasetConfig(root=tmp_path / "other", classes=["glioma"])
        pipeline = FursovPipeline(small_config)
        # Явно переданный config должен использоваться вместо self.config
        with pytest.raises(Exception):
            pipeline.run_stage(
                "vectorize", class_name="glioma", config=other_config, save=False
            )

    def test_canonical_algorithm_stages_via_run_stage(self, small_config):
        pipeline = FursovPipeline(small_config)
        X = pipeline.run_stage("vectorize", class_name="glioma", save=False)

        finder = pipeline.run_stage("global_pair", X=X)
        builder = pipeline.run_stage(
            "reference_centers", X=X, initial_pair=finder.pair_indices_, n_subclasses=4
        )
        attacher = pipeline.run_stage(
            "subclass_seed", X=X, center_indices=builder.center_indices_
        )
        growth = pipeline.run_stage(
            "subclass_growth", X=X, pairs=attacher.pairs_, freeze_basis_at=2
        )

        assert len(growth.subspace_bases_) == 4
        assert len(growth.labels_) == 20

    def test_cluster_stage_matches_fursov_clusterer(self, small_config):
        from subspace_conjugacy.algorithms.fursov_clusterer import FursovClusterer

        pipeline = FursovPipeline(small_config)
        X = pipeline.run_stage("vectorize", class_name="glioma", save=False)

        via_stage = pipeline.run_stage(
            "cluster", X=X, n_subclasses=4, freeze_basis_at=2
        )
        reference = FursovClusterer(n_subclasses=4, freeze_basis_at=2).fit(X)

        for Y_stage, Y_ref in zip(via_stage.subspaces_, reference.subspaces_):
            np.testing.assert_array_equal(Y_stage, Y_ref)


class TestRunStageBinarize:
    """stage "binarize" — Otsu-бинаризация (статья, находка №4), только для
    этапа определения проекции, не для обычной классификации типа опухоли."""

    def test_binarizes_centered_images(self, small_config):
        pipeline = FursovPipeline(small_config)
        outputs = pipeline.run_stage("binarize", class_name="glioma")

        assert len(outputs) == 20  # small_config пишет 20 изображений/класс
        for path in outputs:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            assert set(np.unique(img)) <= {0, 255}

    def test_writes_to_binarized_path_from_config(self, small_config):
        pipeline = FursovPipeline(small_config)
        pipeline.run_stage("binarize", class_name="glioma")

        binarized_dir = small_config.paths["glioma"]["binarized"]
        assert binarized_dir.is_dir()
        assert len(list(binarized_dir.glob("*.png"))) == 20


class TestRunClass:
    def test_run_class_populates_clusterers_and_exports_csv(self, small_config):
        pipeline = FursovPipeline(small_config)
        clusterer = pipeline.run_class("glioma", n_subclasses=4, freeze_basis_at=2)

        assert pipeline.clusterers_["glioma"] is clusterer
        assert len(clusterer.subspaces_) == 4

        csv_path = small_config.get_subclass_bases_path("glioma")
        assert csv_path.exists()
        loaded = np.loadtxt(csv_path, delimiter=",")
        assert loaded.shape == (4 * 2, small_config.n_features)

    def test_run_class_saves_vectors_csv_when_requested(self, small_config):
        pipeline = FursovPipeline(small_config)
        pipeline.run_class("glioma", n_subclasses=4, save_vectors=True)

        vectors_path = small_config.get_vector_csv_path("glioma", "horizontal")
        assert vectors_path.exists()

    def test_run_class_skips_export_when_disabled(self, small_config):
        pipeline = FursovPipeline(small_config)
        pipeline.run_class("glioma", n_subclasses=4, export=False)

        assert not small_config.get_subclass_bases_path("glioma").exists()

    def test_run_class_uses_config_n_subclasses_by_default(self, small_config):
        pipeline = FursovPipeline(small_config)
        clusterer = pipeline.run_class("glioma")
        assert len(clusterer.subspaces_) == small_config.n_subclasses  # 4


class TestRunAllClasses:
    def test_trains_every_class(self, small_config):
        pipeline = FursovPipeline(small_config)
        result = pipeline.run_all_classes(n_subclasses=4, freeze_basis_at=2)

        assert set(result.keys()) == set(small_config.classes)
        assert result is pipeline.clusterers_
        for cls in small_config.classes:
            assert len(pipeline.clusterers_[cls].subspaces_) == 4

    def test_n_subclasses_none_falls_back_to_config_default(self, small_config):
        """small_config.n_subclasses=4 (см. fixture) должен использоваться,
        если run_all_classes() вызван без n_subclasses вовсе."""
        pipeline = FursovPipeline(small_config)
        pipeline.run_all_classes(freeze_basis_at=2)

        for cls in small_config.classes:
            assert len(pipeline.clusterers_[cls].subspaces_) == small_config.n_subclasses

    def test_dict_n_subclasses_trains_each_class_independently(self, small_config):
        per_class = {"glioma": 2, "meningioma": 3, "pituitary": 4}
        pipeline = FursovPipeline(small_config)
        result = pipeline.run_all_classes(n_subclasses=per_class, freeze_basis_at=2)

        for cls, expected in per_class.items():
            assert len(result[cls].subspaces_) == expected

    def test_dict_n_subclasses_missing_class_raises(self, small_config):
        pipeline = FursovPipeline(small_config)
        with pytest.raises(ValueError, match="pituitary"):
            pipeline.run_all_classes(n_subclasses={"glioma": 2, "meningioma": 3})

    def test_build_classifier_after_dict_n_subclasses(self, small_config):
        per_class = {"glioma": 2, "meningioma": 3, "pituitary": 4}
        pipeline = FursovPipeline(small_config)
        pipeline.run_all_classes(n_subclasses=per_class, freeze_basis_at=2)

        classifier = pipeline.build_classifier()

        assert classifier.n_subclasses_by_class_ == per_class


class TestBuildClassifier:
    def test_raises_without_trained_classes(self, small_config):
        pipeline = FursovPipeline(small_config)
        with pytest.raises(RuntimeError):
            pipeline.build_classifier()

    def test_builds_from_clusterers(self, small_config):
        pipeline = FursovPipeline(small_config)
        pipeline.run_all_classes(n_subclasses=4, freeze_basis_at=2)

        classifier = pipeline.build_classifier()

        assert pipeline.classifier_ is classifier
        assert set(classifier.classes_) == set(small_config.classes)
        assert classifier.is_fitted_

    def test_equalize_true_after_unbounded_growth(self, small_config):
        """run_all_classes(freeze_basis_at=None) растит подпространства
        каждого класса независимо до естественного размера; без equalize
        размеры между классами могут отличаться — build_classifier(equalize=True)
        должен привести их к общему минимуму (находка №2, статья Korshikov
        & Fursov: R(x,Y) сопоставим между классами только при равном k)."""
        pipeline = FursovPipeline(small_config)
        # export=False: stage_export_subspaces пишет CSV в формате NB6-7,
        # который требует ровно 2 вектора на подкласс — здесь мы намеренно
        # растим без ограничения (freeze_basis_at=None), экспорт не нужен.
        pipeline.run_all_classes(n_subclasses=4, freeze_basis_at=None, export=False)

        classifier = pipeline.build_classifier(equalize=True)

        all_sizes = {
            Y.shape[1] for bases in classifier.subspaces_.values() for Y in bases
        }
        assert len(all_sizes) == 1
        assert classifier.equalized_basis_size_ == next(iter(all_sizes))

    def test_equalize_false_keeps_natural_unequal_sizes(self, small_config):
        pipeline = FursovPipeline(small_config)
        pipeline.run_all_classes(n_subclasses=4, freeze_basis_at=None, export=False)

        classifier = pipeline.build_classifier(equalize=False)

        assert classifier.equalized_basis_size_ is None


class TestClassifyTest:
    def test_classify_with_explicit_labels(self, small_config):
        _write_synthetic_centered_images(small_config, "test", count=15, seed=99)
        y_true = np.array(
            ["glioma"] * 5 + ["meningioma"] * 5 + ["pituitary"] * 5
        )

        pipeline = FursovPipeline(small_config)
        pipeline.run_all_classes(n_subclasses=4, freeze_basis_at=2)

        result = pipeline.classify_test(y_test=y_true)

        assert "predictions" in result and "report" in result
        assert len(result["predictions"]) == 15
        assert 0.0 <= result["report"]["accuracy"] <= 1.0

    def test_classify_without_labels_uses_positional_fallback(self, small_config):
        _write_synthetic_centered_images(small_config, "test", count=15, seed=99)

        pipeline = FursovPipeline(small_config)
        pipeline.run_all_classes(n_subclasses=4, freeze_basis_at=2)

        explicit = pipeline.classify_test(
            y_test=np.array(["glioma"] * 5 + ["meningioma"] * 5 + ["pituitary"] * 5)
        )
        positional = pipeline.classify_test()

        np.testing.assert_array_equal(
            explicit["predictions"], positional["predictions"]
        )
        assert explicit["report"]["accuracy"] == positional["report"]["accuracy"]

    def test_positional_fallback_wrong_count_raises(self, small_config):
        _write_synthetic_centered_images(small_config, "test", count=7, seed=99)

        pipeline = FursovPipeline(small_config)
        pipeline.run_all_classes(n_subclasses=4, freeze_basis_at=2)

        with pytest.raises(ValueError):
            pipeline.classify_test()

    def test_classify_test_builds_classifier_implicitly(self, small_config):
        _write_synthetic_centered_images(small_config, "test", count=15, seed=99)
        pipeline = FursovPipeline(small_config)
        pipeline.run_all_classes(n_subclasses=4, freeze_basis_at=2)

        assert pipeline.classifier_ is None
        pipeline.classify_test(
            y_test=np.array(["glioma"] * 5 + ["meningioma"] * 5 + ["pituitary"] * 5)
        )
        assert pipeline.classifier_ is not None


class TestRunPreprocessing:
    def test_chains_resize_and_center(self, tmp_path):
        config = DatasetConfig(root=tmp_path, classes=["glioma"])
        config.create_directories(stages=["raw"])

        rng = np.random.default_rng(0)
        raw_dir = config.paths["glioma"]["raw"]
        for i in range(4):
            img = rng.integers(0, 256, size=(300, 300, 3), dtype=np.uint8)
            cv2.imwrite(str(raw_dir / f"src_{i}.jpg"), img)

        pipeline = FursovPipeline(config)
        centered_paths = pipeline.run_preprocessing("glioma", raw_pattern="*.jpg")

        assert len(centered_paths) == 4
        resized_files = list(config.paths["glioma"]["resized"].glob("*.png"))
        assert len(resized_files) == 4
        for path in centered_paths:
            assert cv2.imread(str(path)).shape == (256, 256, 3)


@pytest.mark.notebook_parity
@pytest.mark.slow
class TestFullPipelineOnRealArchive:
    """Полный путь raw archive .jpg -> resize -> center -> cluster -> classify."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not (ARCHIVE_ROOT / "Training" / "glioma").is_dir():
            pytest.skip(f"Brain Tumor MRI Dataset не найден в {ARCHIVE_ROOT}.")

    def test_end_to_end_from_raw_images(self, tmp_path):
        import shutil

        config = DatasetConfig(
            root=tmp_path,
            classes=["glioma", "meningioma", "pituitary"],
            n_subclasses=4,
            test_samples_per_class=5,
        )
        config.create_directories(stages=["raw"])

        for cls in config.classes:
            src_files = sorted((ARCHIVE_ROOT / "Training" / cls).glob("*.jpg"))[:20]
            for f in src_files:
                shutil.copy(f, config.paths[cls]["raw"] / f.name)

        config.paths["test"]["raw"].mkdir(parents=True, exist_ok=True)
        y_true = []
        for cls in config.classes:
            test_files = sorted((ARCHIVE_ROOT / "Training" / cls).glob("*.jpg"))[
                20 : 20 + config.test_samples_per_class
            ]
            for f in test_files:
                shutil.copy(f, config.paths["test"]["raw"] / f"test{len(y_true) + 1}.jpg")
                y_true.append(cls)

        pipeline = FursovPipeline(config)
        for cls in list(config.classes) + ["test"]:
            pipeline.run_preprocessing(cls, raw_pattern="*.jpg")

        clusterers = pipeline.run_all_classes(n_subclasses=4, freeze_basis_at=2)
        assert set(clusterers.keys()) == set(config.classes)

        classifier = pipeline.build_classifier()
        assert classifier.is_fitted_

        report = pipeline.classify_test(y_test=np.array(y_true))
        assert len(report["predictions"]) == len(y_true)
        assert 0.0 <= report["report"]["accuracy"] <= 1.0
