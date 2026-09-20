"""Тесты preprocessing/ (NB1 resize + NB2 normalization/centering, Фаза 1)."""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from subspace_conjugacy.config import DatasetConfig
from subspace_conjugacy.preprocessing.centering import (
    center_image,
    compute_horizontal_delta,
    compute_vertical_delta,
    shift_columns,
    shift_rows,
)
from subspace_conjugacy.preprocessing.binarization import (
    binarize_directory,
    otsu_binarize,
    otsu_threshold,
)
from subspace_conjugacy.preprocessing.normalization import suppress_background
from subspace_conjugacy.preprocessing.preprocessor import ImagePreprocessor
from subspace_conjugacy.preprocessing.resize import (
    resize_directory,
    resize_image,
    resize_image_file,
)

ARCHIVE_ROOT = Path(
    os.environ.get("BRAIN_MRI_ARCHIVE_ROOT", r"C:\Users\nazar\Downloads\archive")
)


def _centroid_deviation(gray: np.ndarray, threshold: int = 10) -> float:
    """L1-отклонение центроида ярких пикселей от геометрического центра."""
    ys, xs = np.where(gray > threshold)
    h, w = gray.shape
    return abs(ys.mean() - h / 2) + abs(xs.mean() - w / 2)


def _make_offset_square(size: int = 64, background: int = 0) -> np.ndarray:
    """Синтетическое изображение с ярким квадратом не по центру."""
    image = np.full((size, size, 3), background, dtype=np.uint8)
    image[10:30, 40:60] = 200
    return image


class TestResizeImage:
    def test_resize_to_target_shape(self):
        for shape in [(400, 300, 3), (512, 512, 3), (200, 600, 3)]:
            img = np.random.randint(0, 256, shape, dtype=np.uint8)
            resized = resize_image(img, size=(256, 256))
            assert resized.shape == (256, 256, 3)

    def test_resize_preserves_grayscale_dimensionality(self):
        img = np.random.randint(0, 256, (100, 150), dtype=np.uint8)
        resized = resize_image(img, size=(64, 64))
        assert resized.shape == (64, 64)

    def test_resize_custom_size(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        resized = resize_image(img, size=(128, 64))
        assert resized.shape == (64, 128, 3)  # (height, width, C)


class TestResizeFileAndDirectory:
    def test_resize_image_file_roundtrip(self, tmp_path):
        input_path = tmp_path / "raw.jpg"
        output_path = tmp_path / "resized.png"
        cv2.imwrite(str(input_path), np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8))

        result_path = resize_image_file(input_path, output_path, size=(256, 256))

        assert result_path == output_path
        loaded = cv2.imread(str(output_path))
        assert loaded.shape == (256, 256, 3)

    def test_resize_image_file_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resize_image_file(tmp_path / "missing.jpg", tmp_path / "out.png")

    def test_resize_directory_batch(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "resized"
        raw_dir.mkdir()

        for i in range(4):
            img = np.random.randint(0, 256, (300 + i * 10, 300, 3), dtype=np.uint8)
            cv2.imwrite(str(raw_dir / f"img_{i}.jpg"), img)

        outputs = resize_directory(raw_dir, out_dir, output_prefix="glioma")

        assert len(outputs) == 4
        assert [p.name for p in outputs] == [
            "glioma1.png", "glioma2.png", "glioma3.png", "glioma4.png",
        ]
        for path in outputs:
            assert cv2.imread(str(path)).shape == (256, 256, 3)

    def test_resize_directory_empty_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError):
            resize_directory(empty_dir, tmp_path / "out")


class TestSuppressBackground:
    def test_grayscale_threshold(self):
        gray = np.array([[5, 50, 200], [3, 80, 9], [120, 4, 250]], dtype=np.uint8)
        result = suppress_background(gray, threshold=10)
        expected = np.array([[0, 50, 200], [0, 80, 0], [120, 0, 250]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_does_not_mutate_input(self):
        gray = np.array([[5, 50]], dtype=np.uint8)
        original = gray.copy()
        suppress_background(gray, threshold=10)
        np.testing.assert_array_equal(gray, original)

    def test_color_with_grayscale_reference(self):
        gray_ref = np.array([[5, 80], [200, 3]], dtype=np.uint8)
        color = np.full((2, 2, 3), 150, dtype=np.uint8)

        result = suppress_background(color, threshold=10, reference=gray_ref)

        np.testing.assert_array_equal(result[0, 0], [0, 0, 0])       # 5 < 10
        np.testing.assert_array_equal(result[0, 1], [150, 150, 150])  # 80 >= 10
        np.testing.assert_array_equal(result[1, 0], [150, 150, 150])  # 200 >= 10
        np.testing.assert_array_equal(result[1, 1], [0, 0, 0])       # 3 < 10

    def test_reference_must_be_2d(self):
        color = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            suppress_background(color, reference=np.zeros((4, 4, 3), dtype=np.uint8))

    def test_reference_shape_mismatch_raises(self):
        color = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            suppress_background(color, reference=np.zeros((2, 2), dtype=np.uint8))


class TestCenteringDeltas:
    def test_zero_delta_for_centered_content(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[27:37, 27:37] = 200  # примерно по центру
        assert abs(compute_horizontal_delta(gray)) < 2
        assert abs(compute_vertical_delta(gray)) < 2

    def test_positive_horizontal_delta_when_content_shifted_down(self):
        """Контент смещён вниз -> больше пустых строк сверху -> delta > 0."""
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[50:60, 20:40] = 200
        assert compute_horizontal_delta(gray) > 0

    def test_negative_horizontal_delta_when_content_shifted_up(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[2:12, 20:40] = 200
        assert compute_horizontal_delta(gray) < 0

    def test_vertical_delta_sign_symmetry(self):
        """Отражение по горизонтали должно менять знак vertical_delta.

        Контент у левого края -> маленький левый отступ, большой правый ->
        delta = (left_empty - right_empty) / 2 < 0 (нужно сдвинуть влево,
        т.е. в сторону контента, чтобы уравнять отступы).
        """
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[20:40, 2:12] = 200  # контент у левого края
        left_delta = compute_vertical_delta(gray)

        gray_mirrored = gray[:, ::-1].copy()
        right_delta = compute_vertical_delta(gray_mirrored)

        assert left_delta < 0
        assert right_delta > 0


class TestShiftFunctions:
    def test_shift_rows_below_min_shift_is_noop(self):
        image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        result = shift_rows(image, delta=1, min_shift=2)
        np.testing.assert_array_equal(result, image)

    def test_shift_rows_applies_roll(self):
        image = np.arange(25).reshape(5, 5).astype(np.uint8)
        shifted = shift_rows(image, delta=2, min_shift=2)
        np.testing.assert_array_equal(shifted, np.roll(image, -2, axis=0))

    def test_shift_columns_applies_roll(self):
        image = np.arange(25).reshape(5, 5).astype(np.uint8)
        shifted = shift_columns(image, delta=-3, min_shift=2)
        np.testing.assert_array_equal(shifted, np.roll(image, 3, axis=1))


class TestCenterImage:
    def test_offset_square_moves_toward_center(self):
        image = _make_offset_square(size=64)
        before_dev = _centroid_deviation(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        centered = center_image(image, background_threshold=10)
        after_dev = _centroid_deviation(cv2.cvtColor(centered, cv2.COLOR_BGR2GRAY))

        assert after_dev < before_dev

    def test_output_shape_and_dtype_preserved(self):
        image = _make_offset_square(size=48)
        centered = center_image(image)
        assert centered.shape == image.shape
        assert centered.dtype == image.dtype

    def test_grayscale_input_supported(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[5:15, 5:15] = 200
        centered = center_image(gray)
        assert centered.shape == gray.shape


class TestImagePreprocessor:
    def test_process_end_to_end_shape(self):
        preprocessor = ImagePreprocessor(target_size=(256, 256))
        raw = np.random.randint(0, 60, (400, 300, 3), dtype=np.uint8)
        raw[150:250, 100:200] = 200

        processed = preprocessor.process(raw)

        assert processed.shape == (256, 256, 3)
        assert processed.dtype == np.uint8

    def test_process_file(self, tmp_path):
        preprocessor = ImagePreprocessor()
        input_path = tmp_path / "raw.jpg"
        cv2.imwrite(str(input_path), np.random.randint(0, 256, (350, 350, 3), dtype=np.uint8))

        output_path = preprocessor.process_file(input_path, tmp_path / "out.png")

        assert output_path.exists()
        assert cv2.imread(str(output_path)).shape == (256, 256, 3)

    def test_process_directory(self, tmp_path):
        preprocessor = ImagePreprocessor()
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        for i in range(3):
            cv2.imwrite(
                str(raw_dir / f"src_{i}.jpg"),
                np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8),
            )

        outputs = preprocessor.process_directory(raw_dir, out_dir, output_prefix="glioma")

        assert [p.name for p in outputs] == ["glioma1.png", "glioma2.png", "glioma3.png"]

    def test_process_class_via_config_two_stage(self, tmp_path):
        preprocessor = ImagePreprocessor()
        config = DatasetConfig(root=tmp_path, classes=["glioma"])
        config.create_directories(stages=["raw"])

        raw_dir = config.paths["glioma"]["raw"]
        for i in range(3):
            cv2.imwrite(
                str(raw_dir / f"src_{i}.jpg"),
                np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8),
            )

        centered_paths = preprocessor.process_class_via_config(
            "glioma", config, pattern="*.jpg"
        )

        assert len(centered_paths) == 3
        resized_files = sorted(config.paths["glioma"]["resized"].glob("*.png"))
        assert len(resized_files) == 3
        for path in centered_paths:
            assert cv2.imread(str(path)).shape == (256, 256, 3)


class TestOtsuBinarization:
    """Otsu-бинаризация (статья, "Data Preprocessing", находка №4) — только
    для этапа определения проекции, статья явно НЕ применяет её для
    определения типа опухоли."""

    @staticmethod
    def _make_bimodal_image(size=64, low=20, high=220, noise=10):
        rng = np.random.default_rng(0)
        image = np.full((size, size), low, dtype=np.uint8)
        image[20:40, 20:40] = high
        noise_arr = rng.integers(-noise, noise + 1, size=image.shape)
        return np.clip(image.astype(int) + noise_arr, 0, 255).astype(np.uint8)

    def test_threshold_between_the_two_modes(self):
        gray = self._make_bimodal_image()
        t = otsu_threshold(gray)
        assert 20 < t < 220

    def test_binarize_produces_only_black_and_white(self):
        gray = self._make_bimodal_image()
        binarized = otsu_binarize(gray)
        assert set(np.unique(binarized)) <= {0, 255}

    def test_binarize_default_bright_is_white(self):
        gray = self._make_bimodal_image()
        binarized = otsu_binarize(gray)
        assert binarized[30, 30] == 255  # яркая область -> белая
        assert binarized[5, 5] == 0      # тёмный фон -> чёрный

    def test_binarize_invert_flips_result(self):
        gray = self._make_bimodal_image()
        normal = otsu_binarize(gray)
        inverted = otsu_binarize(gray, invert=True)
        np.testing.assert_array_equal(inverted, 255 - normal)

    def test_accepts_color_image(self):
        gray = self._make_bimodal_image()
        color = np.stack([gray, gray, gray], axis=-1)
        binarized = otsu_binarize(color)
        assert binarized.ndim == 2
        assert binarized.shape == gray.shape

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            otsu_binarize(np.zeros(64, dtype=np.uint8))


class TestBinarizeDirectory:
    def test_binarizes_all_matching_files(self, tmp_path):
        input_dir = tmp_path / "centered"
        output_dir = tmp_path / "binarized"
        input_dir.mkdir()

        for i in range(1, 4):
            img = TestOtsuBinarization._make_bimodal_image()
            cv2.imwrite(str(input_dir / f"glioma{i}.png"), img)

        outputs = binarize_directory(input_dir, output_dir, pattern="glioma*.png")

        assert len(outputs) == 3
        for path in outputs:
            loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            assert set(np.unique(loaded)) <= {0, 255}

    def test_output_filenames_match_input(self, tmp_path):
        input_dir = tmp_path / "centered"
        output_dir = tmp_path / "binarized"
        input_dir.mkdir()
        cv2.imwrite(
            str(input_dir / "glioma7.png"), TestOtsuBinarization._make_bimodal_image()
        )

        outputs = binarize_directory(input_dir, output_dir)

        assert outputs[0].name == "glioma7.png"

    def test_empty_directory_raises(self, tmp_path):
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        with pytest.raises(ValueError):
            binarize_directory(input_dir, tmp_path / "out")

    def test_creates_output_directory(self, tmp_path):
        input_dir = tmp_path / "centered"
        input_dir.mkdir()
        cv2.imwrite(
            str(input_dir / "a.png"), TestOtsuBinarization._make_bimodal_image()
        )
        output_dir = tmp_path / "nested" / "binarized"

        binarize_directory(input_dir, output_dir)

        assert output_dir.is_dir()


@pytest.mark.notebook_parity
class TestRealBrainTumorArchive:
    """Тесты на реальном Brain Tumor MRI Dataset (Kaggle, archive/Training).

    Датасет не входит в репозиторий — путь задаётся через переменную
    окружения BRAIN_MRI_ARCHIVE_ROOT (по умолчанию путь на машине автора).
    Пропускается, если директория не найдена.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not (ARCHIVE_ROOT / "Training" / "glioma").is_dir():
            pytest.skip(
                f"Brain Tumor MRI Dataset не найден в {ARCHIVE_ROOT}. "
                "Задайте BRAIN_MRI_ARCHIVE_ROOT или пропустите этот тест."
            )

    @pytest.mark.parametrize("class_name", ["glioma", "meningioma", "pituitary"])
    def test_process_real_images_without_crashing(self, class_name):
        preprocessor = ImagePreprocessor()
        class_dir = ARCHIVE_ROOT / "Training" / class_name
        paths = sorted(class_dir.glob("*.jpg"))[:15]
        assert paths, f"Не найдено .jpg файлов в {class_dir}"

        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            processed = preprocessor.process(image)
            assert processed.shape == (256, 256, 3)
            assert processed.dtype == np.uint8

    def test_centering_reduces_average_centroid_deviation(self):
        """На реальных изображениях центрирование в среднем приближает
        содержимое к центру кадра (эмпирически: медианное отклонение падает
        с ~13 до ~5 пикселей на выборке из 90 изображений, см. коммит)."""
        preprocessor = ImagePreprocessor()
        before_devs, after_devs = [], []

        for class_name in ["glioma", "meningioma", "pituitary"]:
            class_dir = ARCHIVE_ROOT / "Training" / class_name
            for path in sorted(class_dir.glob("*.jpg"))[:20]:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                resized = resize_image(image, size=(256, 256))
                processed = preprocessor.process(image)

                before_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                after_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

                if (before_gray > 10).any() and (after_gray > 10).any():
                    before_devs.append(_centroid_deviation(before_gray))
                    after_devs.append(_centroid_deviation(after_gray))

        before_devs = np.array(before_devs)
        after_devs = np.array(after_devs)

        assert len(before_devs) > 30
        assert np.median(after_devs) < np.median(before_devs)
        assert np.mean(after_devs) < np.mean(before_devs)

    def test_full_pipeline_from_raw_archive_to_clusterer(self, tmp_path):
        """Полный путь: сырые .jpg архива -> resize -> center -> vectorize -> cluster."""
        import shutil

        from subspace_conjugacy import FursovClusterer
        from subspace_conjugacy.features.extraction import extract_class_vectors

        config = DatasetConfig(root=tmp_path, classes=["glioma"])
        config.create_directories(stages=["raw"])

        src_files = sorted((ARCHIVE_ROOT / "Training" / "glioma").glob("*.jpg"))[:20]
        for f in src_files:
            shutil.copy(f, config.paths["glioma"]["raw"] / f.name)

        preprocessor = ImagePreprocessor()
        preprocessor.process_class_via_config("glioma", config, pattern="*.jpg")

        X = extract_class_vectors(config, "glioma", stage="centered", count=20)
        assert X.shape == (20, 65536)

        clusterer = FursovClusterer(n_subclasses=4, freeze_basis_at=2)
        clusterer.fit(X)

        assert len(clusterer.subspaces_) == 4
        assert len(clusterer.labels_) == 20
