"""Unit tests для векторизации изображений (features/vectorization.py)."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from subspace_conjugacy.features.vectorization import (
    vectorize_image,
    vectorize_batch,
    load_and_vectorize,
    load_and_vectorize_batch,
    _ensure_grayscale,
)


class TestVectorizeImage:
    """Тесты функции vectorize_image."""

    def test_horizontal_vectorization(self):
        """Horizontal должен совпадать с numpy flatten."""
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        vec = vectorize_image(img, method="horizontal")

        assert vec.shape == (65536,)
        assert np.array_equal(vec, img.flatten())

    def test_vertical_vectorization(self):
        """Vertical должен совпадать с numpy ravel('F')."""
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        vec = vectorize_image(img, method="vertical")

        assert vec.shape == (65536,)
        assert np.array_equal(vec, img.ravel(order="F"))

    def test_different_image_sizes(self):
        """Проверка работы с изображениями разных размеров."""
        sizes = [(128, 128), (256, 256), (512, 512), (100, 200)]

        for h, w in sizes:
            img = np.random.randint(0, 256, (h, w), dtype=np.uint8)
            vec = vectorize_image(img)
            assert vec.shape == (h * w,)

    def test_color_to_grayscale_conversion(self):
        """Цветное изображение должно автоматически конвертироваться."""
        img_color = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        vec = vectorize_image(img_color)

        assert vec.shape == (65536,)
        assert vec.dtype == np.uint8

    def test_invalid_method_raises_error(self):
        """Некорректный метод должен вызывать ValueError."""
        img = np.random.randint(0, 256, (256, 256))

        with pytest.raises(ValueError, match="Unknown vectorization method"):
            vectorize_image(img, method="diagonal")

    def test_dtype_preservation(self):
        """Проверка сохранения типа данных."""
        img_float = np.random.rand(100, 100).astype(np.float32)
        vec = vectorize_image(img_float)

        assert vec.dtype == np.float32


class TestVectorizeBatch:
    """Тесты функции vectorize_batch."""

    def test_batch_from_list(self):
        """Векторизация списка изображений."""
        imgs = [np.random.randint(0, 256, (256, 256)) for _ in range(10)]
        X = vectorize_batch(imgs, method="horizontal")

        assert X.shape == (10, 65536)
        assert isinstance(X, np.ndarray)

    def test_batch_from_3d_array(self):
        """Векторизация 3D массива (N, H, W)."""
        imgs_array = np.random.randint(0, 256, (10, 256, 256))
        X = vectorize_batch(imgs_array)

        assert X.shape == (10, 65536)

    def test_batch_from_4d_array(self):
        """Векторизация 4D массива (N, H, W, C) — цветные изображения."""
        imgs_color = np.random.randint(0, 256, (5, 256, 256, 3))
        X = vectorize_batch(imgs_color)

        assert X.shape == (5, 65536)

    def test_batch_with_vertical_method(self):
        """Проверка vertical метода на батче."""
        imgs = [np.random.randint(0, 256, (100, 100)) for _ in range(5)]
        X = vectorize_batch(imgs, method="vertical")

        assert X.shape == (5, 10000)
        # Проверяем, что каждый вектор соответствует ravel('F')
        for i, img in enumerate(imgs):
            assert np.array_equal(X[i], img.ravel(order="F"))

    def test_invalid_array_dimension_raises_error(self):
        """Некорректная размерность массива должна вызывать ValueError."""
        imgs_5d = np.random.randint(0, 256, (2, 3, 256, 256, 3))

        with pytest.raises(ValueError, match="Expected 3D or 4D array"):
            vectorize_batch(imgs_5d)


class TestLoadAndVectorize:
    """Тесты функций load_and_vectorize."""

    def test_load_nonexistent_file_raises_error(self):
        """Несуществующий файл должен вызывать FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_and_vectorize("/nonexistent/path/image.png")

    @pytest.mark.skipif(
        not hasattr(np, "random"),
        reason="Test requires numpy"
    )
    def test_load_and_vectorize_with_temp_image(self):
        """Загрузка и векторизация временного изображения."""
        # Пропускаем, если cv2 или PIL недоступны
        try:
            import cv2
        except ImportError:
            pytest.skip("OpenCV not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)

            vec = load_and_vectorize(img_path, method="horizontal")
            assert vec.shape == (65536,)

    def test_invalid_backend_raises_error(self):
        """Некорректный backend должен вызывать ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.png"
            # Создаём пустой файл
            img_path.touch()

            with pytest.raises(ValueError, match="Unknown backend"):
                load_and_vectorize(img_path, backend="invalid")


class TestEnsureGrayscale:
    """Тесты функции _ensure_grayscale."""

    def test_grayscale_unchanged(self):
        """Grayscale изображение должно оставаться без изменений."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = _ensure_grayscale(img)

        assert np.array_equal(result, img)
        assert result.shape == (100, 100)

    def test_color_to_grayscale_conversion(self):
        """Цветное изображение должно конвертироваться в grayscale."""
        img_color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = _ensure_grayscale(img_color)

        assert result.shape == (100, 100)
        assert result.ndim == 2

    def test_single_channel_3d_to_2d(self):
        """3D массив с одним каналом (H, W, 1) → (H, W)."""
        img = np.random.randint(0, 256, (100, 100, 1), dtype=np.uint8)
        result = _ensure_grayscale(img)

        assert result.shape == (100, 100)
        assert result.ndim == 2

    def test_invalid_dimension_raises_error(self):
        """Некорректная размерность должна вызывать ValueError."""
        img_1d = np.random.randint(0, 256, 100)

        with pytest.raises(ValueError, match="must be 2D .* or 3D"):
            _ensure_grayscale(img_1d)

    def test_weighted_average_formula(self):
        """Проверка формулы weighted average (ITU-R BT.601)."""
        # Создаём цветное изображение с известными значениями
        img = np.array([
            [[100, 50, 200]],  # RGB
        ], dtype=np.uint8)

        result = _ensure_grayscale(img)

        # Ожидаемое значение: 0.299*100 + 0.587*50 + 0.114*200
        expected = int(0.299 * 100 + 0.587 * 50 + 0.114 * 200)
        assert result[0, 0] == expected


@pytest.mark.theory
class TestVectorizationPerformance:
    """Бенчмарки производительности (маркер theory)."""

    def test_horizontal_faster_than_loops(self):
        """Векторизация должна быть быстрее циклов из NB3."""
        import time

        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

        # Оптимизированная версия
        start = time.perf_counter()
        for _ in range(100):
            _ = vectorize_image(img, method="horizontal")
        time_optimized = time.perf_counter() - start

        # NB3 двойной цикл (медленный)
        def nb3_vectorization(img):
            vector = []
            for line in img:
                for pixel in line:
                    vector.append(pixel)
            return np.array(vector)

        start = time.perf_counter()
        for _ in range(100):
            _ = nb3_vectorization(img)
        time_nb3 = time.perf_counter() - start

        speedup = time_nb3 / time_optimized
        # Должно быть минимум 10x быстрее
        assert speedup > 10, f"Speedup только {speedup:.1f}x (ожидалось >10x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
