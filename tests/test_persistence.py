"""Тесты io/persistence.py — сериализация моделей (pickle/npz/json).

Раньше эти проверки жили только в блоке ``if __name__ == "__main__"`` самого
модуля (Фаза 0 плана рефакторинга требует перенести такие блоки в tests/).
"""

from pathlib import Path

import numpy as np
import pytest

from subspace_conjugacy.io.persistence import (
    export_subspaces_json,
    export_subspaces_npz,
    import_subspaces_json,
    import_subspaces_npz,
    load_model,
    save_model,
)


class _MockClassifier:
    """Минимальная заглушка обученной модели для тестов persistence."""

    def __init__(self, fitted: bool = True):
        self.n_subclasses = 2
        self.n_features_in_ = 16
        self.is_fitted_ = fitted
        self.classes_ = np.array(["glioma", "meningioma"])
        rng = np.random.default_rng(0)
        self.subspaces_ = {
            "glioma": [rng.standard_normal((16, 3)), rng.standard_normal((16, 3))],
            "meningioma": [rng.standard_normal((16, 3)), rng.standard_normal((16, 3))],
        }


class TestSaveLoadModel:
    def test_roundtrip_pickle(self, tmp_path):
        model = _MockClassifier()
        path = tmp_path / "model.pkl"

        save_model(model, path)
        assert path.exists()

        loaded = load_model(path)
        assert loaded.is_fitted_ is True
        assert loaded.n_features_in_ == 16
        np.testing.assert_array_equal(loaded.classes_, model.classes_)

    def test_save_unfitted_model_raises(self, tmp_path):
        model = _MockClassifier(fitted=False)
        with pytest.raises(RuntimeError):
            save_model(model, tmp_path / "model.pkl")

    def test_save_creates_missing_parent_directories(self, tmp_path):
        model = _MockClassifier()
        nested_path = tmp_path / "a" / "b" / "model.pkl"
        save_model(model, nested_path)
        assert nested_path.exists()

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "missing.pkl")

    def test_load_corrupted_file_raises_io_error(self, tmp_path):
        bad_path = tmp_path / "corrupted.pkl"
        bad_path.write_bytes(b"not a valid pickle stream")
        with pytest.raises(IOError):
            load_model(bad_path)


class TestNpzSubspaces:
    def test_roundtrip_npz(self, tmp_path):
        model = _MockClassifier()
        path = tmp_path / "subspaces.npz"

        export_subspaces_npz(model, path)
        assert path.exists()

        loaded = import_subspaces_npz(path)
        assert "glioma" in loaded
        assert len(loaded["glioma"]) == 2
        assert loaded["glioma"][0].shape == (16, 3)
        for Y_orig, Y_loaded in zip(model.subspaces_["glioma"], loaded["glioma"]):
            np.testing.assert_allclose(Y_orig, Y_loaded)

    def test_export_without_subspaces_raises(self, tmp_path):
        class Empty:
            subspaces_ = {}

        with pytest.raises(ValueError):
            export_subspaces_npz(Empty(), tmp_path / "out.npz")

    def test_import_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_subspaces_npz(tmp_path / "missing.npz")


class TestJsonSubspaces:
    def test_roundtrip_json(self, tmp_path):
        model = _MockClassifier()
        path = tmp_path / "subspaces.json"

        export_subspaces_json(model, path)
        assert path.exists()

        loaded = import_subspaces_json(path)
        assert loaded["n_features_in"] == 16
        assert loaded["n_subclasses"] == 2
        assert "meningioma" in loaded["subspaces"]
        assert isinstance(loaded["subspaces"]["meningioma"][0], np.ndarray)
        np.testing.assert_allclose(
            loaded["subspaces"]["glioma"][0], model.subspaces_["glioma"][0]
        )

    def test_export_without_subspaces_raises(self, tmp_path):
        class Empty:
            subspaces_ = {}

        with pytest.raises(ValueError):
            export_subspaces_json(Empty(), tmp_path / "out.json")

    def test_import_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_subspaces_json(tmp_path / "missing.json")

    def test_classes_fallback_to_subspace_keys_without_classes_attr(self, tmp_path):
        class NoClasses:
            n_subclasses = 1
            n_features_in_ = 8
            subspaces_ = {"pituitary": [np.random.randn(8, 2)]}

        path = tmp_path / "no_classes.json"
        export_subspaces_json(NoClasses(), path)

        loaded = import_subspaces_json(path)
        assert loaded["classes"] == ["pituitary"]
