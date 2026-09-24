"""Тесты features/video.py — векторизация видеопоследовательностей для
метода подпространственной сопряжённости (статья 4, "Метод
подпространственной сопряжённости для классификации аудио- и видеоданных",
theory/article_plans/04_audio_video.txt).
"""

import cv2
import numpy as np
import pytest

from subspace_conjugacy.features.video import (
    extract_frames,
    extract_frames_from_paths,
    load_and_vectorize_video,
    load_and_vectorize_video_batch,
    vectorize_frames_concat,
    vectorize_frames_keyframe,
    vectorize_video,
)

FRAME_SIZE = (32, 32)


def _write_video(path, n_frames: int, size=FRAME_SIZE, fps: float = 10.0) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size, isColor=True)
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), (i * 10) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def video_file(tmp_path):
    def _make(n_frames: int) -> "Path":
        path = tmp_path / f"video_{n_frames}.avi"
        _write_video(path, n_frames)
        return path

    return _make


@pytest.fixture
def frame_files(tmp_path):
    def _make(n_frames: int, size=(20, 20)):
        paths = []
        for i in range(n_frames):
            p = tmp_path / f"frame_{i}.png"
            cv2.imwrite(str(p), np.full((size[1], size[0]), (i * 30) % 255, dtype=np.uint8))
            paths.append(p)
        return paths

    return _make


class TestExtractFrames:
    def test_shape_and_resize(self, video_file):
        path = video_file(20)
        frames = extract_frames(path, n_frames=8, resize=(16, 16))
        assert frames.shape == (8, 16, 16)

    def test_short_video_duplicates_last_frame(self, video_file):
        path = video_file(3)
        frames = extract_frames(path, n_frames=8, resize=(16, 16))
        assert frames.shape == (8, 16, 16)
        # Последние кадры должны быть дублированием последнего реального кадра
        np.testing.assert_array_equal(frames[-1], frames[-2])

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_frames("does_not_exist.avi")

    def test_no_resize_keeps_original_size(self, video_file):
        path = video_file(10)
        frames = extract_frames(path, n_frames=5, resize=None)
        assert frames.shape == (5, FRAME_SIZE[1], FRAME_SIZE[0])


class TestExtractFramesFromPaths:
    def test_upsamples_when_fewer_frames_than_requested(self, frame_files):
        paths = frame_files(5)
        frames = extract_frames_from_paths(paths, n_frames=8, resize=(10, 10))
        assert frames.shape == (8, 10, 10)

    def test_downsamples_when_more_frames_than_requested(self, frame_files):
        paths = frame_files(10)
        frames = extract_frames_from_paths(paths, n_frames=3, resize=(10, 10))
        assert frames.shape == (3, 10, 10)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="пустым"):
            extract_frames_from_paths([], n_frames=4)


class TestVectorizeFrames:
    def test_concat_length(self):
        frames = np.random.randint(0, 256, (8, 16, 16), dtype=np.uint8)
        vec = vectorize_frames_concat(frames)
        assert vec.shape == (8 * 16 * 16,)

    def test_keyframe_length(self):
        frames = np.random.randint(0, 256, (8, 16, 16), dtype=np.uint8)
        vec = vectorize_frames_keyframe(frames)
        assert vec.shape == (16 * 16,)

    def test_keyframe_middle_index(self):
        frames = np.stack([np.full((4, 4), i, dtype=np.uint8) for i in range(7)])
        vec = vectorize_frames_keyframe(frames, index="middle")
        assert np.all(vec == 3)  # 7 // 2 == 3

    def test_keyframe_explicit_index(self):
        frames = np.stack([np.full((4, 4), i, dtype=np.uint8) for i in range(7)])
        vec = vectorize_frames_keyframe(frames, index=0)
        assert np.all(vec == 0)

    def test_vectorize_video_dispatch(self):
        frames = np.random.randint(0, 256, (6, 8, 8), dtype=np.uint8)
        concat = vectorize_video(frames, mode="concat")
        keyframe = vectorize_video(frames, mode="keyframe")
        assert concat.shape == (6 * 8 * 8,)
        assert keyframe.shape == (8 * 8,)

    def test_invalid_mode_raises(self):
        frames = np.random.randint(0, 256, (4, 8, 8), dtype=np.uint8)
        with pytest.raises(ValueError, match="mode"):
            vectorize_video(frames, mode="unknown")


class TestLoadAndVectorizeVideoBatch:
    def test_batch_of_variable_length_videos(self, video_file):
        paths = [video_file(n) for n in (3, 10, 25)]
        X = load_and_vectorize_video_batch(paths, n_frames=8, resize=(16, 16), mode="concat")
        assert X.shape == (3, 8 * 16 * 16)

    def test_single_matches_batch(self, video_file):
        path = video_file(10)
        single = load_and_vectorize_video(path, n_frames=5, resize=(8, 8), mode="keyframe")
        batch = load_and_vectorize_video_batch([path], n_frames=5, resize=(8, 8), mode="keyframe")
        np.testing.assert_array_equal(batch[0], single)


class TestClassificationEndToEnd:
    def test_two_distinct_brightness_levels_are_separable(self, tmp_path):
        from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

        rng = np.random.default_rng(0)
        paths, labels = [], []
        for base, label in [(30, "dark"), (200, "bright")]:
            for i in range(8):
                n_frames = rng.integers(5, 15)
                path = tmp_path / f"{label}_{i}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(str(path), fourcc, 10.0, FRAME_SIZE, isColor=True)
                for _ in range(int(n_frames)):
                    noise = rng.integers(-10, 10)
                    value = int(np.clip(base + noise, 0, 255))
                    writer.write(np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), value, dtype=np.uint8))
                writer.release()
                paths.append(path)
                labels.append(label)
        y = np.array(labels)

        X = load_and_vectorize_video_batch(paths, n_frames=6, resize=(12, 12), mode="concat")
        idx = rng.permutation(len(y))
        train_idx, test_idx = idx[:12], idx[12:]

        clf = SubspaceConjugacyClassifier(n_subclasses=2).fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        assert np.mean(preds == y[test_idx]) >= 0.75
