"""Тесты features/audio.py — векторизация аудиосигналов для метода
подпространственной сопряжённости (статья 4, "Метод подпространственной
сопряжённости для классификации аудио- и видеоданных",
theory/article_plans/04_audio_video.txt).
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from subspace_conjugacy.features.audio import (
    compute_mfcc,
    compute_spectrogram,
    load_and_vectorize_audio,
    load_and_vectorize_audio_batch,
    load_waveform,
    mel_filterbank,
    pad_or_truncate_frames,
    vectorize_audio,
)

SAMPLE_RATE = 8000


def _make_tone(freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


@pytest.fixture
def tone_waveform():
    return _make_tone(440.0, 1.0), SAMPLE_RATE


@pytest.fixture
def tmp_wav_file(tmp_path):
    def _write(freq: float, duration: float, sample_rate: int = SAMPLE_RATE) -> Path:
        wav = (_make_tone(freq, duration, sample_rate) * 20000).astype(np.int16)
        path = tmp_path / f"tone_{freq}_{duration}.wav"
        wavfile.write(str(path), sample_rate, wav)
        return path

    return _write


class TestLoadWaveform:
    def test_loads_mono_normalized(self, tmp_wav_file):
        path = tmp_wav_file(220.0, 0.5)
        waveform, sr = load_waveform(path)
        assert sr == SAMPLE_RATE
        assert waveform.ndim == 1
        assert np.max(np.abs(waveform)) <= 1.0 + 1e-9

    def test_averages_stereo_to_mono(self, tmp_path):
        left = (_make_tone(220.0, 0.5) * 20000).astype(np.int16)
        right = (_make_tone(220.0, 0.5) * 10000).astype(np.int16)
        stereo = np.stack([left, right], axis=1)
        path = tmp_path / "stereo.wav"
        wavfile.write(str(path), SAMPLE_RATE, stereo)

        waveform, sr = load_waveform(path)
        assert waveform.ndim == 1
        assert len(waveform) == len(left)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_waveform("does_not_exist.wav")


class TestComputeSpectrogram:
    def test_shape(self, tone_waveform):
        waveform, sr = tone_waveform
        spec = compute_spectrogram(waveform, sr, n_fft=512, hop_length=256)
        assert spec.shape[0] == 512 // 2 + 1
        assert spec.shape[1] > 0

    def test_log_scale_changes_values(self, tone_waveform):
        waveform, sr = tone_waveform
        linear = compute_spectrogram(waveform, sr, log_scale=False)
        logged = compute_spectrogram(waveform, sr, log_scale=True)
        np.testing.assert_allclose(logged, np.log1p(linear))

    def test_non_negative_magnitude(self, tone_waveform):
        waveform, sr = tone_waveform
        spec = compute_spectrogram(waveform, sr)
        assert np.all(spec >= 0)


class TestMelFilterbank:
    def test_shape(self):
        fb = mel_filterbank(n_filters=40, n_fft=512, sample_rate=SAMPLE_RATE)
        assert fb.shape == (40, 512 // 2 + 1)

    def test_non_negative_and_nonzero(self):
        fb = mel_filterbank(n_filters=20, n_fft=512, sample_rate=SAMPLE_RATE)
        assert np.all(fb >= 0)
        assert np.all(fb.sum(axis=1) > 0)


class TestComputeMfcc:
    def test_shape(self, tone_waveform):
        waveform, sr = tone_waveform
        mfcc = compute_mfcc(waveform, sr, n_mels=40, n_mfcc=13)
        assert mfcc.shape[0] == 13
        assert mfcc.shape[1] > 0

    def test_different_n_mfcc(self, tone_waveform):
        waveform, sr = tone_waveform
        mfcc = compute_mfcc(waveform, sr, n_mfcc=20)
        assert mfcc.shape[0] == 20


class TestPadOrTruncateFrames:
    def test_pads_short_matrix_with_zeros(self):
        matrix = np.ones((5, 3))
        padded = pad_or_truncate_frames(matrix, n_frames=6)
        assert padded.shape == (5, 6)
        np.testing.assert_array_equal(padded[:, :3], matrix)
        np.testing.assert_array_equal(padded[:, 3:], np.zeros((5, 3)))

    def test_truncates_long_matrix(self):
        matrix = np.arange(20).reshape(5, 4)
        truncated = pad_or_truncate_frames(matrix, n_frames=2)
        assert truncated.shape == (5, 2)
        np.testing.assert_array_equal(truncated, matrix[:, :2])

    def test_exact_size_unchanged(self):
        matrix = np.ones((5, 4))
        result = pad_or_truncate_frames(matrix, n_frames=4)
        np.testing.assert_array_equal(result, matrix)


class TestVectorizeAudio:
    def test_stft_vector_length(self, tone_waveform):
        waveform, sr = tone_waveform
        vec = vectorize_audio(waveform, sr, representation="stft", n_fft=512, n_frames=50)
        assert vec.shape == (257 * 50,)

    def test_mfcc_vector_length(self, tone_waveform):
        waveform, sr = tone_waveform
        vec = vectorize_audio(waveform, sr, representation="mfcc", n_mfcc=13, n_frames=50)
        assert vec.shape == (13 * 50,)

    def test_invalid_representation_raises(self, tone_waveform):
        waveform, sr = tone_waveform
        with pytest.raises(ValueError, match="representation"):
            vectorize_audio(waveform, sr, representation="unknown")

    def test_variable_duration_gives_same_length(self, tone_waveform):
        waveform, sr = tone_waveform
        short = _make_tone(440.0, 0.3, sr)
        long = _make_tone(440.0, 3.0, sr)
        vec_short = vectorize_audio(short, sr, representation="mfcc", n_frames=40)
        vec_long = vectorize_audio(long, sr, representation="mfcc", n_frames=40)
        assert vec_short.shape == vec_long.shape


class TestLoadAndVectorizeAudioBatch:
    def test_batch_of_variable_length_clips(self, tmp_wav_file):
        paths = [tmp_wav_file(220.0, dur) for dur in (0.4, 1.0, 2.0)]
        X = load_and_vectorize_audio_batch(paths, representation="mfcc", n_frames=30, n_mfcc=13)
        assert X.shape == (3, 13 * 30)

    def test_single_file_matches_batch(self, tmp_wav_file):
        path = tmp_wav_file(220.0, 0.6)
        single = load_and_vectorize_audio(path, representation="stft", n_frames=20)
        batch = load_and_vectorize_audio_batch([path], representation="stft", n_frames=20)
        np.testing.assert_allclose(batch[0], single)


class TestClassificationEndToEnd:
    def test_two_distinct_tones_are_separable(self, tmp_wav_file):
        from subspace_conjugacy.models.classifier import SubspaceConjugacyClassifier

        rng = np.random.default_rng(0)
        paths, labels = [], []
        for freq, label in [(220.0, "low"), (880.0, "high")]:
            for _ in range(10):
                dur = rng.uniform(0.5, 1.2)
                paths.append(tmp_wav_file(freq, dur))
                labels.append(label)
        y = np.array(labels)

        X = load_and_vectorize_audio_batch(paths, representation="mfcc", n_frames=30, n_mfcc=13)
        idx = rng.permutation(len(y))
        train_idx, test_idx = idx[:14], idx[14:]

        clf = SubspaceConjugacyClassifier(n_subclasses=2).fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        assert np.mean(preds == y[test_idx]) >= 0.8
