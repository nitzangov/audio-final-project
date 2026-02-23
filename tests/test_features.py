"""Smoke tests for feature extraction: shape, dtype, normalization, no NaN/Inf."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import extract_cqt, extract_mel

SR = 22050
DURATION = 5.0  # short clip for fast tests


@pytest.fixture
def waveform():
    """Synthetic waveform: 5 seconds of a 440Hz sine wave."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestMelSpectrogram:
    def test_output_shape(self, waveform):
        mel = extract_mel(waveform, sr=SR, n_mels=128, n_fft=2048, hop_length=512)
        assert mel.ndim == 2
        assert mel.shape[0] == 128
        expected_frames = int(len(waveform) / 512) + 1
        assert abs(mel.shape[1] - expected_frames) <= 2

    def test_dtype(self, waveform):
        mel = extract_mel(waveform, sr=SR)
        assert mel.dtype == np.float32

    def test_no_nan_inf(self, waveform):
        mel = extract_mel(waveform, sr=SR)
        assert not np.isnan(mel).any(), "Mel spectrogram contains NaN"
        assert not np.isinf(mel).any(), "Mel spectrogram contains Inf"

    def test_normalized(self, waveform):
        mel = extract_mel(waveform, sr=SR)
        assert abs(mel.mean()) < 0.1, f"Mean should be ~0, got {mel.mean()}"
        assert abs(mel.std() - 1.0) < 0.1, f"Std should be ~1, got {mel.std()}"

    def test_different_params(self, waveform):
        mel = extract_mel(waveform, sr=SR, n_mels=64, n_fft=1024, hop_length=256)
        assert mel.shape[0] == 64


class TestCQT:
    def test_output_shape(self, waveform):
        cqt = extract_cqt(waveform, sr=SR, n_bins=84, hop_length=512)
        assert cqt.ndim == 2
        assert cqt.shape[0] == 84

    def test_dtype(self, waveform):
        cqt = extract_cqt(waveform, sr=SR)
        assert cqt.dtype == np.float32

    def test_no_nan_inf(self, waveform):
        cqt = extract_cqt(waveform, sr=SR)
        assert not np.isnan(cqt).any(), "CQT contains NaN"
        assert not np.isinf(cqt).any(), "CQT contains Inf"

    def test_normalized(self, waveform):
        cqt = extract_cqt(waveform, sr=SR)
        assert abs(cqt.mean()) < 0.1, f"Mean should be ~0, got {cqt.mean()}"
        assert abs(cqt.std() - 1.0) < 0.1, f"Std should be ~1, got {cqt.std()}"
