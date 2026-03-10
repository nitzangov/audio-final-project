"""Smoke tests for FMADataset: item contract, shapes, dtypes, label range."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FMADataset

N_MELS = 128
N_CQT_BINS = 84
T_FRAMES = 200
NUM_CLASSES = 4


@pytest.fixture
def mock_cache(tmp_path):
    """Create a temporary cache directory with synthetic .npy spectrograms."""
    mel_dir = tmp_path / "mel"
    cqt_dir = tmp_path / "cqt"
    mel_dir.mkdir()
    cqt_dir.mkdir()

    track_ids = [1, 2, 3, 4, 5]
    labels = [0, 1, 2, 3, 0]

    for tid in track_ids:
        np.save(mel_dir / f"{tid}.npy", np.random.randn(N_MELS, T_FRAMES).astype(np.float32))
        np.save(cqt_dir / f"{tid}.npy", np.random.randn(N_CQT_BINS, T_FRAMES).astype(np.float32))

    return tmp_path, track_ids, labels


class TestFMADatasetPhase1:
    def test_len(self, mock_cache):
        cache_dir, tids, labels = mock_cache
        ds = FMADataset(tids, labels, cache_dir, phase=1)
        assert len(ds) == len(tids)

    def test_getitem_returns_tuple(self, mock_cache):
        cache_dir, tids, labels = mock_cache
        ds = FMADataset(tids, labels, cache_dir, phase=1)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2  # (mel, label)

    def test_mel_shape(self, mock_cache):
        cache_dir, tids, labels = mock_cache
        ds = FMADataset(tids, labels, cache_dir, phase=1)
        mel, label = ds[0]
        assert mel.ndim == 3  # (1, n_mels, T)
        assert mel.shape[0] == 1
        assert mel.shape[1] == N_MELS

    def test_mel_dtype(self, mock_cache):
        cache_dir, tids, labels = mock_cache
        ds = FMADataset(tids, labels, cache_dir, phase=1)
        mel, _ = ds[0]
        assert mel.dtype == torch.float32

    def test_label_range(self, mock_cache):
        cache_dir, tids, labels = mock_cache
        ds = FMADataset(tids, labels, cache_dir, phase=1)
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0 <= label < NUM_CLASSES

    def test_crop_reduces_time(self, mock_cache):
        cache_dir, tids, labels = mock_cache
        crop = 100
        ds = FMADataset(tids, labels, cache_dir, crop_frames=crop, phase=1)
        mel, _ = ds[0]
        assert mel.shape[2] == crop

    def test_crop_pads_short(self, mock_cache):
        cache_dir, tids, labels = mock_cache
        crop = T_FRAMES + 50
        ds = FMADataset(tids, labels, cache_dir, crop_frames=crop, phase=1)
        mel, _ = ds[0]
        assert mel.shape[2] == crop


class TestFMADatasetPhase2:
    def test_getitem_returns_triple(self, mock_cache):
        cache_dir, tids, labels = mock_cache
        ds = FMADataset(tids, labels, cache_dir, phase=2)
        item = ds[0]
        assert len(item) == 3  # (mel, cqt, label)

    def test_cqt_shape(self, mock_cache):
        cache_dir, tids, labels = mock_cache
        ds = FMADataset(tids, labels, cache_dir, phase=2)
        _, cqt, _ = ds[0]
        assert cqt.shape[0] == 1
        assert cqt.shape[1] == N_CQT_BINS

    def test_mel_cqt_crop_alignment(self, mock_cache):
        """Mel and CQT must receive the same temporal crop for the same sample."""
        cache_dir, tids, labels = mock_cache
        crop = 80
        ds = FMADataset(
            tids, labels, cache_dir, crop_frames=crop, crop_mode="random", phase=2,
        )
        mel_dir = cache_dir / "mel"
        cqt_dir = cache_dir / "cqt"
        tid = tids[0]
        mel_raw = np.load(mel_dir / f"{tid}.npy")
        cqt_raw = np.load(cqt_dir / f"{tid}.npy")

        np.random.seed(42)
        mel1, cqt1, _ = ds[0]
        np.random.seed(42)
        mel2, cqt2, _ = ds[0]

        assert torch.equal(mel1, mel2), "Same seed should produce same mel crop"
        assert torch.equal(cqt1, cqt2), "Same seed should produce same cqt crop"

        start = None
        for s in range(mel_raw.shape[1] - crop + 1):
            if np.allclose(mel_raw[:, s : s + crop], mel1.squeeze(0).numpy()):
                start = s
                break
        assert start is not None, "Could not find mel crop offset in raw mel"
        expected_cqt = cqt_raw[:, start : start + crop]
        assert np.allclose(expected_cqt, cqt1.squeeze(0).numpy()), (
            "CQT crop must use the same start offset as mel"
        )
