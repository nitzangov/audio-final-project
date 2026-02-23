"""PyTorch Dataset and DataLoader factory for FMA spectrograms."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class FMADataset(Dataset):
    """Dataset that loads pre-cached spectrogram .npy files.

    Supports Phase 1 (Mel-only) and Phase 2+ (Mel + CQT).
    Applies temporal cropping: random for training, center for val/test.
    """

    def __init__(
        self,
        track_ids: list[int],
        labels: list[int],
        cache_dir: str | Path,
        crop_frames: int | None = None,
        crop_mode: str = "center",
        phase: int = 1,
    ):
        self.track_ids = track_ids
        self.labels = labels
        self.cache_dir = Path(cache_dir)
        self.crop_frames = crop_frames
        self.crop_mode = crop_mode
        self.phase = phase

    def __len__(self):
        return len(self.track_ids)

    def _load_and_crop(self, path: Path) -> torch.Tensor:
        spec = np.load(path)  # (n_freq, T)

        if self.crop_frames is not None and spec.shape[1] > self.crop_frames:
            if self.crop_mode == "random":
                start = np.random.randint(0, spec.shape[1] - self.crop_frames)
            else:
                start = (spec.shape[1] - self.crop_frames) // 2
            spec = spec[:, start : start + self.crop_frames]
        elif self.crop_frames is not None and spec.shape[1] < self.crop_frames:
            pad_width = self.crop_frames - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode="constant")

        return torch.from_numpy(spec).unsqueeze(0)  # (1, n_freq, T)

    def __getitem__(self, idx):
        tid = self.track_ids[idx]
        label = self.labels[idx]

        mel_path = self.cache_dir / "mel" / f"{tid}.npy"
        mel = self._load_and_crop(mel_path)

        if self.phase >= 2:
            cqt_path = self.cache_dir / "cqt" / f"{tid}.npy"
            cqt = self._load_and_crop(cqt_path)
            return mel, cqt, label

        return mel, label


def _build_label_mapping(genres: list[str]) -> dict[str, int]:
    """Build a consistent genre -> integer mapping sorted alphabetically."""
    unique = sorted(set(genres))
    return {g: i for i, g in enumerate(unique)}


def get_dataloaders(
    config,
    split_csv: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Create train/val/test DataLoaders from a split CSV.

    Returns (train_loader, val_loader, test_loader, label_map).
    """
    split_csv = Path(split_csv or config.data.splits_csv)
    df = pd.read_csv(split_csv)
    cache_dir = Path(config.data.cache_dir)
    phase = config.model.phase

    label_map = _build_label_mapping(df["genre"].tolist())
    logger.info("Label mapping: %s", label_map)

    # Compute a fixed crop size in frames based on duration and hop_length
    sr = config.data.sample_rate
    duration = config.data.duration_sec
    hop = config.features.mel.hop_length
    crop_frames = int(sr * duration / hop) + 1

    # Filter to tracks that have cached features
    valid_tids = set()
    for tid in df["track_id"]:
        mel_path = cache_dir / "mel" / f"{int(tid)}.npy"
        if mel_path.exists():
            if phase >= 2:
                cqt_path = cache_dir / "cqt" / f"{int(tid)}.npy"
                if cqt_path.exists():
                    valid_tids.add(int(tid))
            else:
                valid_tids.add(int(tid))

    df = df[df["track_id"].isin(valid_tids)].copy()
    if len(df) == 0:
        raise RuntimeError(
            "No cached features found. Run scripts/preprocess.py first."
        )

    loaders = {}
    for split_name, crop_mode in [("train", "random"), ("val", "center"), ("test", "center")]:
        split_df = df[df["split"] == split_name]
        tids = split_df["track_id"].astype(int).tolist()
        labels = [label_map[g] for g in split_df["genre"]]

        ds = FMADataset(
            track_ids=tids,
            labels=labels,
            cache_dir=cache_dir,
            crop_frames=crop_frames,
            crop_mode=crop_mode,
            phase=phase,
        )

        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            ds,
            batch_size=config.training.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )
        logger.info(
            "%s set: %d tracks, %d batches",
            split_name, len(ds), len(loaders[split_name]),
        )

    return loaders["train"], loaders["val"], loaders["test"], label_map
