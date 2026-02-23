"""Audio preprocessing: load, mono, resample, trim silence, normalize, segment."""

import logging
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def load_audio(path: Path, target_sr: int = 22050) -> np.ndarray | None:
    """Load an audio file, convert to mono, and resample.

    Tries torchaudio first (fast C++ backend), falls back to librosa for
    files that torchaudio cannot decode.

    Returns a 1-D numpy array at target_sr, or None on failure.
    """
    try:
        waveform, sr = torchaudio.load(str(path))
        # Mono: average channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample (includes anti-alias filter)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0).numpy()
    except Exception as e:
        logger.debug("torchaudio failed for %s (%s), trying librosa", path, e)

    try:
        y, _ = librosa.load(str(path), sr=target_sr, mono=True)
        return y
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def trim_silence(y: np.ndarray, top_db: float = 60.0) -> np.ndarray:
    """Remove leading/trailing silence below the given dB threshold."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def peak_normalize(y: np.ndarray) -> np.ndarray:
    """Peak-normalize waveform to [-1, 1]."""
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak
    return y


def segment_waveform(y: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Pad or truncate waveform to exactly `duration_sec` seconds.

    Stores the full waveform up to duration_sec. If shorter, zero-pads.
    The crop strategy (random vs center) is applied at Dataset load time.
    """
    target_len = int(sr * duration_sec)
    if len(y) >= target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y


def preprocess_track(
    audio_path: Path,
    target_sr: int = 22050,
    duration_sec: float = 30.0,
    top_db: float = 60.0,
) -> np.ndarray | None:
    """Full preprocessing pipeline for a single track.

    Returns a 1-D numpy array of shape (sr * duration_sec,) or None on failure.
    """
    y = load_audio(audio_path, target_sr)
    if y is None:
        return None

    if len(y) == 0:
        logger.warning("Empty audio after loading: %s", audio_path)
        return None

    y = trim_silence(y, top_db=top_db)
    if len(y) == 0:
        logger.warning("Empty audio after trimming: %s", audio_path)
        return None

    y = peak_normalize(y)
    y = segment_waveform(y, target_sr, duration_sec)
    return y


def preprocess_dataset(
    split_csv: Path,
    audio_dir: Path,
    cache_dir: Path,
    target_sr: int = 22050,
    duration_sec: float = 30.0,
) -> tuple[int, int]:
    """Preprocess all tracks listed in the split CSV.

    Saves waveforms as .npy files to cache_dir/waveforms/.
    Returns (num_processed, num_skipped).
    """
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(split_csv)
    waveform_dir = Path(cache_dir) / "waveforms"
    waveform_dir.mkdir(parents=True, exist_ok=True)

    skipped_log = Path(cache_dir) / "skipped_tracks.log"
    num_processed = 0
    num_skipped = 0

    with open(skipped_log, "w") as log_f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            tid = int(row["track_id"])
            tid_str = str(tid).zfill(6)
            subdir = tid_str[:3]

            out_path = waveform_dir / f"{tid}.npy"
            if out_path.exists():
                num_processed += 1
                continue

            audio_path = audio_dir / subdir / f"{tid_str}.mp3"
            y = preprocess_track(audio_path, target_sr, duration_sec)

            if y is None:
                log_f.write(f"{tid}\t{audio_path}\n")
                num_skipped += 1
                continue

            np.save(out_path, y)
            num_processed += 1

    logger.info("Preprocessed %d tracks, skipped %d", num_processed, num_skipped)
    return num_processed, num_skipped
