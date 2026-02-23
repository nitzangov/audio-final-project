"""Feature extraction: Log-Mel spectrogram and Constant-Q Transform."""

import logging
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_mel(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute a log-Mel spectrogram, normalized to zero mean / unit variance.

    Returns array of shape (n_mels, T).
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # Per-spectrogram normalization
    mean = S_db.mean()
    std = S_db.std()
    if std > 0:
        S_db = (S_db - mean) / std
    else:
        S_db = S_db - mean
    return S_db.astype(np.float32)


def extract_cqt(
    y: np.ndarray,
    sr: int = 22050,
    n_bins: int = 84,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute a log-amplitude Constant-Q Transform.

    Returns array of shape (n_bins, T).
    """
    C = np.abs(librosa.cqt(y=y, sr=sr, n_bins=n_bins, hop_length=hop_length))
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    mean = C_db.mean()
    std = C_db.std()
    if std > 0:
        C_db = (C_db - mean) / std
    else:
        C_db = C_db - mean
    return C_db.astype(np.float32)


def extract_features_for_dataset(
    split_csv: str | Path,
    cache_dir: str | Path,
    sr: int = 22050,
    mel_params: dict | None = None,
    cqt_params: dict | None = None,
    extract_cqt_features: bool = False,
):
    """Extract and cache spectrogram features for all preprocessed waveforms.

    Reads waveforms from cache_dir/waveforms/, saves spectrograms to
    cache_dir/mel/ (and optionally cache_dir/cqt/).
    """
    import pandas as pd

    mel_params = mel_params or {}
    cqt_params = cqt_params or {}
    cache_dir = Path(cache_dir)

    df = pd.read_csv(split_csv)
    waveform_dir = cache_dir / "waveforms"
    mel_dir = cache_dir / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)

    if extract_cqt_features:
        cqt_dir = cache_dir / "cqt"
        cqt_dir.mkdir(parents=True, exist_ok=True)

    num_extracted = 0
    num_skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        tid = int(row["track_id"])
        waveform_path = waveform_dir / f"{tid}.npy"

        if not waveform_path.exists():
            num_skipped += 1
            continue

        mel_path = mel_dir / f"{tid}.npy"
        need_mel = not mel_path.exists()
        need_cqt = extract_cqt_features and not (cache_dir / "cqt" / f"{tid}.npy").exists()

        if not need_mel and not need_cqt:
            num_extracted += 1
            continue

        try:
            y = np.load(waveform_path)

            if need_mel:
                mel_spec = extract_mel(y, sr=sr, **mel_params)
                np.save(mel_path, mel_spec)

            if need_cqt:
                cqt_spec = extract_cqt(y, sr=sr, **cqt_params)
                np.save(cache_dir / "cqt" / f"{tid}.npy", cqt_spec)

            num_extracted += 1
        except Exception as e:
            logger.warning("Feature extraction failed for track %d: %s", tid, e)
            num_skipped += 1

    logger.info("Extracted features for %d tracks, skipped %d", num_extracted, num_skipped)
    return num_extracted, num_skipped
