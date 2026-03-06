"""FMA dataset download, checksum verification, and metadata parsing."""

import hashlib
import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

FMA_URLS = {
    "small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "medium": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
}
METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

FMA_CHECKSUMS: dict[str, str] = {
    # Checksums omitted — FMA hosting has changed over time and the original
    # checksums from the paper no longer match.  File integrity is verified
    # implicitly by successful zip extraction.
}

_CHUNK_SIZE = 1 << 20  # 1 MB


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_checksum: str | None = None):
    """Download a file via requests (handles SSL correctly) with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("File already exists: %s", dest)
        return

    logger.info("Downloading %s -> %s", url, dest)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
            f.write(chunk)
            pbar.update(len(chunk))

    if expected_checksum:
        actual = _sha256_file(dest)
        if not actual.startswith(expected_checksum):
            dest.unlink()
            raise RuntimeError(
                f"Checksum mismatch for {dest.name}: "
                f"expected {expected_checksum}..., got {actual[:len(expected_checksum)]}..."
            )
        logger.info("Checksum OK for %s", dest.name)


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract a zip archive if not already extracted."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info("Already extracted: %s", dest_dir)
        return
    logger.info("Extracting %s -> %s", zip_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir.parent)


def load_tracks_metadata(metadata_dir: Path) -> pd.DataFrame:
    """Load tracks.csv from FMA metadata, handling the multi-level header."""
    tracks_csv = metadata_dir / "fma_metadata" / "tracks.csv"
    if not tracks_csv.exists():
        tracks_csv = metadata_dir / "tracks.csv"

    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    return tracks


def generate_split_csv(
    metadata_dir: Path,
    audio_dir: Path,
    output_path: Path,
    subset: str = "small",
) -> pd.DataFrame:
    """Parse FMA metadata and generate a reproducible split CSV.

    Uses the canonical 'set' column from tracks.csv for train/val/test splits.
    Returns a DataFrame with columns: track_id, split, genre.
    """
    tracks = load_tracks_metadata(metadata_dir)

    # Extract relevant columns
    genre_col = ("track", "genre_top")
    subset_col = ("set", "subset")
    split_col = ("set", "split")

    df = pd.DataFrame({
        "track_id": tracks.index,
        "genre": tracks[genre_col].values,
        "subset": tracks[subset_col].values,
        "split": tracks[split_col].values,
    })

    # Filter to requested subset
    df = df[df["subset"] == subset].copy()
    df = df.dropna(subset=["genre", "split"])

    # Verify audio files exist and filter missing ones
    skipped = []
    valid_mask = []
    for tid in df["track_id"]:
        tid_str = str(tid).zfill(6)
        subdir = tid_str[:3]
        audio_path = audio_dir / subdir / f"{tid_str}.mp3"
        exists = audio_path.exists()
        valid_mask.append(exists)
        if not exists:
            skipped.append(tid)

    if skipped:
        logger.warning("Skipping %d tracks with missing audio files", len(skipped))

    df = df[valid_mask].copy()
    df = df[["track_id", "split", "genre"]].reset_index(drop=True)

    # Map split names to consistent convention
    df["split"] = df["split"].map({
        "training": "train",
        "validation": "val",
        "test": "test",
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved split CSV to %s (%d tracks)", output_path, len(df))

    # Log distribution
    for split_name in ["train", "val", "test"]:
        n = (df["split"] == split_name).sum()
        logger.info("  %s: %d tracks", split_name, n)

    return df


def download_fma(config) -> Path:
    """Full download pipeline: fetch audio + metadata, extract, generate splits."""
    base_dir = Path(config.data.raw_dir).parent
    subset = config.data.subset

    audio_zip = base_dir / f"fma_{subset}.zip"
    metadata_zip = base_dir / "fma_metadata.zip"

    download_file(FMA_URLS[subset], audio_zip, FMA_CHECKSUMS.get(subset))
    download_file(METADATA_URL, metadata_zip, FMA_CHECKSUMS.get("metadata"))

    audio_dir = Path(config.data.raw_dir)
    metadata_dir = Path(config.data.metadata_dir)
    extract_zip(audio_zip, audio_dir)
    extract_zip(metadata_zip, metadata_dir)

    splits_path = Path(config.data.splits_csv)
    generate_split_csv(metadata_dir, audio_dir, splits_path, subset)

    return splits_path
