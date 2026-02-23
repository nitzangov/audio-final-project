#!/usr/bin/env python3
"""Run the full preprocessing and feature extraction pipeline."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.features import extract_features_for_dataset
from src.data.preprocessing import preprocess_dataset
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Preprocess FMA audio + extract features")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--extract-cqt", action="store_true",
        help="Also extract CQT features (Phase 2+)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    splits_csv = Path(config.data.splits_csv)
    if not splits_csv.exists():
        print(f"ERROR: Split CSV not found at {splits_csv}. Run scripts/download.py first.")
        sys.exit(1)

    audio_dir = Path(config.data.raw_dir)
    cache_dir = Path(config.data.cache_dir)

    # Step 1: Preprocess waveforms
    print("=== Step 1: Preprocessing waveforms ===")
    n_proc, n_skip = preprocess_dataset(
        split_csv=splits_csv,
        audio_dir=audio_dir,
        cache_dir=cache_dir,
        target_sr=config.data.sample_rate,
        duration_sec=config.data.duration_sec,
    )
    print(f"Preprocessed: {n_proc}, Skipped: {n_skip}")

    # Step 2: Extract features
    print("\n=== Step 2: Extracting features ===")
    mel_params = {
        "n_mels": config.features.mel.n_mels,
        "n_fft": config.features.mel.n_fft,
        "hop_length": config.features.mel.hop_length,
    }
    cqt_params = {
        "n_bins": config.features.cqt.n_bins,
        "hop_length": config.features.cqt.hop_length,
    }

    n_feat, n_skip_feat = extract_features_for_dataset(
        split_csv=splits_csv,
        cache_dir=cache_dir,
        sr=config.data.sample_rate,
        mel_params=mel_params,
        cqt_params=cqt_params,
        extract_cqt_features=args.extract_cqt,
    )
    print(f"Features extracted: {n_feat}, Skipped: {n_skip_feat}")
    print("\nDone.")


if __name__ == "__main__":
    main()
