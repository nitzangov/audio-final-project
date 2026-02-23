#!/usr/bin/env python3
"""Download the FMA dataset and generate train/val/test split CSV."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.download import download_fma
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Download FMA dataset")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    splits_path = download_fma(config)
    print(f"Done. Split CSV saved to: {splits_path}")


if __name__ == "__main__":
    main()
