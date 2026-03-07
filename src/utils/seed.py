"""Reproducibility utilities: global seeding, deterministic flags, run metadata."""

import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set global seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _config_hash(config_path: str | Path) -> str:
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def save_run_metadata(
    config_path: str | Path,
    seed: int,
    output_dir: str | Path,
    phase: int = 1,
    timestamp: str | None = None,
    extra: dict | None = None,
):
    """Save reproducibility metadata for a training run."""
    from src.utils.naming import result_filename

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "seed": seed,
        "phase": phase,
        "config_hash": _config_hash(config_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    if extra:
        meta.update(extra)

    filename = result_filename("run_metadata", "json", phase, timestamp)
    out_path = output_dir / filename
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    return out_path
