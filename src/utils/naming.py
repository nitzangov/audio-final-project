"""Consistent naming for generated result files: phase tag + timestamp."""

import glob
from datetime import datetime
from pathlib import Path


def result_filename(base: str, ext: str, phase: int, timestamp: str | None = None) -> str:
    """Generate a result filename with phase and timestamp.

    Example: result_filename("best_model", "pt", 1) -> "best_model_phase1_20260306_163000.pt"
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_phase{phase}_{timestamp}.{ext}"


def find_latest(directory: str | Path, base: str, ext: str, phase: int | None = None) -> Path | None:
    """Find the most recently modified file matching the naming pattern.

    If phase is None, matches any phase.
    """
    directory = Path(directory)
    if phase is not None:
        pattern = str(directory / f"{base}_phase{phase}_*.{ext}")
    else:
        pattern = str(directory / f"{base}_phase*_*.{ext}")

    matches = glob.glob(pattern)
    if not matches:
        return None
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))
