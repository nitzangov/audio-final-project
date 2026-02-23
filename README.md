# Deep-Sync Audio Classifier

Multimodal deep learning model for music genre classification using the Free Music Archive (FMA) dataset. Dual-input architecture fusing temporal (Mel spectrogram) and spectral (CQT) features through parallel CNN streams with hierarchical temporal attention.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Create virtual environment
uv venv .venv --python 3.11

# Activate it
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Running the Pipeline

All scripts accept `--config path/to/config.yaml` (defaults to `configs/default.yaml`).

### Step 1: Download FMA dataset

Downloads FMA Small (~7.2 GB), extracts audio and metadata, and generates the canonical train/val/test split CSV.

```bash
python scripts/download.py
```

Output: `data/fma_small/` (audio), `data/fma_metadata/` (metadata), `configs/splits/small_split.csv` (split file).

### Step 2: Preprocess audio and extract features

Loads each track, converts to mono, resamples to 22,050 Hz, trims silence, normalizes, segments to 30 seconds, then extracts Log-Mel spectrograms. All results are cached to disk as `.npy` files. Corrupted or unreadable tracks are logged to `data/cache/skipped_tracks.log` and skipped.

```bash
# Phase 1 (Mel-only)
python scripts/preprocess.py

# Phase 2+ (also extract CQT features)
python scripts/preprocess.py --extract-cqt
```

Output: `data/cache/waveforms/`, `data/cache/mel/`, and optionally `data/cache/cqt/`.

### Step 3: Train

Trains the model with AdamW, learning rate scheduling, class-weighted loss, and early stopping. Saves the best checkpoint by validation accuracy.

```bash
python scripts/train.py
```

To override the random seed:

```bash
python scripts/train.py --seed 123
```

Output: `checkpoints/best_model.pt`, `checkpoints/training_history.json`, `checkpoints/run_metadata.json`.

### Step 4: Evaluate

Loads the best checkpoint and evaluates on the test set. Reports accuracy, top-3 accuracy, F1 scores, per-class breakdown, and inference latency. Saves a confusion matrix heatmap.

```bash
python scripts/evaluate.py
```

To evaluate a specific checkpoint:

```bash
python scripts/evaluate.py --checkpoint path/to/model.pt
```

Output: `checkpoints/test_results.json`, `checkpoints/confusion_matrix.png`.

### Step 5: Benchmark latency

Profiles feature extraction (Mel and CQT) and model forward-pass latency separately, reporting mean/p50/p95 timings.

```bash
python scripts/benchmark.py
```

Output: `checkpoints/benchmark_results.json`.

### Step 6: Run tests

```bash
python -m pytest tests/ -v
```

## Project Phases

The model architecture is controlled by `model.phase` in `configs/default.yaml`:

| Phase | Architecture | What changes |
|-------|-------------|--------------|
| 1 (MVP) | Mel-only CNN | Single backbone, global average pooling |
| 2 | Dual-stream (Mel + CQT) | Two parallel CNN backbones, concatenation fusion |
| 3 | Dual-stream + Attention | Temporal attention replaces global average pooling |

Each phase should demonstrate measurable improvement over the previous one before advancing.

**To move between phases:**

1. **Phase 1 -> 2**: Run `python scripts/preprocess.py --extract-cqt`, then set `model.phase: 2` and `model.num_classes: 8` in the config.
2. **Phase 2 -> 3**: Set `model.phase: 3` in the config. No additional preprocessing needed.

## Project Structure

```
audio-final-project/
├── configs/
│   ├── default.yaml              # All hyperparameters
│   └── splits/                   # Generated train/val/test split CSVs
├── src/
│   ├── data/
│   │   ├── download.py           # FMA download + checksum + metadata
│   │   ├── preprocessing.py      # Audio preprocessing pipeline
│   │   ├── features.py           # Mel + CQT feature extraction
│   │   └── dataset.py            # PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── backbone.py           # Lightweight CNN with GroupNorm
│   │   ├── attention.py          # Temporal attention mechanism
│   │   └── deepsync.py           # Full multi-phase classifier
│   ├── training/
│   │   ├── trainer.py            # Training loop + early stopping
│   │   └── metrics.py            # Evaluation metrics + latency profiling
│   └── utils/
│       ├── config.py             # YAML config loader
│       └── seed.py               # Reproducibility utilities
├── scripts/
│   ├── download.py               # Entry point: download FMA
│   ├── preprocess.py             # Entry point: preprocess + extract features
│   ├── train.py                  # Entry point: train model
│   ├── evaluate.py               # Entry point: evaluate on test set
│   └── benchmark.py              # Entry point: latency profiler
├── tests/
│   ├── test_features.py          # Feature extraction smoke tests
│   ├── test_dataset.py           # Dataset contract tests
│   └── test_model.py             # Model forward-pass smoke tests
└── requirements.txt
```
