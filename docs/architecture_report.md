# Deep-Sync Audio Classifier: Architecture and Experiment Report

## 1. Overview

Deep-Sync is a multimodal deep learning system for music genre classification. It fuses temporal (Log-Mel Spectrogram) and spectral (Constant-Q Transform) features through parallel CNN streams, with an optional hierarchical attention mechanism for temporal weighting. The system is designed for CPU-first training and follows a phased development approach where each architectural addition must demonstrate measurable improvement before the next is introduced.

**Dataset:** FMA Small (Free Music Archive) — 8,000 tracks, 8 genres, 30 seconds each at 44.1 kHz.

**Framework:** PyTorch (CPU).

---

## 2. Dataset and Splits

### 2.1 FMA Small

| Property | Value |
|---|---|
| Total tracks | 8,000 |
| Genres | 8 (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock) |
| Tracks per genre | 1,000 (perfectly balanced) |
| Audio format | MP3, 30 seconds, 44.1 kHz |
| Source | [Free Music Archive (GitHub)](https://github.com/mdeff/fma) |

### 2.2 Split Protocol

Splits follow the canonical FMA `set` column from `tracks.csv`, ensuring reproducibility and no data leakage:

| Split | Tracks | Purpose |
|---|---|---|
| Train | 6,400 (80%) | Model training with random temporal cropping |
| Validation | 800 (10%) | Hyperparameter tuning, early stopping, model selection |
| Test | 800 (10%) | Final evaluation (center crop only) |

The resolved split is saved as `configs/splits/small_split.csv` with columns `(track_id, split, genre)` so any collaborator gets identical partitions without re-downloading metadata.

---

## 3. Data Engineering Pipeline

### 3.1 Audio Preprocessing

Each track passes through a sequential pipeline, with processed waveforms cached as `.npy` files:

1. **Load**: `torchaudio.load()` with automatic fallback to `librosa.load()` for corrupted MP3s. Tracks that fail both loaders are logged to `data/cache/skipped_tracks.log` and skipped (3 out of 8,000 tracks were skipped).

2. **Mono conversion**: Multi-channel audio is averaged to a single channel.

3. **Resampling**: Downsampled from 44.1 kHz to 22,050 Hz via `torchaudio.transforms.Resample`, which includes a built-in anti-aliasing filter. No separate low-pass filter is applied, as the resampler's anti-aliasing is sufficient and an additional LPF could remove useful high-frequency cues.

4. **Gate truncation**: Leading and trailing silence is removed using `librosa.effects.trim` with a -60 dB threshold.

5. **Peak normalization**: Waveform is scaled to [-1, 1] by dividing by its peak absolute value.

6. **Segmentation**: Waveform is padded (zero-fill) or truncated to exactly 30 seconds (661,500 samples at 22,050 Hz). The temporal crop position is determined at training time, not during preprocessing.

### 3.2 Feature Extraction

Features are **pre-cached to disk** rather than computed on-the-fly, which is critical for CPU training where CQT computation would otherwise dominate training time.

#### Path A: Log-Mel Spectrogram

Captures timbral density and percussive transients.

| Parameter | Value |
|---|---|
| Number of Mel bands | 128 |
| FFT window size | 2,048 samples (~93 ms) |
| Hop length | 512 samples (~23 ms) |
| Output shape | (128, 1,292) per track |
| Normalization | Per-spectrogram zero mean, unit variance |
| Library | `librosa.feature.melspectrogram` → `librosa.power_to_db` |

#### Path B: Constant-Q Transform (CQT)

Utilizes logarithmic frequency bins for pitch and harmonic detection — particularly effective for distinguishing genres with distinct tonal characteristics.

| Parameter | Value |
|---|---|
| Number of CQT bins | 84 (7 octaves × 12 bins/octave) |
| Hop length | 512 samples (~23 ms) |
| Output shape | (84, 1,292) per track |
| Normalization | Per-spectrogram zero mean, unit variance |
| Library | `librosa.cqt` → `librosa.amplitude_to_db` |

### 3.3 Data Loading and Augmentation

The `FMADataset` PyTorch Dataset class loads pre-cached `.npy` spectrograms and applies temporal cropping:

- **Training**: Random temporal crop — a random 1,292-frame window is selected from the spectrogram, exposing the model to different temporal regions across epochs. This acts as a form of data augmentation and avoids bias toward any fixed section (e.g., intro vs. chorus).

- **Validation/Test**: Center crop — deterministic cropping from the middle of the spectrogram for consistent evaluation.

- **In-memory preloading**: When enabled (`data.preload: true`), all spectrogram files are loaded into RAM at startup, eliminating per-batch disk I/O. With ~8 GB of cached features and 16 GB system RAM, this cuts epoch time roughly in half.

---

## 4. Model Architecture

The architecture follows a phased design where each phase adds a component on top of the previous one:

### 4.1 CNN Backbone (`CNNBackbone`)

A lightweight 3-block convolutional neural network, shared across all phases and used independently for each input stream. Each block consists of:

```
Conv2d(3×3, padding=1) → GroupNorm(8 groups) → ReLU → MaxPool2d(2×2)
```

| Block | Input Channels | Output Channels |
|---|---|---|
| Block 1 | 1 | 32 |
| Block 2 | 32 | 64 |
| Block 3 | 64 | 128 |

After the convolutional blocks, `AdaptiveAvgPool2d((1, None))` collapses the frequency axis to 1 while preserving the temporal dimension, producing output shape `(B, 128, T')` where `T' = T // 8` (each MaxPool halves the dimension, and there are 3 blocks).

**GroupNorm vs. BatchNorm**: GroupNorm with 8 groups was chosen over BatchNorm because BatchNorm statistics become unreliable with small batch sizes (batch_size=16). GroupNorm computes normalization statistics within channel groups of each sample independently, providing stable training regardless of batch size.

### 4.2 Phase 1: Mel-only Baseline

The simplest architecture — a single CNN stream processing only the Log-Mel spectrogram.

```
Input: Mel Spectrogram (B, 1, 128, T)
         │
    CNN Backbone
    (B, 128, T')
         │
  Global Avg Pool (temporal)
    (B, 128)
         │
  FC(128 → 64) → ReLU → Dropout(0.3)
         │
  FC(64 → 8)
         │
    Output: logits (B, 8)
```

**Parameters: 101,896**

### 4.3 Phase 2: Dual-Stream Fusion

Adds a second, independent CNN backbone for the CQT spectrogram. The two streams are concatenated before the classification head.

```
Mel (B, 1, 128, T)          CQT (B, 1, 84, T)
       │                            │
  CNN Backbone A              CNN Backbone B
  (B, 128, T')               (B, 128, T')
       │                            │
       └──── Concat on channel ─────┘
                    │
              (B, 256, T')
                    │
          Global Avg Pool (temporal)
              (B, 256)
                    │
          FC(256 → 64) → ReLU → Dropout(0.3)
                    │
          FC(64 → 8)
                    │
             Output: logits (B, 8)
```

The two CNN backbones have **independent weights** — they are not shared. This allows each stream to learn feature representations tailored to its input modality (timbral vs. harmonic).

Temporal dimension alignment is handled explicitly: if the Mel and CQT streams produce slightly different temporal lengths (due to differing frequency dimensions), the longer one is truncated to match.

**Parameters: 203,208** (~2× Phase 1)

### 4.4 Phase 3: Temporal Attention

Replaces the Global Average Pool in Phase 2 with a learned `TemporalAttention` module that can focus on genre-defining segments (e.g., a distinctive chorus or rhythmic pattern) rather than treating all temporal positions equally.

```
Mel (B, 1, 128, T)          CQT (B, 1, 84, T)
       │                            │
  CNN Backbone A              CNN Backbone B
  (B, 128, T')               (B, 128, T')
       │                            │
       └──── Concat on channel ─────┘
                    │
              (B, 256, T')
                    │
              Permute → (B, T', 256)
                    │
         ┌─── Temporal Attention ───┐
         │                          │
         │  Linear(256 → 128)       │
         │       → Tanh             │
         │  Linear(128 → 1)         │
         │       → Softmax          │
         │                          │
         │  Weighted sum: (B, 256)  │
         │  + Attention weights     │
         └──────────────────────────┘
                    │
          FC(256 → 64) → ReLU → Dropout(0.3)
                    │
          FC(64 → 8)
                    │
             Output: logits (B, 8)
```

The attention mechanism computes a learned importance score for each temporal position:
1. **Project**: Each temporal frame `(D=256)` is projected to a lower-dimensional attention space `(attention_dim=128)` and passed through `Tanh`.
2. **Score**: A linear layer maps each projected frame to a scalar score.
3. **Normalize**: Softmax over the temporal dimension produces attention weights that sum to 1.
4. **Aggregate**: The output is the weighted sum of all temporal frames, producing a single `(B, 256)` vector.

The attention weights are stored and can be extracted via `model.get_attention_weights()` for visualization — allowing analysis of which temporal segments the model considers most discriminative for each genre.

**Parameters: 236,232** (~2.3× Phase 1, ~1.16× Phase 2)

---

## 5. Training Configuration

### 5.1 Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW | Weight decay decoupled from gradient updates; good default for CNNs |
| Learning rate | 1×10⁻³ | Standard starting point for Adam variants |
| Weight decay | 1×10⁻⁴ | Light L2 regularization to prevent overfitting |
| Batch size | 16 | Fits comfortably in CPU memory; GroupNorm ensures stability |
| Max epochs | 50 | Sufficient for convergence on FMA Small |
| LR scheduler | ReduceLROnPlateau (patience=5, factor=0.5) | Halves LR when val loss plateaus for 5 epochs |
| Early stopping | Patience = 10 epochs | Stops training when val loss shows no improvement |
| Loss function | CrossEntropyLoss with class weights | Inverse-frequency weights computed from training set; handles any class imbalance |
| Dropout | 0.3 | Applied in the classification head before the output layer |

### 5.2 Reproducibility

- **Global seeding**: `torch.manual_seed`, `numpy.random.seed`, `random.seed`, and `torch.use_deterministic_algorithms(True)` are set at the start of every run.
- **Config hashing**: A SHA-256 hash of the full configuration is stored with each run.
- **Run metadata**: Seed, config hash, Python/PyTorch versions, timestamp, and phase number are saved to `run_metadata_phase{N}_{timestamp}.json`.
- **Deterministic splits**: The split CSV is version-controlled, guaranteeing identical train/val/test partitions.

### 5.3 Checkpointing and File Naming

All generated files follow the naming convention `{base}_phase{N}_{YYYYMMDD_HHMMSS}.{ext}` to prevent cross-phase overwrites:

- `best_model_phase1_20260306_162301.pt` — saved whenever validation accuracy improves
- `training_history_phase1_20260306_164347.json` — full per-epoch metrics
- `test_results_phase1_20260306_173457.json` — test-set evaluation
- `confusion_matrix_phase1_20260306_173454.png`
- `benchmark_results_phase1_20260306_174206.json`
- Figures: `loss_curves_phase1.png`, `accuracy_f1_phase1.png`, etc.

---

## 6. Evaluation Metrics

### 6.1 Primary KPIs

| Metric | Description |
|---|---|
| **Accuracy** | Overall classification accuracy on the test set |
| **Top-3 Accuracy** | Fraction of samples where the true label is in the top 3 predictions |
| **F1 (macro)** | Unweighted mean of per-class F1 scores — treats all genres equally |
| **F1 (weighted)** | Per-class F1 weighted by class support |
| **Per-class Precision/Recall/F1** | Detailed breakdown via `classification_report` |
| **Confusion Matrix** | Full 8×8 matrix showing prediction patterns |

### 6.2 Latency Metrics

Latency is measured separately for feature extraction and model inference:

| Metric | Description |
|---|---|
| Feature extraction (Mel) | Time to compute Log-Mel from raw waveform |
| Feature extraction (CQT) | Time to compute CQT from raw waveform (Phase 2+) |
| Model forward pass | Time for a single-sample inference through the model |

All latency measurements use 100 timed runs after 10 warmup runs, reporting mean, P50, and P95.

---

## 7. Results

### 7.1 Phase 1: Mel-only CNN Baseline

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 59.3% | 49.3% |
| Top-3 Accuracy | — | 80.5% |
| F1 (macro) | — | 0.480 |
| F1 (weighted) | — | 0.480 |
| Best epoch | 48 / 50 | — |
| Parameters | 101,896 | — |
| Training time | ~15 hours (50 epochs × ~17 min/epoch) | — |

**Latency (Phase 1):**

| Component | Mean | P50 | P95 |
|---|---|---|---|
| Mel extraction | 16.6 ms | 16.2 ms | 18.6 ms |
| Model forward | 14.5 ms | 14.2 ms | 17.2 ms |
| **Total pipeline** | **31.2 ms** | — | — |

### 7.2 Phase 2: Dual-Stream Fusion (Mel + CQT)

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 55.3% | 48.0% |
| Top-3 Accuracy | — | 79.4% |
| F1 (macro) | 0.548 | 0.471 |
| F1 (weighted) | — | 0.471 |
| Best epoch | 43 / 50 | — |
| Parameters | 203,208 | — |
| Training time | ~14 hours (50 epochs × ~17 min/epoch) | — |

**Latency (Phase 2):**

| Component | Mean | P50 | P95 |
|---|---|---|---|
| Model forward | 37.6 ms | 37.5 ms | 40.1 ms |

### 7.3 Phase Comparison

| Metric | Phase 1 (Mel-only) | Phase 2 (Dual-stream) | Delta |
|---|---|---|---|
| Val Accuracy | **59.3%** | 55.3% | -4.0% |
| Test Accuracy | **49.3%** | 48.0% | -1.3% |
| Test F1 (macro) | **0.480** | 0.471 | -0.009 |
| Test Top-3 Acc | **80.5%** | 79.4% | -1.1% |
| Parameters | 101,896 | 203,208 | +99% |
| Forward latency | **14.5 ms** | 37.6 ms | +159% |

**Analysis**: Phase 2 did not improve over the Phase 1 baseline on the test set despite having twice the parameters. The validation accuracy was lower than Phase 1 (55.3% vs. 59.3%), and this gap widened on the test set. Both phases show a significant train-to-test generalization gap, suggesting that the model may benefit from stronger regularization or data augmentation. The CQT stream added computational cost without a corresponding accuracy gain, possibly because the 8-genre FMA Small task does not require the harmonic resolution that CQT provides — the Mel spectrogram may already capture sufficient discriminative information for these genres.

### 7.4 Phase 3: Temporal Attention (pending)

Phase 3 training has not yet been conducted. The temporal attention mechanism adds 33,024 parameters (256→128 projection + 128→1 scoring) and replaces the global average pool, allowing the model to learn which temporal segments are most relevant for genre classification.

---

## 8. Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Normalization layer | GroupNorm (8 groups) | Stable with small CPU batch sizes (16), unlike BatchNorm |
| Feature caching | Pre-compute to `.npy` on disk | CQT is too expensive for on-the-fly computation on CPU |
| In-memory preloading | Optional (config flag) | Eliminates disk I/O bottleneck; halves epoch time |
| Crop strategy | Random (train) / center (eval) | Avoids missing genre-defining sections; acts as augmentation |
| LPF after resample | Removed | `torchaudio.Resample` includes anti-aliasing; extra LPF may discard useful cues |
| Split protocol | FMA metadata `set` column + saved CSV | Reproducible, no leakage, standard FMA protocol |
| Phased model | Phase 1→2→3 gated progression | Proves each component adds value before adding complexity |
| Loss function | CrossEntropyLoss with class weights | Handles potential class imbalance; standard for classification |
| ROC-AUC | Deferred | Noisy in multiclass setups; F1 + confusion matrix are more actionable |
| Error handling | Skip + log corrupted tracks | Pipeline never crashes; issues are auditable |

---

## 9. Project Structure

```
audio-final-project/
├── configs/
│   ├── default.yaml                         # All hyperparameters
│   └── splits/
│       └── small_split.csv                  # Deterministic train/val/test IDs
├── src/
│   ├── data/
│   │   ├── download.py                      # FMA download + checksum verification
│   │   ├── preprocessing.py                 # Trim, normalize, segment
│   │   ├── features.py                      # Mel + CQT extraction (pre-cache)
│   │   └── dataset.py                       # PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── backbone.py                      # Lightweight CNN (GroupNorm)
│   │   ├── attention.py                     # Temporal attention (Phase 3)
│   │   └── deepsync.py                      # Full model: phases 1/2/3
│   ├── training/
│   │   ├── trainer.py                       # Training loop, early stopping
│   │   ├── metrics.py                       # Accuracy, F1, confusion matrix, latency
│   │   └── visualize.py                     # Plot generation
│   └── utils/
│       ├── config.py                        # YAML config loader
│       ├── seed.py                          # Reproducibility + run metadata
│       └── naming.py                        # Phase-safe file naming
├── scripts/
│   ├── download.py                          # Download FMA dataset
│   ├── preprocess.py                        # Preprocess + extract features
│   ├── train.py                             # Train model
│   ├── evaluate.py                          # Evaluate on test set
│   ├── benchmark.py                         # Latency profiling
│   └── visualize.py                         # Regenerate plots
├── tests/
│   ├── test_features.py                     # Feature shape/NaN checks
│   ├── test_dataset.py                      # Dataset contract tests
│   └── test_model.py                        # Forward-pass smoke tests
├── checkpoints/                             # Saved models, results, figures
├── requirements.txt
└── README.md
```

---

## 10. Reproducibility Checklist

- [x] Fixed random seeds (Python, NumPy, PyTorch)
- [x] Deterministic PyTorch operations enabled
- [x] Split CSV version-controlled
- [x] Config hashed and logged per run
- [x] Environment info (Python/PyTorch versions) recorded
- [x] Phase-safe file naming prevents accidental overwrites
- [x] Corrupted tracks logged, not silently dropped
- [x] All dependencies pinned in `requirements.txt`
