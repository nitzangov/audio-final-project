#!/usr/bin/env python3
"""Benchmark throughput and latency for feature extraction and model inference."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data.features import extract_mel, extract_cqt
from src.models.deepsync import DeepSyncClassifier
from src.training.metrics import measure_latency
from src.utils.config import load_config


def benchmark_feature_extraction(config, n_runs: int = 50):
    """Measure feature extraction latency on a synthetic waveform."""
    sr = config.data.sample_rate
    duration = config.data.duration_sec
    y = np.random.randn(int(sr * duration)).astype(np.float32)

    mel_params = {
        "sr": sr,
        "n_mels": config.features.mel.n_mels,
        "n_fft": config.features.mel.n_fft,
        "hop_length": config.features.mel.hop_length,
    }
    cqt_params = {
        "sr": sr,
        "n_bins": config.features.cqt.n_bins,
        "hop_length": config.features.cqt.hop_length,
    }

    # Warmup
    for _ in range(5):
        extract_mel(y, **mel_params)

    mel_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        extract_mel(y, **mel_params)
        mel_times.append((time.perf_counter() - t0) * 1000)

    # Warmup CQT
    for _ in range(3):
        extract_cqt(y, **cqt_params)

    cqt_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        extract_cqt(y, **cqt_params)
        cqt_times.append((time.perf_counter() - t0) * 1000)

    return {
        "mel_extraction": {
            "mean_ms": float(np.mean(mel_times)),
            "p50_ms": float(np.percentile(mel_times, 50)),
            "p95_ms": float(np.percentile(mel_times, 95)),
        },
        "cqt_extraction": {
            "mean_ms": float(np.mean(cqt_times)),
            "p50_ms": float(np.percentile(cqt_times, 50)),
            "p95_ms": float(np.percentile(cqt_times, 95)),
        },
    }


def benchmark_model_forward(config, n_runs: int = 100):
    """Measure model forward-pass latency with random input."""
    model = DeepSyncClassifier.from_config(config)
    model.eval()

    sr = config.data.sample_rate
    duration = config.data.duration_sec
    hop = config.features.mel.hop_length
    n_frames = int(sr * duration / hop) + 1

    mel_input = torch.randn(1, 1, config.features.mel.n_mels, n_frames)

    if config.model.phase >= 2:
        cqt_input = torch.randn(1, 1, config.features.cqt.n_bins, n_frames)
        sample_input = (mel_input, cqt_input)
    else:
        sample_input = (mel_input,)

    return measure_latency(model, sample_input, n_runs=n_runs)


def main():
    parser = argparse.ArgumentParser(description="Benchmark feature extraction and model inference")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--n-runs", type=int, default=50, help="Number of benchmark runs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    print("=" * 50)
    print("Feature Extraction Benchmark")
    print("=" * 50)
    feat_results = benchmark_feature_extraction(config, n_runs=args.n_runs)
    for feat_type, stats in feat_results.items():
        print(f"\n{feat_type}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.2f}")

    print(f"\n{'='*50}")
    print(f"Model Forward-Pass Benchmark (Phase {config.model.phase})")
    print("=" * 50)
    model_results = benchmark_model_forward(config, n_runs=args.n_runs)
    for k, v in model_results.items():
        print(f"  {k}: {v:.2f}")

    # Compute total pipeline latency estimate
    mel_ms = feat_results["mel_extraction"]["mean_ms"]
    model_ms = model_results["model_forward_mean_ms"]
    total = mel_ms + model_ms
    if config.model.phase >= 2:
        cqt_ms = feat_results["cqt_extraction"]["mean_ms"]
        total += cqt_ms

    print(f"\n{'='*50}")
    print(f"Estimated Total Pipeline Latency")
    print("=" * 50)
    print(f"  Feature extraction (Mel): {mel_ms:.2f} ms")
    if config.model.phase >= 2:
        print(f"  Feature extraction (CQT): {cqt_ms:.2f} ms")
    print(f"  Model forward:            {model_ms:.2f} ms")
    print(f"  Total:                    {total:.2f} ms")

    # Save results
    all_results = {
        "feature_extraction": feat_results,
        "model_forward": model_results,
        "total_pipeline_mean_ms": total,
        "phase": config.model.phase,
    }
    output_path = Path(config.checkpoint_dir) / "benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
