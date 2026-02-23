#!/usr/bin/env python3
"""Evaluate a trained DeepSync model on the test set."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.dataset import get_dataloaders
from src.models.deepsync import DeepSyncClassifier
from src.training.metrics import compute_metrics, measure_latency, save_confusion_matrix
from src.training.trainer import evaluate
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSync classifier")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: checkpoints/best_model.pt)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    set_seed(config.seed)

    checkpoint_path = Path(args.checkpoint or f"{config.checkpoint_dir}/best_model.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    _, _, test_loader, label_map = get_dataloaders(config)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

    model = DeepSyncClassifier.from_config(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, y_true, y_pred, y_prob = evaluate(
        model, test_loader, criterion, phase=config.model.phase,
    )

    metrics = compute_metrics(y_true, y_pred, y_prob, label_names)

    print(f"\n{'='*50}")
    print(f"Test Results (Phase {config.model.phase})")
    print(f"{'='*50}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    if "top3_accuracy" in metrics:
        print(f"Top-3 Acc:     {metrics['top3_accuracy']:.4f}")
    print(f"F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"\n{metrics.get('classification_report', '')}")

    # Save confusion matrix
    cm_path = Path(config.checkpoint_dir) / "confusion_matrix.png"
    save_confusion_matrix(metrics["confusion_matrix"], label_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Measure latency
    phase = config.model.phase
    sample_batch = next(iter(test_loader))
    if phase >= 2:
        sample_mel = sample_batch[0][:1]
        sample_cqt = sample_batch[1][:1]
        sample_input = (sample_mel, sample_cqt)
    else:
        sample_mel = sample_batch[0][:1]
        sample_input = (sample_mel,)

    latency = measure_latency(model, sample_input)
    print(f"\nLatency (model forward, single sample):")
    print(f"  Mean: {latency['model_forward_mean_ms']:.2f} ms")
    print(f"  P50:  {latency['model_forward_p50_ms']:.2f} ms")
    print(f"  P95:  {latency['model_forward_p95_ms']:.2f} ms")

    # Save all results
    results = {
        "test_loss": test_loss,
        "accuracy": metrics["accuracy"],
        "top3_accuracy": metrics.get("top3_accuracy"),
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        **latency,
    }
    results_path = Path(config.checkpoint_dir) / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
