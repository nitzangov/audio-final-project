#!/usr/bin/env python3
"""Evaluate a trained DeepSync model on the test set."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.metrics import classification_report

from src.data.dataset import get_dataloaders
from src.models.deepsync import DeepSyncClassifier
from src.training.metrics import compute_metrics, measure_latency, save_confusion_matrix
from src.training.trainer import evaluate
from src.training.visualize import generate_eval_plots
from src.utils.config import load_config
from src.utils.naming import find_latest, result_filename
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepSync classifier")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: latest best_model for current phase)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    phase = config.model.phase
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_seed(config.seed)

    checkpoint_dir = Path(config.checkpoint_dir)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest(checkpoint_dir, "best_model", "pt", phase)
        if checkpoint_path is None:
            # Fallback to legacy name
            checkpoint_path = checkpoint_dir / "best_model.pt"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    _, _, test_loader, label_map = get_dataloaders(config)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

    model = DeepSyncClassifier.from_config(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: {checkpoint_path.name} (epoch {checkpoint['epoch']})")

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, y_true, y_pred, y_prob = evaluate(
        model, test_loader, criterion, phase=phase,
    )

    metrics = compute_metrics(y_true, y_pred, y_prob, label_names)

    print(f"\n{'='*50}")
    print(f"Test Results (Phase {phase})")
    print(f"{'='*50}")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    if "top3_accuracy" in metrics:
        print(f"Top-3 Acc:     {metrics['top3_accuracy']:.4f}")
    print(f"F1 (macro):    {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"\n{metrics.get('classification_report', '')}")

    # Save confusion matrix
    cm_name = result_filename("confusion_matrix", "png", phase, eval_ts)
    cm_path = checkpoint_dir / cm_name
    save_confusion_matrix(metrics["confusion_matrix"], label_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Measure latency
    sample_batch = next(iter(test_loader))
    if phase >= 2:
        sample_input = (sample_batch[0][:1], sample_batch[1][:1])
    else:
        sample_input = (sample_batch[0][:1],)

    latency = measure_latency(model, sample_input)
    print(f"\nLatency (model forward, single sample):")
    print(f"  Mean: {latency['model_forward_mean_ms']:.2f} ms")
    print(f"  P50:  {latency['model_forward_p50_ms']:.2f} ms")
    print(f"  P95:  {latency['model_forward_p95_ms']:.2f} ms")

    # Save results
    results = {
        "test_loss": test_loss,
        "accuracy": metrics["accuracy"],
        "top3_accuracy": metrics.get("top3_accuracy"),
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        **latency,
    }
    results_name = result_filename("test_results", "json", phase, eval_ts)
    results_path = checkpoint_dir / results_name
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    history_path = find_latest(checkpoint_dir, "training_history", "json", phase)
    if history_path is None:
        # Fallback to legacy name
        history_path = checkpoint_dir / "training_history.json"

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        per_class = classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0,
        )
        per_class = {k: v for k, v in per_class.items() if k in label_names}

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        start_epoch = max(1, checkpoint["epoch"] - len(history["train_loss"]) + 1)

        figures_dir = checkpoint_dir / "figures"
        generate_eval_plots(
            history=history,
            start_epoch=start_epoch,
            output_dir=figures_dir,
            phase=phase,
            n_params=n_params,
            test_results=results,
            per_class=per_class,
        )
        print(f"All plots saved to {figures_dir}/")
    else:
        print("Note: training_history not found, skipping plot generation.")


if __name__ == "__main__":
    main()
