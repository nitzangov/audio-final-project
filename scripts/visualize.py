#!/usr/bin/env python3
"""Standalone script to regenerate all visualizations from saved results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from sklearn.metrics import classification_report

from src.data.dataset import get_dataloaders
from src.models.deepsync import DeepSyncClassifier
from src.training.metrics import compute_metrics
from src.training.trainer import evaluate
from src.training.visualize import generate_eval_plots, generate_training_plots
from src.utils.config import load_config
from src.utils.naming import find_latest
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Generate training/evaluation visualizations")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--start-epoch", type=int, default=None,
        help="First epoch in training history (auto-detected from checkpoint if omitted)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    phase = config.model.phase
    checkpoint_dir = Path(config.checkpoint_dir)
    figures_dir = checkpoint_dir / "figures"

    history_path = find_latest(checkpoint_dir, "training_history", "json", phase)
    if history_path is None:
        history_path = checkpoint_dir / "training_history.json"
    if not history_path.exists():
        print(f"ERROR: No training_history found for phase {phase}. Run training first.")
        sys.exit(1)

    with open(history_path) as f:
        history = json.load(f)

    ckpt_path = find_latest(checkpoint_dir, "best_model", "pt", phase)
    if ckpt_path is None:
        ckpt_path = checkpoint_dir / "best_model.pt"

    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        start_epoch = max(1, ckpt["epoch"] - len(history["train_loss"]) + 1)
    else:
        start_epoch = 1

    test_results_path = find_latest(checkpoint_dir, "test_results", "json", phase)
    if test_results_path is None:
        test_results_path = checkpoint_dir / "test_results.json"

    if test_results_path.exists():
        with open(test_results_path) as f:
            test_results = json.load(f)

        set_seed(config.seed)
        _, _, test_loader, label_map = get_dataloaders(config)
        label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

        model = DeepSyncClassifier.from_config(config)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])

        criterion = torch.nn.CrossEntropyLoss()
        _, y_true, y_pred, _ = evaluate(model, test_loader, criterion, phase=phase)

        per_class = classification_report(
            y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0,
        )
        per_class = {k: v for k, v in per_class.items() if k in label_names}
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        generate_eval_plots(
            history=history,
            start_epoch=start_epoch,
            output_dir=figures_dir,
            phase=phase,
            n_params=n_params,
            test_results=test_results,
            per_class=per_class,
        )
    else:
        generate_training_plots(history, start_epoch, figures_dir, phase=phase)

    print(f"All figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
