#!/usr/bin/env python3
"""Train the DeepSync genre classifier."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_dataloaders
from src.models.deepsync import DeepSyncClassifier
from src.training.trainer import train
from src.utils.config import load_config
from src.utils.seed import save_run_metadata, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train DeepSync classifier")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from (default: None)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    checkpoint_dir = Path(config.checkpoint_dir)
    save_run_metadata(args.config, seed, checkpoint_dir)

    train_loader, val_loader, _, label_map = get_dataloaders(config)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

    model = DeepSyncClassifier.from_config(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: phase={config.model.phase}, params={n_params:,}")

    resume_path = args.resume
    if resume_path:
        print(f"Resuming from: {resume_path}")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        checkpoint_dir=checkpoint_dir,
        label_names=label_names,
        resume_checkpoint=resume_path,
    )

    print(f"\nTraining complete.")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best val accuracy: {history['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
