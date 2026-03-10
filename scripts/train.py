#!/usr/bin/env python3
"""Train the DeepSync genre classifier."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import get_dataloaders
from src.models.deepsync import DeepSyncClassifier
from src.training.trainer import train
from src.training.visualize import generate_training_plots
from src.utils.config import load_config
from src.utils.naming import result_filename
from src.utils.seed import save_run_metadata, set_seed


def _load_phase2_weights(model, checkpoint_path: str) -> int:
    """Transfer backbone/classifier weights from a Phase 2 checkpoint into a
    Phase 3 model.  Attention parameters (which don't exist in Phase 2) keep
    their fresh initialization.  Returns the number of transferred keys."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    phase2_state = ckpt["model_state_dict"]
    phase3_state = model.state_dict()
    transferred = 0
    for key in phase2_state:
        if key in phase3_state:
            phase3_state[key] = phase2_state[key]
            transferred += 1
    model.load_state_dict(phase3_state)
    return transferred


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
    parser.add_argument(
        "--from-phase2", type=str, default=None,
        help="Path to a Phase 2 checkpoint.  Transfers backbone and classifier "
             "weights into the Phase 3 model (attention starts fresh).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.seed
    phase = config.model.phase
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_seed(seed)

    checkpoint_dir = Path(config.checkpoint_dir)
    save_run_metadata(args.config, seed, checkpoint_dir, phase=phase, timestamp=run_ts)

    train_loader, val_loader, _, label_map = get_dataloaders(config)
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

    model = DeepSyncClassifier.from_config(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: phase={phase}, params={n_params:,}")

    if args.from_phase2:
        n = _load_phase2_weights(model, args.from_phase2)
        print(f"Transferred {n} parameter tensors from Phase 2 checkpoint")

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
    print(f"Checkpoint: {history.get('checkpoint_filename', 'N/A')}")

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"] + 1

    figures_dir = checkpoint_dir / "figures"
    generate_training_plots(history, start_epoch, figures_dir, phase=phase)
    print(f"Training plots saved to {figures_dir}/")


if __name__ == "__main__":
    main()
