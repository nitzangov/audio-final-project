"""Training loop with early stopping, checkpointing, and logging."""

import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


def compute_class_weights(dataloader: DataLoader) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training set."""
    counts: Counter = Counter()
    for batch in dataloader:
        labels = batch[-1]
        counts.update(labels.tolist())

    num_classes = max(counts.keys()) + 1
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls, count in counts.items():
        weights[cls] = total / (num_classes * count)
    return weights


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    phase: int = 1,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if phase >= 2:
            mel, cqt, labels = batch
            logits = model(mel, cqt)
        else:
            mel, labels = batch
            logits = model(mel)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    phase: int = 1,
) -> tuple[float, list[int], list[int], np.ndarray]:
    """Evaluate the model. Returns (avg_loss, y_true, y_pred, y_prob)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true = []
    all_pred = []
    all_prob = []

    for batch in dataloader:
        if phase >= 2:
            mel, cqt, labels = batch
            logits = model(mel, cqt)
        else:
            mel, labels = batch
            logits = model(mel)

        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_true.extend(labels.tolist())
        all_pred.extend(preds.tolist())
        all_prob.append(probs.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    y_prob = np.concatenate(all_prob, axis=0) if all_prob else np.array([])
    return avg_loss, all_true, all_pred, y_prob


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    checkpoint_dir: str | Path = "checkpoints",
    label_names: list[str] | None = None,
) -> dict:
    """Full training loop with early stopping and checkpointing.

    Returns a dict of training history and best metrics.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    phase = config.model.phase

    class_weights = compute_class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.training.scheduler_patience,
        factor=config.training.scheduler_factor,
    )
    early_stopping = EarlyStopping(patience=config.training.early_stop_patience)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_macro": [],
        "lr": [],
    }
    best_val_acc = 0.0
    best_epoch = 0

    logger.info("Starting training: %d epochs, batch_size=%d, phase=%d",
                config.training.epochs, config.training.batch_size, phase)

    for epoch in range(1, config.training.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, phase)
        val_loss, y_true, y_pred, y_prob = evaluate(model, val_loader, criterion, phase)

        metrics = compute_metrics(y_true, y_pred, y_prob, label_names)
        val_acc = metrics["accuracy"]
        val_f1 = metrics["f1_macro"]
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1_macro"].append(val_f1)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d [%.1fs] - train_loss: %.4f, val_loss: %.4f, "
            "val_acc: %.4f, val_f1: %.4f, lr: %.2e",
            epoch, config.training.epochs, elapsed,
            train_loss, val_loss, val_acc, val_f1, current_lr,
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1,
                "config": config.to_dict(),
            }, checkpoint_dir / "best_model.pt")
            logger.info("  -> New best model saved (acc=%.4f)", val_acc)

        scheduler.step(val_loss)

        if early_stopping.step(val_loss):
            logger.info("Early stopping at epoch %d", epoch)
            break

    history["best_epoch"] = best_epoch
    history["best_val_accuracy"] = best_val_acc

    # Save training history
    with open(checkpoint_dir / "training_history.json", "w") as f:
        serializable = {k: v for k, v in history.items()}
        json.dump(serializable, f, indent=2)

    return history
