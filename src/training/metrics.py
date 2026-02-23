"""Evaluation metrics: accuracy, F1, confusion matrix, latency profiling."""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict:
    """Compute classification metrics.

    Args:
        y_true: ground-truth labels
        y_pred: predicted labels
        y_prob: (N, C) predicted probabilities (for top-k accuracy)
        label_names: ordered genre names for the report
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_prob is not None and y_prob.shape[1] >= 3:
        results["top3_accuracy"] = top_k_accuracy_score(
            y_true, y_prob, k=3, labels=list(range(y_prob.shape[1])),
        )

    if label_names:
        results["classification_report"] = classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0,
        )

    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    return results


def save_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    output_path: str | Path,
):
    """Save a confusion matrix heatmap as a PNG image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def measure_latency(
    model: torch.nn.Module,
    sample_input: tuple[torch.Tensor, ...],
    n_runs: int = 100,
    warmup: int = 10,
) -> dict:
    """Measure model forward-pass latency (CPU).

    Returns dict with mean, p50, p95 latency in milliseconds.
    """
    model.eval()
    times = []

    with torch.no_grad():
        for i in range(warmup + n_runs):
            start = time.perf_counter()
            model(*sample_input)
            elapsed = (time.perf_counter() - start) * 1000
            if i >= warmup:
                times.append(elapsed)

    times = np.array(times)
    return {
        "model_forward_mean_ms": float(times.mean()),
        "model_forward_p50_ms": float(np.percentile(times, 50)),
        "model_forward_p95_ms": float(np.percentile(times, 95)),
    }
