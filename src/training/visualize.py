"""Visualization utilities for training history and evaluation results."""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

PHASE_NAMES = {1: "Phase 1 (Mel-only CNN)", 2: "Phase 2 (Dual-stream)", 3: "Phase 3 (+Attention)"}


def _fig_path(output_dir: Path, name: str, phase: int) -> Path:
    return output_dir / f"{name}_phase{phase}.png"


def plot_loss_curves(history: dict, start_epoch: int, output_dir: Path, phase: int = 1):
    epochs = list(range(start_epoch, start_epoch + len(history["train_loss"])))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    best_epoch = history.get("best_epoch")
    if best_epoch and best_epoch in epochs:
        idx = epochs.index(best_epoch)
        ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best (epoch {best_epoch})")
        ax.scatter([best_epoch], [history["val_loss"][idx]], color="red", s=80, zorder=5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_fig_path(output_dir, "loss_curves", phase), dpi=150)
    plt.close(fig)


def plot_accuracy_f1(history: dict, start_epoch: int, output_dir: Path, phase: int = 1):
    epochs = list(range(start_epoch, start_epoch + len(history["val_accuracy"])))
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_acc, color_f1 = "#2196F3", "#FF9800"
    ax1.plot(epochs, history["val_accuracy"], color=color_acc, linewidth=2, label="Val Accuracy")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12, color=color_acc)
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_f1_macro"], color=color_f1, linewidth=2, linestyle="--", label="Val F1 (macro)")
    ax2.set_ylabel("F1 Score", fontsize=12, color=color_f1)
    ax2.tick_params(axis="y", labelcolor=color_f1)
    best_epoch = history.get("best_epoch")
    if best_epoch and best_epoch in epochs:
        idx = epochs.index(best_epoch)
        ax1.scatter([best_epoch], [history["val_accuracy"][idx]], color=color_acc, s=80, zorder=5, edgecolors="black")
        ax2.scatter([best_epoch], [history["val_f1_macro"][idx]], color=color_f1, s=80, zorder=5, edgecolors="black")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc="lower right")
    ax1.set_title("Validation Accuracy & F1 Score", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_fig_path(output_dir, "accuracy_f1", phase), dpi=150)
    plt.close(fig)


def plot_learning_rate(history: dict, start_epoch: int, output_dir: Path, phase: int = 1):
    epochs = list(range(start_epoch, start_epoch + len(history["lr"])))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(epochs, history["lr"], linewidth=2, color="#4CAF50")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(_fig_path(output_dir, "learning_rate", phase), dpi=150)
    plt.close(fig)


def plot_per_class_performance(
    per_class: dict[str, dict[str, float]],
    phase: int,
    output_dir: Path,
):
    """Per-class precision/recall/F1 grouped bar chart."""
    genres = list(per_class.keys())
    precision = [per_class[g]["precision"] for g in genres]
    recall = [per_class[g]["recall"] for g in genres]
    f1 = [per_class[g]["f1-score"] for g in genres]

    x = np.arange(len(genres))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, recall, width, label="Recall", color="#FF9800", alpha=0.85)
    bars3 = ax.bar(x + width, f1, width, label="F1", color="#4CAF50", alpha=0.85)
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Class Test Performance ({PHASE_NAMES.get(phase, f'Phase {phase}')})", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(genres, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(_fig_path(output_dir, "per_class_performance", phase), dpi=150)
    plt.close(fig)


def plot_summary_dashboard(
    history: dict,
    start_epoch: int,
    output_dir: Path,
    phase: int = 1,
    n_params: int | None = None,
    test_results: dict | None = None,
    per_class: dict[str, dict[str, float]] | None = None,
):
    """Combined dashboard with all key metrics."""
    phase_name = PHASE_NAMES.get(phase, f"Phase {phase}")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Deep-Sync {phase_name}: Training Summary", fontsize=16, fontweight="bold")

    epochs = list(range(start_epoch, start_epoch + len(history["train_loss"])))
    best_epoch = history.get("best_epoch")

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], label="Val", linewidth=1.5)
    if best_epoch and best_epoch in epochs:
        ax1.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Loss", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(epochs, history["val_accuracy"], color="#2196F3", linewidth=1.5)
    if best_epoch and best_epoch in epochs:
        idx = epochs.index(best_epoch)
        ax2.scatter([best_epoch], [history["val_accuracy"][idx]], color="red", s=60, zorder=5)
        ax2.annotate(f"{history['val_accuracy'][idx]:.1%}",
                     (best_epoch, history["val_accuracy"][idx]),
                     textcoords="offset points", xytext=(10, -10), fontsize=10, fontweight="bold")
    ax2.set_title("Validation Accuracy", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    if per_class:
        genres = list(per_class.keys())
        f1_scores = [per_class[g]["f1-score"] for g in genres]
        colors = ["#4CAF50" if f > 0.5 else "#FF9800" if f > 0.3 else "#F44336" for f in f1_scores]
        bars = ax3.barh(genres, f1_scores, color=colors, alpha=0.85)
        ax3.set_xlabel("F1 Score")
        ax3.set_title("Per-Class Test F1", fontsize=12)
        ax3.set_xlim(0, 1.0)
        for bar, score in zip(bars, f1_scores):
            ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{score:.2f}", va="center", fontsize=9)
        ax3.grid(True, axis="x", alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Run evaluate.py\nfor per-class metrics",
                 ha="center", va="center", fontsize=12, transform=ax3.transAxes)
        ax3.set_title("Per-Class Test F1", fontsize=12)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    summary_lines = [
        ("Model", phase_name),
        ("Parameters", f"{n_params:,}" if n_params else "N/A"),
        ("Best Val Accuracy", f"{history.get('best_val_accuracy', 0):.1%}"),
        ("Best Epoch", str(best_epoch)),
        ("", ""),
    ]
    if test_results:
        summary_lines += [
            ("Test Accuracy", f"{test_results['accuracy']:.1%}"),
            ("Test Top-3 Accuracy", f"{test_results.get('top3_accuracy', 0):.1%}"),
            ("Test F1 (macro)", f"{test_results['f1_macro']:.4f}"),
            ("Inference Latency", f"{test_results.get('model_forward_mean_ms', 0):.1f} ms"),
        ]
    y = 0.9
    for label, value in summary_lines:
        if label:
            ax4.text(0.1, y, label + ":", fontsize=11, fontweight="bold",
                     transform=ax4.transAxes, verticalalignment="top")
            ax4.text(0.65, y, value, fontsize=11,
                     transform=ax4.transAxes, verticalalignment="top")
        y -= 0.09

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_fig_path(output_dir, "summary_dashboard", phase), dpi=150)
    plt.close(fig)


def generate_training_plots(
    history: dict,
    start_epoch: int,
    output_dir: Path,
    phase: int = 1,
):
    """Generate plots available after training (no test metrics needed)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating training plots...")
    plot_loss_curves(history, start_epoch, output_dir, phase)
    plot_accuracy_f1(history, start_epoch, output_dir, phase)
    plot_learning_rate(history, start_epoch, output_dir, phase)
    logger.info("Training plots saved to %s", output_dir)


def generate_eval_plots(
    history: dict,
    start_epoch: int,
    output_dir: Path,
    phase: int,
    n_params: int | None,
    test_results: dict,
    per_class: dict[str, dict[str, float]],
):
    """Generate all plots including test evaluation results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating evaluation plots...")
    plot_loss_curves(history, start_epoch, output_dir, phase)
    plot_accuracy_f1(history, start_epoch, output_dir, phase)
    plot_learning_rate(history, start_epoch, output_dir, phase)
    plot_per_class_performance(per_class, phase, output_dir)
    plot_summary_dashboard(
        history, start_epoch, output_dir,
        phase=phase, n_params=n_params,
        test_results=test_results, per_class=per_class,
    )
    logger.info("All plots saved to %s", output_dir)
