"""
Visualization utilities for quantum state tomography results.

Plots:
    - Fidelity vs. shot count (DDPM vs MLE)
    - Density matrix visualizations (real and imaginary parts)
    - Training curves
    - State type breakdown
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, List, Dict, Tuple

# Use non-interactive backend
matplotlib.use("Agg")


def plot_fidelity_vs_shots(
    shot_levels: List[int],
    ddpm_fidelities: np.ndarray,
    ddpm_std: np.ndarray,
    mle_fidelities: np.ndarray,
    mle_std: np.ndarray,
    linear_fidelities: Optional[np.ndarray] = None,
    linear_std: Optional[np.ndarray] = None,
    title: str = "Fidelity vs. Measurement Shots",
    save_path: Optional[str] = None,
):
    """
    Plot fidelity vs. number of measurement shots for DDPM and MLE.

    Args:
        shot_levels: List of shot counts.
        ddpm_fidelities: Mean DDPM fidelity per shot level, shape (len(shot_levels),).
        ddpm_std: Standard deviation of DDPM fidelity.
        mle_fidelities: Mean MLE fidelity per shot level.
        mle_std: Standard deviation of MLE fidelity.
        linear_fidelities: Optional linear inversion baseline.
        linear_std: Optional linear inversion std.
        title: Plot title.
        save_path: Path to save the figure. If None, display.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    shot_levels = np.array(shot_levels)

    ax.errorbar(
        shot_levels, ddpm_fidelities, yerr=ddpm_std,
        marker="o", linewidth=2, markersize=6, capsize=3,
        label="DDPM (ours)", color="#2196F3",
    )
    ax.errorbar(
        shot_levels, mle_fidelities, yerr=mle_std,
        marker="s", linewidth=2, markersize=6, capsize=3,
        label="MLE", color="#FF5722",
    )

    if linear_fidelities is not None:
        lin_yerr = linear_std if linear_std is not None else np.zeros_like(shot_levels)
        ax.errorbar(
            shot_levels, linear_fidelities, yerr=lin_yerr,
            marker="^", linewidth=2, markersize=6, capsize=3,
            label="Linear Inversion", color="#9E9E9E", linestyle="--",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Measurement Shots", fontsize=12)
    ax.set_ylabel("Fidelity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_density_matrix(
    rho_true: np.ndarray,
    rho_pred: np.ndarray,
    title: str = "Density Matrix Comparison",
    save_path: Optional[str] = None,
):
    """
    Visualize true vs. predicted density matrix (real and imaginary parts).

    Args:
        rho_true: True density matrix of shape (d, d).
        rho_pred: Predicted density matrix of shape (d, d).
        title: Plot title.
        save_path: Path to save the figure.
    """
    d = rho_true.shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    vmin = min(np.min(np.real(rho_true)), np.min(np.real(rho_pred)))
    vmax = max(np.max(np.real(rho_true)), np.max(np.real(rho_pred)))

    # Real parts
    im0 = axes[0, 0].imshow(np.real(rho_true), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("True (Real)", fontsize=11)
    axes[0, 0].set_xticks(range(d))
    axes[0, 0].set_yticks(range(d))

    im1 = axes[0, 1].imshow(np.real(rho_pred), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Predicted (Real)", fontsize=11)
    axes[0, 1].set_xticks(range(d))
    axes[0, 1].set_yticks(range(d))

    diff_real = np.real(rho_pred - rho_true)
    vmax_diff = max(np.max(np.abs(diff_real)), 1e-10)
    im2 = axes[0, 2].imshow(diff_real, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)
    axes[0, 2].set_title("Difference (Real)", fontsize=11)
    axes[0, 2].set_xticks(range(d))
    axes[0, 2].set_yticks(range(d))
    plt.colorbar(im2, ax=axes[0, 2])

    # Imaginary parts
    vmin_imag = min(np.min(np.imag(rho_true)), np.min(np.imag(rho_pred)))
    vmax_imag = max(np.max(np.imag(rho_true)), np.max(np.imag(rho_pred)))

    im3 = axes[1, 0].imshow(np.imag(rho_true), cmap="RdBu_r",
                             vmin=vmin_imag, vmax=vmax_imag)
    axes[1, 0].set_title("True (Imag)", fontsize=11)
    axes[1, 0].set_xticks(range(d))
    axes[1, 0].set_yticks(range(d))

    im4 = axes[1, 1].imshow(np.imag(rho_pred), cmap="RdBu_r",
                             vmin=vmin_imag, vmax=vmax_imag)
    axes[1, 1].set_title("Predicted (Imag)", fontsize=11)
    axes[1, 1].set_xticks(range(d))
    axes[1, 1].set_yticks(range(d))

    diff_imag = np.imag(rho_pred - rho_true)
    vmax_diff_i = max(np.max(np.abs(diff_imag)), 1e-10)
    im5 = axes[1, 2].imshow(diff_imag, cmap="RdBu_r",
                             vmin=-vmax_diff_i, vmax=vmax_diff_i)
    axes[1, 2].set_title("Difference (Imag)", fontsize=11)
    axes[1, 2].set_xticks(range(d))
    axes[1, 2].set_yticks(range(d))
    plt.colorbar(im5, ax=axes[1, 2])

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    """
    Plot training and validation loss curves.

    Args:
        metrics_history: Dict with "train_loss" and "val_loss" lists.
        save_path: Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax = axes[0]
    train_loss = metrics_history["train_loss"]
    val_loss = metrics_history.get("val_loss", [])
    val_epochs = np.linspace(0, len(train_loss) - 1, len(val_loss))

    ax.plot(train_loss, label="Train Loss", alpha=0.7, linewidth=1)
    if val_loss:
        ax.plot(val_epochs, val_loss, label="Val Loss", marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if train_loss:
        ax.set_yscale("log")

    # Learning rate
    ax = axes[1]
    lr_history = metrics_history.get("lr", [])
    ax.plot(lr_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_state_type_breakdown(
    labels: List[str],
    ddpm_fidelities: np.ndarray,
    mle_fidelities: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing DDPM and MLE fidelity by state type.

    Args:
        labels: State type names.
        ddpm_fidelities: Mean DDPM fidelity per type.
        mle_fidelities: Mean MLE fidelity per type.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, ddpm_fidelities, width, label="DDPM", color="#2196F3")
    ax.bar(x + width / 2, mle_fidelities, width, label="MLE", color="#FF5722")

    ax.set_xlabel("State Type")
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity by State Type")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
