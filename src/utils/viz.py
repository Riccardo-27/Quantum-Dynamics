import os
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np


# Setup stile grafico coerente per report scientifici
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
})


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_indices: Optional[List[int]] = None,
    feature_names: Optional[List[str]] = None,
    time_steps: Optional[np.ndarray] = None,
    title: str = "Confronto Dinamica Quantistica: Reale vs Predetta",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Traccia il confronto tra traiettoria reale e predetta per un sottoinsieme di variabili.
    Salva la figura con risoluzione >= 300 DPI per il report.
    """
    if feature_indices is None:
        feature_indices = [0, 1, 2, 9, 10, 20]  # Esempi di magnetizzazioni e correlazioni

    n_features = len(feature_indices)
    fig, axes = plt.subplots(
        nrows=(n_features + 1) // 2,
        ncols=2,
        figsize=(14, 3.5 * ((n_features + 1) // 2)),
        sharex=True,
    )
    axes = np.atleast_1d(axes).flatten()

    steps = np.arange(len(y_true)) if time_steps is None else time_steps

    for i, feat_idx in enumerate(feature_indices):
        ax = axes[i]
        label_name = feature_names[i] if feature_names and i < len(feature_names) else f"Feature {feat_idx}"

        ax.plot(steps, y_true[:, feat_idx], label="Reale", color="#1f77b4", linestyle="-")
        ax.plot(steps, y_pred[:, feat_idx], label="Predetto", color="#d62728", linestyle="--")
        ax.set_title(label_name)
        ax.set_ylabel("Valore")
        ax.grid(True)
        ax.legend(loc="best")

    # Nascondi subplot in eccesso se dispari
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    axes[-1].set_xlabel("Passi Temporali / Tempo t")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Grafico salvato in: {save_path} ({dpi} DPI)")

    plt.show()


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Curve di Apprendimento (Loss)",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """
    Traccia le curve di Training e Validation Loss nel corso delle epoche.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train Loss (MSE)", color="#2ca02c")
    ax.plot(epochs, val_losses, label="Val Loss (MSE)", color="#ff7f0e", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Mean Squared Error")
    ax.set_yscale("log")
    ax.grid(True, which="both")
    ax.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Curva di Loss salvata in: {save_path} ({dpi} DPI)")

    plt.show()
