from typing import Dict
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcola le metriche di errore fondamentali tra valori reali e predetti.
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }


def compute_horizon_mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcola l'MSE per ciascun passo temporale futuro dell'orizzonte di previsione.
    Forma input attesa: (N_samples, horizon_steps, n_features)
    Ritorna:
        array di forma (horizon_steps,)
    """
    diff_squared = (y_true - y_pred) ** 2
    # Media su campioni e feature, mantenendo l'asse temporale
    return np.mean(diff_squared, axis=(0, 2))
