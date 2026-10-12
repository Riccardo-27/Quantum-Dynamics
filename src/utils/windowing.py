from typing import Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset


# Griglia di configurazioni raccomandate per la Sfida 1 (Time Window Optimization)
WINDOW_CONFIGS: List[Dict[str, int]] = [
    {"name": "short_window", "input_len": 50, "output_len": 100},
    {"name": "medium_window", "input_len": 100, "output_len": 100},
    {"name": "long_window", "input_len": 200, "output_len": 100},
]


def create_sliding_windows(
    trajectories: np.ndarray,
    input_len: int,
    output_len: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae finestre scorrevoli (X, Y) da un insieme di traiettorie 3D (N_traj, T, F).
    Ritorna:
        X: array (num_samples, input_len, F)
        Y: array (num_samples, output_len, F)
    """
    n_traj, n_points, n_features = trajectories.shape
    total_window = input_len + output_len

    if n_points < total_window:
        raise ValueError(
            f"La lunghezza temporale ({n_points}) deve essere >= input_len + output_len ({total_window})"
        )

    x_list = []
    y_list = []

    for traj_idx in range(n_traj):
        traj = trajectories[traj_idx]
        for start in range(0, n_points - total_window + 1, stride):
            x_w = traj[start : start + input_len]
            y_w = traj[start + input_len : start + total_window]
            x_list.append(x_w)
            y_list.append(y_w)

    X = np.asarray(x_list, dtype=np.float32)
    Y = np.asarray(y_list, dtype=np.float32)
    return X, Y


class QuantumWindowDataset(Dataset):
    """
    Dataset PyTorch per campioni a finestra scorrevole.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
