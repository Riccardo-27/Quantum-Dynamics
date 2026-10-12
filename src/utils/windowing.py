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
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.Y = torch.from_numpy(Y) if isinstance(Y, np.ndarray) else Y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def get_dataloaders(
    features: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    config_name: str = "short_window",
    stride_train: int = 5,
    stride_eval: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, int]]:
    """
    Genera i DataLoader PyTorch per Train, Validation e Test a partire dalla configurazione scelta.
    """
    config = next((c for c in WINDOW_CONFIGS if c["name"] == config_name), None)
    if config is None:
        valid_names = [c["name"] for c in WINDOW_CONFIGS]
        raise ValueError(f"Configurazione '{config_name}' non valida. Opzioni: {valid_names}")

    in_len = config["input_len"]
    out_len = config["output_len"]

    train_trajs = features[split_indices["train_idx"]]
    val_trajs = features[split_indices["val_idx"]]
    test_trajs = features[split_indices["test_idx"]]

    X_train, Y_train = create_sliding_windows(train_trajs, in_len, out_len, stride=stride_train)
    X_val, Y_val = create_sliding_windows(val_trajs, in_len, out_len, stride=stride_eval)
    X_test, Y_test = create_sliding_windows(test_trajs, in_len, out_len, stride=stride_eval)

    train_dataset = QuantumWindowDataset(X_train, Y_train)
    val_dataset = QuantumWindowDataset(X_val, Y_val)
    test_dataset = QuantumWindowDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, config
