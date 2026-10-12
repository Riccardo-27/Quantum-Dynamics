import os
from typing import Tuple
import numpy as np
import pandas as pd


def convert_csv_to_npy(
    csv_path: str = "data/trajectories.csv",
    npy_path: str = "data/trajectories.npy",
    time_path: str = "data/time_grid.npy",
    num_trajectories: int = 400,
    num_points: int = 1001,
    num_features: int = 55,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legge trajectories.csv una sola volta e lo salva in formato binario (.npy).
    Ritorna:
        features: array float32 di forma (400, 1001, 55)
        time_grid: array float32 di forma (1001,)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV non trovato in: {csv_path}")

    print(f"Caricamento {csv_path} in corso (operazione una-tantum)...")
    df = pd.read_csv(csv_path, header=None, dtype=np.float32)

    total_expected_rows = num_trajectories * num_points
    if df.shape[0] != total_expected_rows or df.shape[1] != (1 + num_features):
        print(f"Attenzione: forma inattesa {df.shape}, attesa ({total_expected_rows}, {1 + num_features})")

    # Colonna 0 è il tempo
    time_grid = df.iloc[:num_points, 0].to_numpy(dtype=np.float32)

    # Colonne 1..55 sono le feature (10 magnetizzazioni + 45 correlazioni)
    features_flat = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    features = features_flat.reshape(num_trajectories, num_points, num_features)

    # Salvataggio binario veloce
    np.save(npy_path, features)
    np.save(time_path, time_grid)
    print(f"Dataset convertito e salvato con successo in:")
    print(f"  - {npy_path} -> forma: {features.shape}, dimensione: {os.path.getsize(npy_path) / (1024**2):.1f} MB")
    print(f"  - {time_path} -> forma: {time_grid.shape}")

    return features, time_grid


def load_dataset(
    data_dir: str = "data",
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica istantaneamente il dataset dal file .npy se esiste,
    altrimenti esegue la conversione dal CSV.
    """
    npy_path = os.path.join(data_dir, "trajectories.npy")
    time_path = os.path.join(data_dir, "time_grid.npy")
    csv_path = os.path.join(data_dir, "trajectories.csv")

    if not force_recompute and os.path.exists(npy_path) and os.path.exists(time_path):
        features = np.load(npy_path)
        time_grid = np.load(time_path)
        return features, time_grid

    return convert_csv_to_npy(csv_path, npy_path, time_path)


def split_trajectories(
    num_trajectories: int = 400,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partiziona gli indici delle traiettorie in Train, Validation e Test.
    Garantisce che lo split avvenga per intere traiettorie indipendenti (no data leakage).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

    rng = np.random.default_rng(seed)
    indices = np.arange(num_trajectories)
    rng.shuffle(indices)

    n_train = int(num_trajectories * train_ratio)
    n_val = int(num_trajectories * val_ratio)

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train : n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val :])

    return train_idx, val_idx, test_idx
