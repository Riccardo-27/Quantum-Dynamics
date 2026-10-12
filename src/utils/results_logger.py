import os
import csv
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


DEFAULT_LOG_PATH = "results/results_log.csv"

FIELDNAMES = [
    "run_id",
    "timestamp",
    "model_name",
    "family",               # RNN oppure Transformer
    "window_name",
    "input_len",
    "output_len",
    "epochs",
    "learning_rate",
    "batch_size",
    "train_mse",
    "val_mse",
    "test_mse",
    "test_mae",
    "num_parameters",
    "notes",
]


def log_experiment_result(
    data: Dict[str, Any],
    log_path: str = DEFAULT_LOG_PATH,
) -> None:
    """
    Registra i risultati di un trial o di un modello in append su results_log.csv.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0

    if "timestamp" not in data:
        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "run_id" not in data:
        data["run_id"] = f"{data.get('model_name', 'model')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    row = {field: data.get(field, "") for field in FIELDNAMES}

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Risultato registrato con successo in {log_path} [run_id: {data['run_id']}]")


def load_experiment_results(log_path: str = DEFAULT_LOG_PATH) -> Optional[pd.DataFrame]:
    """
    Carica i risultati registrati come DataFrame Pandas.
    """
    if not os.path.exists(log_path):
        return None
    return pd.read_csv(log_path)
