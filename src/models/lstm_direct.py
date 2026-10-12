import torch
import torch.nn as nn


class LSTMDirectForecaster(nn.Module):
    """
    Modello M1: LSTM Diretto Multi-Step per Quantum Dynamics.
    
    Data una finestra passata di osservabili X in R^(B x T_in x 55),
    un encoder LSTM a più strati estrae la rappresentazione densa
    della dinamica quantistica e una testa di proiezione mappa direttamente
    in single-shot sull'intero orizzonte futuro Y in R^(B x T_out x 55).
    """
    def __init__(
        self,
        input_features: int = 55,
        output_features: int = 55,
        input_len: int = 50,
        output_len: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder Ricorrente LSTM
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer Normalization per stabilizzare la rappresentazione di contesto
        self.norm = nn.LayerNorm(hidden_dim)

        # Projection Head (MLP a 2 stadi per mappare sull'orizzonte multi-step)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_len * output_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Argomenti:
            x: Tensor di forma (batch_size, input_len, input_features)
        Ritorna:
            y_pred: Tensor di forma (batch_size, output_len, output_features)
        """
        batch_size = x.size(0)

        # Passaggio nell'LSTM
        out, (h_n, _) = self.lstm(x)

        # Usiamo l'ultimo stato nascosto dell'ultimo layer LSTM: (batch_size, hidden_dim)
        context = h_n[-1]
        context = self.norm(context)

        # Proiezione diretta sull'orizzonte futuro srotolato
        flat_pred = self.head(context)

        # Reshape a (batch_size, output_len, output_features)
        y_pred = flat_pred.view(batch_size, self.output_len, self.output_features)
        return y_pred

    def count_parameters(self) -> int:
        """Restituisce il numero totale di parametri addestrabili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
