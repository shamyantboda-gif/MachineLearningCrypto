"""A small multi-asset LSTM.

The design is driven by one arithmetic fact. Each asset contributes roughly two
thousand daily observations, and the signal to noise ratio in daily crypto
direction is close to zero. At that sample size capacity is the enemy: a
network with enough parameters will fit the training fold's noise perfectly and
report it as a finding. So this is one or two layers, a few dozen hidden units,
and heavy dropout, and it is not apologised for.

The model is fitted once across all four assets rather than once per asset, and
the asset identity enters through a small embedding concatenated to the final
hidden state. That does two things. It roughly quadruples the effective sample
size, which is the single biggest lever available here. And it lets whatever
structure is shared across the assets, which for crypto is most of it, be
learned from all of the data at once, while the embedding absorbs the part that
is genuinely per asset, such as a different unconditional drift or a different
typical volatility.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.torch_common import BaseTorchModel


class _LSTMNetwork(nn.Module):
    """Recurrent encoder, asset embedding, linear head to one output."""

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        asset_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch applies this dropout between stacked layers only, so with
            # a single layer it would be a no op and warns about it.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.asset_embedding = nn.Embedding(n_assets, asset_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + asset_embedding_dim, 1)

    def forward(self, window: torch.Tensor, asset_code: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(window)
        # Final layer's last hidden state summarises the whole window.
        summary = hidden[-1]
        joined = torch.cat([summary, self.asset_embedding(asset_code)], dim=1)
        return self.head(self.dropout(joined)).squeeze(-1)


class LSTMModel(BaseTorchModel):
    """Multi-asset LSTM over a rolling window of standardised features."""

    name = "lstm"

    def _build_network(self, n_features: int, n_assets: int) -> nn.Module:
        # Two layers is the ceiling. Anything deeper has no defensible sample
        # size behind it at this panel length.
        num_layers = min(max(int(self.params.get("num_layers", 1)), 1), 2)
        return _LSTMNetwork(
            n_features=n_features,
            n_assets=n_assets,
            hidden_size=int(self.params.get("hidden_size", 48)),
            num_layers=num_layers,
            dropout=float(self.params.get("dropout", 0.3)),
            asset_embedding_dim=int(self.params.get("asset_embedding_dim", 4)),
        )
