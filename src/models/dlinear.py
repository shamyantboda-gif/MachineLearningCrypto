"""DLinear, after Zeng et al., "Are Transformers Effective for Time Series
Forecasting?" (AAAI 2023).

That paper's result was that a single linear layer applied to a decomposed
input matched or beat a shelf full of transformer forecasters on most standard
benchmarks. It is included here for exactly that rhetorical purpose. If the
LSTM and the CNN cannot beat a model that is two linear layers and a moving
average, then their recurrence, their dilated convolutions and the days of CPU
time spent fitting them have not earned their place, and the honest conclusion
is that the extra machinery bought nothing.

The mechanism is simple. Each input window is split into a moving average trend
and the seasonal remainder, each part gets its own linear map, and the two are
summed. Nothing else. It is kept genuinely small, a few thousand parameters,
because a fair test of "is the complicated model worth it" requires the simple
model to actually be simple.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.torch_common import BaseTorchModel


class _MovingAverage(nn.Module):
    """Trend component: a centred moving average with edge padding.

    Edge padding replicates the first and last observation rather than padding
    with zeros, so the trend at the ends of the window is not dragged toward
    zero by pad values that were never observed.
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = max(int(kernel_size), 1)
        self.left = (self.kernel_size - 1) // 2
        self.right = self.kernel_size - 1 - self.left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch, time, features); avg_pool1d works on (batch, channels, time).
        moved = x.transpose(1, 2)
        padded = F.pad(moved, (self.left, self.right), mode="replicate")
        smoothed = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1)
        return smoothed.transpose(1, 2)


class _DLinearNetwork(nn.Module):
    """Trend and seasonal linear maps, an asset embedding, one output."""

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        lookback: int,
        moving_avg_kernel: int,
        asset_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.decompose = _MovingAverage(moving_avg_kernel)
        flat = lookback * n_features
        # Both maps land in the embedding space so the asset vector can simply
        # be added, which is the additive per asset offset DLinear would
        # otherwise get from a per series bias.
        self.trend = nn.Linear(flat, asset_embedding_dim)
        self.seasonal = nn.Linear(flat, asset_embedding_dim)
        self.asset_embedding = nn.Embedding(n_assets, asset_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(asset_embedding_dim, 1)

    def forward(self, window: torch.Tensor, asset_code: torch.Tensor) -> torch.Tensor:
        trend = self.decompose(window)
        seasonal = window - trend
        batch = window.shape[0]
        mixed = self.trend(trend.reshape(batch, -1)) + self.seasonal(seasonal.reshape(batch, -1))
        mixed = mixed + self.asset_embedding(asset_code)
        return self.head(self.dropout(mixed)).squeeze(-1)


class DLinearModel(BaseTorchModel):
    """Decomposition plus linear. The control the other two must beat."""

    name = "dlinear"

    def _build_network(self, n_features: int, n_assets: int) -> nn.Module:
        # dlinear.yaml carries neither of these, so the defaults matter. A
        # small embedding and light dropout keep the parameter count in the low
        # thousands, which is the whole point of the comparison.
        return _DLinearNetwork(
            n_features=n_features,
            n_assets=n_assets,
            lookback=self.lookback,
            moving_avg_kernel=int(self.params.get("moving_avg_kernel", 7)),
            asset_embedding_dim=int(self.params.get("asset_embedding_dim", 4)),
            dropout=float(self.params.get("dropout", 0.1)),
        )
