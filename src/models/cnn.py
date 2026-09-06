"""A WaveNet style dilated temporal convolution.

The convolutional alternative to the LSTM. Stacked dilated convolutions reach
the same thirty day receptive field as the recurrent model with a fraction of
the sequential work, because the dilations widen geometrically: with kernel
size 3 and dilations 1, 2, 4, 8 the top layer sees thirty one timesteps.

Capacity is held to the same standard as the LSTM and for the same reason. Two
thousand daily observations per asset does not support a wide network, so the
channel count stays in the tens, dropout is heavy, and the whole sequence is
collapsed by global average pooling rather than a large flattening layer that
would multiply the parameter count by the lookback.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.torch_common import BaseTorchModel


class _CausalBlock(nn.Module):
    """One dilated convolution with left only padding and a residual path."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        # Pad only on the left. A symmetric or right side pad would let the
        # convolution at position t read positions after t, which for a
        # forecasting model is leakage from the future inside the window: the
        # feature vector at t would end up carrying information from t+1. The
        # left pad width is exactly the amount the kernel reaches backward, so
        # the output length matches the input length with no lookahead.
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # A residual connection only makes sense where the channel counts line
        # up. The first block changes n_features into n_channels, so it has no
        # residual; every later block does.
        self.residual = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(F.pad(x, (self.left_pad, 0)))
        out = self.dropout(self.activation(out))
        return x + out if self.residual else out


class _CNNNetwork(nn.Module):
    """Dilated convolution stack, global average pool, asset embedding, head."""

    def __init__(
        self,
        n_features: int,
        n_assets: int,
        channels: int,
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        asset_embedding_dim: int,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = n_features
        for dilation in dilations:
            blocks.append(_CausalBlock(in_channels, channels, kernel_size, int(dilation), dropout))
            in_channels = channels
        self.blocks = nn.ModuleList(blocks)
        self.asset_embedding = nn.Embedding(n_assets, asset_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels + asset_embedding_dim, 1)

    def forward(self, window: torch.Tensor, asset_code: torch.Tensor) -> torch.Tensor:
        # Conv1d wants (batch, channels, time); windows arrive as
        # (batch, time, features).
        x = window.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        summary = x.mean(dim=2)
        joined = torch.cat([summary, self.asset_embedding(asset_code)], dim=1)
        return self.head(self.dropout(joined)).squeeze(-1)


class CNNModel(BaseTorchModel):
    """Causal dilated CNN over a rolling window of standardised features."""

    name = "cnn"

    def _build_network(self, n_features: int, n_assets: int) -> nn.Module:
        dilations = [int(d) for d in self.params.get("dilations", [1, 2, 4, 8])]
        if not dilations:
            raise ValueError("cnn needs at least one dilation")
        return _CNNNetwork(
            n_features=n_features,
            n_assets=n_assets,
            channels=int(self.params.get("channels", 48)),
            kernel_size=int(self.params.get("kernel_size", 3)),
            dilations=dilations,
            dropout=float(self.params.get("dropout", 0.3)),
            asset_embedding_dim=int(self.params.get("asset_embedding_dim", 4)),
        )
