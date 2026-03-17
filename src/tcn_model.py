from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("CausalBatchNorm1d expects input shaped (batch, channels, seq_len).")
        batch_size, channels, seq_len = x.shape
        if channels != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels, received {channels}.")

        cumulative_sum = x.sum(dim=0).cumsum(dim=1)
        cumulative_sq_sum = (x * x).sum(dim=0).cumsum(dim=1)
        counts = (
            torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype)
            .view(1, seq_len)
            .mul(batch_size)
        )
        mean = cumulative_sum / counts
        variance = (cumulative_sq_sum / counts) - mean.pow(2)
        variance = variance.clamp_min(0.0)

        mean = mean.unsqueeze(0)
        variance = variance.unsqueeze(0)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return (
            normalized * self.weight.view(1, channels, 1)
            + self.bias.view(1, channels, 1)
        )


class TemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = CausalBatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = CausalBatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        hidden = self.conv1(F.pad(x, (self.left_padding, 0)))
        hidden = self.norm1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(F.pad(hidden, (self.left_padding, 0)))
        hidden = self.norm2(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return hidden + residual


class TCNPredictor(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        hidden_channels: int = 32,
        n_blocks: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_assets = n_assets
        self.hidden_channels = hidden_channels
        self.input_projection = nn.Linear(n_features, hidden_channels)
        dilations = [2**idx for idx in range(n_blocks)]
        self.blocks = nn.ModuleList(
            [
                TemporalConvBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.output_head = nn.Linear(hidden_channels, 2)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input shaped (batch, seq_len, n_assets, n_features).")
        batch_size, seq_len, n_assets, n_features = x.shape
        if n_assets != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} assets, received {n_assets}.")
        if n_features != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, received {n_features}.")

        projected = self.input_projection(x.reshape(batch_size * seq_len * n_assets, n_features))
        projected = projected.view(batch_size, seq_len, n_assets, self.hidden_channels)
        projected = projected.permute(0, 2, 3, 1).reshape(batch_size * n_assets, self.hidden_channels, seq_len)
        hidden = projected
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden.transpose(1, 2).reshape(batch_size, n_assets, seq_len, self.hidden_channels)
        return hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        pooled = encoded.mean(dim=2)
        output = self.output_head(pooled)
        return output

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape[:-1] != targets.shape or predictions.shape[-1] != 2:
            raise ValueError("Predictions must be shaped (batch, n_assets, 2) and targets (batch, n_assets).")
        mean = predictions[..., 0]
        log_var = predictions[..., 1].clamp(min=-10.0, max=10.0)
        variance = torch.exp(log_var).clamp_min(1e-6)
        nll = 0.5 * (log_var + ((targets - mean) ** 2) / variance)
        return nll.mean()
