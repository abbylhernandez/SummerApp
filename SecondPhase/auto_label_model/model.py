"""Hybrid temporal convolution + Transformer model for EMG auto-labeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    input_channels: int = 3
    hidden_channels: int = 96
    transformer_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.15
    stride: int = 4
    window_samples: int = 500

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.norm = nn.GroupNorm(8, channels)
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.output = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(inputs)
        hidden = self.norm(hidden)
        values, gates = self.pointwise(hidden).chunk(2, dim=1)
        hidden = values * torch.sigmoid(gates)
        hidden = self.dropout(self.output(hidden))
        return inputs + hidden


class AutoLabelNet(nn.Module):
    """Predict Act identity for clips and two start distributions for raw trials.

    The convolutional encoder reduces the timeline to one logit every four raw
    samples. Dilated temporal blocks capture local EMG morphology; a compact
    Transformer captures long-range Act ordering and trial context.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        channels = self.config.hidden_channels
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.config.input_channels,
                64,
                kernel_size=9,
                stride=2,
                padding=4,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.temporal_blocks = nn.Sequential(*[
            ResidualTemporalBlock(channels, dilation, self.config.dropout)
            for dilation in (1, 2, 4, 8, 16, 32)
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=self.config.attention_heads,
            dim_feedforward=channels * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_layers,
            norm=nn.LayerNorm(channels),
            enable_nested_tensor=False,
        )
        self.clip_head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(channels, 2),
        )
        self.location_head = nn.Conv1d(channels, 2, kernel_size=1)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Expected [batch, time, channels] or [batch, channels, time]")
        if inputs.shape[-1] == self.config.input_channels:
            inputs = inputs.transpose(1, 2)
        hidden = self.stem(inputs)
        hidden = self.temporal_blocks(hidden)
        hidden = self.transformer(hidden.transpose(1, 2))
        return hidden

    def classify_clips(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(inputs)
        pooled = hidden.mean(dim=1)
        return self.clip_head(pooled)

    def localize(
        self,
        inputs: torch.Tensor,
        raw_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.encode(inputs)
        logits = self.location_head(hidden.transpose(1, 2))
        if raw_lengths is not None:
            for batch_index, raw_length in enumerate(raw_lengths.tolist()):
                max_start = max(0, int(raw_length) - self.config.window_samples)
                max_bin = min(logits.shape[-1] - 1, max_start // self.config.stride)
                if max_bin + 1 < logits.shape[-1]:
                    logits[batch_index, :, max_bin + 1:] = -torch.inf
        return logits


def gaussian_location_loss(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    sigma_bins: float = 12.5,
) -> torch.Tensor:
    """Soft-boundary loss tolerant to small human annotation differences."""
    positions = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    finite_mask = torch.isfinite(logits)
    targets = []
    for batch_index in range(logits.shape[0]):
        per_act = []
        for act_index in range(2):
            center = target_bins[batch_index, act_index].to(logits.dtype)
            distribution = torch.exp(-0.5 * ((positions - center) / sigma_bins) ** 2)
            distribution = distribution * finite_mask[batch_index, act_index]
            distribution = distribution / distribution.sum().clamp_min(1e-12)
            per_act.append(distribution)
        targets.append(torch.stack(per_act))
    soft_targets = torch.stack(targets)
    safe_logits = logits.masked_fill(
        ~finite_mask,
        torch.finfo(logits.dtype).min,
    )
    return -(soft_targets * F.log_softmax(safe_logits, dim=-1)).sum(dim=-1).mean()


def ordering_loss(logits: torch.Tensor, window_bins: int) -> torch.Tensor:
    """Penalize predictions that violate Act 1 then non-overlapping Act 2."""
    probabilities = F.softmax(logits, dim=-1)
    positions = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    expected = (probabilities * positions).sum(dim=-1)
    violation = expected[:, 0] + window_bins - expected[:, 1]
    return F.relu(violation).pow(2).mean() / max(1, window_bins ** 2)


def prediction_from_logits(logits: torch.Tensor, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits, dim=-1)
    confidence, bins = probabilities.max(dim=-1)
    return bins * stride, confidence
