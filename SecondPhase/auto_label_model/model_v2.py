"""Small full-trial TCN for dense EMG activity and boundary detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelV2Config:
    input_channels: int = 3
    hidden_channels: int = 48
    stride: int = 4
    window_samples: int = 500
    dropout: float = 0.12
    act1_prior_fraction: float = 0.47
    act2_prior_fraction: float = 0.61
    prior_sigma_fraction: float = 0.10
    evidence_scale: float = 2.0
    inference_evidence_multiplier: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class TCNBlock(nn.Module):
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
        self.expand = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.project = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(self.depthwise(inputs))
        values, gates = self.expand(hidden).chunk(2, dim=1)
        hidden = values * torch.sigmoid(gates)
        return inputs + self.dropout(self.project(hidden))


class FullTrialTCN(nn.Module):
    """Detect background/Act regions and exact starts from complete trials."""

    def __init__(self, config: ModelV2Config | None = None):
        super().__init__()
        self.config = config or ModelV2Config()
        channels = self.config.hidden_channels
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.config.input_channels + 1,
                channels,
                kernel_size=9,
                stride=2,
                padding=4,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            TCNBlock(channels, dilation, self.config.dropout)
            for dilation in (1, 2, 4, 8, 16, 32, 16, 8)
        ])
        self.class_head = nn.Conv1d(channels, 3, kernel_size=1)
        self.boundary_head = nn.Conv1d(channels, 2, kernel_size=1)

    def _with_position(
        self, inputs: torch.Tensor, raw_lengths: torch.Tensor
    ) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("Expected [batch, time, channels]")
        if inputs.shape[-1] != self.config.input_channels:
            raise ValueError(f"Expected {self.config.input_channels} EMG channels")
        batch, samples, _ = inputs.shape
        indices = torch.arange(samples, device=inputs.device, dtype=inputs.dtype)
        denominator = (raw_lengths.to(inputs.dtype) - 1).clamp_min(1).view(batch, 1)
        position = (indices.view(1, samples) / denominator).clamp(0, 1)
        position = position.mul(2).sub(1).unsqueeze(-1)
        return torch.cat((inputs, position), dim=-1).transpose(1, 2)

    def forward(
        self, inputs: torch.Tensor, raw_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.blocks(self.stem(self._with_position(inputs, raw_lengths)))
        class_logits = self.class_head(hidden)
        evidence = self.config.evidence_scale * torch.tanh(self.boundary_head(hidden))

        bins = evidence.shape[-1]
        sample_positions = (
            torch.arange(bins, device=inputs.device, dtype=inputs.dtype)
            * self.config.stride
        )
        fractions = sample_positions.view(1, 1, bins) / raw_lengths.to(
            inputs.dtype
        ).clamp_min(1).view(-1, 1, 1)
        prior_centers = inputs.new_tensor([
            self.config.act1_prior_fraction,
            self.config.act2_prior_fraction,
        ]).view(1, 2, 1)
        prior_logits = -0.5 * (
            (fractions - prior_centers) / self.config.prior_sigma_fraction
        ).square()
        location_logits = (
            prior_logits + self.config.inference_evidence_multiplier * evidence
        )

        for batch_index, raw_length in enumerate(raw_lengths.tolist()):
            max_start = max(0, int(raw_length) - self.config.window_samples)
            max_bin = min(bins - 1, max_start // self.config.stride)
            if max_bin + 1 < bins:
                location_logits[batch_index, :, max_bin + 1:] = -torch.inf
        return class_logits, location_logits, evidence


def dense_targets(
    starts: torch.Tensor,
    bins: int,
    stride: int,
    window_samples: int,
) -> torch.Tensor:
    targets = torch.zeros(
        (starts.shape[0], bins), dtype=torch.long, device=starts.device
    )
    window_bins = ceil(window_samples / stride)
    for batch_index in range(starts.shape[0]):
        for act_index in range(2):
            first = max(0, int(starts[batch_index, act_index].item()) // stride)
            last = min(bins, first + window_bins)
            targets[batch_index, first:last] = act_index + 1
    return targets


def gaussian_boundary_loss(
    logits: torch.Tensor,
    starts: torch.Tensor,
    stride: int,
    sigma_samples: float = 60.0,
) -> torch.Tensor:
    positions = torch.arange(
        logits.shape[-1], device=logits.device, dtype=logits.dtype
    )
    finite = torch.isfinite(logits)
    centers = starts.to(logits.dtype) / stride
    sigma_bins = sigma_samples / stride
    distributions = torch.exp(
        -0.5 * ((positions.view(1, 1, -1) - centers.unsqueeze(-1)) / sigma_bins).square()
    )
    distributions = distributions * finite
    distributions = distributions / distributions.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    safe_logits = logits.masked_fill(~finite, torch.finfo(logits.dtype).min)
    return -(distributions * F.log_softmax(safe_logits, dim=-1)).sum(dim=-1).mean()


def ordering_loss(logits: torch.Tensor, window_samples: int, stride: int) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    positions = torch.arange(
        logits.shape[-1], device=logits.device, dtype=logits.dtype
    )
    expected = (probabilities * positions).sum(dim=-1)
    window_bins = ceil(window_samples / stride)
    violation = expected[:, 0] + window_bins - expected[:, 1]
    return F.relu(violation).square().mean() / max(1, window_bins**2)


def select_ordered_pair(logits: torch.Tensor, window_bins: int) -> tuple[int, int]:
    """Select the highest-scoring non-overlapping Act 1/Act 2 pair."""
    act1, act2 = logits[0], logits[1]
    length = logits.shape[-1]
    if length <= window_bins:
        return 0, max(0, length - 1)
    suffix_values, suffix_indices = torch.cummax(act2.flip(0), dim=0)
    suffix_values = suffix_values.flip(0)
    suffix_indices = (length - 1 - suffix_indices).flip(0)
    candidates = act1[: length - window_bins] + suffix_values[window_bins:]
    act1_bin = int(candidates.argmax().item())
    act2_bin = int(suffix_indices[act1_bin + window_bins].item())
    return act1_bin, act2_bin
