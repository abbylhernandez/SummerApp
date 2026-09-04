"""Compact U-Time-style network for full-trial EMG segmentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class UTimeConfig:
    input_channels: int = 9
    base_channels: int = 24
    window_samples: int = 500
    dropout: float = 0.12
    region_score_weight: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float):
        super().__init__()
        groups = 8 if output_channels % 8 == 0 else 4
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, 5, padding=2),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_channels, output_channels, 5, padding=2),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
                groups=channels,
            ),
            nn.GroupNorm(8, channels),
            nn.Conv1d(channels, channels * 2, 1),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class UTimeEMG(nn.Module):
    """Map an arbitrary-length trial to dense classes and two start scores."""

    def __init__(self, config: UTimeConfig | None = None):
        super().__init__()
        self.config = config or UTimeConfig()
        base = self.config.base_channels
        self.session_adapter = nn.Sequential(
            nn.Conv1d(self.config.input_channels, self.config.input_channels, 1),
            nn.GroupNorm(3, self.config.input_channels),
            nn.Tanh(),
        )
        self.encoder1 = ConvBlock(self.config.input_channels, base, self.config.dropout)
        self.encoder2 = ConvBlock(base, base * 2, self.config.dropout)
        self.encoder3 = ConvBlock(base * 2, base * 4, self.config.dropout)
        self.encoder4 = ConvBlock(base * 4, base * 6, self.config.dropout)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.bottleneck = nn.Sequential(
            ConvBlock(base * 6, base * 8, self.config.dropout),
            DilatedBlock(base * 8, 1, self.config.dropout),
            DilatedBlock(base * 8, 2, self.config.dropout),
            DilatedBlock(base * 8, 4, self.config.dropout),
        )
        self.decoder4 = ConvBlock(base * 8 + base * 6, base * 6, self.config.dropout)
        self.decoder3 = ConvBlock(base * 6 + base * 4, base * 4, self.config.dropout)
        self.decoder2 = ConvBlock(base * 4 + base * 2, base * 2, self.config.dropout)
        self.decoder1 = ConvBlock(base * 2 + base, base, self.config.dropout)
        self.class_head = nn.Conv1d(base, 3, 1)
        self.boundary_head = nn.Conv1d(base, 2, 1)

    @staticmethod
    def _upsample(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(inputs, size=skip.shape[-1], mode="linear", align_corners=False)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.config.input_channels:
            raise ValueError(
                f"Expected [batch, samples, {self.config.input_channels}]"
            )
        hidden = inputs.transpose(1, 2)
        hidden = hidden + 0.10 * self.session_adapter(hidden)
        level1 = self.encoder1(hidden)
        level2 = self.encoder2(self.pool(level1))
        level3 = self.encoder3(self.pool(level2))
        level4 = self.encoder4(self.pool(level3))
        center = self.bottleneck(self.pool(level4))
        decoded4 = self.decoder4(torch.cat((self._upsample(center, level4), level4), 1))
        decoded3 = self.decoder3(torch.cat((self._upsample(decoded4, level3), level3), 1))
        decoded2 = self.decoder2(torch.cat((self._upsample(decoded3, level2), level2), 1))
        decoded1 = self.decoder1(torch.cat((self._upsample(decoded2, level1), level1), 1))
        return self.class_head(decoded1), self.boundary_head(decoded1)


def dense_labels(starts: torch.Tensor, samples: int, window_samples: int) -> torch.Tensor:
    labels = torch.zeros(
        (starts.shape[0], samples), dtype=torch.long, device=starts.device
    )
    for batch_index in range(starts.shape[0]):
        for act_index in range(2):
            first = max(0, int(starts[batch_index, act_index].item()))
            last = min(samples, first + window_samples)
            labels[batch_index, first:last] = act_index + 1
    return labels


def focal_segmentation_loss(
    logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor
) -> torch.Tensor:
    cross_entropy = F.cross_entropy(
        logits, labels, weight=class_weights, reduction="none"
    )
    probabilities = torch.softmax(logits, dim=1)
    correct_probability = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    focal = ((1.0 - correct_probability).square() * cross_entropy).mean()

    one_hot = F.one_hot(labels, num_classes=3).permute(0, 2, 1).to(logits.dtype)
    activity_probabilities = probabilities[:, 1:]
    activity_targets = one_hot[:, 1:]
    intersection = (activity_probabilities * activity_targets).sum(dim=-1)
    denominator = activity_probabilities.sum(dim=-1) + activity_targets.sum(dim=-1)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return focal + 0.5 * dice_loss


def boundary_distribution_loss(
    logits: torch.Tensor, starts: torch.Tensor, sigma_samples: float = 35.0
) -> torch.Tensor:
    positions = torch.arange(
        logits.shape[-1], device=logits.device, dtype=logits.dtype
    )
    centers = starts.to(logits.dtype).unsqueeze(-1)
    targets = torch.exp(-0.5 * ((positions.view(1, 1, -1) - centers) / sigma_samples).square())
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def combined_start_scores(
    class_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    window_samples: int,
    region_weight: float,
) -> torch.Tensor:
    """Combine boundary evidence with mean Act probability over each window."""
    log_probabilities = F.log_softmax(class_logits, dim=1)[:, 1:]
    valid = F.avg_pool1d(log_probabilities, kernel_size=window_samples, stride=1)
    region_scores = torch.full_like(boundary_logits, float("-inf"))
    region_scores[:, :, : valid.shape[-1]] = valid
    finite = torch.isfinite(region_scores)
    safe_region = region_scores.masked_fill(~finite, 0.0)

    def standardize(values: torch.Tensor, mask: torch.Tensor | None = None):
        if mask is None:
            mean = values.mean(dim=-1, keepdim=True)
            std = values.std(dim=-1, keepdim=True).clamp_min(1e-5)
        else:
            count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
            mean = (values * mask).sum(dim=-1, keepdim=True) / count
            variance = ((values - mean).square() * mask).sum(dim=-1, keepdim=True) / count
            std = variance.sqrt().clamp_min(1e-5)
        return (values - mean) / std

    scores = standardize(boundary_logits) + region_weight * standardize(
        safe_region, finite
    )
    return scores.masked_fill(~finite, float("-inf"))
