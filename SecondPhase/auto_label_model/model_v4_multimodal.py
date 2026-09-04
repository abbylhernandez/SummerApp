"""Temporal Video+EMG fusion model using cached pretrained frame embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class MultimodalConfig:
    visual_dimensions: int = 576
    emg_dimensions: int = 18
    hidden_dimensions: int = 96
    dropout: float = 0.18
    window_samples: int = 500
    region_score_weight: float = 0.5
    act1_prior_fraction: float = 0.45
    act2_prior_fraction: float = 0.65
    prior_sigma_fraction: float = 0.10
    evidence_multiplier: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class TemporalResidual(nn.Module):
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


class VideoEMGFusionNet(nn.Module):
    def __init__(self, config: MultimodalConfig | None = None):
        super().__init__()
        self.config = config or MultimodalConfig()
        hidden = self.config.hidden_dimensions
        self.appearance_projection = nn.Sequential(
            nn.LayerNorm(self.config.visual_dimensions),
            nn.Linear(self.config.visual_dimensions, hidden),
            nn.GELU(),
        )
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(self.config.visual_dimensions),
            nn.Linear(self.config.visual_dimensions, hidden // 2),
            nn.GELU(),
        )
        self.emg_projection = nn.Sequential(
            nn.LayerNorm(self.config.emg_dimensions),
            nn.Linear(self.config.emg_dimensions, hidden // 2),
            nn.GELU(),
        )
        fused = hidden * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.recurrent = nn.GRU(
            hidden,
            hidden,
            num_layers=2,
            dropout=self.config.dropout,
            bidirectional=True,
            batch_first=True,
        )
        temporal_channels = hidden * 2
        self.temporal = nn.Sequential(
            TemporalResidual(temporal_channels, 1, self.config.dropout),
            TemporalResidual(temporal_channels, 2, self.config.dropout),
            TemporalResidual(temporal_channels, 4, self.config.dropout),
        )
        self.class_head = nn.Conv1d(temporal_channels, 3, 1)
        self.boundary_head = nn.Conv1d(temporal_channels, 2, 1)

    def forward(
        self,
        appearance: torch.Tensor,
        motion: torch.Tensor,
        emg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat((
            self.appearance_projection(appearance),
            self.motion_projection(motion),
            self.emg_projection(emg),
        ), dim=-1)
        hidden, _ = self.recurrent(self.fusion(fused))
        hidden = self.temporal(hidden.transpose(1, 2))
        return self.class_head(hidden), self.boundary_head(hidden)
