"""EMG-specific, per-trial features for the V3 temporal U-Net."""

from __future__ import annotations

import numpy as np


def _moving_average(data: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(int(width), len(data)))
    if width == 1:
        return data.copy()
    left = width // 2
    right = width - 1 - left
    padded = np.pad(data, ((left, right), (0, 0)), mode="reflect")
    kernel = np.full(width, 1.0 / width, dtype=np.float32)
    return np.column_stack([
        np.convolve(padded[:, channel], kernel, mode="valid")
        for channel in range(data.shape[1])
    ]).astype(np.float32)


def _robust_scale(data: np.ndarray) -> np.ndarray:
    median = np.median(data, axis=0, keepdims=True)
    centered = data - median
    mad = np.median(np.abs(centered), axis=0, keepdims=True)
    scale = np.maximum(1.4826 * mad, data.std(axis=0, keepdims=True) * 0.1)
    scale = np.maximum(scale, 1e-6)
    return np.clip(centered / scale, -10.0, 10.0).astype(np.float32)


def make_emg_features(raw: np.ndarray) -> np.ndarray:
    """Return high-pass, RMS-envelope, and derivative features (N x 9)."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("Expected raw EMG with shape [samples, 3]")
    trend = _moving_average(raw, 129)
    high_pass = raw - trend
    envelope = np.sqrt(np.maximum(_moving_average(np.square(high_pass), 65), 0.0))
    derivative = np.diff(high_pass, axis=0, prepend=high_pass[:1])
    return np.concatenate([
        _robust_scale(high_pass),
        _robust_scale(envelope),
        _robust_scale(derivative),
    ], axis=1).astype(np.float32)
