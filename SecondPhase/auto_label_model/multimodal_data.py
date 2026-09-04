"""Alignment and feature loading for cached video plus raw EMG trials."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .dataset_builder import load_emg_channels
    from .features_v3 import make_emg_features
except ImportError:
    from dataset_builder import load_emg_channels
    from features_v3 import make_emg_features


def load_emg_times(path: Path) -> np.ndarray:
    timestamps = []
    with path.open("r", encoding="utf-8-sig") as handle:
        _ = handle.readline()
        for line in handle:
            if not line.strip():
                continue
            first = line.split(",", 1)[0].strip()
            try:
                timestamps.append(int(first))
            except ValueError as error:
                raise ValueError(f"Unsupported timestamp in {path}: {first}") from error
    if not timestamps:
        raise ValueError(f"No timestamps in {path}")
    unwrapped = [timestamps[0]]
    offset = 0
    previous = timestamps[0]
    for timestamp in timestamps[1:]:
        if timestamp < previous:
            offset += 2**32
        unwrapped.append(timestamp + offset)
        previous = timestamp
    values = np.asarray(unwrapped, dtype=np.int64)
    return (values - values[0]).astype(np.float64) / 1e9


def aggregate_emg_by_frame(features, times, frame_times):
    output = []
    half_interval = 0.5 * np.median(np.diff(frame_times)) if len(frame_times) > 1 else 1 / 60
    for frame_time in frame_times:
        first = int(np.searchsorted(times, frame_time - half_interval, side="left"))
        last = int(np.searchsorted(times, frame_time + half_interval, side="right"))
        first = max(0, min(first, len(features) - 1))
        last = max(first + 1, min(last, len(features)))
        segment = features[first:last]
        output.append(np.concatenate((segment.mean(axis=0), segment.std(axis=0))))
    return np.asarray(output, dtype=np.float32)


def load_multimodal_trial(row: dict, cache_dir: Path) -> dict:
    with np.load(cache_dir / f"{row['pair_id']}.npz") as cached:
        visual = cached["features"].astype(np.float32)
        fps = float(cached["fps"])
    times = load_emg_times(Path(row["raw_path"]))
    raw = load_emg_channels(Path(row["raw_path"]))
    if len(times) != len(raw):
        raise ValueError(f"Timestamp/sample mismatch in {row['raw_path']}")
    frame_times = np.arange(len(visual), dtype=np.float64) / fps
    duration_difference = frame_times[-1] - times[-1]
    appearance = visual - visual.mean(axis=0, keepdims=True)
    motion = np.diff(visual, axis=0, prepend=visual[:1])
    emg = aggregate_emg_by_frame(make_emg_features(raw), times, frame_times)
    sample_indices = np.searchsorted(times, frame_times, side="left")
    sample_indices = np.clip(sample_indices, 0, len(times) - 1).astype(np.int64)
    starts_samples = np.asarray(
        [int(row["act1_start"]), int(row["act2_start"])], dtype=np.int64
    )
    starts_frames = np.asarray([
        int(np.argmin(np.abs(frame_times - times[start])))
        for start in starts_samples
    ], dtype=np.int64)
    end_sample = min(len(times) - 1, starts_samples[0] + int(row["window_samples"]))
    window_frames = max(
        1,
        int(round((times[end_sample] - times[starts_samples[0]]) * fps)),
    )
    return {
        "appearance": appearance.astype(np.float32),
        "motion": motion.astype(np.float32),
        "emg": emg,
        "sample_indices": sample_indices,
        "starts_samples": starts_samples,
        "starts_frames": starts_frames,
        "window_frames": window_frames,
        "fps": fps,
        "duration_difference": float(duration_difference),
    }
