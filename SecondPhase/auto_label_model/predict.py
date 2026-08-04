"""Predict Act 1 and Act 2 500-sample starts for one raw EMG trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .dataset_builder import DEFAULT_OUTPUT, load_emg_channels
    from .train import robust_normalize
except ImportError:
    from dataset_builder import DEFAULT_OUTPUT, load_emg_channels
    from train import robust_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_file", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT / "checkpoints" / "auto_label_model.pt",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "PyTorch is required for prediction. Install requirements.txt first."
        ) from error
    try:
        from .model import AutoLabelNet, ModelConfig
    except ImportError:
        from model import AutoLabelNet, ModelConfig
    return torch, AutoLabelNet, ModelConfig


def select_ordered_pair(logits, window_bins: int):
    """Maximize joint Act scores while enforcing Act1+window <= Act2."""
    act1 = logits[0]
    act2 = logits[1]
    length = logits.shape[-1]
    if length <= window_bins:
        return 0, max(0, length - 1)

    suffix_values = act2.clone()
    suffix_indices = np.arange(length, dtype=np.int64)
    best_value = float("-inf")
    best_index = length - 1
    for index in range(length - 1, -1, -1):
        value = float(act2[index].item())
        if value >= best_value:
            best_value = value
            best_index = index
        suffix_values[index] = best_value
        suffix_indices[index] = best_index

    best_score = float("-inf")
    best_pair = (0, window_bins)
    for act1_index in range(0, length - window_bins):
        act2_floor = act1_index + window_bins
        score = float(act1[act1_index].item() + suffix_values[act2_floor].item())
        if score > best_score:
            best_score = score
            best_pair = (act1_index, int(suffix_indices[act2_floor]))
    return best_pair


def predict(trial_file: Path, checkpoint_path: Path, device_name: str) -> dict:
    torch, AutoLabelNet, ModelConfig = require_torch()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = ModelConfig(**payload["model_config"])
    model = AutoLabelNet(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    raw = load_emg_channels(trial_file)
    normalized = robust_normalize(raw)
    inputs = torch.from_numpy(normalized).unsqueeze(0).to(device)
    lengths = torch.tensor([len(raw)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model.localize(inputs, lengths)[0]
        probabilities = torch.softmax(logits, dim=-1)

    window_bins = config.window_samples // config.stride
    act1_bin, act2_bin = select_ordered_pair(logits, window_bins)
    max_start = max(0, len(raw) - config.window_samples)
    act1_start = min(max_start, act1_bin * config.stride)
    act2_start = min(max_start, act2_bin * config.stride)
    return {
        "trial_file": str(trial_file.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset_version": payload.get("dataset_version", "unknown"),
        "window_samples": config.window_samples,
        "act1": {
            "start_sample": act1_start,
            "end_sample_exclusive": act1_start + config.window_samples,
            "confidence": float(probabilities[0, act1_bin].item()),
        },
        "act2": {
            "start_sample": act2_start,
            "end_sample_exclusive": act2_start + config.window_samples,
            "confidence": float(probabilities[1, act2_bin].item()),
        },
        "device": str(device),
    }


def main() -> None:
    args = parse_args()
    result = predict(args.trial_file, args.checkpoint, args.device)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
