"""Predict two ordered 500-sample labels using the V2 full-trial TCN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .dataset_builder import DEFAULT_OUTPUT, load_emg_channels
    from .model_v2 import FullTrialTCN, ModelV2Config, select_ordered_pair
    from .train import robust_normalize
except ImportError:
    from dataset_builder import DEFAULT_OUTPUT, load_emg_channels
    from model_v2 import FullTrialTCN, ModelV2Config, select_ordered_pair
    from train import robust_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_file", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT / "checkpoints" / "auto_label_model_v2.pt",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def predict(trial_file: Path, checkpoint: Path, requested_device: str) -> dict:
    import torch

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    config = ModelV2Config(**payload["model_config"])
    model = FullTrialTCN(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    raw = load_emg_channels(trial_file)
    inputs = torch.from_numpy(robust_normalize(raw)).unsqueeze(0).to(device)
    raw_length = torch.tensor([len(raw)], dtype=torch.long, device=device)
    with torch.no_grad():
        class_logits, location_logits, _ = model(inputs, raw_length)
        location_probabilities = torch.softmax(location_logits[0], dim=-1)
        class_probabilities = torch.softmax(class_logits[0], dim=0)
    window_bins = config.window_samples // config.stride
    bins = select_ordered_pair(location_logits[0], window_bins)
    maximum = max(0, len(raw) - config.window_samples)
    acts = {}
    for act_index, selected_bin in enumerate(bins):
        start = min(maximum, selected_bin * config.stride)
        region_last = min(
            class_probabilities.shape[-1], selected_bin + window_bins
        )
        region_probability = class_probabilities[
            act_index + 1, selected_bin:region_last
        ].mean()
        acts[f"act{act_index + 1}"] = {
            "start_sample": start,
            "end_sample_exclusive": start + config.window_samples,
            "boundary_confidence": float(
                location_probabilities[act_index, selected_bin].item()
            ),
            "region_probability": float(region_probability.item()),
        }
    return {
        "trial_file": str(trial_file.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "dataset_version": payload.get("dataset_version", "unknown"),
        "window_samples": config.window_samples,
        **acts,
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
