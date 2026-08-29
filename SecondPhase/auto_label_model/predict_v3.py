"""Predict ordered Act windows using the V3 U-Time checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .dataset_builder import DEFAULT_OUTPUT, load_emg_channels
    from .features_v3 import make_emg_features
    from .model_v2 import select_ordered_pair
    from .model_v3_utime import UTimeConfig, UTimeEMG, combined_start_scores
except ImportError:
    from dataset_builder import DEFAULT_OUTPUT, load_emg_channels
    from features_v3 import make_emg_features
    from model_v2 import select_ordered_pair
    from model_v3_utime import UTimeConfig, UTimeEMG, combined_start_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_file", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT / "checkpoints" / "auto_label_model_v3_utime.pt",
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
    config = UTimeConfig(**payload["model_config"])
    model = UTimeEMG(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    raw = load_emg_channels(trial_file)
    features = torch.from_numpy(make_emg_features(raw)).unsqueeze(0).to(device)
    with torch.no_grad():
        class_logits, boundary_logits = model(features)
        scores = combined_start_scores(
            class_logits,
            boundary_logits,
            config.window_samples,
            config.region_score_weight,
        )[0]
        probabilities = torch.softmax(scores, dim=-1)
    starts = select_ordered_pair(scores, config.window_samples)
    maximum = max(0, len(raw) - config.window_samples)
    acts = {}
    for act_index, predicted in enumerate(starts):
        start = min(maximum, predicted)
        acts[f"act{act_index + 1}"] = {
            "start_sample": start,
            "end_sample_exclusive": start + config.window_samples,
            "confidence": float(probabilities[act_index, predicted].item()),
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
