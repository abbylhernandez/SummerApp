"""Cache pretrained MobileNetV3 frame features for every labeled trial video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .dataset_builder import DEFAULT_OUTPUT
except ImportError:
    from dataset_builder import DEFAULT_OUTPUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def make_encoder(device):
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT
    backbone = mobilenet_v3_small(weights=weights).features.to(device).eval()
    model = nn.Sequential(backbone, nn.AdaptiveAvgPool2d(1), nn.Flatten(1)).to(device)
    model.eval()
    return model


def prepare_frames(frames: list[np.ndarray], device):
    import torch
    from torch.nn import functional as F

    array = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    tensor = tensor.div_(255.0)
    mean = tensor.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = tensor.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (tensor - mean) / std


def encode_video(path: Path, encoder, device, batch_size: int):
    import torch

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    batches = []
    pending = []
    with torch.no_grad():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pending.append(frame)
            if len(pending) >= batch_size:
                batches.append(encoder(prepare_frames(pending, device)).cpu().numpy())
                pending.clear()
        if pending:
            batches.append(encoder(prepare_frames(pending, device)).cpu().numpy())
    capture.release()
    if not batches:
        raise ValueError(f"No readable frames in {path}")
    features = np.concatenate(batches).astype(np.float16)
    return features, fps, expected_frames


def main() -> None:
    args = parse_args()
    import torch

    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(requested)
    rows = read_rows(args.artifacts / "localization_manifest.csv")
    cache_dir = args.artifacts / "video_features_mobilenet_v3_small"
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder = make_encoder(device)
    inventory = []
    for index, row in enumerate(rows, start=1):
        output = cache_dir / f"{row['pair_id']}.npz"
        if output.exists() and not args.overwrite:
            with np.load(output) as cached:
                features = cached["features"]
                fps = float(cached["fps"])
                expected_frames = int(cached["expected_frames"])
        else:
            features, fps, expected_frames = encode_video(
                Path(row["video_path"]), encoder, device, args.batch_size
            )
            np.savez_compressed(
                output,
                features=features,
                fps=np.float32(fps),
                expected_frames=np.int32(expected_frames),
                video_path=np.asarray(row["video_path"]),
            )
        inventory.append({
            "pair_id": row["pair_id"],
            "split": row["split"],
            "source_group": row["source_group"],
            "frames": int(len(features)),
            "reported_frames": expected_frames,
            "fps": fps,
            "feature_dimensions": int(features.shape[1]),
            "cache_path": str(output.resolve()),
        })
        print(
            f"[{index:03d}/{len(rows):03d}] {row['pair_id']} "
            f"frames={len(features)} features={features.shape[1]}"
        )
    summary = {
        "encoder": "torchvision_mobilenet_v3_small_imagenet1k_v1",
        "device": str(device),
        "trials": len(inventory),
        "total_frames": sum(item["frames"] for item in inventory),
        "inventory": inventory,
    }
    (cache_dir / "inventory.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "inventory"}, indent=2))


if __name__ == "__main__":
    main()
