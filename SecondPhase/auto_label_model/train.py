"""Train the removable EMG Act classifier and automatic window localizer.

Use ``--rescan`` whenever new Phase 2 labels are available. The dataset
builder re-audits duplicates, updates manifests, and keeps related trials and
all derivatives in the same split before training begins.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np

try:
    from .dataset_builder import (
        DEFAULT_EXTERNAL,
        DEFAULT_OUTPUT,
        DEFAULT_TRIAL_LOGS,
        build_dataset,
        load_emg_channels,
    )
except ImportError:
    from dataset_builder import (
        DEFAULT_EXTERNAL,
        DEFAULT_OUTPUT,
        DEFAULT_TRIAL_LOGS,
        build_dataset,
        load_emg_channels,
    )


PACKAGE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescan", action="store_true", help="Rebuild/audit manifests first")
    parser.add_argument("--trial-logs", type=Path, default=DEFAULT_TRIAL_LOGS)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--localize-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true", help="One epoch per stage for a smoke test")
    return parser.parse_args()


def require_torch():
    try:
        import torch
        from torch.nn import functional as functional
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit(
            "PyTorch is required for training. From this folder run:\n"
            "  ..\\.venv\\Scripts\\python.exe -m pip install "
            "-r auto_label_model\\requirements.txt"
        ) from error
    try:
        from .model import (
            AutoLabelNet,
            ModelConfig,
            gaussian_location_loss,
            ordering_loss,
            prediction_from_logits,
        )
    except ImportError:
        from model import (
            AutoLabelNet,
            ModelConfig,
            gaussian_location_loss,
            ordering_loss,
            prediction_from_logits,
        )
    return {
        "torch": torch,
        "F": functional,
        "DataLoader": DataLoader,
        "AutoLabelNet": AutoLabelNet,
        "ModelConfig": ModelConfig,
        "gaussian_location_loss": gaussian_location_loss,
        "ordering_loss": ordering_loss,
        "prediction_from_logits": prediction_from_logits,
    }


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def robust_normalize(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    median = np.median(data, axis=0, keepdims=True)
    mad = np.median(np.abs(data - median), axis=0, keepdims=True)
    scale = np.maximum(mad * 1.4826, data.std(axis=0, keepdims=True) * 0.1)
    scale = np.maximum(scale, 1e-5)
    return np.clip((data - median) / scale, -12.0, 12.0).astype(np.float32)


def augment_clip(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = data.copy()
    output *= rng.uniform(0.90, 1.10, size=(1, output.shape[1])).astype(np.float32)
    output += rng.normal(0.0, 0.015, size=output.shape).astype(np.float32)
    if rng.random() < 0.35:
        shift = int(rng.integers(-20, 21))
        output = np.roll(output, shift, axis=0)
    return output


class ClipDataset:
    def __init__(self, rows: list[dict], torch_module, training: bool, seed: int):
        self.rows = rows
        self.torch = torch_module
        self.training = training
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        data = robust_normalize(load_emg_channels(Path(row["canonical_path"])))
        if self.training:
            data = augment_clip(data, self.rng)
        return self.torch.from_numpy(data), int(row["label"])


class LocalizationDataset:
    def __init__(self, rows: list[dict], torch_module):
        self.rows = rows
        self.torch = torch_module

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        data = robust_normalize(load_emg_channels(Path(row["raw_path"])))
        starts = np.asarray([int(row["act1_start"]), int(row["act2_start"])], dtype=np.int64)
        return (
            self.torch.from_numpy(data),
            self.torch.from_numpy(starts),
            self.torch.tensor(len(data), dtype=self.torch.long),
        )


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(torch, requested: str):
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but no CUDA-capable PyTorch device is available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def classifier_accuracy(model, loader, device, torch) -> float:
    if len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            predictions = model.classify_clips(inputs).argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += len(labels)
    return correct / max(1, total)


def pretrain_classifier(model, loaders, args, modules, device) -> dict:
    torch = modules["torch"]
    functional = modules["F"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = -1.0
    history = []
    for epoch in range(1, args.pretrain_epochs + 1):
        model.train()
        losses = []
        for inputs, labels in loaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model.classify_clips(inputs), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation = classifier_accuracy(model, loaders["validation"], device, torch)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "validation_accuracy": validation,
        })
        print(
            f"pretrain {epoch:03d}/{args.pretrain_epochs}: "
            f"loss={history[-1]['loss']:.4f} val_accuracy={validation:.3f}"
        )
        score = validation if np.isfinite(validation) else -1.0
        if score > best_validation:
            best_validation = score
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        "history": history,
        "validation_accuracy": classifier_accuracy(
            model, loaders["validation"], device, torch
        ),
        "test_accuracy": classifier_accuracy(model, loaders["test"], device, torch),
    }


def localization_metrics(model, loader, device, modules) -> dict:
    torch = modules["torch"]
    predict = modules["prediction_from_logits"]
    errors = []
    confidences = []
    model.eval()
    with torch.no_grad():
        for inputs, starts, raw_length in loader:
            inputs = inputs.to(device)
            starts = starts.to(device)
            raw_length = raw_length.to(device)
            logits = model.localize(inputs, raw_length)
            predicted, confidence = predict(logits, model.config.stride)
            errors.extend(torch.abs(predicted - starts).cpu().numpy().reshape(-1).tolist())
            confidences.extend(confidence.cpu().numpy().reshape(-1).tolist())
    error_array = np.asarray(errors, dtype=float)
    return {
        "mae_samples": float(error_array.mean()) if len(error_array) else float("nan"),
        "median_error_samples": float(np.median(error_array)) if len(error_array) else float("nan"),
        "within_25_samples": float(np.mean(error_array <= 25)) if len(error_array) else float("nan"),
        "within_50_samples": float(np.mean(error_array <= 50)) if len(error_array) else float("nan"),
        "within_100_samples": float(np.mean(error_array <= 100)) if len(error_array) else float("nan"),
        "mean_confidence": float(np.mean(confidences)) if confidences else float("nan"),
    }


def train_localizer(model, loaders, args, modules, device) -> dict:
    torch = modules["torch"]
    gaussian_loss = modules["gaussian_location_loss"]
    order_loss = modules["ordering_loss"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate * 0.5, weight_decay=2e-3
    )
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    history = []
    stride = model.config.stride
    window_bins = model.config.window_samples // stride
    for epoch in range(1, args.localize_epochs + 1):
        model.train()
        losses = []
        for inputs, starts, raw_length in loaders["train"]:
            inputs = inputs.to(device)
            starts = starts.to(device)
            raw_length = raw_length.to(device)
            target_bins = torch.round(starts.float() / stride).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model.localize(inputs, raw_length)
            location = gaussian_loss(logits, target_bins)
            ordering = order_loss(logits, window_bins)
            loss = location + 0.15 * ordering
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation = localization_metrics(model, loaders["validation"], device, modules)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else float("nan"),
            **{f"validation_{key}": value for key, value in validation.items()},
        })
        print(
            f"localize {epoch:03d}/{args.localize_epochs}: "
            f"loss={history[-1]['loss']:.4f} "
            f"val_mae={validation['mae_samples']:.1f} samples"
        )
        if validation["mae_samples"] < best_mae:
            best_mae = validation["mae_samples"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        "history": history,
        "validation": localization_metrics(model, loaders["validation"], device, modules),
        "test": localization_metrics(model, loaders["test"], device, modules),
    }


def atomic_torch_save(torch, payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    handle.close()
    try:
        torch.save(payload, handle.name)
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.remove(handle.name)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.pretrain_epochs = 1
        args.localize_epochs = 1
    if args.rescan:
        build_dataset(args.trial_logs, args.external, args.artifacts)

    clip_manifest = args.artifacts / "clip_manifest.csv"
    localization_manifest = args.artifacts / "localization_manifest.csv"
    audit_path = args.artifacts / "dataset_audit.json"
    for required in (clip_manifest, localization_manifest, audit_path):
        if not required.exists():
            raise SystemExit(f"Missing {required}; run dataset_builder.py first or use --rescan")

    modules = require_torch()
    torch = modules["torch"]
    DataLoader = modules["DataLoader"]
    set_seed(torch, args.seed)
    device = select_device(torch, args.device)
    print(f"Training on {device}")

    clip_rows = read_csv(clip_manifest)
    localization_rows = read_csv(localization_manifest)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    blocking = {
        "local_errors": audit.get("local_errors", []),
        "near_duplicate_warnings": audit.get("near_duplicate_warnings", []),
        "clip_overlap_warnings": audit.get("clip_overlap_warnings", []),
    }
    if any(blocking.values()):
        raise SystemExit(
            "Training stopped because dataset_audit.json contains matching or "
            "overlap warnings. Resolve them before training."
        )
    clip_loaders = {}
    localization_loaders = {}
    for split in ("train", "validation", "test"):
        split_clip_rows = [row for row in clip_rows if row["split"] == split]
        clip_loaders[split] = DataLoader(
            ClipDataset(split_clip_rows, torch, split == "train", args.seed),
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
        split_localization_rows = [
            row for row in localization_rows if row["split"] == split
        ]
        localization_loaders[split] = DataLoader(
            LocalizationDataset(split_localization_rows, torch),
            batch_size=1,
            shuffle=split == "train",
            num_workers=0,
        )

    config = modules["ModelConfig"]()
    model = modules["AutoLabelNet"](config).to(device)
    classifier_results = pretrain_classifier(
        model, clip_loaders, args, modules, device
    )
    localization_results = train_localizer(
        model, localization_loaders, args, modules, device
    )
    checkpoint = args.artifacts / "checkpoints" / "auto_label_model.pt"
    metrics = {
        "dataset_version": audit["dataset_version"],
        "device": str(device),
        "seed": args.seed,
        "classifier": classifier_results,
        "localization": localization_results,
    }
    atomic_torch_save(torch, {
        "model_state": model.state_dict(),
        "model_config": config.to_dict(),
        "dataset_version": audit["dataset_version"],
        "metrics": metrics,
    }, checkpoint)
    metrics_path = args.artifacts / "checkpoints" / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint}")
    print(json.dumps(localization_results["test"], indent=2))


if __name__ == "__main__":
    main()
