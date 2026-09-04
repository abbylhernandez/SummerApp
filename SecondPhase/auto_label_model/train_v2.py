"""Train the V2 full-trial dense TCN without modifying the V1 checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import statistics
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
    from .model_v2 import (
        FullTrialTCN,
        ModelV2Config,
        dense_targets,
        gaussian_boundary_loss,
        ordering_loss,
        select_ordered_pair,
    )
    from .train import robust_normalize
except ImportError:
    from dataset_builder import (
        DEFAULT_EXTERNAL,
        DEFAULT_OUTPUT,
        DEFAULT_TRIAL_LOGS,
        build_dataset,
        load_emg_channels,
    )
    from model_v2 import (
        FullTrialTCN,
        ModelV2Config,
        dense_targets,
        gaussian_boundary_loss,
        ordering_loss,
        select_ordered_pair,
    )
    from train import robust_normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescan", action="store_true")
    parser.add_argument("--trial-logs", type=Path, default=DEFAULT_TRIAL_LOGS)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class FullTrialDataset:
    def __init__(self, rows: list[dict], torch_module, training: bool, seed: int):
        self.rows = rows
        self.torch = torch_module
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        if index not in self.cache:
            raw = load_emg_channels(Path(self.rows[index]["raw_path"]))
            self.cache[index] = robust_normalize(raw)
        data = self.cache[index]
        if self.training:
            data = data.copy()
            data *= self.rng.uniform(0.92, 1.08, size=(1, data.shape[1])).astype(np.float32)
            data += self.rng.normal(
                0.0, self.rng.uniform(0.0, 0.025), size=data.shape
            ).astype(np.float32)
        row = self.rows[index]
        starts = np.asarray(
            [int(row["act1_start"]), int(row["act2_start"])], dtype=np.int64
        )
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


def choose_device(torch, requested: str):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return torch.device(requested)


def baseline_metrics(train_rows: list[dict], rows: list[dict]) -> dict:
    fractions = [
        statistics.median(
            int(row[f"act{act}_start"]) / int(row["raw_samples"])
            for row in train_rows
        )
        for act in (1, 2)
    ]
    errors = [[], []]
    for row in rows:
        length = int(row["raw_samples"])
        for act_index in range(2):
            predicted = round(fractions[act_index] * length)
            actual = int(row[f"act{act_index + 1}_start"])
            errors[act_index].append(abs(predicted - actual))
    return summarize_errors(errors, confidences=[])


def summarize_errors(errors: list[list[float]], confidences: list[float]) -> dict:
    combined = np.asarray(errors[0] + errors[1], dtype=float)
    result = {
        "mae_samples": float(combined.mean()),
        "median_error_samples": float(np.median(combined)),
        "within_25_samples": float(np.mean(combined <= 25)),
        "within_50_samples": float(np.mean(combined <= 50)),
        "within_100_samples": float(np.mean(combined <= 100)),
        "act1_mae_samples": float(np.mean(errors[0])),
        "act2_mae_samples": float(np.mean(errors[1])),
    }
    if confidences:
        result["mean_confidence"] = float(np.mean(confidences))
    return result


def evaluate(model, loader, device, torch, evidence_multiplier=None) -> dict:
    model.eval()
    errors: list[list[float]] = [[], []]
    confidences = []
    with torch.no_grad():
        for inputs, starts, raw_length in loader:
            inputs = inputs.to(device)
            starts = starts.to(device)
            raw_length = raw_length.to(device)
            _, location_logits, evidence = model(inputs, raw_length)
            if evidence_multiplier is not None:
                prior_logits = (
                    location_logits
                    - model.config.inference_evidence_multiplier * evidence
                )
                location_logits = prior_logits + evidence_multiplier * evidence
            logits = location_logits[0]
            act1_bin, act2_bin = select_ordered_pair(
                logits, model.config.window_samples // model.config.stride
            )
            predicted = [
                act1_bin * model.config.stride,
                act2_bin * model.config.stride,
            ]
            probabilities = torch.softmax(logits, dim=-1)
            for act_index in range(2):
                actual = int(starts[0, act_index].item())
                errors[act_index].append(abs(predicted[act_index] - actual))
                confidences.append(
                    float(probabilities[act_index, [act1_bin, act2_bin][act_index]].item())
                )
    return summarize_errors(errors, confidences)


def atomic_save(torch, payload: dict, path: Path) -> None:
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
        args.epochs = 1
        args.patience = 1
    if args.rescan:
        build_dataset(args.trial_logs, args.external, args.artifacts)

    manifest = args.artifacts / "localization_manifest.csv"
    audit_path = args.artifacts / "dataset_audit.json"
    if not manifest.exists() or not audit_path.exists():
        raise SystemExit("Missing audited dataset; run with --rescan")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    blocking = (
        audit.get("local_errors", [])
        + audit.get("near_duplicate_warnings", [])
        + audit.get("clip_overlap_warnings", [])
    )
    if blocking:
        raise SystemExit("Dataset audit contains blocking overlap/matching warnings")

    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    set_seed(torch, args.seed)
    device = choose_device(torch, args.device)
    print(f"V2 training on {device}")
    rows = read_csv(manifest)
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    loaders = {
        split: DataLoader(
            FullTrialDataset(data, torch, split == "train", args.seed),
            batch_size=1,
            shuffle=split == "train",
            num_workers=0,
        )
        for split, data in split_rows.items()
    }

    prior_fractions = [
        statistics.median(
            int(row[f"act{act}_start"]) / int(row["raw_samples"])
            for row in split_rows["train"]
        )
        for act in (1, 2)
    ]
    config = ModelV2Config(
        act1_prior_fraction=prior_fractions[0],
        act2_prior_fraction=prior_fractions[1],
    )
    model = FullTrialTCN(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )
    class_weights = torch.tensor([0.15, 1.0, 1.0], device=device)
    baselines = {
        split: baseline_metrics(split_rows["train"], split_rows[split])
        for split in ("validation", "test")
    }
    print(
        "Position baseline: "
        f"val_mae={baselines['validation']['mae_samples']:.1f} "
        f"test_mae={baselines['test']['mae_samples']:.1f} samples"
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, starts, raw_length in loaders["train"]:
            inputs = inputs.to(device)
            starts = starts.to(device)
            raw_length = raw_length.to(device)
            optimizer.zero_grad(set_to_none=True)
            class_logits, location_logits, evidence = model(inputs, raw_length)
            targets = dense_targets(
                starts,
                class_logits.shape[-1],
                config.stride,
                config.window_samples,
            )
            segmentation = F.cross_entropy(class_logits, targets, weight=class_weights)
            boundary = gaussian_boundary_loss(location_logits, starts, config.stride)
            ordered = ordering_loss(
                location_logits, config.window_samples, config.stride
            )
            evidence_regularizer = evidence.square().mean()
            loss = boundary + 0.65 * segmentation + 0.10 * ordered + 0.005 * evidence_regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))

        validation = evaluate(model, loaders["validation"], device, torch)
        mean_loss = float(np.mean(losses))
        scheduler.step(validation["mae_samples"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "learning_rate": learning_rate,
            **{f"validation_{key}": value for key, value in validation.items()},
        })
        print(
            f"v2 {epoch:03d}/{args.epochs}: loss={mean_loss:.4f} "
            f"val_mae={validation['mae_samples']:.1f} "
            f"within100={validation['within_100_samples']:.3f} "
            f"lr={learning_rate:.1e}"
        )
        if validation["mae_samples"] < best_validation - 1.0:
            best_validation = validation["mae_samples"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    model.load_state_dict(best_state)
    calibration_candidates = [
        0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0
    ]
    calibration_results = {
        str(multiplier): evaluate(
            model,
            loaders["validation"],
            device,
            torch,
            evidence_multiplier=multiplier,
        )
        for multiplier in calibration_candidates
    }
    selected_multiplier = min(
        calibration_candidates,
        key=lambda value: calibration_results[str(value)]["mae_samples"],
    )
    config.inference_evidence_multiplier = selected_multiplier
    print(
        f"Calibrated EMG evidence multiplier={selected_multiplier:g} "
        f"(val_mae={calibration_results[str(selected_multiplier)]['mae_samples']:.1f})"
    )
    results = {
        "dataset_version": audit["dataset_version"],
        "device": str(device),
        "seed": args.seed,
        "model": "full_trial_tcn_v2",
        "position_priors": prior_fractions,
        "evidence_calibration": {
            "selected_multiplier": selected_multiplier,
            "validation_candidates": calibration_results,
        },
        "baseline": baselines,
        "history": history,
        "validation": evaluate(model, loaders["validation"], device, torch),
        "test": evaluate(model, loaders["test"], device, torch),
    }
    checkpoint = args.artifacts / "checkpoints" / "auto_label_model_v2.pt"
    atomic_save(torch, {
        "model_state": model.state_dict(),
        "model_config": config.to_dict(),
        "dataset_version": audit["dataset_version"],
        "metrics": results,
    }, checkpoint)
    metrics_path = args.artifacts / "checkpoints" / "training_metrics_v2.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved V2 checkpoint: {checkpoint}")
    print(json.dumps(results["test"], indent=2))


if __name__ == "__main__":
    main()
