"""Train and evaluate the isolated V3 U-Time full-trial EMG model."""

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
    from .features_v3 import make_emg_features
    from .model_v2 import select_ordered_pair
    from .model_v3_utime import (
        UTimeConfig,
        UTimeEMG,
        boundary_distribution_loss,
        combined_start_scores,
        dense_labels,
        focal_segmentation_loss,
    )
except ImportError:
    from dataset_builder import (
        DEFAULT_EXTERNAL,
        DEFAULT_OUTPUT,
        DEFAULT_TRIAL_LOGS,
        build_dataset,
        load_emg_channels,
    )
    from features_v3 import make_emg_features
    from model_v2 import select_ordered_pair
    from model_v3_utime import (
        UTimeConfig,
        UTimeEMG,
        boundary_distribution_loss,
        combined_start_scores,
        dense_labels,
        focal_segmentation_loss,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescan", action="store_true")
    parser.add_argument("--trial-logs", type=Path, default=DEFAULT_TRIAL_LOGS)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--calibration-trials", type=int, default=5)
    parser.add_argument("--calibration-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class UTimeTrialDataset:
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
            self.cache[index] = make_emg_features(
                load_emg_channels(Path(self.rows[index]["raw_path"]))
            )
        features = self.cache[index]
        if self.training:
            features = features.copy()
            features *= self.rng.uniform(0.93, 1.07, size=(1, 9)).astype(np.float32)
            features += self.rng.normal(
                0.0, self.rng.uniform(0.0, 0.02), size=features.shape
            ).astype(np.float32)
            if self.rng.random() < 0.15:
                channel = int(self.rng.integers(0, 3))
                features[:, [channel, channel + 3, channel + 6]] = 0.0
        row = self.rows[index]
        starts = np.asarray(
            [int(row["act1_start"]), int(row["act2_start"])], dtype=np.int64
        )
        return self.torch.from_numpy(features), self.torch.from_numpy(starts)


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


def summarize_errors(errors: list[list[float]], confidences: list[float]) -> dict:
    combined = np.asarray(errors[0] + errors[1], dtype=float)
    result = {
        "starts_evaluated": int(len(combined)),
        "mae_samples": float(combined.mean()),
        "median_error_samples": float(np.median(combined)),
        "within_25_samples": float(np.mean(combined <= 25)),
        "within_50_samples": float(np.mean(combined <= 50)),
        "within_100_samples": float(np.mean(combined <= 100)),
        "within_250_samples": float(np.mean(combined <= 250)),
        "within_500_samples": float(np.mean(combined <= 500)),
        "act1_mae_samples": float(np.mean(errors[0])),
        "act2_mae_samples": float(np.mean(errors[1])),
    }
    if confidences:
        result["mean_confidence"] = float(np.mean(confidences))
    return result


def evaluate(model, loader, device, torch, region_weight=None) -> dict:
    model.eval()
    errors: list[list[float]] = [[], []]
    confidences = []
    weight = model.config.region_score_weight if region_weight is None else region_weight
    with torch.no_grad():
        for inputs, starts in loader:
            inputs = inputs.to(device)
            starts = starts.to(device)
            class_logits, boundary_logits = model(inputs)
            scores = combined_start_scores(
                class_logits,
                boundary_logits,
                model.config.window_samples,
                weight,
            )[0]
            bins = select_ordered_pair(scores, model.config.window_samples)
            probabilities = torch.softmax(scores, dim=-1)
            for act_index, predicted in enumerate(bins):
                actual = int(starts[0, act_index].item())
                errors[act_index].append(abs(predicted - actual))
                confidences.append(float(probabilities[act_index, predicted].item()))
    return summarize_errors(errors, confidences)


def training_loss(model, inputs, starts, class_weights):
    class_logits, boundary_logits = model(inputs)
    labels = dense_labels(starts, inputs.shape[1], model.config.window_samples)
    segmentation = focal_segmentation_loss(class_logits, labels, class_weights)
    boundary = boundary_distribution_loss(boundary_logits, starts)
    return boundary + 0.75 * segmentation


def make_loader(rows, torch, training, seed):
    from torch.utils.data import DataLoader

    return DataLoader(
        UTimeTrialDataset(rows, torch, training, seed),
        batch_size=1,
        shuffle=training,
        num_workers=0,
    )


def calibrate_session(model, rows, args, device, torch, class_weights):
    adapted = copy.deepcopy(model)
    for parameter in adapted.parameters():
        parameter.requires_grad = False
    trainable_modules = (
        adapted.session_adapter,
        adapted.decoder1,
        adapted.class_head,
        adapted.boundary_head,
    )
    parameters = []
    for module in trainable_modules:
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad = True
            parameters.append(parameter)
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-3)
    loader = make_loader(rows, torch, True, args.seed + 91)
    adapted.eval()
    for _ in range(args.calibration_epochs):
        for inputs, starts in loader:
            inputs = inputs.to(device)
            starts = starts.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(adapted, inputs, starts, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            optimizer.step()
    return adapted


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
        args.calibration_epochs = 1
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
        raise SystemExit("Dataset audit contains blocking warnings")

    import torch

    set_seed(torch, args.seed)
    device = choose_device(torch, args.device)
    print(f"V3 U-Time training on {device}")
    rows = read_csv(manifest)
    split_rows = {
        split: sorted(
            [row for row in rows if row["split"] == split],
            key=lambda row: (row["session"], int(row["trial_number"])),
        )
        for split in ("train", "validation", "test")
    }
    loaders = {
        split: make_loader(data, torch, split == "train", args.seed)
        for split, data in split_rows.items()
    }
    config = UTimeConfig()
    model = UTimeEMG(config).to(device)
    class_weights = torch.tensor([0.08, 1.0, 1.0], device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for inputs, starts in loaders["train"]:
            inputs = inputs.to(device)
            starts = starts.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, inputs, starts, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation = evaluate(model, loaders["validation"], device, torch)
        scheduler.step(validation["mae_samples"])
        learning_rate = float(optimizer.param_groups[0]["lr"])
        mean_loss = float(np.mean(losses))
        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "learning_rate": learning_rate,
            **{f"validation_{key}": value for key, value in validation.items()},
        })
        print(
            f"v3 {epoch:03d}/{args.epochs}: loss={mean_loss:.4f} "
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
    weight_candidates = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]
    weight_metrics = {
        str(weight): evaluate(
            model, loaders["validation"], device, torch, region_weight=weight
        )
        for weight in weight_candidates
    }
    config.region_score_weight = min(
        weight_candidates,
        key=lambda weight: weight_metrics[str(weight)]["mae_samples"],
    )
    print(
        f"Selected region weight={config.region_score_weight:g} "
        f"(val_mae={weight_metrics[str(config.region_score_weight)]['mae_samples']:.1f})"
    )

    zero_shot_validation = evaluate(model, loaders["validation"], device, torch)
    zero_shot_test = evaluate(model, loaders["test"], device, torch)
    calibration_count = min(args.calibration_trials, len(split_rows["validation"]) - 1)
    validation_adapter = calibrate_session(
        model,
        split_rows["validation"][:calibration_count],
        args,
        device,
        torch,
        class_weights,
    )
    validation_calibrated = evaluate(
        validation_adapter,
        make_loader(split_rows["validation"][calibration_count:], torch, False, args.seed),
        device,
        torch,
    )
    test_calibration_count = min(args.calibration_trials, len(split_rows["test"]) - 1)
    test_adapter = calibrate_session(
        model,
        split_rows["test"][:test_calibration_count],
        args,
        device,
        torch,
        class_weights,
    )
    test_calibrated = evaluate(
        test_adapter,
        make_loader(split_rows["test"][test_calibration_count:], torch, False, args.seed),
        device,
        torch,
    )

    results = {
        "dataset_version": audit["dataset_version"],
        "device": str(device),
        "seed": args.seed,
        "model": "utime_emg_v3",
        "history": history,
        "region_weight_validation": weight_metrics,
        "selected_region_weight": config.region_score_weight,
        "zero_shot_validation": zero_shot_validation,
        "zero_shot_test": zero_shot_test,
        "calibration_protocol": {
            "labeled_trials_per_session": args.calibration_trials,
            "epochs": args.calibration_epochs,
            "trainable": "session_adapter_final_decoder_and_output_heads",
        },
        "calibrated_validation_remaining_trials": validation_calibrated,
        "calibrated_test_remaining_trials": test_calibrated,
    }
    checkpoint_dir = args.artifacts / "checkpoints"
    checkpoint = checkpoint_dir / "auto_label_model_v3_utime.pt"
    atomic_save(torch, {
        "model_state": model.state_dict(),
        "model_config": config.to_dict(),
        "dataset_version": audit["dataset_version"],
        "metrics": results,
    }, checkpoint)
    atomic_save(torch, {
        "model_state": test_adapter.state_dict(),
        "model_config": config.to_dict(),
        "dataset_version": audit["dataset_version"],
        "calibrated_source_group": split_rows["test"][0]["source_group"],
        "calibration_trial_numbers": [
            row["trial_number"] for row in split_rows["test"][:test_calibration_count]
        ],
        "metrics": test_calibrated,
    }, checkpoint_dir / "auto_label_model_v3_test_calibrated.pt")
    metrics_path = checkpoint_dir / "training_metrics_v3.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved V3 checkpoint: {checkpoint}")
    print(json.dumps({
        "zero_shot_test": zero_shot_test,
        "calibrated_test_remaining_trials": test_calibrated,
    }, indent=2))


if __name__ == "__main__":
    main()
