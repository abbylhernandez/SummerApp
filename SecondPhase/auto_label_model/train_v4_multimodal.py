"""Train the isolated pretrained-Video+EMG temporal boundary model."""

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
    from .dataset_builder import DEFAULT_OUTPUT
    from .model_v2 import select_ordered_pair
    from .model_v3_utime import (
        boundary_distribution_loss,
        combined_start_scores,
        dense_labels,
        focal_segmentation_loss,
    )
    from .model_v4_multimodal import MultimodalConfig, VideoEMGFusionNet
    from .multimodal_data import load_emg_times, load_multimodal_trial
except ImportError:
    from dataset_builder import DEFAULT_OUTPUT
    from model_v2 import select_ordered_pair
    from model_v3_utime import (
        boundary_distribution_loss,
        combined_start_scores,
        dense_labels,
        focal_segmentation_loss,
    )
    from model_v4_multimodal import MultimodalConfig, VideoEMGFusionNet
    from multimodal_data import load_emg_times, load_multimodal_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def synchronized_rows(rows: list[dict], cache_dir: Path, tolerance: float = 0.5):
    accepted, excluded = [], []
    for row in rows:
        with np.load(cache_dir / f"{row['pair_id']}.npz") as cached:
            frames = len(cached["features"])
            fps = float(cached["fps"])
        emg_duration = float(load_emg_times(Path(row["raw_path"]))[-1])
        video_duration = (frames - 1) / fps
        difference = video_duration - emg_duration
        if abs(difference) > tolerance:
            excluded.append({
                "pair_id": row["pair_id"],
                "split": row["split"],
                "video_path": row["video_path"],
                "duration_difference_seconds": difference,
            })
        else:
            accepted.append(row)
    return accepted, excluded


class MultimodalDataset:
    def __init__(self, rows, cache_dir, torch_module, training, seed):
        self.rows = rows
        self.cache_dir = cache_dir
        self.torch = torch_module
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index not in self.cache:
            self.cache[index] = load_multimodal_trial(self.rows[index], self.cache_dir)
        item = self.cache[index]
        appearance = item["appearance"]
        motion = item["motion"]
        emg = item["emg"]
        if self.training:
            appearance = appearance.copy()
            motion = motion.copy()
            emg = emg.copy()
            appearance += self.rng.normal(0, 0.01, appearance.shape).astype(np.float32)
            motion += self.rng.normal(0, 0.01, motion.shape).astype(np.float32)
            emg += self.rng.normal(0, 0.015, emg.shape).astype(np.float32)
            choice = self.rng.random()
            if choice < 0.08:
                emg[:] = 0
            elif choice < 0.16:
                motion[:] = 0
        return (
            self.torch.from_numpy(appearance),
            self.torch.from_numpy(motion),
            self.torch.from_numpy(emg),
            self.torch.from_numpy(item["sample_indices"]),
            self.torch.from_numpy(item["starts_samples"]),
            self.torch.from_numpy(item["starts_frames"]),
            self.torch.tensor(item["window_frames"], dtype=self.torch.long),
            self.torch.tensor(int(self.rows[index]["raw_samples"]), dtype=self.torch.long),
        )


def make_loader(rows, cache_dir, torch, training, seed):
    from torch.utils.data import DataLoader
    return DataLoader(
        MultimodalDataset(rows, cache_dir, torch, training, seed),
        batch_size=1,
        shuffle=training,
        num_workers=0,
    )


def set_seed(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_from_errors(errors, confidences):
    combined = np.asarray(errors[0] + errors[1], dtype=float)
    return {
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
        "mean_confidence": float(np.mean(confidences)),
    }


def evaluate(
    model,
    loader,
    device,
    torch,
    region_weight=None,
    modality="both",
    evidence_multiplier=None,
):
    model.eval()
    errors = [[], []]
    confidences = []
    weight = model.config.region_score_weight if region_weight is None else region_weight
    with torch.no_grad():
        for appearance, motion, emg, sample_indices, starts_samples, _, window_frames, raw_samples in loader:
            appearance, motion, emg = appearance.to(device), motion.to(device), emg.to(device)
            if modality == "video":
                emg.zero_()
            elif modality == "emg":
                appearance.zero_()
                motion.zero_()
            class_logits, boundary_logits = model(appearance, motion, emg)
            frames = int(window_frames.item())
            scores = combined_start_scores(
                class_logits, boundary_logits, frames, weight
            )[0]
            if evidence_multiplier is not None:
                fractions = sample_indices[0].to(
                    device=device, dtype=scores.dtype
                ) / raw_samples.to(device=device, dtype=scores.dtype).clamp_min(1)
                centers = scores.new_tensor([
                    model.config.act1_prior_fraction,
                    model.config.act2_prior_fraction,
                ]).view(2, 1)
                prior = -0.5 * (
                    (fractions.view(1, -1) - centers)
                    / model.config.prior_sigma_fraction
                ).square()
                scores = prior + evidence_multiplier * scores
            predicted_frames = select_ordered_pair(scores, frames)
            probabilities = torch.softmax(scores, dim=-1)
            for act_index, frame in enumerate(predicted_frames):
                predicted_sample = int(sample_indices[0, frame].item())
                actual_sample = int(starts_samples[0, act_index].item())
                errors[act_index].append(abs(predicted_sample - actual_sample))
                confidences.append(float(probabilities[act_index, frame].item()))
    return metrics_from_errors(errors, confidences)


def baseline_metrics(train_rows, evaluation_rows):
    fractions = [
        statistics.median(
            int(row[f"act{act}_start"]) / int(row["raw_samples"])
            for row in train_rows
        )
        for act in (1, 2)
    ]
    errors = [[], []]
    for row in evaluation_rows:
        for act_index in range(2):
            predicted = round(fractions[act_index] * int(row["raw_samples"]))
            actual = int(row[f"act{act_index + 1}_start"])
            errors[act_index].append(abs(predicted - actual))
    return metrics_from_errors(errors, [0.0] * (2 * len(evaluation_rows)))


def training_loss(model, batch, device, class_weights):
    appearance, motion, emg, _, _, starts_frames, window_frames, _ = batch
    appearance, motion, emg = appearance.to(device), motion.to(device), emg.to(device)
    starts_frames = starts_frames.to(device)
    class_logits, boundary_logits = model(appearance, motion, emg)
    labels = dense_labels(
        starts_frames, class_logits.shape[-1], int(window_frames.item())
    )
    segmentation = focal_segmentation_loss(class_logits, labels, class_weights)
    boundary = boundary_distribution_loss(boundary_logits, starts_frames, sigma_samples=1.0)
    return boundary + 0.8 * segmentation


def atomic_save(torch, payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp")
    handle.close()
    try:
        torch.save(payload, handle.name)
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.remove(handle.name)


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 1
        args.patience = 1
    import torch
    set_seed(torch, args.seed)
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device = torch.device(requested)
    print(f"V4 Video+EMG training on {device}")
    cache_dir = args.artifacts / "video_features_mobilenet_v3_small"
    rows, excluded = synchronized_rows(
        read_rows(args.artifacts / "localization_manifest.csv"), cache_dir
    )
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    print(
        f"Synchronized trials: train={len(split_rows['train'])} "
        f"validation={len(split_rows['validation'])} test={len(split_rows['test'])}; "
        f"excluded={len(excluded)}"
    )
    loaders = {
        split: make_loader(data, cache_dir, torch, split == "train", args.seed)
        for split, data in split_rows.items()
    }
    prior_fractions = [
        statistics.median(
            int(row[f"act{act}_start"]) / int(row["raw_samples"])
            for row in split_rows["train"]
        )
        for act in (1, 2)
    ]
    config = MultimodalConfig(
        act1_prior_fraction=prior_fractions[0],
        act2_prior_fraction=prior_fractions[1],
    )
    model = VideoEMGFusionNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )
    class_weights = torch.tensor([0.05, 1.0, 1.0], device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model, batch, device, class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation = evaluate(model, loaders["validation"], device, torch)
        scheduler.step(validation["mae_samples"])
        mean_loss = float(np.mean(losses))
        lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "learning_rate": lr,
            **{f"validation_{key}": value for key, value in validation.items()},
        })
        print(
            f"v4 {epoch:03d}/{args.epochs}: loss={mean_loss:.4f} "
            f"val_mae={validation['mae_samples']:.1f} "
            f"within100={validation['within_100_samples']:.3f} lr={lr:.1e}"
        )
        if validation["mae_samples"] < best_validation - 1.0:
            best_validation = validation["mae_samples"]
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break
    model.load_state_dict(best_state)
    region_candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    region_metrics = {
        str(weight): evaluate(
            model, loaders["validation"], device, torch, region_weight=weight
        )
        for weight in region_candidates
    }
    config.region_score_weight = min(
        region_candidates,
        key=lambda weight: region_metrics[str(weight)]["mae_samples"],
    )
    evidence_candidates = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
    evidence_metrics = {
        str(multiplier): evaluate(
            model,
            loaders["validation"],
            device,
            torch,
            evidence_multiplier=multiplier,
        )
        for multiplier in evidence_candidates
    }
    config.evidence_multiplier = min(
        evidence_candidates,
        key=lambda multiplier: evidence_metrics[str(multiplier)]["mae_samples"],
    )
    results = {
        "model": "mobilenet_v3_small_bigru_video_emg_v4",
        "device": str(device),
        "seed": args.seed,
        "excluded_sync_outliers": excluded,
        "split_counts": {key: len(value) for key, value in split_rows.items()},
        "history": history,
        "selected_region_weight": config.region_score_weight,
        "region_weight_validation": region_metrics,
        "position_priors": prior_fractions,
        "selected_evidence_multiplier": config.evidence_multiplier,
        "evidence_multiplier_validation": evidence_metrics,
        "position_baseline_test": baseline_metrics(split_rows["train"], split_rows["test"]),
        "validation": evaluate(model, loaders["validation"], device, torch),
        "test_video_emg": evaluate(model, loaders["test"], device, torch, modality="both"),
        "test_video_only": evaluate(model, loaders["test"], device, torch, modality="video"),
        "test_emg_only": evaluate(model, loaders["test"], device, torch, modality="emg"),
        "test_position_video_emg": evaluate(
            model,
            loaders["test"],
            device,
            torch,
            modality="both",
            evidence_multiplier=config.evidence_multiplier,
        ),
    }
    checkpoint_dir = args.artifacts / "checkpoints"
    checkpoint = checkpoint_dir / "auto_label_model_v4_video_emg.pt"
    atomic_save(torch, {
        "model_state": model.state_dict(),
        "model_config": config.to_dict(),
        "video_encoder": "torchvision_mobilenet_v3_small_imagenet1k_v1",
        "metrics": results,
    }, checkpoint)
    (checkpoint_dir / "training_metrics_v4.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Selected region weight={config.region_score_weight:g}; "
        f"evidence multiplier={config.evidence_multiplier:g}"
    )
    print(f"Saved V4 checkpoint: {checkpoint}")
    print(json.dumps({
        "position_baseline_test": results["position_baseline_test"],
        "test_video_emg": results["test_video_emg"],
        "test_video_only": results["test_video_only"],
        "test_emg_only": results["test_emg_only"],
        "test_position_video_emg": results["test_position_video_emg"],
    }, indent=2))


if __name__ == "__main__":
    main()
