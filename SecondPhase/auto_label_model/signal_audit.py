"""Audit cross-session separability of background, Act 1, and Act 2 EMG windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .dataset_builder import DEFAULT_OUTPUT, load_emg_channels
    from .features_v3 import make_emg_features
except ImportError:
    from dataset_builder import DEFAULT_OUTPUT, load_emg_channels
    from features_v3 import make_emg_features


CLASS_NAMES = ("background", "act1", "act2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def overlaps(start: int, other_start: int, width: int) -> bool:
    return start < other_start + width and other_start < start + width


def background_starts(length: int, starts: list[int], width: int) -> list[int]:
    candidates = [
        start for start in range(0, max(1, length - width + 1), max(50, width // 2))
        if all(not overlaps(start, activity, width) for activity in starts)
    ]
    if not candidates:
        return []
    target_fractions = (0.20, 0.80)
    chosen = []
    for fraction in target_fractions:
        target = fraction * max(0, length - width)
        available = [value for value in candidates if value not in chosen]
        if available:
            chosen.append(min(available, key=lambda value: abs(value - target)))
    return chosen


def summarize_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    rms = np.sqrt(np.mean(np.square(window), axis=0))
    absolute_mean = np.mean(np.abs(window), axis=0)
    q10, q25, q50, q75, q90 = np.percentile(window, (10, 25, 50, 75, 90), axis=0)
    maximum = np.max(np.abs(window), axis=0)
    waveform_length = np.mean(np.abs(np.diff(window, axis=0)), axis=0)
    zero_crossing = np.mean(window[:-1] * window[1:] < 0, axis=0)
    return np.concatenate([
        mean, std, rms, absolute_mean,
        q10, q25, q50, q75, q90,
        maximum, waveform_length, zero_crossing,
    ]).astype(np.float32)


def build_examples(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    features = []
    labels = []
    metadata = []
    for row in rows:
        raw = load_emg_channels(Path(row["raw_path"]))
        signal = make_emg_features(raw)
        width = int(row["window_samples"])
        starts = [int(row["act1_start"]), int(row["act2_start"])]
        examples = [(starts[0], 1), (starts[1], 2)]
        examples.extend((start, 0) for start in background_starts(len(raw), starts, width))
        for start, label in examples:
            window = signal[start:start + width]
            if len(window) != width:
                continue
            features.append(summarize_window(window))
            labels.append(label)
            metadata.append({
                "split": row["split"],
                "source_group": row["source_group"],
                "session": row["session"],
                "trial_number": int(row["trial_number"]),
                "start": start,
                "label": CLASS_NAMES[label],
            })
    return np.stack(features), np.asarray(labels, dtype=np.int64), metadata


def confusion_matrix(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.int64)
    for truth, guess in zip(actual, predicted):
        matrix[int(truth), int(guess)] += 1
    return matrix


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = confusion_matrix(actual, predicted)
    recalls = [
        float(matrix[index, index] / max(1, matrix[index].sum()))
        for index in range(3)
    ]
    act_mask = actual != 0
    act_only = float(np.mean(predicted[act_mask] == actual[act_mask]))
    return {
        "accuracy": float(np.mean(predicted == actual)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": dict(zip(CLASS_NAMES, recalls)),
        "act1_vs_act2_with_background_errors_counted": act_only,
        "confusion_actual_rows_predicted_columns": matrix.tolist(),
    }


def nearest_centroid(train_x, train_y, evaluation_x) -> np.ndarray:
    centers = np.stack([train_x[train_y == label].mean(axis=0) for label in range(3)])
    distances = np.square(evaluation_x[:, None, :] - centers[None, :, :]).mean(axis=-1)
    return distances.argmin(axis=1)


def within_group_controls(feature_matrix, labels, metadata) -> dict:
    controls = {}
    groups = sorted({item["source_group"] for item in metadata})
    for group in groups:
        group_indices = np.asarray([
            index for index, item in enumerate(metadata)
            if item["source_group"] == group
        ], dtype=np.int64)
        evaluation_indices = np.asarray([
            index for index in group_indices
            if metadata[index]["trial_number"] % 5 == 0
        ], dtype=np.int64)
        training_indices = np.asarray([
            index for index in group_indices
            if metadata[index]["trial_number"] % 5 != 0
        ], dtype=np.int64)
        if not len(training_indices) or not len(evaluation_indices):
            continue
        mean = feature_matrix[training_indices].mean(axis=0, keepdims=True)
        std = feature_matrix[training_indices].std(axis=0, keepdims=True)
        training = (feature_matrix[training_indices] - mean) / np.maximum(std, 1e-6)
        evaluation = (feature_matrix[evaluation_indices] - mean) / np.maximum(std, 1e-6)
        prediction = nearest_centroid(
            training,
            labels[training_indices],
            evaluation,
        )
        controls[group] = {
            "training_examples": int(len(training_indices)),
            "evaluation_examples": int(len(evaluation_indices)),
            **metrics(labels[evaluation_indices], prediction),
        }
    return controls


def train_mlp(train_x, train_y, validation_x, validation_y, epochs, seed, device_name):
    import torch
    from torch import nn

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 96),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(96, 48),
        nn.GELU(),
        nn.Linear(48, 3),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(train_y).to(device)
    validation_tensor = torch.from_numpy(validation_x).to(device)
    best_state = None
    best_balanced = -1.0
    stale = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = model(validation_tensor).argmax(dim=1).cpu().numpy()
        score = metrics(validation_y, predicted)["balanced_accuracy"]
        if score > best_balanced + 1e-6:
            best_balanced = score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 60:
                break
    model.load_state_dict(best_state)
    return model, device


def main() -> None:
    args = parse_args()
    manifest = args.artifacts / "localization_manifest.csv"
    rows = read_csv(manifest)
    feature_matrix, labels, metadata = build_examples(rows)
    split_indices = {
        split: np.asarray(
            [index for index, item in enumerate(metadata) if item["split"] == split],
            dtype=np.int64,
        )
        for split in ("train", "validation", "test")
    }
    train_indices = split_indices["train"]
    mean = feature_matrix[train_indices].mean(axis=0, keepdims=True)
    std = feature_matrix[train_indices].std(axis=0, keepdims=True)
    standardized = ((feature_matrix - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    train_x, train_y = standardized[train_indices], labels[train_indices]
    validation_x = standardized[split_indices["validation"]]
    validation_y = labels[split_indices["validation"]]
    test_x = standardized[split_indices["test"]]
    test_y = labels[split_indices["test"]]

    centroid_validation = metrics(
        validation_y, nearest_centroid(train_x, train_y, validation_x)
    )
    centroid_test = metrics(test_y, nearest_centroid(train_x, train_y, test_x))
    model, device = train_mlp(
        train_x,
        train_y,
        validation_x,
        validation_y,
        args.epochs,
        args.seed,
        args.device,
    )
    import torch
    model.eval()
    with torch.no_grad():
        mlp_validation_prediction = model(
            torch.from_numpy(validation_x).to(device)
        ).argmax(dim=1).cpu().numpy()
        mlp_test_prediction = model(
            torch.from_numpy(test_x).to(device)
        ).argmax(dim=1).cpu().numpy()
    mlp_validation = metrics(validation_y, mlp_validation_prediction)
    mlp_test = metrics(test_y, mlp_test_prediction)
    result = {
        "classes": list(CLASS_NAMES),
        "chance_balanced_accuracy": 1.0 / 3.0,
        "examples": {
            split: {
                "total": int(len(indices)),
                "by_class": {
                    CLASS_NAMES[label]: int(np.sum(labels[indices] == label))
                    for label in range(3)
                },
            }
            for split, indices in split_indices.items()
        },
        "nearest_centroid": {
            "validation": centroid_validation,
            "test": centroid_test,
        },
        "summary_feature_mlp": {
            "validation": mlp_validation,
            "test": mlp_test,
        },
        "within_group_nearest_centroid_controls": within_group_controls(
            feature_matrix, labels, metadata
        ),
        "decision": {
            "minimum_test_balanced_accuracy_for_inception": 0.50,
            "passes": bool(mlp_test["balanced_accuracy"] >= 0.50),
        },
    }
    output = args.artifacts / "signal_separability_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved audit: {output}")


if __name__ == "__main__":
    main()
