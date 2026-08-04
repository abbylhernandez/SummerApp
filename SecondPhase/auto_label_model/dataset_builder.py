"""Build leakage-safe EMG manifests for automatic Act 1/Act 2 localization.

The builder never edits source recordings. It creates a small canonical
500-sample Act dataset plus manifests under this package's ``artifacts``
directory. Re-running it discovers newly labeled trials and updates the
manifests used by ``train.py --rescan``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_TRIAL_LOGS = REPO_ROOT / "FirstPhase" / "trial_logs"
DEFAULT_EXTERNAL = (
    REPO_ROOT.parents[1] / "DataDeep" / "Combine Dataset"
)
DEFAULT_OUTPUT = PACKAGE_DIR / "artifacts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_emg_channels(path: Path) -> np.ndarray:
    """Read old/new EMG text formats and return rounded ch1/ch2/ch3."""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        next(handle, None)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                if "," in line:
                    parts = [part.strip() for part in line.split(",")]
                    values = [float(parts[1]), float(parts[2]), float(parts[3])]
                else:
                    parts = line.split()
                    values = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
            except (IndexError, ValueError):
                continue
            rows.append(values)
    if not rows:
        raise ValueError(f"No EMG samples found in {path}")
    return np.round(np.asarray(rows, dtype=np.float64), 6)


def signal_hash(data: np.ndarray) -> str:
    normalized = np.round(np.asarray(data, dtype=np.float64), 6).astype("<f8")
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def find_exact_window(raw: np.ndarray, clip: np.ndarray) -> list[int]:
    """Return all exact clip starts in a rounded raw EMG recording."""
    if len(clip) > len(raw):
        return []
    candidates = np.where(np.all(raw == clip[0], axis=1))[0]
    starts = []
    for start in candidates:
        end = int(start) + len(clip)
        if end <= len(raw) and np.array_equal(raw[int(start):end], clip):
            starts.append(int(start))
    return starts


def trial_number(path: Path, underscore: bool | None = None) -> int | None:
    if underscore is True:
        match = re.search(r"trial_(\d+)\.txt$", path.name, re.IGNORECASE)
    elif underscore is False:
        match = re.search(r"trial(\d+)\.txt$", path.name, re.IGNORECASE)
    else:
        match = re.search(r"trial_?(\d+)\.txt$", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def safe_slug(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return value or "dataset"


def infer_group(session_name: str) -> str:
    """Conservative subject/session families used to prevent leakage."""
    lower = session_name.lower()
    if "subject4" in lower or "validation4" in lower:
        return "subject4"
    if "subject2" in lower:
        return "subject2"
    if lower.startswith("validation2data"):
        return "validation2_primary"
    if "mikul" in lower or lower == "testdata":
        return "mikul_validation"
    return f"session_{safe_slug(session_name)}"


def find_raw_trial(session: Path, number: int) -> Path | None:
    for name in (f"trial{number}.txt", f"trial_{number}.txt"):
        candidate = session / name
        if candidate.exists():
            return candidate
    return None


def find_video(session: Path, number: int) -> Path | None:
    for name in (f"video{number}.avi", f"video_{number}.avi", f"trial{number}.avi"):
        candidate = session / name
        if candidate.exists():
            return candidate
    return None


def discover_local_pairs(trial_logs: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    errors: list[dict] = []
    for session in sorted(path for path in trial_logs.iterdir() if path.is_dir()):
        act1_dir = session / "ResultClip" / "act1"
        act2_dir = session / "ResultClip" / "act2"
        if not act1_dir.is_dir() or not act2_dir.is_dir():
            continue
        act1 = {
            n: path for path in act1_dir.glob("trial_*.txt")
            if (n := trial_number(path, underscore=True)) is not None
        }
        act2 = {
            n: path for path in act2_dir.glob("trial_*.txt")
            if (n := trial_number(path, underscore=True)) is not None
        }
        for number in sorted(set(act1) & set(act2)):
            raw_path = find_raw_trial(session, number)
            if raw_path is None:
                errors.append({
                    "kind": "missing_raw",
                    "session": str(session),
                    "trial": number,
                })
                continue
            raw = load_emg_channels(raw_path)
            act1_data = load_emg_channels(act1[number])
            act2_data = load_emg_channels(act2[number])
            act1_starts = find_exact_window(raw, act1_data)
            act2_starts = find_exact_window(raw, act2_data)
            if len(act1_starts) != 1 or len(act2_starts) != 1:
                errors.append({
                    "kind": "window_match",
                    "session": str(session),
                    "trial": number,
                    "act1_starts": act1_starts,
                    "act2_starts": act2_starts,
                })
                continue
            video_path = find_video(session, number)
            records.append({
                "source_type": "local_full_trial",
                "session": session.name,
                "source_group": infer_group(session.name),
                "trial_number": number,
                "raw_path": str(raw_path.resolve()),
                "video_path": str(video_path.resolve()) if video_path else "",
                "raw_samples": len(raw),
                "raw_signal_hash": signal_hash(raw),
                "raw_file_hash": sha256_file(raw_path),
                "act1_path": str(act1[number].resolve()),
                "act2_path": str(act2[number].resolve()),
                "act1_start": act1_starts[0],
                "act2_start": act2_starts[0],
                "act1_samples": len(act1_data),
                "act2_samples": len(act2_data),
                "act1_hash": signal_hash(act1_data),
                "act2_hash": signal_hash(act2_data),
            })
    return records, errors


def resolve_external_root(root: Path) -> Path:
    current = root
    for _ in range(3):
        if (current / "act 1").is_dir() and (current / "act 2").is_dir():
            return current
        nested = current / "Combine Dataset"
        if nested.is_dir():
            current = nested
        else:
            break
    raise FileNotFoundError(f"Could not find 'act 1' and 'act 2' under {root}")


def discover_external_pairs(external_root: Path) -> tuple[list[dict], dict]:
    root = resolve_external_root(external_root)
    act_dirs = {1: root / "act 1", 2: root / "act 2"}
    base_by_act: dict[int, dict[int, Path]] = {}
    variant_counts: dict[str, int] = Counter()
    for act, directory in act_dirs.items():
        base: dict[int, Path] = {}
        for path in directory.glob("*.txt"):
            match = re.match(r"(.+)_trial_(\d+)\.txt$", path.name, re.IGNORECASE)
            if not match:
                continue
            prefix, number_text = match.groups()
            variant_counts[f"act{act}:{prefix}"] += 1
            if prefix == "ResultClip":
                base[int(number_text)] = path
        base_by_act[act] = base

    records = []
    for number in sorted(set(base_by_act[1]) & set(base_by_act[2])):
        act1_path = base_by_act[1][number]
        act2_path = base_by_act[2][number]
        act1_data = load_emg_channels(act1_path)
        act2_data = load_emg_channels(act2_path)
        records.append({
            "source_type": "external_clip_only",
            "session": "external_combine_dataset",
            "source_group": "external_combine_dataset",
            "trial_number": number,
            "raw_path": "",
            "video_path": "",
            "raw_samples": "",
            "raw_signal_hash": "",
            "raw_file_hash": "",
            "act1_path": str(act1_path.resolve()),
            "act2_path": str(act2_path.resolve()),
            "act1_start": "",
            "act2_start": "",
            "act1_samples": len(act1_data),
            "act2_samples": len(act2_data),
            "act1_hash": signal_hash(act1_data),
            "act2_hash": signal_hash(act2_data),
        })
    return records, {
        "resolved_root": str(root.resolve()),
        "variant_counts": dict(sorted(variant_counts.items())),
    }


def raw_duplicate_groups(trial_logs: Path) -> list[dict]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for session in sorted(path for path in trial_logs.iterdir() if path.is_dir()):
        for path in session.glob("trial*.txt"):
            if trial_number(path) is not None:
                by_hash[sha256_file(path)].append(str(path.resolve()))
    return [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(by_hash.items())
        if len(paths) > 1
    ]


def deduplicate_pairs(records: list[dict]) -> tuple[list[dict], list[dict]]:
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        pair_hash = hashlib.sha256(
            f"{record['act1_hash']}:{record['act2_hash']}".encode("ascii")
        ).hexdigest()
        record["pair_hash"] = pair_hash
        by_pair[pair_hash].append(record)

    kept = []
    excluded = []
    for pair_hash, group in sorted(by_pair.items()):
        group.sort(key=lambda row: (row["source_type"] != "local_full_trial", row["session"]))
        kept.append(group[0])
        for duplicate in group[1:]:
            excluded.append({
                "pair_hash": pair_hash,
                "kept": f"{group[0]['session']}:{group[0]['trial_number']}",
                "excluded": f"{duplicate['session']}:{duplicate['trial_number']}",
            })
    return kept, excluded


def near_duplicate_pairs(records: list[dict], threshold: float = 0.995) -> list[dict]:
    cached = []
    for row in records:
        a1 = load_emg_channels(Path(row["act1_path"])).reshape(-1)
        a2 = load_emg_channels(Path(row["act2_path"])).reshape(-1)
        vector = np.concatenate([a1, a2])
        vector = (vector - vector.mean()) / (vector.std() + 1e-12)
        cached.append((row, vector))
    warnings = []
    for left in range(len(cached)):
        row_a, vec_a = cached[left]
        for right in range(left + 1, len(cached)):
            row_b, vec_b = cached[right]
            if vec_a.shape != vec_b.shape:
                continue
            correlation = float(np.mean(vec_a * vec_b))
            if correlation >= threshold:
                warnings.append({
                    "correlation": correlation,
                    "left": f"{row_a['session']}:{row_a['trial_number']}",
                    "right": f"{row_b['session']}:{row_b['trial_number']}",
                })
    return warnings


def clip_overlap_warnings(records: list[dict]) -> list[dict]:
    """Find canonical Act signals reused by more than one kept pair."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        for act in (1, 2):
            by_hash[row[f"act{act}_hash"]].append({
                "pair": f"{row['session']}:{row['trial_number']}",
                "act": act,
                "path": row[f"act{act}_path"],
            })
    return [
        {"signal_hash": digest, "occurrences": occurrences}
        for digest, occurrences in sorted(by_hash.items())
        if len(occurrences) > 1
    ]


def assign_group_splits(records: list[dict]) -> dict[str, str]:
    counts = Counter(
        row["source_group"] for row in records
        if row["source_type"] == "local_full_trial"
    )
    ordered = sorted(counts, key=lambda group: (-counts[group], group))
    assignment: dict[str, str] = {}
    if ordered:
        assignment[ordered[0]] = "train"
    if len(ordered) > 1:
        assignment[ordered[1]] = "test"
    if len(ordered) > 2:
        assignment[ordered[2]] = "validation"
    split_counts = Counter({split: 0 for split in ("train", "validation", "test")})
    for group, split in assignment.items():
        split_counts[split] += counts[group]
    total = sum(counts.values()) or 1
    targets = {"train": 0.70 * total, "validation": 0.15 * total, "test": 0.15 * total}
    for group in ordered[3:]:
        split = max(targets, key=lambda name: targets[name] - split_counts[name])
        assignment[group] = split
        split_counts[split] += counts[group]
    for row in records:
        assignment.setdefault(row["source_group"], "train")
    return assignment


def copy_if_changed(source: Path, destination: Path, source_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and signal_hash(load_emg_channels(destination)) == source_hash:
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.remove(handle.name)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.remove(handle.name)


def build_dataset(trial_logs: Path, external_root: Path, output_dir: Path) -> dict:
    trial_logs = trial_logs.resolve()
    external_root = external_root.resolve()
    output_dir = output_dir.resolve()
    local, local_errors = discover_local_pairs(trial_logs)
    external, external_audit = discover_external_pairs(external_root)
    records, excluded_pairs = deduplicate_pairs(local + external)
    split_map = assign_group_splits(records)

    clip_rows = []
    localization_rows = []
    combined_root = output_dir / "combined_clips"
    for index, record in enumerate(
        sorted(records, key=lambda row: (row["source_type"], row["session"], row["trial_number"])),
        start=1,
    ):
        pair_id = f"pair_{index:04d}"
        record["pair_id"] = pair_id
        split = split_map[record["source_group"]]
        copied_paths = {}
        for act in (1, 2):
            source = Path(record[f"act{act}_path"])
            destination = combined_root / f"act{act}" / f"{pair_id}.txt"
            copy_if_changed(source, destination, record[f"act{act}_hash"])
            copied_paths[act] = destination
            clip_rows.append({
                "pair_id": pair_id,
                "act": act,
                "label": act - 1,
                "split": split,
                "source_group": record["source_group"],
                "source_type": record["source_type"],
                "session": record["session"],
                "trial_number": record["trial_number"],
                "canonical_path": str(destination),
                "original_path": record[f"act{act}_path"],
                "signal_hash": record[f"act{act}_hash"],
                "pair_hash": record["pair_hash"],
            })
        if record["source_type"] == "local_full_trial":
            localization_rows.append({
                "pair_id": pair_id,
                "split": split,
                "source_group": record["source_group"],
                "session": record["session"],
                "trial_number": record["trial_number"],
                "raw_path": record["raw_path"],
                "video_path": record["video_path"],
                "raw_samples": record["raw_samples"],
                "raw_signal_hash": record["raw_signal_hash"],
                "act1_start": record["act1_start"],
                "act2_start": record["act2_start"],
                "window_samples": record["act1_samples"],
                "act1_clip_path": str(copied_paths[1]),
                "act2_clip_path": str(copied_paths[2]),
            })

    clip_fields = [
        "pair_id", "act", "label", "split", "source_group", "source_type",
        "session", "trial_number", "canonical_path", "original_path",
        "signal_hash", "pair_hash",
    ]
    localization_fields = [
        "pair_id", "split", "source_group", "session", "trial_number",
        "raw_path", "video_path", "raw_samples", "raw_signal_hash",
        "act1_start", "act2_start", "window_samples", "act1_clip_path",
        "act2_clip_path",
    ]
    atomic_write_csv(output_dir / "clip_manifest.csv", clip_fields, clip_rows)
    atomic_write_csv(
        output_dir / "localization_manifest.csv",
        localization_fields,
        localization_rows,
    )

    split_counts = Counter(row["split"] for row in localization_rows)
    group_counts = Counter(row["source_group"] for row in localization_rows)
    dataset_version = hashlib.sha256(
        "".join(sorted(record["pair_hash"] for record in records)).encode("ascii")
    ).hexdigest()[:16]
    near_duplicates = near_duplicate_pairs(records)
    clip_overlaps = clip_overlap_warnings(records)
    audit = {
        "dataset_version": dataset_version,
        "source_roots": {
            "trial_logs": str(trial_logs),
            "external": str(external_root),
        },
        "unique_pairs": len(records),
        "canonical_clips": len(clip_rows),
        "localization_trials": len(localization_rows),
        "external_clip_only_pairs": sum(
            row["source_type"] == "external_clip_only" for row in records
        ),
        "local_errors": local_errors,
        "excluded_exact_pair_duplicates": excluded_pairs,
        "near_duplicate_warnings": near_duplicates,
        "clip_overlap_warnings": clip_overlaps,
        "raw_duplicate_groups": raw_duplicate_groups(trial_logs),
        "split_counts": dict(sorted(split_counts.items())),
        "group_counts": dict(sorted(group_counts.items())),
        "group_split_assignment": dict(sorted(split_map.items())),
        "external_inventory": external_audit,
        "notes": [
            "Resampled variants are derivatives and are never independent split units.",
            "testdata and mikulvalidationcheckData raw trials are exact duplicates.",
            "Only canonical 500-sample Act clips are copied into combined_clips.",
            "Clip-only external pairs can pretrain Act classification but cannot supervise raw-trial localization.",
        ],
    }
    atomic_write_json(output_dir / "dataset_audit.json", audit)
    summary = (
        f"Dataset version: {dataset_version}\n"
        f"Unique paired trials: {len(records)}\n"
        f"Canonical Act clips: {len(clip_rows)}\n"
        f"Full raw localization trials: {len(localization_rows)}\n"
        f"External clip-only pairs: {audit['external_clip_only_pairs']}\n"
        f"Localization splits: {dict(sorted(split_counts.items()))}\n"
        f"Exact raw duplicate groups: {len(audit['raw_duplicate_groups'])}\n"
        f"Near-duplicate pair warnings: {len(audit['near_duplicate_warnings'])}\n"
        f"Exact clip overlap warnings: {len(audit['clip_overlap_warnings'])}\n"
    )
    (output_dir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-logs", type=Path, default=DEFAULT_TRIAL_LOGS)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_dataset(args.trial_logs, args.external, args.output)
    print(json.dumps({
        "dataset_version": audit["dataset_version"],
        "unique_pairs": audit["unique_pairs"],
        "canonical_clips": audit["canonical_clips"],
        "localization_trials": audit["localization_trials"],
        "split_counts": audit["split_counts"],
        "raw_duplicate_groups": len(audit["raw_duplicate_groups"]),
        "near_duplicate_warnings": len(audit["near_duplicate_warnings"]),
        "clip_overlap_warnings": len(audit["clip_overlap_warnings"]),
    }, indent=2))
    if (
        audit["local_errors"]
        or audit["near_duplicate_warnings"]
        or audit["clip_overlap_warnings"]
    ):
        raise SystemExit(
            "Dataset audit has blocking errors or overlap warnings; inspect "
            "dataset_audit.json before training."
        )


if __name__ == "__main__":
    main()
