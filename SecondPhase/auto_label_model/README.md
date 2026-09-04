# Optional Act 1 / Act 2 Auto-Label Model

This directory is intentionally isolated from `EMGVideoViewer.py`. Delete the
entire `auto_label_model` folder to remove the experiment without breaking the
manual Phase 2 viewer.

The current viewer is **not automatically connected to an unvalidated model**.
Train and evaluate the model first. Integration should happen only after the
held-out test metrics are acceptable.

## Audited data available now

- 157 paired trials have full raw EMG plus exact Act 1 and Act 2 start indices.
- 16 additional external pairs are clip-only and are used for Act encoder
  pretraining, not raw-trial localization.
- 173 unique paired trials produce 346 canonical 500-sample clips.
- The current audit finds no exact raw duplicates, near-duplicate warnings, or
  clip overlap between splits.
- Saved 200-1000 sample variants are derivatives of their parent trial. They
  are not counted as independent examples and never cross dataset splits.

Generated manifests, copied canonical clips, audit reports, metrics, and model
checkpoints live under `artifacts/` and are ignored by Git.

## Model

`model.py` defines a hybrid network with:

1. Strided temporal convolutions for local EMG patterns.
2. Six gated, dilated residual temporal blocks.
3. A compact Transformer for long-range trial context.
4. An Act 1/Act 2 clip-classification head for pretraining.
5. Two temporal probability heads that predict the fixed 500-sample starts.
6. A soft-boundary loss and an ordering/non-overlap constraint.

The 16 external clip pairs contribute to encoder pretraining. Only examples
with a recoverable full raw trial supervise localization.

## Build or refresh the dataset

From `SecondPhase`:

```powershell
..\.venv\Scripts\python.exe auto_label_model\dataset_builder.py
```

Every run rescans both configured data roots, hashes raw trials and canonical
signals, reconstructs label start indices, checks exact/near duplicates, and
atomically updates:

- `artifacts/combined_clips/act1`
- `artifacts/combined_clips/act2`
- `artifacts/clip_manifest.csv`
- `artifacts/localization_manifest.csv`
- `artifacts/dataset_audit.json`
- `artifacts/SUMMARY.txt`

The manifests, rather than every old resampling variant, define the training
dataset. Stale files are harmless because training reads only current manifest
rows.

## Install the optional model dependency

PyTorch is not part of the existing application environment. Install it only
if this experiment will be trained:

```powershell
.\auto_label_model\install_gpu.ps1
```

That script installs the official CUDA 13.0 PyTorch wheel for this computer's
RTX 3060 and verifies that CUDA is available. The training command defaults to
`--device auto`, so it will use the GPU automatically. To intentionally set up
a CPU-only environment instead, install `requirements.txt` directly.

## Train and automatically include new labels

```powershell
..\.venv\Scripts\python.exe auto_label_model\train.py --rescan
```

Or run:

```powershell
.\auto_label_model\retrain.ps1
```

`--rescan` is the retraining switch: newly completed Phase 2 labels are audited
and added before training. Use `--quick` for a one-epoch smoke test.

The default group-safe split currently contains 107 training, 25 validation,
and 25 held-out localization trials. `subject2` remains the validation group,
`validation2_primary` remains the test group, and all other complete sessions
train the model. Subject/session families and exact copies stay together. The
final checkpoint is:

```text
artifacts/checkpoints/auto_label_model.pt
```

## Predict one raw trial

```powershell
..\.venv\Scripts\python.exe auto_label_model\predict.py `
  "..\FirstPhase\trial_logs\testdata\trial5.txt"
```

The output contains Act 1 and Act 2 start samples, fixed end samples, and model
confidence. Predictions enforce Act 1 before a non-overlapping Act 2.

## Integration gate

Do not let predictions move the Phase 2 sliders automatically until the held-
out report is reviewed. A reasonable first gate is:

- Median start error below 100 samples for both Acts.
- At least 80% of starts within 100 samples.
- No duplicate or near-duplicate warnings in `dataset_audit.json`.
- Manual review on a completely new recording session.

After that gate passes, `predict.py` can be called asynchronously when Phase 2
loads a trial, with low-confidence predictions shown as suggestions rather than
automatic labels.

## V2 full-trial experiment

V2 is isolated in `model_v2.py`, `train_v2.py`, and `predict_v2.py`. It uses
complete raw trials, adds normalized trial position, learns dense background /
Act 1 / Act 2 regions, predicts both boundaries, and anchors predictions with
position priors learned only from the training split. It has early stopping and
always reports a simple position-only baseline for comparison. After training,
the influence of learned EMG evidence is calibrated on validation data before
the held-out test set is evaluated.

Train V2 on the GPU:

```powershell
.\auto_label_model\retrain_v2.ps1 -Device cuda
```

Its separate checkpoint is
`artifacts/checkpoints/auto_label_model_v2.pt`. V1 is not overwritten.

## V3 U-Time experiment

V3 is isolated in `features_v3.py`, `model_v3_utime.py`, `train_v3.py`, and
`predict_v3.py`. It follows a compact U-Time/1D U-Net design for complete
physiological time series. Inputs include high-pass EMG, a short RMS envelope,
and first differences. The network predicts dense background/Act regions and
two boundary distributions without a fixed trial-position prior.

V3 reports two separate protocols:

- **Zero-shot:** an entire unseen collection session is evaluated with no
  labels from that session.
- **Rapid calibration:** five trials from an unseen session tune only a small
  session adapter and the output heads; the remaining trials are evaluated.

Run the full GPU experiment with:

```powershell
.\auto_label_model\retrain_v3.ps1 -Device cuda
```

The base and session-calibrated checkpoints are separate from V1 and V2 under
`artifacts/checkpoints/`.

## V4 pretrained Video + EMG experiment

V4 uses the synchronized external-camera videos as the primary modality and
EMG as secondary evidence. `cache_video_features.py` extracts one pretrained
MobileNetV3 embedding per native 30 FPS frame. `model_v4_multimodal.py` fuses
centered appearance, frame-to-frame visual motion, and frame-aligned EMG with
a bidirectional temporal model. The cached embeddings are ignored by Git and
reused by subsequent training runs.

Run the repeatable pipeline with:

```powershell
.\auto_label_model\retrain_v4.ps1 -Device cuda
```

One held-out recording whose EMG and video durations differ by more than one
second is excluded and reported in `training_metrics_v4.json`. V4 saves a
separate `auto_label_model_v4_video_emg.pt` checkpoint and is not connected to
the manual viewer unless it passes the integration gate.
