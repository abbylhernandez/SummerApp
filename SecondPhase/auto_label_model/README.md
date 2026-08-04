# Optional Act 1 / Act 2 Auto-Label Model

This directory is intentionally isolated from `EMGVideoViewer.py`. Delete the
entire `auto_label_model` folder to remove the experiment without breaking the
manual Phase 2 viewer.

The current viewer is **not automatically connected to an unvalidated model**.
Train and evaluate the model first. Integration should happen only after the
held-out test metrics are acceptable.

## Audited data available now

- 110 paired trials have full raw EMG plus exact Act 1 and Act 2 start indices.
- 16 additional external pairs are clip-only and are used for Act encoder
  pretraining, not raw-trial localization.
- 126 unique paired trials produce 252 canonical 500-sample clips.
- `testdata` and `mikulvalidationcheckData...` contain 51 exact duplicate raw
  trials. The builder groups/excludes them so copies cannot cross splits.
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

The default group-safe split currently contains 60 training, 25 validation,
and 25 held-out localization trials. Subject/session families and exact copies
stay together. The final checkpoint is:

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
