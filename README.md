# SenseSeek Neural Decoding

Decode a participant's **cognitive activity stage** during an information-seeking
session from a sliding window of preprocessed EEG. Six-class classification:

| Index | Stage  | Description                          |
|-------|--------|--------------------------------------|
| 0     | IN     | Information Need (recognising a knowledge gap) |
| 1     | QF     | Query Formulation (composing a search query)   |
| 2     | LISTEN | Listening to spoken content          |
| 3     | READ   | Reading retrieved documents          |
| 4     | TYPE   | Typing a query or response           |
| 5     | SPEAK  | Speaking an answer aloud             |

---

## Dataset

Built from the [SenseSeek dataset](https://doi.org/10.17863/CAM.116355) — 19 participants
performing 12 TREC information-seeking tasks with simultaneous EEG (EMOTIV EPOC Flex),
eye-tracking, wristband (Empatica E4), and screen recording.

### Default configuration (5 s windows, 2.5 s stride)

| Metric            | Value                        |
|-------------------|------------------------------|
| Total windows     | 8,236                        |
| EEG channels      | 60 (common across all participants) |
| Samples / window  | 1,280  (256 Hz × 5 s)        |
| Signals shape     | (8236, 60, 1280)  float32    |
| Participants      | 19  (PA5 skipped — no event PKL) |
| Topics            | 12  (TREC IDs 314–743)       |

#### Label distribution

| Stage  | Windows | %     |
|--------|---------|-------|
| IN     | 1,850   | 22.5% |
| QF     | 684     |  8.3% |
| LISTEN | 2,823   | 34.3% |
| READ   | 2,094   | 25.4% |
| TYPE   | 647     |  7.9% |
| SPEAK  | 138     |  1.7% |

> **Why 5 s?** QF and SPEAK have max durations of ~10 s, so a 30 s window
> would yield zero samples for those stages. 5 s captures all six classes.

### HDF5 schema

```
/signals   float32  (N, C, T)   — EEG windows, gzip-compressed
/labels    int32    (N,)        — stage index 0–5
/pid       bytes    (N,)        — participant ID, e.g. b'PA11'
/topic_id  int32    (N,)        — 0-indexed topic ID
/task_num  int32    (N,)        — task occurrence number (1–12)
```

Attributes:

| Dataset    | Attribute       | Content                              |
|------------|-----------------|--------------------------------------|
| `/signals` | `eeg_channels`  | list of 60 channel names             |
| `/signals` | `eeg_sfreq`     | 256 (Hz)                             |
| `/signals` | `window_sec`    | window length used at build time     |
| `/signals` | `stage_names`   | `['IN','QF','LISTEN','READ','TYPE','SPEAK']` |
| `/labels`  | `stage_names`   | same as above                        |
| `/topic_id`| `trec_ids`      | raw TREC topic IDs (int32 array)     |
| `/topic_id`| `topic_names`   | topic name strings                   |

---

## EEG preprocessing

Applied per participant inside `load_and_preprocess_eeg()`:

1. Load EDF with MNE
2. Pick EEG channels; fall back to all channels if none typed as EEG
3. Strip non-EEG channels (`BATTERY`, `BATTERY_PERCENT`, `COUNTER`, `INTERPOLATED`, `MARKER_HARDWARE`, `MarkerIndex`, `MarkerType`, `MarkerValueInt`, `OR_TIME_STAMP_ms`, `OR_TIME_STAMP_s`, `TIME_STAMP_ms`, `TIME_STAMP_s`)
4. Retain only channels present in **every** participant (intersection → 60 channels)
5. Common-average reference (CAR)
6. Bandpass filter  1–40 Hz  (IIR Butterworth)
7. Resample to 256 Hz
8. Slice into sliding windows (no cross-boundary padding)
9. Per-participant, per-channel z-score normalisation over all windows

---

## Usage

### Prerequisites

```bash
# Option A — install script (detects OS and shows AWS CLI install command)
bash install_requirements.sh

# Option B — manual
pip install mne h5py numpy pandas openpyxl awscli
```

### 1 — Download raw data from S3

The dataset is hosted in a public S3 bucket (`s3://sense-seek-dataset/`). No AWS
account is needed.

**EEG only** (recommended — everything needed to build the HDF5 dataset, ~few GB):

```bash
python make_dataset_sense_seek.py download \
    --output-dir data/raw \
    --eeg-only
```

This downloads:

| Folder / file | Contents |
|---|---|
| `HEADSET/raw/EEG/` | EDF recordings (primary model input) |
| `HEADSET/raw/EEG_config/` | Channel layout JSON files |
| `HEADSET/processed/events/` | Stage timing PKL files |
| `task_materials.xlsx` | TREC topic ID → topic name mapping |
| other metadata CSVs | Participant IDs, attention checks, ratings |

**Full multimodal download** (also includes wristband, eye-tracker, screen, timing):

```bash
python make_dataset_sense_seek.py download --output-dir data/raw
```

Additionally downloads:

| Folder | Contents |
|---|---|
| `WRISTBAND/raw/ACC/` | Empatica E4 accelerometer (32 Hz) |
| `WRISTBAND/raw/EDA/` | Empatica E4 electrodermal activity (4 Hz) |
| `EYETRACKER/raw/EYE/` | Gaze CSV files |
| `HEADSET/raw/head_motion/` | Head motion data |
| `TIME/` | Cross-modal timestamp CSVs |
| `SCREEN/` | Screen recordings |

Dry-run (show what would be downloaded without writing anything):

```bash
python make_dataset_sense_seek.py download --output-dir data/raw --eeg-only --dry-run
```

### 2 — Build the HDF5 dataset

```bash
python make_dataset_sense_seek.py process \
    --raw-dir   data/raw \
    --output-dir dataset/
```

With custom window parameters:

```bash
python make_dataset_sense_seek.py process \
    --raw-dir       data/raw \
    --output-dir    dataset/ \
    --window-sec    5.0 \
    --window-stride 2.5
```

Outputs two matched files:

```
dataset/
  dataset_20260529_213343.h5          ← HDF5 dataset
  dataset_20260529_213343_report.txt  ← human-readable data report
```

### 3 — Download + process in one shot

```bash
# EEG only (recommended)
python make_dataset_sense_seek.py all \
    --raw-dir    data/raw \
    --output-dir dataset/ \
    --eeg-only

# Full multimodal
python make_dataset_sense_seek.py all \
    --raw-dir    data/raw \
    --output-dir dataset/
```

---

## CLI parameters

| Argument          | Sub-command       | Default     | Description                        |
|-------------------|-------------------|-------------|------------------------------------|
| `--raw-dir`       | `process`, `all`  | `raw_data`  | Root of downloaded raw data        |
| `--output-dir`    | all               | `dataset`   | Output directory for HDF5 + report |
| `--window-sec`    | `process`, `all`  | `5.0`       | Window length in seconds           |
| `--window-stride` | `process`, `all`  | `2.5`       | Stride in seconds (overlap = 1 − stride/window) |
| `--eeg-only`      | `download`, `all` | off         | Download only EEG + event files (skip wristband, eye-tracker, screen) |
| `--dry-run`       | `download`, `all` | off         | Show S3 operations without writing |
| `--list-only`     | `download`        | off         | List all S3 objects and exit        |

---

## Loading the dataset (Python)

```python
import h5py
import numpy as np

with h5py.File('dataset/dataset_20260529_213343.h5', 'r') as f:
    # metadata
    stage_names = [s.decode() for s in f['labels'].attrs['stage_names']]
    ch_names    = [c.decode() for c in f['signals'].attrs['eeg_channels']]
    topic_names = [t.decode() for t in f['topic_id'].attrs['topic_names']]

    # single window
    x = f['signals'][0]          # (60, 1280)  float32
    y = f['labels'][0]           # int  0–5
    pid       = f['pid'][0].decode()              # e.g. 'PA11'
    topic     = topic_names[f['topic_id'][0]]     # e.g. 'Antarctica exploration'
    task_num  = f['task_num'][0]                  # e.g. 3

    # all windows for one stage
    labels = f['labels'][:]
    read_mask = labels == 3      # READ
    X_read = f['signals'][read_mask]   # (N_read, 60, 1280)

    # all windows for one participant
    pids = np.array([p.decode() for p in f['pid'][:]])
    X_pa11 = f['signals'][pids == 'PA11']
```

---

---

## Training a model

### Subject-based splits

Participants are pre-assigned to non-overlapping train / val / test splits in
`subject_splits.csv` (no window from the same participant appears in two splits):

| Split | PIDs | Windows (approx.) |
|-------|------|-------------------|
| train | PA6, PA8, PA9, PA11, PA12, PA13, PA17, PA18, PA19, PA20, PA21, PA22, PA33 | ~5,700 |
| val   | PA26, PA27, PA29 | ~1,300 |
| test  | PA30, PA31, PA32 | ~1,200 |

### Quick start

```bash
python train_with_dataloader.py \
    --h5          dataset/dataset_<timestamp>.h5 \
    --splits      subject_splits.csv \
    --epochs      50 \
    --batch-size  64 \
    --lr          1e-3 \
    --output-dir  runs/exp1
```

### CLI parameters

| Argument        | Default              | Description |
|-----------------|----------------------|-------------|
| `--h5`          | *(required)*         | Path to the HDF5 dataset file |
| `--splits`      | `subject_splits.csv` | CSV with `pid,split` columns |
| `--epochs`      | `50`                 | Number of training epochs |
| `--batch-size`  | `64`                 | Mini-batch size |
| `--lr`          | `1e-3`               | Adam learning rate |
| `--weight-decay`| `1e-4`               | Adam weight decay |
| `--dropout`     | `0.5`                | Dropout probability |
| `--model`       | `mlp`                | Architecture — currently `mlp` |
| `--num-workers` | `4`                  | DataLoader worker processes (use `0` on Windows) |
| `--output-dir`  | `runs/exp1`          | Where to write checkpoints and logs |
| `--device`      | `auto`               | `auto` \| `cpu` \| `cuda` \| `mps` |

### Outputs

```
runs/exp1/
  best_model.pt      ← best checkpoint (keyed on val accuracy)
  training_log.csv   ← epoch-by-epoch train/val loss and accuracy
  test_results.txt   ← per-class accuracy + confusion matrix on test set
```

### Using the DataLoader in your own script

```python
from train_with_dataloader import SenseSeekDataset, build_loaders
import h5py

# Build subject-split loaders
train_loader, val_loader, test_loader = build_loaders(
    h5_path    = 'dataset/dataset_<timestamp>.h5',
    splits_csv = 'subject_splits.csv',
    batch_size = 64,
    num_workers = 4,
)

# Each batch: x (B, 60, 1280) float32,  y (B,) int64
for x, y in train_loader:
    print(x.shape, y.shape)   # torch.Size([64, 60, 1280])  torch.Size([64])
    break

# Or build a loader for a custom list of participants
from train_with_dataloader import SenseSeekDataset
from torch.utils.data import DataLoader

ds = SenseSeekDataset('dataset/dataset_<timestamp>.h5', pids=['PA11', 'PA12'])
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
```

---

## Project structure

```
make_dataset_sense_seek.py    — pipeline: S3 download + EEG preprocessing + HDF5 builder
train_with_dataloader.py      — subject-split DataLoader + MLP training script
subject_splits.csv            — train/val/test participant assignments
test.py                       — dry-run / unit test for single participant (PA11)
data/raw/                     — downloaded raw data (gitignored)
  HEADSET/raw/EEG/            — EDF recordings
  HEADSET/processed/events/   — stage timing PKL files
  WRISTBAND/raw/{ACC,EDA}/    — Empatica E4 wristband signals
  EYETRACKER/raw/EYE/         — gaze CSV files
  task_materials.xlsx          — TREC topic ID → topic name mapping
dataset/                      — output HDF5 files (gitignored)
runs/                         — training outputs (gitignored)
```

---

## Dependencies

```
mne
h5py
numpy
pandas
openpyxl   # for reading task_materials.xlsx
awscli     # for S3 download
torch      # for train_with_dataloader.py
```
