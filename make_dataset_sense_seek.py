#!/usr/bin/env python3
"""
make_dataset_sense_seek.py

Objective
---------
Decode a participant's **cognitive activity stage** during an information-seeking
session from a 30-second window of raw EEG.  Six-class classification:

  0  IN      – Information Need   (recognising a knowledge gap)
  1  QF      – Query Formulation  (composing a search query)
  2  LISTEN  – Listening to spoken content
  3  READ    – Reading retrieved documents
  4  TYPE    – Typing a query or response
  5  SPEAK   – Speaking an answer aloud

These stages were identified from the SenseSeek dataset notebook
(prepare_training_data.ipynb) where `stage` is the prediction target with
values ['IN', 'QF', 'LISTEN', 'READ', 'TYPE', 'SPEAK'].

Output
------
  One HDF5 file:  <output_dir>/dataset.h5

    /signals  float32  (N, C, 1280)  – EEG  256 Hz × 5 s  (default; set by --window-sec)
                                       C = channels common to all participants
    /labels   int32    (N,)           – stage index (0–5)
    /pid       bytes    (N,)           – participant ID, e.g. b'PA11'
    /topic_id  int32    (N,)           – 0-indexed topic ID
    /task_num  int32    (N,)           – task occurrence number within topic (1–12)

  All five datasets share the same index – window i belongs to:
      f['signals'][i], f['labels'][i], f['pid'][i], f['topic_id'][i], f['task_num'][i]

  Attributes on /topic_id: trec_ids (raw 3-digit TREC IDs as int32), topic_names (strings)
  Attributes on /signals:  eeg_channels, eeg_sfreq, window_sec, stage_names

Usage
-----
  # 1. Download raw data only
  python make_dataset_sense_seek.py download --output-dir raw_data

  # 2. Build signals/labels from downloaded raw data
  python make_dataset_sense_seek.py process --raw-dir raw_data --output-dir dataset

  # 3. Download + process in one shot
  python make_dataset_sense_seek.py all
"""

import os
import re
import subprocess
import argparse
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

# ── configuration ─────────────────────────────────────────────────────────────

BUCKET = "s3://sense-seek-dataset"

SFREQ_TARGET  = 256        # resample all recordings to this rate (Hz)
WINDOW_SEC    = 5.0        # window length in seconds
WINDOW_STRIDE = 2.5        # stride in seconds; 50 % overlap → more training samples
BANDPASS         = (1.0, 40.0)  # Hz – standard cognitive EEG band
ACC_SFREQ_TARGET = 32           # Hz – Empatica E4 accelerometer
EDA_SFREQ_TARGET = 4            # Hz – Empatica E4 electrodermal activity

# Six cognitive activity stages – label mapping preserved from original notebook
STAGES       = ['IN', 'QF', 'LISTEN', 'READ', 'TYPE', 'SPEAK']
STAGE_TO_INT = {s: i for i, s in enumerate(STAGES)}

# ── S3 manifest ───────────────────────────────────────────────────────────────

# Folders always needed to build the EEG dataset
EEG_PREFIXES = [
    "HEADSET/raw/EEG/",          # EDF recordings – primary model input
    "HEADSET/raw/EEG_config/",   # channel layout JSON files
    "HEADSET/processed/events/", # per-participant stage timing PKL files
]

# Additional multimodal folders (skip with --eeg-only)
OPTIONAL_PREFIXES = [
    "TIME/",                     # cross-modal timestamp CSVs
    "HEADSET/raw/head_motion/",  # head motion auxiliary modality
    "WRISTBAND/raw/ACC/",        # Empatica E4 accelerometer
    "WRISTBAND/raw/EDA/",        # Empatica E4 electrodermal activity
    "EYETRACKER/raw/EYE/",       # gaze data
    "SCREEN/",                   # screen recordings
]

REQUIRED_FILES = [
    "PID with consent.txt",
    "attention_checks.csv",
    "demographic.xlsx",
    "self_ratings.csv",
    "survey_durations.csv",
    "task_materials.xlsx",
]

# Pre-computed feature/processed folders – never downloaded
EXCLUDED_PREFIXES = [
    "HEADSET/features/",
    "HEADSET/processed/EEG/",
    "HEADSET/processed/head motion/",
    "EYETRACKER/features/",
    "EYETRACKER/processed/",
    "WRISTBAND/features/",
    "WRISTBAND/processed/",
]

# ── S3 download helpers ───────────────────────────────────────────────────────

def list_s3_bucket(s3_uri: str) -> str:
    cmd = ["aws", "s3", "ls", s3_uri, "--recursive", "--no-sign-request"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def _check_aws_cli() -> None:
    """Raise a clear error if the AWS CLI is not installed or not executable."""
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"AWS CLI not found or not working ({exc}).\n"
            "Install it with:  pip install awscli"
        ) from exc


def _sync_prefix(prefix: str, local_root: Path, dry_run: bool = False) -> None:
    src = f"{BUCKET}/{prefix}"
    dst = str(local_root / prefix)
    cmd = ["aws", "s3", "sync", src, dst, "--no-sign-request", "--no-progress"]
    if dry_run:
        cmd.append("--dryrun")
    print(f"  syncing  {src}  ->  {dst}")
    subprocess.run(cmd, check=True)


def _copy_file(key: str, local_root: Path, dry_run: bool = False) -> None:
    src = f"{BUCKET}/{key}"
    dst = local_root / key
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", src, str(dst), "--no-sign-request", "--no-progress"]
    if dry_run:
        cmd.append("--dryrun")
    print(f"  copying  {src}  ->  {dst}")
    subprocess.run(cmd, check=True)


def download_dataset(local_root: Path, dry_run: bool = False,
                     eeg_only: bool = False) -> None:
    """
    Download raw Sense-Seek data from s3://sense-seek-dataset/ into local_root.

    eeg_only=True  – download only EEG recordings + event PKLs (fastest; enough
                     to build the HDF5 dataset).
    eeg_only=False – also download wristband, eye-tracker, screen, and timing
                     data (full multimodal archive, ~several GB extra).
    """
    _check_aws_cli()
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    prefixes = EEG_PREFIXES + ([] if eeg_only else OPTIONAL_PREFIXES)

    print("=== Downloading raw data folders ===")
    if eeg_only:
        print("  (--eeg-only: skipping wristband, eye-tracker, screen, and timing data)")
    for prefix in prefixes:
        _sync_prefix(prefix, local_root, dry_run=dry_run)

    print("\n=== Downloading metadata files ===")
    for key in REQUIRED_FILES:
        _copy_file(key, local_root, dry_run=dry_run)

    print("\nDownload complete." if not dry_run else "\nDry-run complete (nothing written).")

# ── EEG processing helpers ────────────────────────────────────────────────────

def load_and_preprocess_eeg(edf_path: Path) -> tuple:
    """
    Load EDF, pick EEG channels, apply common-average reference,
    bandpass 1–40 Hz, resample to SFREQ_TARGET.

    Returns (data: float32 (C, T), sfreq, ch_names, meas_date_unix)
    meas_date_unix is the recording start as a unix timestamp (None if not set).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False,
                               stim=False, exclude='bads')
    if len(eeg_picks) == 0:
        eeg_picks = list(range(len(raw.ch_names)))
    raw.pick(eeg_picks)

    try:
        raw.set_eeg_reference('average', projection=False, verbose=False)
    except Exception:
        pass

    raw.filter(l_freq=BANDPASS[0], h_freq=BANDPASS[1],
               method='iir', verbose=False)

    if abs(raw.info['sfreq'] - SFREQ_TARGET) > 0.5:
        raw.resample(SFREQ_TARGET, verbose=False)

    meas_date = raw.info.get('meas_date')
    meas_date_unix = meas_date.timestamp() if meas_date is not None else None

    data = raw.get_data().astype(np.float32)
    return data, float(raw.info['sfreq']), list(raw.ch_names), meas_date_unix


def _parse_empatica_csv(path: Path) -> tuple:
    """
    Parse an Empatica E4 CSV (ACC or EDA).
    Format: row 0 = unix start timestamp, row 1 = sample rate, rows 2+ = data.
    Returns (data: float32 (C, T), sfreq: float, start_unix: float)
    """
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    start_unix = float(lines[0].split(',')[0])
    sfreq      = float(lines[1].split(',')[0])
    rows = [[float(v) for v in ln.split(',')] for ln in lines[2:]]
    data = np.array(rows, dtype=np.float32)   # (T, C)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data.T, sfreq, start_unix           # (C, T)


def _slice_modality(
    data: np.ndarray,
    sfreq: float,
    data_start_unix: float,
    eeg_start_unix: float,
    win_start_sec: float,
    target_samples: int,
) -> np.ndarray:
    """
    Extract a fixed-length window from a wristband signal aligned to EEG timing.
    Returns float32 (C, target_samples); zero-padded if the window is short.
    Returns NaN array if alignment is impossible (no EEG meas_date).
    """
    if eeg_start_unix is None:
        return np.full((data.shape[0], target_samples), np.nan, dtype=np.float32)

    offset = (eeg_start_unix - data_start_unix) * sfreq
    s = int(round(offset + win_start_sec * sfreq))
    e = s + target_samples

    if s >= data.shape[1] or e <= 0:
        return np.zeros((data.shape[0], target_samples), dtype=np.float32)

    s_clip, e_clip = max(0, s), min(data.shape[1], e)
    seg = data[:, s_clip:e_clip]
    pad_l = s_clip - s
    pad_r = target_samples - seg.shape[1] - pad_l
    if pad_l > 0 or pad_r > 0:
        seg = np.pad(seg, ((0, 0), (max(0, pad_l), max(0, pad_r))))
    return seg[:, :target_samples].astype(np.float32)


def load_events(pkl_path: Path) -> pd.DataFrame:
    """
    Parse a SenseSeek event PKL into a tidy DataFrame with columns:
        stage, task, start_sec, end_sec

    The PKL has the structure:
        {'event_details': [starts, durations, labels], 'conditions': [...]}
    where each label is '{task_id}{stage}' (e.g. '3531IN') and conditions
    holds just the stage suffix (e.g. 'IN').  Transition markers (+1..+4),
    quiet-sitting (QS), self-rating (SR) and Baseline rows are dropped.
    """
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)

    starts_     = obj['event_details'][0]
    durations_  = obj['event_details'][1]
    labels_     = obj['event_details'][2]
    conditions_ = obj['conditions']

    rows = []
    for start, dur, label, cond in zip(starts_, durations_, labels_, conditions_):
        cond = str(cond)
        if cond not in STAGES:
            continue
        code         = label[: len(label) - len(cond)] if label.endswith(cond) else '000'
        topic_id_raw = int(code[:3]) if len(code) >= 3 and code[:3].isdigit() else 0
        task_num     = int(code[3:]) if len(code) > 3 and code[3:].isdigit() else 0
        rows.append({
            'stage':        cond,
            'topic_id_raw': topic_id_raw,   # 3-digit TREC topic ID (e.g. 314)
            'task_num':     task_num,        # task occurrence number (e.g. 1)
            'start_sec':    float(start),
            'end_sec':      float(start) + float(dur),
        })

    if not rows:
        raise ValueError(f"{pkl_path.name}: no labelled stage events found")
    return pd.DataFrame(rows)


def z_score_normalize(arr: np.ndarray) -> np.ndarray:
    """Per-channel z-score over a (N, C, T) float32 array."""
    mean = arr.mean(axis=(0, 2), keepdims=True)
    std  = arr.std(axis=(0, 2),  keepdims=True) + 1e-8
    return ((arr - mean) / std).astype(np.float32)


def _sanitize(s: str) -> str:
    """Make a string safe for use as a filename component."""
    return re.sub(r'[^\w\-]', '_', str(s)).strip('_') or 'unknown'


def _write_report(out_path: Path, report_path: Path, params: dict) -> None:
    """
    Write a human-readable data report alongside the HDF5 dataset.
    Reads all metadata back from the closed HDF5 file so the report is
    guaranteed to reflect exactly what was written.
    """
    with h5py.File(out_path, 'r') as hf:
        labels     = hf['labels'][:]
        pids_bytes = hf['pid'][:]
        topic_ids  = hf['topic_id'][:]
        task_nums  = hf['task_num'][:]
        N, C, T    = hf['signals'].shape
        ch_names   = [c.decode() for c in hf['signals'].attrs['eeg_channels']]
        stage_names= [s.decode() for s in hf['labels'].attrs['stage_names']]
        trec_ids   = [int(x) for x in hf['topic_id'].attrs['trec_ids']]
        topic_names= [t.decode() for t in hf['topic_id'].attrs['topic_names']]

    pids = [p.decode() for p in pids_bytes]
    file_size_mb = os.path.getsize(out_path) / 1024 ** 2

    SEP1 = '=' * 70
    SEP2 = '-' * 70

    lines = [
        SEP1,
        'SenseSeek EEG Dataset Report',
        SEP1,
        f"Generated      : {params['timestamp']}",
        f"Dataset file   : {out_path.name}",
        f"File size      : {file_size_mb:.1f} MB",
        '',
        SEP2,
        'Generation Parameters',
        SEP2,
        f"  window_sec    : {params['window_sec']} s",
        f"  window_stride : {params['window_stride']} s",
        f"  sfreq_target  : {SFREQ_TARGET} Hz",
        f"  bandpass      : {BANDPASS[0]} – {BANDPASS[1]} Hz",
        f"  normalization : per-participant per-channel z-score",
        f"  stages        : {', '.join(stage_names)}",
        '',
        SEP2,
        'Overall Statistics',
        SEP2,
        f"  Total windows  : {N}",
        f"  EEG channels   : {C}",
        f"  Samples/window : {T}  ({T / SFREQ_TARGET:.2f} s @ {SFREQ_TARGET} Hz)",
        f"  Participants   : {len(set(pids))}",
        f"  Topics         : {len(set(topic_ids.tolist()))}",
        f"  Signals shape  : ({N}, {C}, {T})  float32",
        '',
        SEP2,
        'Label Distribution',
        SEP2,
        f"  {'idx':<4}  {'stage':<8}  {'count':>6}  {'%':>6}",
    ]
    for i, name in enumerate(stage_names):
        cnt = int((labels == i).sum())
        pct = 100 * cnt / N if N else 0
        lines.append(f"  {i:<4}  {name:<8}  {cnt:>6}  {pct:>5.1f}%")

    lines += [
        '',
        SEP2,
        'Per-Participant Summary',
        SEP2,
        '  ' + f"{'PID':<8}" + ''.join(f"  {s:>6}" for s in stage_names) + f"  {'total':>6}",
    ]
    for pid in sorted(set(pids)):
        mask = np.array([p == pid for p in pids])
        pid_labels = labels[mask]
        row = f"  {pid:<8}"
        for i in range(len(stage_names)):
            row += f"  {int((pid_labels == i).sum()):>6}"
        row += f"  {int(mask.sum()):>6}"
        lines.append(row)

    lines += [
        '',
        SEP2,
        'Per-Topic Summary',
        SEP2,
        f"  {'trec_id':<8}  {'idx':<4}  {'topic_name':<35}  {'windows':>7}",
    ]
    for idx, (tid, tname) in enumerate(zip(trec_ids, topic_names)):
        cnt = int((topic_ids == idx).sum())
        lines.append(f"  {tid:<8}  {idx:<4}  {tname:<35}  {cnt:>7}")

    lines += [
        '',
        SEP2,
        'HDF5 Schema',
        SEP2,
        f"  /signals   float32  ({N}, {C}, {T})",
        f"  /labels    int32    ({N},)",
        f"  /pid       bytes    ({N},)",
        f"  /topic_id  int32    ({N},)",
        f"  /task_num  int32    ({N},)",
        '',
        '  Attributes:',
        f"    /signals[eeg_channels]  : {ch_names}",
        f"    /signals[eeg_sfreq]     : {SFREQ_TARGET}",
        f"    /signals[window_sec]    : {params['window_sec']}",
        f"    /signals[stage_names]   : {stage_names}",
        f"    /labels[stage_names]    : {stage_names}",
        f"    /topic_id[trec_ids]     : {trec_ids}",
        f"    /topic_id[topic_names]  : {topic_names}",
        SEP1,
    ]

    report_path.write_text('\n'.join(lines))
    print(f"  Report  : {report_path}")


# ── dataset builder ───────────────────────────────────────────────────────────

def _collect_common_channels(edf_files: list, events_dir: Path) -> list:
    """
    Pass 1: read EDF headers (no data) to find channels present in every participant.
    Returns sorted list of common channel names.
    """
    ch_sets = []
    for edf_path in edf_files:
        pid      = edf_path.stem.replace('_eeg', '')
        pkl_path = events_dir / f"{pid}_event_infos.pkl"
        if not pkl_path.exists():
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = mne.io.read_raw_edf(str(edf_path), preload=False,
                                       verbose=False).info
        picks = mne.pick_types(info, eeg=True, meg=False,
                               stim=False, exclude='bads')
        names = [info['ch_names'][p] for p in picks] if len(picks) else info['ch_names']
        ch_sets.append(set(names))

    if not ch_sets:
        raise RuntimeError("No valid (EDF + PKL) pairs found.")

    common = sorted(ch_sets[0].intersection(*ch_sets[1:]))
    print(f"  Common EEG channels : {len(common)}")
    return common


def build_dataset(raw_dir: Path, output_dir: Path,
                  window_sec: float = WINDOW_SEC,
                  window_stride: float = WINDOW_STRIDE) -> None:
    """
    Build two non-sequentially accessible on-disk arrays in a single HDF5 file
    (dataset/dataset.h5):

        /signals  float32  (N, C, 7680)  – EEG   256 Hz × 30 s
                                           C = channels common to all participants
        /labels    int32    (N,)           – stage index 0–5
        /pid       bytes    (N,)           – participant ID
        /topic_id  int32    (N,)           – 0-indexed topic ID
        /task_num  int32    (N,)           – task occurrence number (1–12)

    Random access (all arrays share the same index):
        import h5py
        with h5py.File('dataset/dataset.h5', 'r') as f:
            x          = f['signals'][i]                                        # (C, 7680)
            y          = f['labels'][i]                                         # stage 0-5
            pid        = f['pid'][i].decode()                                   # e.g. 'PA11'
            topic_name = f['topic_id'].attrs['topic_names'][f['topic_id'][i]]   # e.g. 'Antarctica exploration'
            task_num   = f['task_num'][i]                                       # e.g. 3

    attrs on /topic_id: trec_ids (raw TREC IDs, int32), topic_names (string list)
    """
    raw_dir    = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = output_dir / f"dataset_{ts}.h5"

    eeg_dir    = raw_dir / "HEADSET" / "raw" / "EEG"
    events_dir = raw_dir / "HEADSET" / "processed" / "events"
    acc_dir    = raw_dir / "WRISTBAND" / "raw" / "ACC"
    eda_dir    = raw_dir / "WRISTBAND" / "raw" / "EDA"

    win_acc = int(window_sec * ACC_SFREQ_TARGET)
    win_eda = int(window_sec * EDA_SFREQ_TARGET)
    # win_eeg/stride are computed from SFREQ_TARGET (the target after resampling).
    # An assert inside the loop confirms every loaded file actually has that rate.
    win_eeg = int(window_sec    * SFREQ_TARGET)
    stride  = int(window_stride * SFREQ_TARGET)

    edf_files = sorted(eeg_dir.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found in {eeg_dir}")

    # ── Pass 1: discover common channels and build task/topic ID maps ─────────
    print("=== Pass 1: discovering common EEG channels ===")
    common_channels = _collect_common_channels(edf_files, events_dir)
    n_ch = len(common_channels)

    # Load topic name mapping from task_materials.xlsx
    task_mat_path = raw_dir / "task_materials.xlsx"
    if task_mat_path.exists():
        tm = pd.read_excel(task_mat_path)[["Topic ID", "Topic"]]
        trec_to_name = {int(r["Topic ID"]): str(r["Topic"]) for _, r in tm.iterrows()}
    else:
        print("  [warn] task_materials.xlsx not found – topic names will be TREC IDs")
        trec_to_name = {}
    unique_trec_ids = sorted(trec_to_name.keys()) if trec_to_name else []
    trec_to_idx     = {tid: i for i, tid in enumerate(unique_trec_ids)}
    topic_names_arr = [trec_to_name.get(tid, str(tid)) for tid in unique_trec_ids]
    print(f"  Topic mapping loaded: {len(unique_trec_ids)} topics")

    # ── Pass 2: extract windows and write to a single resizable HDF5 ─────────
    print(f"\n=== Pass 2: extracting windows -> {out_path} ===")

    with h5py.File(out_path, 'w') as hf:
        # Create resizable datasets; chunks = one window at a time for fast
        # random-access reads by downstream DataLoaders.
        sig_ds = hf.create_dataset(
            'signals',
            shape=(0, n_ch, win_eeg),
            maxshape=(None, n_ch, win_eeg),
            dtype='float32',
            chunks=(1, n_ch, win_eeg),
            compression='gzip',
            compression_opts=4,
        )
        lab_ds = hf.create_dataset(
            'labels',
            shape=(0,),
            maxshape=(None,),
            dtype='int32',
            chunks=(256,),
        )
        str_dt      = h5py.special_dtype(vlen=bytes)
        pid_ds      = hf.create_dataset('pid',      shape=(0,), maxshape=(None,), dtype=str_dt,  chunks=(256,))
        topic_id_ds = hf.create_dataset('topic_id', shape=(0,), maxshape=(None,), dtype='int32', chunks=(256,))
        task_num_ds = hf.create_dataset('task_num', shape=(0,), maxshape=(None,), dtype='int32', chunks=(256,))

        # store metadata as attributes so the file is self-describing
        sig_ds.attrs['eeg_channels']       = np.array(common_channels,  dtype='S')
        sig_ds.attrs['eeg_sfreq']          = float(SFREQ_TARGET)
        sig_ds.attrs['window_sec']         = float(window_sec)
        sig_ds.attrs['stage_names']        = np.array(STAGES,           dtype='S')
        lab_ds.attrs['stage_names']        = np.array(STAGES,           dtype='S')
        topic_id_ds.attrs['trec_ids']      = np.array(unique_trec_ids,  dtype=np.int32)
        topic_id_ds.attrs['topic_names']   = np.array(topic_names_arr,  dtype='S')

        total = 0

        for edf_path in edf_files:
            pid      = edf_path.stem.replace('_eeg', '')
            pkl_path = events_dir / f"{pid}_event_infos.pkl"
            if not pkl_path.exists():
                print(f"  [skip] {pid}: no event PKL")
                continue

            print(f"\n── {pid} ──")
            eeg, sfreq, ch_names, meas_date_unix = load_and_preprocess_eeg(edf_path)
            assert abs(sfreq - SFREQ_TARGET) < 0.5, (
                f"{pid}: expected sfreq={SFREQ_TARGET} after resampling, got {sfreq}"
            )
            n_times = eeg.shape[1]

            # reorder / select to common_channels
            ch_idx = [ch_names.index(c) for c in common_channels if c in ch_names]
            eeg    = eeg[ch_idx]
            print(f"  EEG  {eeg.shape}  {sfreq} Hz  (selected {len(ch_idx)} common channels)")

            # wristband (graceful fallback)
            acc_data = acc_sfreq = acc_start = None
            eda_data = eda_sfreq = eda_start = None

            acc_path = acc_dir / f"{pid}_ACC.csv"
            if acc_path.exists():
                try:
                    acc_data, acc_sfreq, acc_start = _parse_empatica_csv(acc_path)
                except Exception as exc:
                    print(f"  [warn] ACC: {exc}")

            eda_path = eda_dir / f"{pid}_EDA.csv"
            if eda_path.exists():
                try:
                    eda_data, eda_sfreq, eda_start = _parse_empatica_csv(eda_path)
                except Exception as exc:
                    print(f"  [warn] EDA: {exc}")

            events = load_events(pkl_path)
            events = events[events['stage'].isin(STAGES)].copy()
            if events.empty:
                print(f"  [skip] {pid}: no labelled events")
                continue

            eeg_segs, labels = [], []
            pids, topic_ids, task_nums = [], [], []

            for _, row in events.iterrows():
                stage = str(row['stage']).strip()
                label = STAGE_TO_INT[stage]
                ss = max(0, int(row['start_sec'] * sfreq))
                se = min(n_times, int(row['end_sec'] * sfreq))

                win_starts = list(range(ss, se - win_eeg + 1, stride))
                if not win_starts:
                    # stage shorter than one full window – skip entirely
                    continue

                for ws in win_starts:
                    seg_e = eeg[:, ws: ws + win_eeg]   # exactly win_eeg samples
                    eeg_segs.append(seg_e)
                    labels.append(label)
                    pids.append(pid.encode())
                    topic_ids.append(trec_to_idx.get(int(row['topic_id_raw']), 0))
                    task_nums.append(int(row['task_num']))

            if not eeg_segs:
                continue

            eeg_arr    = z_score_normalize(np.stack(eeg_segs))  # (N, C, 7680)
            labels_arr = np.array(labels, dtype=np.int32)
            n_new      = eeg_arr.shape[0]

            # append to the resizable datasets
            for ds in (sig_ds, lab_ds, pid_ds, topic_id_ds, task_num_ds):
                ds.resize(total + n_new, axis=0)
            sig_ds[total: total + n_new]      = eeg_arr
            lab_ds[total: total + n_new]      = labels_arr
            pid_ds[total: total + n_new]      = pids
            topic_id_ds[total: total + n_new] = np.array(topic_ids, dtype=np.int32)
            task_num_ds[total: total + n_new] = np.array(task_nums, dtype=np.int32)
            total += n_new
            print(f"  +{n_new} windows  (total so far: {total})")

    print(f"\nDone. {out_path.name} written to {output_dir}")
    print(f"  signals : shape=({total}, {n_ch}, {win_eeg})  dtype=float32")
    print(f"  labels  : shape=({total},)  dtype=int32")

    report_path = output_dir / f"dataset_{ts}_report.txt"
    _write_report(
        out_path,
        report_path,
        params={
            'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'window_sec':    window_sec,
            'window_stride': window_stride,
        },
    )

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download Sense-Seek dataset from S3 and/or build a training-ready "
            "neural decoding dataset (per-session HDF5 files)."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── download sub-command ──
    dl = subparsers.add_parser("download", help="Download raw data from S3.")
    dl.add_argument("--output-dir", default="raw_data",
                    help="Local root for downloaded files (default: raw_data).")
    dl.add_argument("--dry-run", action="store_true",
                    help="Show what would be downloaded without writing files.")
    dl.add_argument("--list-only", action="store_true",
                    help="List all S3 objects and exit.")
    dl.add_argument("--eeg-only", action="store_true",
                    help="Download only EEG + event files (skip wristband, eye-tracker, screen).")

    # ── process sub-command ──
    pr = subparsers.add_parser(
        "process",
        help="Build per-session HDF5 files from downloaded raw data."
    )
    pr.add_argument("--raw-dir",       default="raw_data",
                    help="Root of downloaded raw data (default: raw_data).")
    pr.add_argument("--output-dir",    default="dataset",
                    help="Where to write the arrays (default: dataset).")
    pr.add_argument("--window-sec",    type=float, default=WINDOW_SEC,
                    help=f"Window length in seconds (default: {WINDOW_SEC}).")
    pr.add_argument("--window-stride", type=float, default=WINDOW_STRIDE,
                    help=f"Window stride in seconds (default: {WINDOW_STRIDE}).")

    # ── all-in-one sub-command ──
    ac = subparsers.add_parser("all", help="Download then process in one step.")
    ac.add_argument("--raw-dir",       default="raw_data")
    ac.add_argument("--output-dir",    default="dataset")
    ac.add_argument("--dry-run",       action="store_true")
    ac.add_argument("--eeg-only", action="store_true",
                    help="Download only EEG + event files (skip wristband, eye-tracker, screen).")
    ac.add_argument("--window-sec",    type=float, default=WINDOW_SEC,
                    help=f"Window length in seconds (default: {WINDOW_SEC}).")
    ac.add_argument("--window-stride", type=float, default=WINDOW_STRIDE,
                    help=f"Window stride in seconds (default: {WINDOW_STRIDE}).")

    args = parser.parse_args()

    if args.command == "download":
        if args.list_only:
            print(list_s3_bucket(f"{BUCKET}/"))
        else:
            download_dataset(local_root=args.output_dir, dry_run=args.dry_run,
                             eeg_only=args.eeg_only)

    elif args.command == "process":
        build_dataset(raw_dir=args.raw_dir, output_dir=args.output_dir,
                      window_sec=args.window_sec, window_stride=args.window_stride)

    elif args.command == "all":
        download_dataset(local_root=args.raw_dir, dry_run=args.dry_run,
                         eeg_only=args.eeg_only)
        if not args.dry_run:
            build_dataset(raw_dir=args.raw_dir, output_dir=args.output_dir,
                          window_sec=args.window_sec, window_stride=args.window_stride)

    else:
        parser.print_help()
