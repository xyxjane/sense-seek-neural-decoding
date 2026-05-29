#!/usr/bin/env python3
"""
test.py

Dry-run test for one participant (PA11):
  1. Inspect the raw event PKL and show its structure
  2. Parse events into a tidy DataFrame using the (corrected) logic
  3. Load and preprocess EEG
  4. Run the windowing loop and show per-window stats
  5. Verify that the 'task' / 'Topic' columns exist and are sensible

Run from repo root:
    python test.py
"""

import pickle
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

RAW_DIR    = Path("data/raw")
PID        = "PA11"
EDF_PATH   = RAW_DIR / "HEADSET" / "raw" / "EEG"   / f"{PID}_eeg.edf"
PKL_PATH   = RAW_DIR / "HEADSET" / "processed" / "events" / f"{PID}_event_infos.pkl"

# ── import helpers from the main script ───────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from make_dataset_sense_seek import (
    STAGES, STAGE_TO_INT,
    WINDOW_SEC, WINDOW_STRIDE, SFREQ_TARGET,
    load_and_preprocess_eeg,
    z_score_normalize,
)

SEP = "─" * 70


# ══════════════════════════════════════════════════════════════════════════════
# 1. Raw PKL inspection
# ══════════════════════════════════════════════════════════════════════════════

print(SEP)
print("1. RAW PKL STRUCTURE")
print(SEP)

with open(PKL_PATH, "rb") as f:
    raw = pickle.load(f)

print(f"type : {type(raw).__name__}")
print(f"keys : {list(raw.keys())}")

starts     = raw["event_details"][0]
durations  = raw["event_details"][1]
labels     = raw["event_details"][2]
conditions = raw["conditions"]

print(f"\nevent_details[0]  (start_sec)   – {len(starts)} entries")
print(f"event_details[1]  (durations)   – {len(durations)} entries")
print(f"event_details[2]  (labels)      – {len(labels)} entries")
print(f"conditions        (stage names) – {len(conditions)} entries")

print("\nFirst 20 events:")
print(f"{'idx':>4}  {'label':<22}  {'cond':<12}  {'start_sec':>10}  {'dur_sec':>8}")
for i in range(min(20, len(labels))):
    print(f"{i:>4}  {labels[i]:<22}  {str(conditions[i]):<12}  {starts[i]:>10.3f}  {durations[i]:>8.3f}")

print(f"\nAll unique conditions: {sorted(set(conditions))}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Corrected event parsing
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("2. PARSED EVENT DATAFRAME")
print(SEP)

def parse_events(pkl_path: Path) -> pd.DataFrame:
    """
    Parse the SenseSeek event PKL into a tidy DataFrame with columns:
        stage      – one of STAGES (IN / QF / LISTEN / READ / TYPE / SPEAK)
        task       – numeric task ID string extracted from the label prefix
        start_sec  – stage start time relative to EEG recording start (seconds)
        end_sec    – stage end time (seconds)

    Rows for transition markers (+1…+4), self-rating (SR), quiet-sitting (QS),
    and Baseline events are dropped.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    starts_    = obj["event_details"][0]
    durations_ = obj["event_details"][1]
    labels_    = obj["event_details"][2]
    conditions_= obj["conditions"]

    rows = []
    for start, dur, label, cond in zip(starts_, durations_, labels_, conditions_):
        cond = str(cond)
        if cond not in STAGES:
            continue
        # split label prefix: first 3 digits = TREC topic ID, rest = task number
        code         = label[: len(label) - len(cond)] if label.endswith(cond) else "000"
        topic_id_raw = int(code[:3]) if len(code) >= 3 and code[:3].isdigit() else 0
        task_num     = int(code[3:]) if len(code) > 3 and code[3:].isdigit() else 0
        rows.append({
            "stage":        cond,
            "topic_id_raw": topic_id_raw,   # 3-digit TREC topic ID (e.g. 314)
            "task_num":     task_num,        # task occurrence number (e.g. 1)
            "start_sec":    float(start),
            "end_sec":      float(start) + float(dur),
        })

    if not rows:
        raise ValueError(f"{pkl_path.name}: no labelled stage events found")
    return pd.DataFrame(rows)


events = parse_events(PKL_PATH)
print(f"Shape : {events.shape}")
print(f"Columns: {list(events.columns)}")
print(f"\nStage counts:\n{events['stage'].value_counts().to_string()}")
print(f"\nUnique topic IDs : {sorted(events['topic_id_raw'].unique())}")
print(f"Unique task nums : {sorted(events['task_num'].unique())}")
print(f"\nFirst 10 rows:")
print(events.head(10).to_string(index=False))

# Duration stats per stage
print(f"\nDuration stats (seconds) per stage:")
dur = events.copy()
dur["dur"] = dur["end_sec"] - dur["start_sec"]
print(dur.groupby("stage")["dur"].describe()[["count", "min", "mean", "max"]].to_string())


# # ══════════════════════════════════════════════════════════════════════════════
# # 3. EEG preprocessing
# # ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("3. EEG PREPROCESSING")
print(SEP)

print(f"Loading {EDF_PATH} …")
eeg, sfreq, ch_names, meas_date_unix = load_and_preprocess_eeg(EDF_PATH)
print(f"  shape      : {eeg.shape}  (channels × samples)")
print(f"  sfreq      : {sfreq} Hz")
print(f"  duration   : {eeg.shape[1] / sfreq:.1f} s  ({eeg.shape[1] / sfreq / 60:.2f} min)")
print(f"  meas_date  : {meas_date_unix}")
print(f"  channels   : {ch_names}")
print(f"  value range: {eeg.min():.4e} … {eeg.max():.4e}")

# mirrors build_dataset: reorder to common channels
# (for a single-participant test the "common" set is just this participant's channels)
common_channels = sorted(ch_names)
ch_idx = [ch_names.index(c) for c in common_channels]
eeg    = eeg[ch_idx]
ch_names = common_channels
print(f"  → reordered to {len(ch_names)} channels (sorted alphabetically for single-case test)")


# # ══════════════════════════════════════════════════════════════════════════════
# # 4. Windowing loop
# # ══════════════════════════════════════════════════════════════════════════════

print(f"\n{SEP}")
print("4. WINDOWING")
print(SEP)

assert abs(sfreq - SFREQ_TARGET) < 0.5, (
    f"expected sfreq={SFREQ_TARGET} after resampling, got {sfreq}"
)
win_eeg = int(WINDOW_SEC    * sfreq)
stride  = int(WINDOW_STRIDE * sfreq)
n_times = eeg.shape[1]

print(f"window = {WINDOW_SEC}s  ({win_eeg} samples)   stride = {WINDOW_STRIDE}s  ({stride} samples)")

eeg_segs, labels_list, topic_id_list, task_num_list, stage_list = [], [], [], [], []

n_skipped = 0
for _, row in events.iterrows():
    stage = str(row["stage"]).strip()
    if stage not in STAGE_TO_INT:
        continue
    label = STAGE_TO_INT[stage]
    ss    = max(0, int(row["start_sec"] * sfreq))
    se    = min(n_times, int(row["end_sec"] * sfreq))

    # only emit windows that fit entirely within this stage
    win_starts = list(range(ss, se - win_eeg + 1, stride))
    if not win_starts:
        n_skipped += 1
        continue

    for ws in win_starts:
        seg = eeg[:, ws: ws + win_eeg]   # exactly win_eeg samples
        eeg_segs.append(seg)
        labels_list.append(label)
        topic_id_list.append(int(row["topic_id_raw"]))
        task_num_list.append(int(row["task_num"]))
        stage_list.append(stage)

print(f"\nStages skipped (too short for one full window): {n_skipped}")
print(f"Total windows extracted : {len(eeg_segs)}")
if eeg_segs:
    all_segs = np.stack(eeg_segs)   # (N, C, T)
    print(f"Stack shape             : {all_segs.shape}")

    print(f"\nWindow counts per stage:")
    from collections import Counter
    for s, n in sorted(Counter(stage_list).items()):
        print(f"  {s:<8} {n}")

    print(f"\nWindow counts per topic ID:")
    for t, n in sorted(Counter(topic_id_list).items()):
        print(f"  {t:<10} {n}")

    # ── 5. Normalisation check ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("5. Z-SCORE NORMALISATION")
    print(SEP)

    norm = z_score_normalize(all_segs)
    print(f"  Before – mean={all_segs.mean():.4e}  std={all_segs.std():.4e}")
    print(f"  After  – mean={norm.mean():.4e}  std={norm.std():.4e}")
    print(f"  dtype  : {norm.dtype}  shape: {norm.shape}")

    # ── 6. Topic ID mapping (mirrors build_dataset) ──────────────────────────
    print(f"\n{SEP}")
    print("6. TOPIC ID MAPPING")
    print(SEP)

    tm_path = RAW_DIR / "task_materials.xlsx"
    if tm_path.exists():
        tm_df = pd.read_excel(tm_path)[["Topic ID", "Topic"]]
        trec_to_name = {int(r["Topic ID"]): str(r["Topic"]) for _, r in tm_df.iterrows()}
    else:
        trec_to_name = {}

    unique_trec_ids = sorted(set(topic_id_list))
    trec_to_idx     = {tid: i for i, tid in enumerate(unique_trec_ids)}
    topic_names     = [trec_to_name.get(tid, str(tid)) for tid in unique_trec_ids]
    topic_idx_arr   = np.array([trec_to_idx[t] for t in topic_id_list], dtype=np.int32)
    task_num_arr    = np.array(task_num_list, dtype=np.int32)

    print(f"  trec_id → idx → name:")
    for tid, idx in trec_to_idx.items():
        print(f"    {tid}  →  {idx}  →  {topic_names[idx]}")
    print(f"  task_num range  : {task_num_arr.min()} – {task_num_arr.max()}")
    print(f"  topic_idx sample: {topic_idx_arr[:10]} … (first 10)")

    # ── 7. Write mini HDF5 and verify random access ───────────────────────────
    import h5py
    print(f"\n{SEP}")
    print("7. HDF5 WRITE + RANDOM ACCESS CHECK")
    print(SEP)

    OUT = Path("test_output") / "single_case.h5"
    OUT.parent.mkdir(exist_ok=True)

    labels_arr = np.array(labels_list, dtype=np.int32)
    str_dt     = h5py.special_dtype(vlen=bytes)
    N, C, T    = norm.shape
    ch_sz      = min(256, N)   # chunk must not exceed dataset size

    with h5py.File(OUT, "w") as hf:
        sig_ds      = hf.create_dataset("signals",   data=norm,          chunks=(1, C, T), compression="gzip", compression_opts=4)
        lab_ds      = hf.create_dataset("labels",    data=labels_arr,    chunks=(ch_sz,))
        pid_ds      = hf.create_dataset("pid",       data=np.array([PID.encode()] * N, dtype=object), dtype=str_dt)
        topic_id_ds = hf.create_dataset("topic_id",  data=topic_idx_arr, chunks=(ch_sz,))
        task_num_ds = hf.create_dataset("task_num",  data=task_num_arr,  chunks=(ch_sz,))

        sig_ds.attrs["eeg_channels"]      = np.array(ch_names,        dtype="S")
        sig_ds.attrs["eeg_sfreq"]         = float(sfreq)
        sig_ds.attrs["window_sec"]        = float(WINDOW_SEC)
        sig_ds.attrs["stage_names"]       = np.array(STAGES,          dtype="S")
        lab_ds.attrs["stage_names"]       = np.array(STAGES,          dtype="S")
        topic_id_ds.attrs["trec_ids"]     = np.array(unique_trec_ids, dtype=np.int32)
        topic_id_ds.attrs["topic_names"]  = np.array(topic_names,     dtype="S")

    print(f"  Written to {OUT}")
    print(f"  Datasets: signals{norm.shape}  labels{labels_arr.shape}  topic_id{topic_idx_arr.shape}  task_num{task_num_arr.shape}")

    # verify random access on 3 random indices
    rng = np.random.default_rng(42)
    idxs = rng.integers(0, N, size=3)
    with h5py.File(OUT, "r") as hf:
        print(f"\n  Random-access spot check (indices {idxs}):")
        for i in int(idxs[0]), int(idxs[1]), int(idxs[2]):
            x          = hf["signals"][i]
            y          = hf["labels"][i]
            p          = hf["pid"][i].decode()
            t_idx      = hf["topic_id"][i]
            t_name     = hf["topic_id"].attrs["topic_names"][t_idx].decode()
            t_num      = hf["task_num"][i]
            stage_name = hf["labels"].attrs["stage_names"][y].decode()
            print(f"    [{i:>4}]  pid={p}  stage={stage_name}  topic={t_name}  task_num={t_num}  "
                  f"signal shape={x.shape}  mean={x.mean():.4f}")

print(f"\n{SEP}")
print("DONE")
print(SEP)
print(SEP)
