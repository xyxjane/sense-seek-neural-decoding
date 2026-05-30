#!/usr/bin/env python3
"""
train_cnn.py

Train a basic 1D-CNN on the SenseSeek EEG dataset for 6-class
cognitive stage classification.

Subject-based splits are defined in subject_splits.csv:
    pid,split
    PA11,train
    PA26,val
    PA30,test
    ...

No participant's windows appear in more than one split, preventing
data leakage across train / val / test.

Usage
-----
  python train_cnn.py \
      --h5     dataset/dataset_20260529_213343.h5 \
      --splits subject_splits.csv \
      --epochs 50 \
      --batch-size 64 \
      --lr 1e-3 \
      --output-dir runs/exp1
"""

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── dataset ───────────────────────────────────────────────────────────────────

class SenseSeekDataset(Dataset):
    """
    Lazy-loading PyTorch Dataset backed by the HDF5 file produced by
    make_dataset_sense_seek.py.

    Only windows belonging to the given participant IDs are exposed.
    The HDF5 file is opened fresh on every __getitem__ call so the
    Dataset is safe to use with DataLoader(num_workers > 0).
    """

    def __init__(self, h5_path: str | Path, pids: list[str]):
        self.h5_path = str(h5_path)

        with h5py.File(self.h5_path, 'r') as hf:
            all_pids = np.array([p.decode() for p in hf['pid'][:]])
            self.stage_names = [s.decode() for s in hf['labels'].attrs['stage_names']]
            self.n_channels  = hf['signals'].shape[1]
            self.n_times     = hf['signals'].shape[2]

        missing = set(pids) - set(all_pids)
        if missing:
            print(f"  [warn] PIDs not found in HDF5 (will be ignored): {sorted(missing)}")

        mask = np.isin(all_pids, list(pids))
        self.indices = np.where(mask)[0]

        if len(self.indices) == 0:
            raise ValueError(f"No windows found for PIDs: {pids}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        with h5py.File(self.h5_path, 'r') as hf:
            x = hf['signals'][idx]          # (C, T)  float32
            y = int(hf['labels'][idx])
        return torch.from_numpy(x.copy()), torch.tensor(y, dtype=torch.long)


def build_loaders(h5_path: str, splits_csv: str, batch_size: int,
                  num_workers: int = 4) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Read subject_splits.csv and return (train_loader, val_loader, test_loader).
    """
    df = pd.read_csv(splits_csv)
    df['pid']   = df['pid'].str.strip()
    df['split'] = df['split'].str.strip()

    def pids_for(split_name):
        return df.loc[df['split'] == split_name, 'pid'].tolist()

    train_pids = pids_for('train')
    val_pids   = pids_for('val')
    test_pids  = pids_for('test')

    print(f"  train PIDs ({len(train_pids)}): {train_pids}")
    print(f"  val   PIDs ({len(val_pids)}):   {val_pids}")
    print(f"  test  PIDs ({len(test_pids)}):  {test_pids}")

    train_ds = SenseSeekDataset(h5_path, train_pids)
    val_ds   = SenseSeekDataset(h5_path, val_pids)
    test_ds  = SenseSeekDataset(h5_path, test_pids)

    print(f"  windows — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader


# ── model ─────────────────────────────────────────────────────────────────────

class EEGConvNet(nn.Module):
    """
    Basic 1D-CNN for EEG classification.

    Input:  (B, C, T)  –  B = batch, C = EEG channels, T = time samples
    Output: (B, n_classes)  –  raw logits

    Architecture
    ------------
    Three Conv1d blocks (temporal feature extraction) with BatchNorm,
    ReLU, and increasing channel depth, followed by global average
    pooling and a linear classifier.
    """

    def __init__(self, n_channels: int = 60, n_times: int = 1280,
                 n_classes: int = 6, dropout: float = 0.5):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1 – coarse temporal features
            nn.Conv1d(n_channels, 64,  kernel_size=32, stride=4, padding=14),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            # Block 2 – mid-range features
            nn.Conv1d(64,  128, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            # Block 3 – fine features
            nn.Conv1d(128, 256, kernel_size=8,  stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Global average pool → (B, 256)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# ── training utilities ────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module,
             device: torch.device,
             stage_names: list[str]) -> dict:
    model.eval()
    n_cls = len(stage_names)
    total_loss, correct, total = 0.0, 0, 0
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        preds  = logits.argmax(1)
        total_loss += loss.item() * len(y)
        correct    += (preds == y).sum().item()
        total      += len(y)
        for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
            conf[t, p] += 1

    per_class_acc = {
        stage_names[i]: conf[i, i] / conf[i].sum() if conf[i].sum() else 0.0
        for i in range(n_cls)
    }
    return {
        'loss':          total_loss / total,
        'accuracy':      correct / total,
        'per_class_acc': per_class_acc,
        'confusion':     conf,
    }


def print_results(tag: str, res: dict) -> None:
    print(f"  {tag:5s}  loss={res['loss']:.4f}  acc={res['accuracy']:.4f}")
    for stage, acc in res['per_class_acc'].items():
        print(f"         {stage:<8} {acc:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a basic CNN on the SenseSeek EEG dataset."
    )
    parser.add_argument('--h5',          required=True,
                        help='Path to the HDF5 dataset file.')
    parser.add_argument('--splits',      default='subject_splits.csv',
                        help='CSV with pid,split columns (default: subject_splits.csv).')
    parser.add_argument('--epochs',      type=int,   default=50)
    parser.add_argument('--batch-size',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--weight-decay',type=float, default=1e-4)
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--num-workers', type=int,   default=4,
                        help='DataLoader worker processes (0 = main process).')
    parser.add_argument('--output-dir',  default='runs/exp1',
                        help='Directory for checkpoints and logs.')
    parser.add_argument('--device',      default='auto',
                        help='Device: auto | cpu | cuda | mps.')
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ── data ──────────────────────────────────────────────────────────────────
    print("\n=== Building data loaders ===")
    train_loader, val_loader, test_loader = build_loaders(
        args.h5, args.splits, args.batch_size, args.num_workers
    )

    # Read metadata from dataset
    with h5py.File(args.h5, 'r') as hf:
        stage_names = [s.decode() for s in hf['labels'].attrs['stage_names']]
        n_channels  = hf['signals'].shape[1]
        n_times     = hf['signals'].shape[2]
    n_classes = len(stage_names)
    print(f"  Input shape : ({n_channels}, {n_times})   Classes: {n_classes} → {stage_names}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = EEGConvNet(n_channels=n_channels, n_times=n_times,
                       n_classes=n_classes, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # Class-weighted loss to handle label imbalance
    # (compute weights from training set label counts)
    with h5py.File(args.h5, 'r') as hf:
        all_pids   = np.array([p.decode() for p in hf['pid'][:]])
        all_labels = hf['labels'][:]
    train_pids_set = set(pd.read_csv(args.splits).query("split=='train'")['pid'].str.strip())
    train_mask     = np.isin(all_pids, list(train_pids_set))
    train_labels   = all_labels[train_mask]
    counts         = np.bincount(train_labels, minlength=n_classes).astype(float)
    weights        = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / 'best_model.pt'
    log_path  = out_dir / 'training_log.csv'

    log_rows = []
    best_val_acc = 0.0

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\n=== Training for {args.epochs} epochs ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_res = evaluate(model, val_loader, criterion, device, stage_names)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train loss={train_loss:.4f} acc={train_acc:.4f}  "
              f"val loss={val_res['loss']:.4f} acc={val_res['accuracy']:.4f}  "
              f"({elapsed:.1f}s)")

        # Save best checkpoint
        if val_res['accuracy'] > best_val_acc:
            best_val_acc = val_res['accuracy']
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_acc':     best_val_acc,
                'args':        vars(args),
            }, best_ckpt)
            print(f"  ✓ saved best model (val acc={best_val_acc:.4f})")

        log_rows.append({
            'epoch':     epoch,
            'train_loss': train_loss,
            'train_acc':  train_acc,
            'val_loss':   val_res['loss'],
            'val_acc':    val_res['accuracy'],
        })

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"\nTraining log saved to {log_path}")

    # ── test evaluation ───────────────────────────────────────────────────────
    print("\n=== Test evaluation (best checkpoint) ===")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    test_res = evaluate(model, test_loader, criterion, device, stage_names)
    print_results('test', test_res)

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = '      ' + ''.join(f'{s:>8}' for s in stage_names)
    print(header)
    for i, row in enumerate(test_res['confusion']):
        print(f"  {stage_names[i]:<6}" + ''.join(f'{v:>8}' for v in row))

    # Save test results
    results_path = out_dir / 'test_results.txt'
    with open(results_path, 'w') as fh:
        fh.write(f"Test accuracy : {test_res['accuracy']:.4f}\n")
        fh.write(f"Test loss     : {test_res['loss']:.4f}\n\n")
        fh.write("Per-class accuracy:\n")
        for stage, acc in test_res['per_class_acc'].items():
            fh.write(f"  {stage:<8} {acc:.4f}\n")
        fh.write("\nConfusion matrix (rows=true, cols=pred):\n")
        fh.write(header + '\n')
        for i, row in enumerate(test_res['confusion']):
            fh.write(f"  {stage_names[i]:<6}" + ''.join(f'{v:>8}' for v in row) + '\n')
    print(f"Test results saved to {results_path}")


if __name__ == '__main__':
    main()
