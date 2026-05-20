"""parse_train_log.py — Quick overfit diagnostic from log.txt.

Parses train_hlbs.py log.txt:
  - Train per-step lines:  [E/MAX][step][Train] recon-lbs: ... total: ...
  - Val per-epoch lines:   [E] Val: [parts] total: ... (Best: ...)
    (parts may be empty in older logs → just `total` plotted then)

Saves CSV per run + matplotlib overlay plot (train smoothed + val).

Usage:
  python tools/analyze/parse_train_log.py <run_dir> [<run_dir> ...] \
      [--out_dir _diag] [--smooth 50] [--ymax 1.0]
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAIN_RE = re.compile(r'\[(\d+)/(\d+)\]\[(\d+)\]\[Train\]\s+(.*?)\s+total:\s+([\d.eE+-]+)')
VAL_RE   = re.compile(r'\[(\d+)\]\s+Val:\s+(.*?)\s+total:\s+([\d.eE+-]+)')
PAIR_RE  = re.compile(r'([A-Za-z][\w-]*)\s*:\s+([\d.eE+-]+)')


def parse_log(path):
    train_rows, val_rows = [], []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            m = TRAIN_RE.match(line)
            if m:
                ep = int(m.group(1)); step = int(m.group(3))
                parts = dict(PAIR_RE.findall(m.group(4)))
                parts['total'] = float(m.group(5))
                for k in parts:
                    if k != 'total':
                        parts[k] = float(parts[k])
                parts.update(epoch=ep, step=step)
                train_rows.append(parts)
                continue
            m = VAL_RE.match(line)
            if m:
                ep = int(m.group(1))
                parts = dict(PAIR_RE.findall(m.group(2)))
                parts['total'] = float(m.group(3))
                for k in parts:
                    if k != 'total':
                        parts[k] = float(parts[k])
                parts['epoch'] = ep
                val_rows.append(parts)
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows)


def plot_run(run_dir, train_df, val_df, out_dir, smooth, ymax):
    name = os.path.basename(run_dir.rstrip('/'))
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    if not train_df.empty:
        train_df.to_csv(out_dir / f'{name}__train.csv', index=False)
    if not val_df.empty:
        val_df.to_csv(out_dir / f'{name}__val.csv', index=False)

    loss_cols_train = [c for c in train_df.columns
                       if c not in ('epoch', 'step') and c != 'total'
                       and train_df[c].astype(float).abs().sum() > 0]
    loss_cols_val   = [c for c in val_df.columns
                       if c not in ('epoch',) and c != 'total'] if not val_df.empty else []

    if train_df.empty:
        print(f'  no train rows  -> skip plot'); return
    train_df = train_df.copy()
    train_df['t'] = train_df['epoch'] + train_df['step'] / (train_df['step'].max() + 1)
    if smooth > 1:
        for c in loss_cols_train + ['total']:
            train_df[c+'_sm'] = train_df[c].rolling(smooth, min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    for c in loss_cols_train:
        col = c + '_sm' if smooth > 1 else c
        ax.plot(train_df['t'], train_df[col], label=c, alpha=0.9, lw=1.2)
    ax.plot(train_df['t'], (train_df['total_sm'] if smooth > 1 else train_df['total']),
            label='total', color='k', lw=1.5)
    if ymax: ax.set_ylim(0, ymax)
    ax.set_xlabel('epoch'); ax.set_ylabel('train loss (smoothed)')
    ax.set_title(f'Train per-component\n{name}', fontsize=9)
    ax.legend(loc='upper right', fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    if not val_df.empty:
        if loss_cols_val:
            for c in loss_cols_val:
                ax.plot(val_df['epoch'], val_df[c], marker='o', ms=3, label=c)
        ax.plot(val_df['epoch'], val_df['total'], marker='s', ms=4, color='k', label='total', lw=1.5)
        ax.set_xlabel('epoch'); ax.set_ylabel('val loss')
        ax.set_title(f'Val (per-comp if available)', fontsize=9)
        ax.legend(loc='upper right', fontsize=7); ax.grid(alpha=0.3)
        if ymax: ax.set_ylim(0, ymax)
    else:
        ax.text(0.5, 0.5, '(no Val lines)', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    out_png = out_dir / f'{name}.png'
    fig.savefig(out_png, dpi=110); plt.close(fig)
    print(f'  saved {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dirs', nargs='+')
    ap.add_argument('--out_dir', default='_diag')
    ap.add_argument('--smooth', type=int, default=50,
                    help='Rolling window for train smoothing (steps).')
    ap.add_argument('--ymax', type=float, default=0.0,
                    help='y-axis upper limit (0 = auto).')
    args = ap.parse_args()

    for rd in args.run_dirs:
        log = os.path.join(rd, 'log.txt')
        if not os.path.exists(log):
            print(f'[skip] {log} not found'); continue
        print(f'[{rd}]')
        train_df, val_df = parse_log(log)
        print(f'  train rows: {len(train_df)}, val rows: {len(val_df)}')
        if not val_df.empty:
            print(f'  val epoch range: [{val_df["epoch"].min()}, {val_df["epoch"].max()}]')
            print(f'  val cols: {sorted([c for c in val_df.columns if c != "epoch"])}')
        plot_run(rd, train_df, val_df, args.out_dir,
                 args.smooth, args.ymax if args.ymax > 0 else None)


if __name__ == '__main__':
    main()
