"""
Migrate old NFS checkpoints to mesh-agnostic format.

After the encoder.py fix (precomputes → register_buffer(persistent=False)),
old ckpts still contain mesh-specific DiffusionNet precomputes as parameters.
This script strips them so the ckpt can load cleanly with strict=True on
ANY topology (ICT / MF / VOCA / BIWI / etc.).

Usage:
    # Single file
    python migrate_nfs_ckpt.py ckpts_comparison/NFS-best/model_best.pth

    # Directory (recursively all .pth)
    python migrate_nfs_ckpt.py ckpts_comparison/NFS-best/

    # Dry-run (report only)
    python migrate_nfs_ckpt.py ckpts_comparison/NFS-best/ --dry-run

Writes backup as {file}.bak before overwriting.
"""
import argparse
import glob
import os
import shutil
import sys
import torch


PRECOMPUTE_SUFFIXES = (
    '.mass', '.L_ind', '.L_val', '.evals', '.evecs',
    '.grad_X_ind', '.grad_X_val', '.grad_Y_ind', '.grad_Y_val',
    '.faces',
)


def _is_precompute_key(k: str) -> bool:
    return any(k.endswith(sfx) for sfx in PRECOMPUTE_SUFFIXES)


def migrate_ckpt(path: str, dry_run: bool = False, backup: bool = True) -> int:
    state = torch.load(path, map_location='cpu', weights_only=False)

    if not isinstance(state, dict):
        print(f'[skip] {path}: not a state_dict')
        return 0

    to_drop = [k for k in state.keys() if _is_precompute_key(k)]
    if not to_drop:
        print(f'[clean] {path}: no precompute keys found (already migrated)')
        return 0

    print(f'[migrate] {path}: dropping {len(to_drop)} keys')
    for k in to_drop:
        shape = tuple(state[k].shape) if hasattr(state[k], 'shape') else '?'
        print(f'   - {k}  {shape}')

    if dry_run:
        return len(to_drop)

    if backup:
        bak = path + '.bak'
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
            print(f'   backup → {bak}')

    cleaned = {k: v for k, v in state.items() if not _is_precompute_key(k)}
    torch.save(cleaned, path)
    print(f'   saved: {path}  ({len(state)} → {len(cleaned)} keys)')
    return len(to_drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=str, help='ckpt file or directory')
    ap.add_argument('--dry-run', action='store_true',
                    help='Report only; do not modify files')
    ap.add_argument('--no-backup', action='store_true',
                    help='Skip .bak backup')
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, '**', '*.pth'),
                                 recursive=True))
    elif os.path.isfile(args.path):
        files = [args.path]
    else:
        print(f'not found: {args.path}', file=sys.stderr)
        sys.exit(1)

    if not files:
        print('no .pth files found')
        sys.exit(0)

    total_dropped = 0
    for fp in files:
        total_dropped += migrate_ckpt(fp, dry_run=args.dry_run,
                                      backup=not args.no_backup)
    print(f'\ndone. total keys {"would be " if args.dry_run else ""}dropped: '
          f'{total_dropped} across {len(files)} file(s).')


if __name__ == '__main__':
    main()
