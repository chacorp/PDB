"""
precompute_sigma_targets.py — Assign per-joint σ target from a YAML category spec.

Category-based (Maya-independent): each joint's σ is decided by matching its name
against keyword lists in sigma_categories.yml. No GT skin weights needed.

Outputs:
    sigma_targets.npy   [J]  per-joint target σ

Usage:
    python precompute_sigma_targets.py \
        --rig_path maya_rig/hybrid \
        --categories maya_rig/hybrid/sigma_categories.yml
"""
import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils.rig_loader import load_rig


def _assign_sigma(joint_name: str, cats, default_sigma: float):
    """Return (sigma, category_name) — first matching category wins."""
    for cat in cats:
        for kw in cat['keywords']:
            if kw in joint_name:
                return cat['sigma'], cat['name']
    return default_sigma, 'default'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rig_path', type=str, required=True)
    ap.add_argument('--categories', type=str, default=None,
                    help='YAML with category keyword spec '
                         '(default: {rig_path}/sigma_categories.yml)')
    ap.add_argument('--out', type=str, default=None,
                    help='Output .npy (default: {rig_path}/sigma_targets.npy)')
    args = ap.parse_args()

    cat_path = args.categories or os.path.join(args.rig_path, 'sigma_categories.yml')
    with open(cat_path) as f:
        spec = yaml.safe_load(f)
    cats          = spec['categories']
    default_sigma = spec.get('default_sigma', 0.25)

    rig = load_rig(args.rig_path, device='cpu')
    J = len(rig.joint_names)
    sigma = np.zeros(J, dtype=np.float32)
    cat_of = [''] * J

    for j, name in enumerate(rig.joint_names):
        sigma[j], cat_of[j] = _assign_sigma(name, cats, default_sigma)

    out = args.out or os.path.join(args.rig_path, 'sigma_targets.npy')
    np.save(out, sigma)

    # Report
    print(f'\n=== σ target assignment (rig={args.rig_path}) ===')
    print(f'  J={J}, spec: {cat_path}')
    print(f'  σ range: [{sigma.min():.4f}, {sigma.max():.4f}]\n')

    per_cat = {}
    for j in range(J):
        per_cat.setdefault(cat_of[j], []).append(j)

    for cat_name in ['narrow', 'medium', 'ears', 'wide', 'default']:
        js = per_cat.get(cat_name, [])
        if not js:
            continue
        sig = sigma[js[0]]
        print(f'  [{cat_name}] σ={sig:.3f}  ({len(js)} joints)')
        for j in js:
            print(f'      {j:3d}  {rig.joint_names[j]}')
        print()

    print(f'Saved → {out}')

    # Bar plot
    fig, ax = plt.subplots(1, 1, figsize=(max(12, J * 0.2), 5))
    xs = np.arange(J)
    order = np.argsort(-sigma)
    color_map = {'narrow': 'crimson', 'medium': 'goldenrod', 'ears': 'gray',
                 'wide': 'steelblue', 'default': 'lightgray'}
    colors = [color_map.get(cat_of[j], 'black') for j in order]
    ax.bar(xs, sigma[order], color=colors)
    labels = [rig.joint_names[j].replace('hybrid_jnt_', '')[:22] for j in order]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel('target σ (world units)')
    ax.set_title('Per-joint σ target  '
                 '(red=narrow, gold=medium, gray=ears, blue=wide)')
    fig.tight_layout()
    png = out.replace('.npy', '.png')
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot  → {png}')


if __name__ == '__main__':
    main()
