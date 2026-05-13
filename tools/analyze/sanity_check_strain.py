"""
Sanity check: visualize GT strain for multiface training data.
For each identity, load a random real frame, compute Green-Lagrange strain,
and render a heatmap on the mesh.
"""
import os
import sys
import glob
import pickle
import random
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
from utils.mesh_utils import compute_vertex_strain
from utils.matplotlib_rnd import vis_mesh_key_weight

# ---- config ----
DATA_BASE = '/data/sihun/multiface_align'
TEMPLATE_PKL = f'{DATA_BASE}/mf_templates.pkl'
SEN_DIR = f'{DATA_BASE}/SEN/train/vertices_npy'
ROM_DIR = f'{DATA_BASE}/ROM/train/vertices_npy'
OUT_DIR = 'sanity_check_strain_output'
os.makedirs(OUT_DIR, exist_ok=True)

# ---- load templates & faces ----
with open(TEMPLATE_PKL, 'rb') as f:
    mf_mesh = pickle.load(f)
faces_np = mf_mesh['face']
faces = torch.tensor(faces_np).long()

# ---- gather identities ----
id_dirs_sen = sorted(glob.glob(f'{SEN_DIR}/m--*'))
id_dirs_rom = sorted(glob.glob(f'{ROM_DIR}/m--*'))

def load_random_frame(id_dir):
    """Load a random .npy frame from a random sentence/sequence."""
    seq_dirs = [d for d in glob.glob(f'{id_dir}/*') if os.path.isdir(d)]
    if not seq_dirs:
        return None
    seq = random.choice(seq_dirs)
    frames = sorted(glob.glob(f'{seq}/*.npy'))
    if not frames:
        return None
    # pick a frame in the middle (more likely to have expression)
    mid = len(frames) // 2
    idx = random.randint(max(0, mid - 10), min(len(frames) - 1, mid + 10))
    return np.load(frames[idx])

print(f"Found {len(id_dirs_sen)} SEN identities, {len(id_dirs_rom)} ROM identities")

# ---- process each identity ----
for source, id_dirs in [('SEN', id_dirs_sen), ('ROM', id_dirs_rom)]:
    for id_dir in id_dirs:
        id_name = os.path.basename(id_dir)

        if id_name not in mf_mesh:
            print(f"  [skip] {id_name} not in template pkl")
            continue

        template_np = mf_mesh[id_name]  # (V, 3)
        deformed_np = load_random_frame(id_dir)
        if deformed_np is None:
            print(f"  [skip] {id_name}: no frames found")
            continue

        # compute strain
        template_t = torch.tensor(template_np).float().unsqueeze(0)  # [1,V,3]
        deformed_t = torch.tensor(deformed_np).float().unsqueeze(0)  # [1,V,3]

        strain_norm, strain_trace = compute_vertex_strain(
            deformed_t, template_t, faces, return_trace=True
        )
        # [1,V,1] -> [V]
        strain_norm_np = strain_norm[0, :, 0].numpy()
        strain_trace_np = strain_trace[0, :, 0].numpy()

        # also compute template->template strain (should be ~0)
        strain_zero = compute_vertex_strain(template_t, template_t, faces)
        zero_max = strain_zero.max().item()

        short_id = id_name.split('--')[1][:4] + '_' + id_name.split('--')[3][:6]

        print(f"[{source}] {short_id}: "
              f"||E||_F  min={strain_norm_np.min():.6f} max={strain_norm_np.max():.6f} mean={strain_norm_np.mean():.6f} | "
              f"trace(E) min={strain_trace_np.min():.6f} max={strain_trace_np.max():.6f} | "
              f"zero-check max={zero_max:.2e}")

        # ---- visualize ||E||_F ----
        strain_for_vis = strain_norm_np.reshape(-1, 1)  # [V, 1]
        vis_mesh_key_weight(
            template_np, faces_np, strain_for_vis, cage_idx=0,
            cmap='hot', vmin=0, vmax=max(strain_norm_np.max(), 1e-6),
            title=f'{source} {short_id} ||E||_F',
            save_path=f'{OUT_DIR}/{source}_{short_id}_strain_norm.png',
        )

        # ---- visualize trace(E) (signed) ----
        strain_trace_vis = strain_trace_np.reshape(-1, 1)
        absmax = max(abs(strain_trace_np.min()), abs(strain_trace_np.max()), 1e-6)
        vis_mesh_key_weight(
            template_np, faces_np, strain_trace_vis, cage_idx=0,
            cmap='bwr', vmin=-absmax, vmax=absmax,
            title=f'{source} {short_id} trace(E)',
            save_path=f'{OUT_DIR}/{source}_{short_id}_strain_trace.png',
        )

print(f"\nDone! Results saved to {OUT_DIR}/")
