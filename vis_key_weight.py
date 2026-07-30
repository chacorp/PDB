"""
Render per-cage-vertex key_weight visualizations for identity meshes across datasets.

Usage:
    python vis_key_weight.py --dataset multiface --split test
"""
import os
import argparse
import pickle

import numpy as np
import torch
import igl
import yaml
import trimesh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm import tqdm

from eval_CBD import Trainer
from utils.matplotlib_rnd import translate, ortho, yrotate
from utils.keys import get_data_splits

CKPT_NAME = "2026-04-02-02-04-44-NGBCv5-dist"
CKPT_DIR = f"./ckpts_CBD/{CKPT_NAME}"

MF_TEMPLATE_PKL = "/data/sihun/pca/multiface_align/mf_templates.pkl"
VOCA_COMA_TEMPLATE_PKL = "/data/sihun/pca/VOCA-COMA/voca_templates.pkl"
BIWI_TEMPLATE_PKL = "./test-mesh/biwi_templates.pkl"
BIWI_PLY = "./test-mesh/BIWI.ply"
BIWI_ALIGN_NPY = "./utils/biwi/align.npy"
ICT_TEMPLATE_PKL = "/data/sihun/ICT-audio2face/split_set/ict_real_templates.pkl"


def build_trainer():
    config = "config/train_CBD.yml"
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
    opts = argparse.Namespace(**opts_yaml)

    opts.ckpt = CKPT_DIR
    opts.in_type = 1
    opts.out_type = 1
    opts.num_cage_v = 512
    opts.version = 5
    opts.last_activation = "relu"
    opts.no_pou = False
    opts.align_latent = True
    opts.use_data2 = False
    opts.use_data3 = False

    return Trainer(opts)


def get_identity_meshes(dataset, split):
    """
    Returns a list of (id_name, verts (N,3) float32, faces (F,3) int) for the given dataset/split.
    split='all' means every identity present in that dataset's template file (used for
    multiface/coma/biwi to cover train+val+test in one go).
    """
    ict_split, voca_split, biwi_split, mf_split, _ = get_data_splits()

    if dataset == "multiface":
        with open(MF_TEMPLATE_PKL, "rb") as f:
            mesh = pickle.load(f)
        faces = mesh["face"]
        id_names = [k for k in mesh.keys() if k != "face"] if split == "all" else mf_split[split]
        return [(id_name, mesh[id_name].astype(np.float32), faces) for id_name in id_names]

    elif dataset == "coma":
        with open(VOCA_COMA_TEMPLATE_PKL, "rb") as f:
            mesh = pickle.load(f)
        faces = mesh["face"]
        id_names = [k for k in mesh.keys() if k != "face"] if split == "all" else voca_split[split]
        return [(id_name, mesh[id_name].astype(np.float32), faces) for id_name in id_names]

    elif dataset == "biwi":
        biwi_trimesh = trimesh.load(BIWI_PLY)
        with open(BIWI_TEMPLATE_PKL, "rb") as f:
            mesh = pickle.load(f)
        faces = biwi_trimesh.faces
        m_align = np.load(BIWI_ALIGN_NPY)
        if split == "all":
            id_names = [k for k in mesh.keys() if k != "face"]
        else:
            id_names = biwi_split[split]
        out = []
        for id_name in id_names:
            v = mesh[id_name]
            v = np.concatenate((v, np.ones((v.shape[0], 1))), axis=1) @ m_align.T
            out.append((id_name, v[:, :3].astype(np.float32), faces))
        return out

    elif dataset == "ict":
        with open(ICT_TEMPLATE_PKL, "rb") as f:
            mesh = pickle.load(f)
        faces = mesh["face"]
        id_names = [k for k in mesh.keys() if k != "face"] if split == "all" else ict_split[split]
        return [(id_name, mesh[id_name].astype(np.float32), faces) for id_name in id_names]

    else:
        raise NotImplementedError(f"dataset '{dataset}' not supported yet")


def normalize_homogeneous(V):
    return np.concatenate([V, np.ones((V.shape[0], 1))], axis=1)


def normalize_bbox(verts):
    """Center verts on their bbox center and scale so the largest axis extent is 1.0."""
    bbox_min = verts.min(0)
    bbox_max = verts.max(0)
    center = (bbox_min + bbox_max) / 2
    scale = (bbox_max - bbox_min).max()
    return (verts - center) / scale


def project_view(verts_norm, faces, yrot):
    """
    Precomputes the sorted 2D triangle geometry (independent of cage_idx) for one view angle.
    Orthographic projection is used so apparent scale doesn't depend on camera distance,
    keeping the mesh consistently framed across all rotation angles.
    """
    V = normalize_homogeneous(verts_norm)

    view = translate(0, 0, -2.0)
    proj = ortho(-0.75, 0.75, -0.75, 0.75, 0.1, 10.0)
    MV = proj @ view

    model = yrotate(yrot)
    V_model = V @ model.T  # verts_norm is already centered at the origin

    V_proj = V_model @ MV.T
    V_proj = V_proj[:, :3] / V_proj[:, 3:4]

    VF = V_proj[faces]
    T = VF[:, :, :2]
    Z = -VF[:, :, 2].mean(1)
    order = np.argsort(Z)

    return T[order], order


def render_key_weight_view(
    T_sorted, order, faces, key_weight, cage_idx, yrot,
    SIZE=2, cmap="viridis", vmin=None, vmax=None, show_cbar=True,
):
    """
    Renders a single (precomputed) view of a mesh colored by key_weight[:, cage_idx].
    """
    Wv = key_weight[:, cage_idx]
    Wf = Wv[faces].mean(axis=1)

    if vmin is None:
        vmin = float(Wf.min())
    if vmax is None:
        vmax = float(Wf.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_fn = plt.get_cmap(cmap)

    W_sorted = Wf[order]
    C = cmap_fn(norm(W_sorted))

    # extra width reserves room for the colorbar + its tick labels so they aren't clipped
    fig = plt.figure(figsize=(SIZE * 1.4, SIZE))
    mesh_w = 1 / 1.4
    ax = fig.add_axes([0, 0, mesh_w, 1], xlim=[-1, 1], ylim=[-1, 1], aspect=1, frameon=False)
    coll = PolyCollection(T_sorted, closed=True, linewidth=0.1, facecolor=C, edgecolor=C)
    ax.add_collection(coll)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    if show_cbar:
        sm = ScalarMappable(norm=norm, cmap=cmap_fn)
        sm.set_array([])
        cbar_ax = fig.add_axes([mesh_w + 0.06, 0.12, 0.05, 0.76])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(f"key_weight[:, {cage_idx}]", fontsize=7)
        ticks = [vmin, (vmin + vmax) / 2, vmax] if vmax > vmin else [vmin]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.4f}" for t in ticks])
        cbar.ax.tick_params(labelsize=6)

    ax.set_title(f"cage {cage_idx} | view yrot={yrot}", fontsize=8)

    return fig


def visualize_identity(trainer, dataset, id_name, verts, faces, out_root, view_yrots=(0, 90, 180), resume=False):
    device = trainer.device
    v_th = torch.from_numpy(verts)[None].float().to(device)
    n_th = torch.from_numpy(
        igl.per_vertex_normals(verts.astype(np.float64), faces)
    )[None].float().to(device)

    with torch.no_grad():
        key_weight = trainer.model.predict_coordinate(v_th, n_th)
    key_weight = key_weight[0].detach().cpu().numpy()

    verts_norm = normalize_bbox(verts)

    # geometry (rotation/projection/depth-sort) only depends on view angle, not cage_idx
    view_geom = {yrot: project_view(verts_norm, faces, yrot) for yrot in view_yrots}

    num_cage = key_weight.shape[1]
    out_dir = os.path.join(out_root, dataset, id_name)
    os.makedirs(out_dir, exist_ok=True)

    for cage_idx in tqdm(range(num_cage), desc=f"{dataset}/{id_name}"):
        out_paths = {
            yrot: os.path.join(out_dir, f"cage{cage_idx:03d}_view{yrot:03d}.png")
            for yrot in view_yrots
        }
        if resume and all(os.path.exists(p) for p in out_paths.values()):
            continue

        vmin = float(key_weight[:, cage_idx].min())
        vmax = float(key_weight[:, cage_idx].max())
        for yrot in view_yrots:
            T_sorted, order = view_geom[yrot]
            fig = render_key_weight_view(
                T_sorted, order, faces, key_weight, cage_idx, yrot, vmin=vmin, vmax=vmax
            )
            fig.savefig(out_paths[yrot], dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="multiface", choices=["multiface", "coma", "biwi", "ict"])
    parser.add_argument("--split", type=str, default="all", help="'all' = every identity in the template file, or a split name like 'test'")
    parser.add_argument("--out_root", type=str, default=f"./vis_weight/{CKPT_NAME}")
    parser.add_argument("--resume", action="store_true", help="skip cage indices whose 3 view images already exist")
    args = parser.parse_args()

    trainer = build_trainer()
    identities = get_identity_meshes(args.dataset, args.split)
    print(f"{args.dataset}/{args.split}: {len(identities)} identity(ies) -> {[n for n,_,_ in identities]}", flush=True)

    for i, (id_name, verts, faces) in enumerate(identities):
        visualize_identity(trainer, args.dataset, id_name, verts, faces, args.out_root, resume=args.resume)
        print(f"[DONE] {args.dataset}/{id_name} ({i+1}/{len(identities)})", flush=True)

    print(f"[ALL DONE] {args.dataset}", flush=True)


if __name__ == "__main__":
    main()
