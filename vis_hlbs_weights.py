"""
vis_hlbs_weights.py — Visualize HLBS skin weights per joint.

Loads a trained HLBS checkpoint, runs forward on the mean template to get
predicted skin weights W [N, J], then renders per-joint heatmaps.

Usage:
    python vis_hlbs_weights.py \
        --ckpt ./ckpts_hlbs/2026-03-26-07-47-48-HLBS-mf-s0-jTrans/model_hlbs_200.pth \
        --rig_path maya_rig --topo_key mf
"""
import os
import sys
import argparse
import numpy as np
import yaml
import torch

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt

from utils.rig_loader import load_rig
from models.hierarchical_lbs import HierarchicalLBS
from utils.matplotlib_rnd import vis_mesh_key_weight, vis_mesh_all_cage_weights


def _render_all_joints_overview(verts, faces, W, joint_names, out_dir, tag):
    """Render all-joint overview using existing vis_mesh_all_cage_weights."""
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for saving

    for mode in ['argmax', 'blend']:
        save_path = os.path.join(out_dir, f"W_{tag}_all_{mode}.png")
        vis_mesh_all_cage_weights(
            verts, faces, W,
            mode=mode,
            view_yrots=(0, 45, -45),
        )
        fig = plt.gcf()
        fig.suptitle(f"All joints ({tag}) — {mode}", fontsize=12)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="HLBS checkpoint .pth")
    parser.add_argument("--rig_path", type=str, default="maya_rig")
    parser.add_argument("--topo_key", type=str, default="mf")
    parser.add_argument("--hid_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--freeze_adapt", action="store_true")
    parser.add_argument("--use_joint_trans", action="store_true")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir (default: {ckpt_dir}/vis_weights)")
    parser.add_argument("--joints", type=str, default=None,
                        help="Comma-separated joint indices to visualize (default: all)")
    parser.add_argument("--show_maya_init", action="store_true",
                        help="Also render Maya init weights for comparison")
    parser.add_argument("--all_only", action="store_true",
                        help="Only render all-joints overview (skip per-joint)")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def load_template_verts_faces(topo_key):
    """Load mean template vertices and faces."""
    import pickle
    if topo_key == 'mf':
        std = np.load("utils/mf/standardization.npy", allow_pickle=True).item()
        faces = np.array(std['new_f'], dtype=np.int32)
        with open("/data/sihun/pca/multiface_align/mf_templates.pkl", 'rb') as f:
            templates = pickle.load(f)
        # Use first identity or mean
        first_id = [k for k in templates if k != 'face'][0]
        verts = templates[first_id].astype(np.float32)
        if 'face' in templates:
            faces = np.array(templates['face'], dtype=np.int32)
    else:
        raise NotImplementedError(f"topo_key '{topo_key}' not implemented")
    return verts, faces


def auto_load_opts(ckpt_path):
    """Try to load train_opts.yml from checkpoint directory."""
    ckpt_dir = os.path.dirname(ckpt_path)
    opts_path = os.path.join(ckpt_dir, "train_opts.yml")
    if os.path.exists(opts_path):
        with open(opts_path) as f:
            return yaml.safe_load(f)
    return None


def main():
    args = parse_args()

    # Try to inherit model config from train_opts.yml
    saved_opts = auto_load_opts(args.ckpt)
    if saved_opts:
        for key in ['hid_dim', 'num_layers', 'freeze_adapt', 'use_joint_trans']:
            if key in saved_opts and not any(f'--{key}' in a for a in sys.argv):
                setattr(args, key, saved_opts[key])
        print(f"Inherited model config from: {os.path.dirname(args.ckpt)}/train_opts.yml")

    device = torch.device(args.device)

    # Load rig + model
    rig = load_rig(args.rig_path)
    model = HierarchicalLBS(
        rig=rig,
        topology=args.topo_key,
        in_dim_exp=12,
        hid_dim=args.hid_dim,
        num_layers=args.num_layers,
        device=str(device),
        freeze_adapt=args.freeze_adapt,
        use_joint_trans=args.use_joint_trans,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Loaded: {args.ckpt}")

    # Load template
    verts, faces = load_template_verts_faces(args.topo_key)
    src_v = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)  # [1, V, 3]

    # Get predicted skin weights
    with torch.no_grad():
        W_pred, delta_W = model._get_skinning_weights(src_v)  # [1, N, J]

    W_np = W_pred[0].cpu().numpy()          # [N, J]
    delta_W_np = delta_W[0].cpu().numpy()   # [N, J]
    J = W_np.shape[1]

    # Maya init weights (softmax of logit_W_base with delta=0)
    import torch.nn.functional as F
    W_maya = F.softmax(model.logit_W_base, dim=-1).cpu().numpy()  # [N, J]

    print(f"Vertices: {verts.shape[0]}, Joints: {J}")
    print(f"delta_W range: [{delta_W_np.min():.4f}, {delta_W_np.max():.4f}]")
    print(f"W_pred  range: [{W_np.min():.6f}, {W_np.max():.6f}]")

    # Output dir
    if args.out_dir is None:
        ckpt_dir = os.path.dirname(args.ckpt)
        args.out_dir = os.path.join(ckpt_dir, "vis_weights")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── All-joints overview (always rendered) ───────────────────────────
    _render_all_joints_overview(verts, faces, W_np, model.joint_names, args.out_dir, "pred")
    if args.show_maya_init:
        _render_all_joints_overview(verts, faces, W_maya, model.joint_names, args.out_dir, "maya")

    if args.all_only:
        print(f"Done (--all_only). Output: {args.out_dir}")
        return

    # Which joints to visualize
    if args.joints:
        joint_indices = [int(x) for x in args.joints.split(",")]
    else:
        # Visualize joints with significant weight (max weight > 0.05)
        max_per_joint = W_np.max(axis=0)
        joint_indices = [j for j in range(J) if max_per_joint[j] > 0.05]
        print(f"Visualizing {len(joint_indices)}/{J} joints with max weight > 0.05")

    joint_names = model.joint_names

    for j in joint_indices:
        name = joint_names[j] if j < len(joint_names) else f"joint_{j}"
        # Per-joint vmax: shows gradient transition clearly
        j_max_pred = W_np[:, j].max()
        j_max_maya = W_maya[:, j].max()
        j_vmax = max(j_max_pred, j_max_maya, 0.01)

        save_path = os.path.join(args.out_dir, f"W_pred_{j:02d}_{name}.png")
        vis_mesh_key_weight(
            verts, faces, W_np, cage_idx=j,
            cmap='magma', vmin=0, vmax=j_vmax,
            view_yrots=(0, 45, -45),
            save_path=save_path,
            title=f"Predicted W — {name} (j{j}, max={j_max_pred:.3f})",
            shade=True,
        )

        if args.show_maya_init:
            save_path_maya = os.path.join(args.out_dir, f"W_maya_{j:02d}_{name}.png")
            vis_mesh_key_weight(
                verts, faces, W_maya, cage_idx=j,
                cmap='magma', vmin=0, vmax=j_vmax,
                view_yrots=(0, 45, -45),
                save_path=save_path_maya,
                title=f"Maya Init W — {name} (j{j}, max={j_max_maya:.3f})",
                shade=True,
            )

    print(f"\nSaved {len(joint_indices)} per-joint weight visualizations to: {args.out_dir}")

    # ── All-joints overview: blended + argmax ────────────────────────────
    _render_all_joints_overview(verts, faces, W_np, joint_names, args.out_dir, "pred")
    if args.show_maya_init:
        _render_all_joints_overview(verts, faces, W_maya, joint_names, args.out_dir, "maya")

    # Summary: top-10 most active joints
    max_per_joint = W_np.max(axis=0)
    top5 = np.argsort(max_per_joint)[::-1][:10]
    print("\nTop-10 joints by max weight:")
    for j in top5:
        name = joint_names[j] if j < len(joint_names) else f"joint_{j}"
        print(f"  [{j:2d}] {name:30s}  max_W={max_per_joint[j]:.4f}  mean_W={W_np[:, j].mean():.6f}")


if __name__ == "__main__":
    main()
