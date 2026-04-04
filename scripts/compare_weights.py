"""
Compare ICT original vs MF transferred skinning weights side by side.

Usage:
    python scripts/compare_weights.py --rig_dir maya_rig/hybrid
"""
import os, sys, argparse
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.matplotlib_rnd import vis_mesh_all_cage_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig_dir", type=str, default="maya_rig/hybrid")
    parser.add_argument("--out_dir", type=str, default="analysis_output/weight_transfer")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load meshes
    mesh_ict = trimesh.load("utils/ict/ict_aligned_mean.obj", process=False)
    V_ict, F_ict = np.array(mesh_ict.vertices, dtype=np.float32), np.array(mesh_ict.faces)

    mesh_mf = trimesh.load("utils/mf/mf_aligned_mean.obj", process=False)
    V_mf, F_mf = np.array(mesh_mf.vertices, dtype=np.float32), np.array(mesh_mf.faces)

    # Load weights
    W_ict = np.load(os.path.join(args.rig_dir, "skin_weights_ict.npy"))
    W_mf = np.load(os.path.join(args.rig_dir, "skin_weights_mf.npy"))

    # Load joint names
    import json
    rig_info_path = os.path.join(args.rig_dir, "rig_info_ict.json")
    with open(rig_info_path) as f:
        rig_info = json.load(f)
    joint_names = [None] * len(rig_info['joints'])
    for j in rig_info['joints']:
        joint_names[j['index']] = j['name']

    print(f"ICT: {V_ict.shape[0]} verts, {W_ict.shape}")
    print(f"MF:  {V_mf.shape[0]} verts, {W_mf.shape}")
    print(f"Joints: {len(joint_names)}")

    # Render argmax overview for both
    for tag, verts, faces, W in [('ict', V_ict, F_ict, W_ict),
                                  ('mf_transferred', V_mf, F_mf, W_mf)]:
        for mode in ['argmax']:
            save_path = os.path.join(args.out_dir, f"W_{tag}_{mode}.png")
            vis_mesh_all_cage_weights(
                verts, faces, W,
                mode=mode,
                view_yrots=(0, 45, -45),
            )
            fig = plt.gcf()
            fig.suptitle(f"{tag} — {mode}", fontsize=12)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")

    # Per-joint comparison for key joints (lip-related)
    lip_joints = [i for i, name in enumerate(joint_names)
                  if name and any(k in name.lower() for k in ['lip', 'jaw', 'mouth'])]
    print(f"\nLip-related joints: {[(i, joint_names[i]) for i in lip_joints]}")

    for ji in lip_joints:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, tag, verts, faces, W in [(axes[0], 'ICT', V_ict, F_ict, W_ict),
                                          (axes[1], 'MF transferred', V_mf, F_mf, W_mf)]:
            w = W[:, ji]
            ax.tripcolor(verts[:, 0], verts[:, 1], faces, w,
                         cmap='hot', vmin=0, vmax=1)
            ax.set_title(f"{tag}: {joint_names[ji]}")
            ax.set_aspect('equal')
            ax.invert_yaxis()

        save_path = os.path.join(args.out_dir, f"joint_{ji:02d}_{joint_names[ji]}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {save_path}")

    print(f"\nAll saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
