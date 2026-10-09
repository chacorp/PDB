"""Per-cell PNG figure for the comparison baselines (NC / NFS / NFR / PDB),
mf <-> ict and mf / ict -> coma / biwi scenarios, same layout as
render_ablation_mask_figure.py:

  left  : source neutral / source expression (GT, frame-captioned) / target neutral
  right : rows = self / cross / cyclic retargeting, cols = NC / NFS / NFR / PDB

Each cell is a single PNG. Diff heatmaps (YlOrRd) are normalized per row
against that row's diff_base (source template for self/cyclic, target template
for cross). NC is normalized on its own scale (it diverges, so sharing a scale
would wash out the other three); NFS/NFR/PDB share one scale per row.

Cache-only: reads the existing eval_CBD (self) and eval_cyc (cross/cyclic)
`-masked-laplacian` results; coma / biwi cross/cyclic come from
render/cache_cross_cyc_comparison.py and ict self from cache_ict_self_comparison.py. Missing caches are rendered as "missing" cells
rather than triggering new inference.

Run from repo root:
    python render/render_comparison_figure.py [scenario_key ...]
"""
import os
import sys
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
# matplotrender does `from utils import xrotate, ...` expecting the site-packages
# utils.py; the repo's utils/ package would shadow it, so import it with the repo
# path hidden, then drop the cached `utils` module so the repo's utils still loads.
def _is_repo(p):
    return os.path.abspath(p or ".").startswith(str(REPO_ROOT))


_repo_paths = [p for p in sys.path if _is_repo(p)]
sys.path = [p for p in sys.path if not _is_repo(p)]
try:
    from matplotrender import plot_mesh_gouraud, calc_face_norm  # noqa: E402
finally:
    sys.path[:0] = _repo_paths
    for _m in [m for m in sys.modules if m == "utils" or m.startswith("utils.")]:
        del sys.modules[_m]

import render_ablation_frames_comparison as cmp_mod  # noqa: E402

M_SCALE = 0.6
M_TRNS = np.array([0, 0, 0])
SIZE = 8
LIGHT_DIR = np.array([0, 0, 1])

OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "figure"

# (name, ckpt dir name, eval_CBD self folder name, scale group)
COLUMNS = [
    ("NC", "2025-10-20-00-15-40-CBD", "2025-10-20-00-15-40-CBD-eval", "NC"),
    ("NFS", "2024-08-18-23-32-29-all", "2024-08-18-23-32-29-all-eval", "shared"),
    ("NFR", "exp_019_ICT_MF-jacob_NFR", "exp_019_ICT_MF-jacob_NFR-eval", "shared"),
    ("PDB", "2026-09-16-02-19-03-NGBCv5", "2026-09-16-02-19-03-NGBCv5-eval", "shared"),
]

import cache_cross_cyc_comparison as cc_mod  # noqa: E402

# mf <-> ict (eval_cyc caches) + coma / biwi targets (cross_cyc caches)
SCENARIOS = [s for s in cmp_mod.SCENARIOS if s[0] in ("mf_to_ict", "ict_to_mf_m00", "ict_to_mf_m02")]
SCENARIOS += cc_mod.SCENARIOS
CROSS_CYC_KEYS = {s[0] for s in cc_mod.SCENARIOS}

FACES = dict(cmp_mod.FACES)
FACES.update({sel: cmp_mod.faces_for(sel) for sel in ["coma", "biwi"]})


def eval_cyc_dir(ckpt_dir, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num):
    """Laplacian-suffixed eval_cyc folder (the one produced by eval_CBD_cyc.py --laplacian)."""
    if src_sel == "ict-cap":
        fn = f"ict-cap-ID_{src_mesh:03d}_test-to-{tgt_sel}-ID_{tgt_mesh:03d}_test_{exp_num:02d}"
    else:
        fn = f"{src_sel}-src{src_mesh:03d}_test-to-{tgt_sel}_test-ID_{tgt_mesh:03d}"
    return REPO_ROOT / "eval_cyc" / ckpt_dir / f"{fn}-masked-laplacian"


def eval_cbd_self_dir(eval_folder, src_sel):
    """eval_CBD.py self-retargeting folder (mf_ROM / mf_SEN / ict)."""
    name = {"mf_ROM": "mf_ROM", "mf_SEN": "mf_SEN", "ict-cap": "ict"}[src_sel]
    return REPO_ROOT / "eval_CBD" / eval_folder / f"{name}-masked-laplacian"


def load_npy(path, idx):
    p = path / f"{idx:06d}.npy"
    return np.load(p) if p.exists() else None


def gather(scenario):
    """Return dict[(row, col_name)] -> vertices or None, plus raw/template data."""
    key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, _filter_dn, idxs = scenario
    results = {}
    for col_name, ckpt_dir, eval_folder, _grp in COLUMNS:
        cyc_dir = eval_cyc_dir(ckpt_dir, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num)
        self_dir = eval_cbd_self_dir(eval_folder, src_sel)
        for frame_idx in idxs:
            # self: mf sources come from eval_CBD; ict-cap sources come from the
            # per-frame cache written by render/cache_ict_self_comparison.py.
            # coma / biwi scenarios reuse the same source clips/frames as
            # ict_to_mf_m00 / m02, so their ict self cache is shared.
            if src_sel == "ict-cap":
                self_key = "ict_to_mf_" + key.rsplit("_", 1)[-1]
                p = REPO_ROOT / "vis_CBD" / "vis_comparison" / "self_ict" / ckpt_dir / self_key / f"frame_{frame_idx:06d}.npy"
                results[(frame_idx, "self", col_name)] = np.load(p) if p.exists() else None
            else:
                results[(frame_idx, "self", col_name)] = load_npy(self_dir / "verts", frame_idx)
            if key in CROSS_CYC_KEYS:
                for row in ("cross", "cyclic"):
                    p = cc_mod.OUT_ROOT / ckpt_dir / key / f"frame_{frame_idx:06d}_{row}.npy"
                    results[(frame_idx, row, col_name)] = np.load(p) if p.exists() else None
                continue
            results[(frame_idx, "cross", col_name)] = (
                load_npy(cyc_dir / "verts", frame_idx) if cyc_dir.exists() else None
            )
            results[(frame_idx, "cyclic", col_name)] = (
                load_npy(cyc_dir / "verts_cyc", frame_idx) if cyc_dir.exists() else None
            )
    return results


def get_template(data_name, id_num, exp_num):
    if data_name in ("coma", "biwi"):
        # same template eval_CBD_cyc.Pipeline.get_mesh uses as the retarget target
        with open(cmp_mod._FACE_PKL[data_name], "rb") as f:
            mesh = pickle.load(f)
        ids = [k for k in mesh.keys() if k != "face"]
        return np.asarray(mesh[ids[id_num]])
    return cmp_mod.get_raw_vertices(data_name, id_num, exp_num, 0)


def save_plain_mesh(V, F, out_path):
    plot_mesh_gouraud(
        [V], [F], mesh_scale=M_SCALE, mesh_trans=M_TRNS,
        is_diff=False, backface_culling=True, rot_list=[[0, 0, 0]],
        size=SIZE, mode="shade", bg_black=False,
        logdir=str(out_path.parent), name=out_path.stem, save=True, show=False,
    )


def joint_colors(Vs, F, diff_base):
    """Heatmap colors for a list of meshes sharing one min/max (diff vs diff_base)."""
    Vs_np = np.stack([np.asarray(v) for v in Vs])
    Cdist = np.linalg.norm(np.abs(Vs_np - np.asarray(diff_base)), axis=-1)
    cmin, cmax = Cdist.min(), Cdist.max()
    Cnorm = (Cdist - cmin) / (cmax - cmin + 1e-12)
    colors = []
    for k, V in enumerate(Vs):
        V_scaled = np.asarray(V) * M_SCALE + M_TRNS
        vn = calc_face_norm(V_scaled, F, mode='v')
        shading = np.clip(vn @ LIGHT_DIR, 0, 1)
        vc = shading[..., None].repeat(3, axis=-1) * 0.6 + 0.3
        Sc = plt.get_cmap("YlOrRd")(Cnorm[k])[..., :3]
        mask = Cnorm[k][:, None]
        colors.append(vc * (1 - mask) + Sc * mask)
    return colors


def save_diff_mesh(V, F, color, out_path):
    plot_mesh_gouraud(
        [V], [F], Cs=[color], mesh_scale=M_SCALE, mesh_trans=M_TRNS,
        is_color=True, backface_culling=True, rot_list=[[0, 0, 0]],
        size=SIZE, mode="shade", bg_black=False,
        logdir=str(out_path.parent), name=out_path.stem, save=True, show=False,
    )


def render_row(row, vs_by_col, F, diff_base, out_dir):
    """vs_by_col: col_name -> V or None. NC on its own scale; the rest share one."""
    present = {c: v for c, v in vs_by_col.items() if v is not None}
    shared_cols = [c for c, _, _, g in COLUMNS if g == "shared" and c in present]
    nc_cols = [c for c, _, _, g in COLUMNS if g == "NC" and c in present]
    colors = {}
    if shared_cols:
        for c, col in zip(shared_cols, joint_colors([present[c] for c in shared_cols], F, diff_base)):
            colors[c] = col
    for c in nc_cols:
        colors[c] = joint_colors([present[c]], F, diff_base)[0]
    for c, _, _, _ in COLUMNS:
        out_path = out_dir / f"{row}_{c}.png"
        if c in colors:
            save_diff_mesh(present[c], F, colors[c], out_path)
        else:
            out_path.unlink(missing_ok=True)


def run_scenario(scenario):
    key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, _filter_dn, idxs = scenario
    print(f"\n========== scenario={key} ==========")
    F_src, F_tgt = FACES[src_sel], FACES[tgt_sel]
    src_template = get_template(src_sel, src_mesh, exp_num)
    tgt_template = get_template(tgt_sel, tgt_mesh, exp_num)
    results = gather(scenario)

    for frame_idx in idxs:
        out_dir = OUT_ROOT / key / f"frame_{frame_idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        gt = cmp_mod.get_raw_vertices(src_sel, src_mesh, exp_num, frame_idx)

        save_plain_mesh(src_template, F_src, out_dir / "source_neutral.png")
        save_plain_mesh(tgt_template, F_tgt, out_dir / "target_neutral.png")
        gt_color = joint_colors([gt], F_src, src_template)[0]
        save_diff_mesh(gt, F_src, gt_color, out_dir / "source_expression.png")

        for row, F, base in [("self", F_src, src_template),
                             ("cross", F_tgt, tgt_template),
                             ("cyclic", F_src, src_template)]:
            vs = {c[0]: results[(frame_idx, row, c[0])] for c in COLUMNS}
            render_row(row, vs, F, base, out_dir)
        print(f"  [saved] {out_dir}")


def main():
    keys = sys.argv[1:]  # optional: only these scenario keys
    for scenario in SCENARIOS:
        if keys and scenario[0] not in keys:
            continue
        run_scenario(scenario)
    print("\nALL DONE:", OUT_ROOT)


if __name__ == "__main__":
    main()
