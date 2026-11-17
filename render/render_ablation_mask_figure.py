"""
Per-cell PNG figure generator for the mask_cage_consist ablation qualitative
comparison table -- same structure/approach as render_design_study_figure.py,
just against render_ablation_frames_mask.py's 4 MODELS (baseline / w_consist /
w_cyclic / w_both) instead of design_study's 8.

For each of the 9 SCENARIOS and each of its 4 frames, renders:
  - 3 identity-reference images, once per (scenario, frame), independent of
    self/cross/cyclic: source_neutral.png, source_expression.png (real
    captured GT deformation, own independent diff-heatmap color scale),
    target_neutral.png.
  - per row (self / cross / cyclic retargeting), 4 variant-checkpoint output
    images (baseline, w_consist, w_cyclic, w_both), diff-heatmap colors
    normalized jointly per row (shared min/max across all 4, against that
    row's diff_base) -- no variant has needed a separate scale here (unlike
    design_study's softplus), but SEPARATE_SCALE_COLS is kept for parity/an
    easy override if one shows up.
Each image is an INDIVIDUAL PNG, one file per cell. No new GPU inference for
cache hits (get_or_run reuses eval_cyc/ caches already on disk); cache misses
(e.g. baseline's self direction) fall back to a cheap filtered pass.

Run from repo root:
    python render/render_ablation_mask_figure.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
# see render_design_study_figure.py for why this must come before any
# `import render.xxx` -- render/render.py (an unrelated psbody/mediapy video
# helper) shadows the render/ package when this script's own directory gets
# auto-inserted at sys.path[0] by a direct `python render/....py` invocation.
sys.path.insert(0, str(REPO_ROOT / "render"))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotrender import plot_mesh_gouraud  # noqa: E402
from matplotrender.utils import calc_face_norm  # noqa: E402

import render_ablation_frames_mask as ds_mod  # noqa: E402
import eval_CBD_cyc  # noqa: E402
from eval_CBD_cyc import Pipeline  # noqa: E402
from eval_CBD import Trainer  # noqa: E402

M_SCALE = 0.6
M_TRNS = np.array([0, 0, 0])
SIZE = 8
LIGHT_DIR = np.array([0, 0, 1])

OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_ablation_mask" / "figure"

# column key -> MODELS-entry label (from render_ablation_frames_mask.MODELS)
COLUMN_MODEL = {
    "baseline": "baseline",
    "w_consist": "cage-consistency+mask",
    "w_cyclic": "cyclic+mask",
    "w_both": "cage+cyclic+mask",
}

SEPARATE_SCALE_COLS = set()


def get_template_vertices(data_name, id_num, exp_num):
    key = (data_name, id_num, exp_num)
    if key not in ds_mod._raw_ds_cache:
        ds_mod.get_raw_vertices(data_name, id_num, exp_num, 0)
    ds = ds_mod._raw_ds_cache[key]
    return ds[0][1].numpy()


def save_plain_mesh(V, F, out_path):
    plot_mesh_gouraud(
        [V], [F], mesh_scale=M_SCALE, mesh_trans=M_TRNS,
        is_diff=False, backface_culling=True, rot_list=[[0, 0, 0]],
        size=SIZE, mode="shade", bg_black=False,
        logdir=str(out_path.parent), name=out_path.stem, save=True, show=False,
    )


def row_joint_colors(Vs, F, diff_base):
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
        vc = vc * (1 - mask) + Sc * mask
        colors.append(vc)
    return colors


def save_diff_mesh(V, F, color, out_path):
    plot_mesh_gouraud(
        [V], [F], Cs=[color], mesh_scale=M_SCALE, mesh_trans=M_TRNS,
        is_color=True, backface_culling=True, rot_list=[[0, 0, 0]],
        size=SIZE, mode="shade", bg_black=False,
        logdir=str(out_path.parent), name=out_path.stem, save=True, show=False,
    )


def save_reference_images(src_template, gt_v, tgt_template, f_src, f_tgt, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_plain_mesh(src_template, f_src, out_dir / "source_neutral.png")
    save_plain_mesh(tgt_template, f_tgt, out_dir / "target_neutral.png")
    gt_color = row_joint_colors([gt_v], f_src, src_template)[0]
    save_diff_mesh(gt_v, f_src, gt_color, out_dir / "source_expression.png")


def run_row(row_name, variant_Vs, variant_F, diff_base, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_keys = list(COLUMN_MODEL.keys())
    joint_keys = [k for k in variant_keys if k not in SEPARATE_SCALE_COLS]
    separate_keys = [k for k in variant_keys if k in SEPARATE_SCALE_COLS]

    joint_colors = row_joint_colors([variant_Vs[k] for k in joint_keys], variant_F, diff_base)
    for k, color in zip(joint_keys, joint_colors):
        save_diff_mesh(variant_Vs[k], variant_F, color, out_dir / f"{row_name}_{k}.png")

    for k in separate_keys:
        color = row_joint_colors([variant_Vs[k]], variant_F, diff_base)[0]
        save_diff_mesh(variant_Vs[k], variant_F, color, out_dir / f"{row_name}_{k}.png")


def resolve_column_ckpts():
    by_label = {name: ckpt for name, ckpt, _override in ds_mod.MODELS}
    ckpt_by_col = {}
    for col_key, label in COLUMN_MODEL.items():
        assert label in by_label, f"model label not found in MODELS: {label}"
        ckpt_by_col[col_key] = by_label[label]
    return ckpt_by_col


def load_pipelines(ckpt_by_col):
    pipelines = {}
    for col_key, ckpt in ckpt_by_col.items():
        label = COLUMN_MODEL[col_key]
        override = next(ov for name, c, ov in ds_mod.MODELS if name == label)
        print(f"  [load model] col={col_key} ckpt={ckpt}")
        opts = ds_mod.build_opts(ckpt, override)
        trainer = Trainer(opts)
        pipeline = Pipeline(opts)
        pipeline.model = trainer.model
        pipelines[col_key] = pipeline
    return pipelines


def run_scenario(ckpt_by_col, pipelines, scenario):
    key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, filter_dn, idxs = scenario
    f_self_cyc, f_cross = ds_mod.FACES[src_sel], ds_mod.FACES[tgt_sel]

    print(f"\n========== scenario={key} ==========")
    src_template = get_template_vertices(src_sel, src_mesh, exp_num)
    tgt_template = get_template_vertices(tgt_sel, tgt_mesh, exp_num)

    self_by_col, cross_by_col, cyclic_by_col = {}, {}, {}
    for col_key, ckpt in ckpt_by_col.items():
        print(f"  [load cache] col={col_key} ckpt={ckpt}")
        # see render_design_study_figure.py's run_scenario for why this global
        # re-sync is needed before every get_or_run call.
        eval_CBD_cyc.opts = pipelines[col_key].opts
        v_self, _ = ds_mod.get_or_run(pipelines[col_key], src_sel, src_mesh, src_sel, src_mesh, exp_num,
                                       filter_dn, idxs, need_cyc=False, ckpt=ckpt)
        v_cross, v_cyclic = ds_mod.get_or_run(pipelines[col_key], src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num,
                                               filter_dn, idxs, need_cyc=True, ckpt=ckpt)
        self_by_col[col_key] = v_self
        cross_by_col[col_key] = v_cross
        cyclic_by_col[col_key] = v_cyclic

    for k, frame_idx in enumerate(idxs):
        print(f"  -- frame {frame_idx} --")
        gt_v = ds_mod.get_raw_vertices(src_sel, src_mesh, exp_num, frame_idx)
        out_dir = OUT_ROOT / key / f"frame_{frame_idx:06d}"

        save_reference_images(src_template, gt_v, tgt_template, f_self_cyc, f_cross, out_dir)
        run_row("self",
                {ck: self_by_col[ck][k] for ck in ckpt_by_col}, f_self_cyc,
                src_template, out_dir)
        run_row("cross",
                {ck: cross_by_col[ck][k] for ck in ckpt_by_col}, f_cross,
                tgt_template, out_dir)
        run_row("cyclic",
                {ck: cyclic_by_col[ck][k] for ck in ckpt_by_col}, f_self_cyc,
                src_template, out_dir)
        print(f"  [saved] {out_dir}")


def main():
    ckpt_by_col = resolve_column_ckpts()
    pipelines = load_pipelines(ckpt_by_col)
    for scenario in ds_mod.SCENARIOS:
        run_scenario(ckpt_by_col, pipelines, scenario)
    print("\nALL DONE:", OUT_ROOT)


if __name__ == "__main__":
    main()
