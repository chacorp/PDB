"""
Per-cell PNG figure generator for vis_ablation_mask_prev.html -- the
PRE-2026-09-21-refresh mask ablation gallery snapshot. Same approach as
render_ablation_mask_figure.py, but against the 5 checkpoints that gallery
actually used (confirmed via each checkpoint's train_opts.yml:
use_cage_consistency_loss/use_cyclic_loss/mask_cage_consist=None -- i.e. the
PLAIN (non-masked) cage-consistency/cyclic/cage+cyclic ablation, matching
render_ablation_frames.py's MODELS list, NOT the later "+mask" refresh).
baseline is ckpts_CBD7/2026-08-03-09-09-59-NGBCv5 (align_latent=True,
use_data1=True, out_type=1, no consist/cyclic loss) -- matching the other 3
ablation variants' config profile exactly (same script's/report's baseline),
NOT ckpts_CBD/2025-10-23-17-32-49-NGBCv5 (align_latent=False, a different,
unrelated "current relu-chosen" checkpoint from that era that was initially
assumed but doesn't match this trio's training setup):

    Ours-dagger (baseline): ckpts_CBD7/2026-08-03-09-09-59-NGBCv5
    cage-consistency:       ckpts_CBD7/2026-08-02-19-04-17-NGBCv5
    cyclic:                 ckpts_CBD_cyclic/2026-08-04-08-48-34-NGBCv5
    cage+cyclic:             ckpts_CBD7/2026-08-05-00-55-47-NGBCv5
    PDBplus:                ckpts_CBD8/2026-09-15-23-20-58-PDBplusv6 (version=6)

Output under vis_CBD/vis_ablation_mask/figure_prev/ (kept separate from
figure/, which belongs to the current 4-model refresh).

Run from repo root:
    python render/render_ablation_mask_prev_figure.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
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

OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_ablation_mask" / "figure_prev"

# NOTE: ds_mod.build_opts()'s yaml/argparse merge (`opts_yaml.update(opts_)`)
# actually lets argparse's own defaults silently overwrite the checkpoint's
# recorded yaml values for any field not in the per-model `override` dict --
# harmless for the current 2026-09-21 set (all align_latent=False, matching
# argparse's default), but these older checkpoints are align_latent=True, so
# it must be forced explicitly here per model (confirmed via each ckpt's own
# train_opts.yml) or the model gets built with the wrong hid_dim (128 vs 256)
# and load_state_dict fails on a shape mismatch.
MODELS = [
    ("baseline", "./ckpts_CBD7/2026-08-03-09-09-59-NGBCv5", {"align_latent": True}),
    ("consist", "./ckpts_CBD7/2026-08-02-19-04-17-NGBCv5", {"align_latent": True}),
    ("cyclic", "./ckpts_CBD_cyclic/2026-08-04-08-48-34-NGBCv5", {"align_latent": True}),
    ("both", "./ckpts_CBD7/2026-08-05-00-55-47-NGBCv5", {"align_latent": True}),
    ("pdbplus", "./ckpts_CBD8/2026-09-15-23-20-58-PDBplusv6", {"version": 6, "align_latent": False}),
]

COLUMN_MODEL = {name: ckpt for name, ckpt, _override in MODELS}
COLUMN_OVERRIDE = {name: override for name, ckpt, override in MODELS}

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


def load_pipelines():
    pipelines = {}
    for col_key, ckpt in COLUMN_MODEL.items():
        override = COLUMN_OVERRIDE[col_key]
        print(f"  [load model] col={col_key} ckpt={ckpt}")
        opts = ds_mod.build_opts(ckpt, override)
        trainer = Trainer(opts)
        pipeline = Pipeline(opts)
        pipeline.model = trainer.model
        pipelines[col_key] = pipeline
    return pipelines


def run_scenario(pipelines, scenario):
    key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, filter_dn, idxs = scenario
    f_self_cyc, f_cross = ds_mod.FACES[src_sel], ds_mod.FACES[tgt_sel]

    print(f"\n========== scenario={key} ==========")
    src_template = get_template_vertices(src_sel, src_mesh, exp_num)
    tgt_template = get_template_vertices(tgt_sel, tgt_mesh, exp_num)

    self_by_col, cross_by_col, cyclic_by_col = {}, {}, {}
    for col_key, ckpt in COLUMN_MODEL.items():
        print(f"  [load cache] col={col_key} ckpt={ckpt}")
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
                {ck: self_by_col[ck][k] for ck in COLUMN_MODEL}, f_self_cyc,
                src_template, out_dir)
        run_row("cross",
                {ck: cross_by_col[ck][k] for ck in COLUMN_MODEL}, f_cross,
                tgt_template, out_dir)
        run_row("cyclic",
                {ck: cyclic_by_col[ck][k] for ck in COLUMN_MODEL}, f_self_cyc,
                src_template, out_dir)
        print(f"  [saved] {out_dir}")


def main():
    pipelines = load_pipelines()
    for scenario in ds_mod.SCENARIOS:
        run_scenario(pipelines, scenario)
    print("\nALL DONE:", OUT_ROOT)


if __name__ == "__main__":
    main()
