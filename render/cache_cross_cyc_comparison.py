"""Cross / cyclic retargeting outputs for the coma / biwi target scenarios
(mf_to_coma, mf_to_biwi, ict_to_coma_m00/m02, ict_to_biwi_m00/m02), for the
comparison baselines NC / NFS / NFR / PDB, saved as one .npy per frame under
vis_CBD/vis_comparison/cross_cyc/<ckpt>/<scenario>/frame_XXXXXX_{cross,cyclic}.npy
so that render/render_comparison_figure.py can read them without re-running
inference.

Targets / frames follow vis_ablation_mask.html (coma FaceTalk_170809_00138_TA =
coma mesh idx 6, biwi F2 = biwi mesh idx 1; mf_ROM frames 8880/5840/8404/6638);
ict sources reuse the mf<->ict scenarios' m00 / m02 clips and frames.

Uses render_ablation_frames_comparison.get_or_run (filtered pass for just the
figure frames; the temporary eval_cyc folder is removed afterwards). Run from
repo root:
    python render/cache_cross_cyc_comparison.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import numpy as np  # noqa: E402

import render_ablation_frames_comparison as cmp_mod  # noqa: E402
import eval_CBD_cyc  # noqa: E402
from eval_CBD_cyc import Pipeline  # noqa: E402
from eval_CBD import Trainer  # noqa: E402

OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "cross_cyc"

COMA_MESH_IDX = 6  # FaceTalk_170809_00138_TA, as in render_ablation_frames_mask.py
BIWI_MESH_IDX = 1  # F2, as in render_ablation_frames_mask.py

# (key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, filter_dn, frames)
SCENARIOS = [
    ("mf_to_coma", "mf_ROM", cmp_mod.MF_TEST_IDX, "coma", COMA_MESH_IDX, 0, "mf_ROM", cmp_mod.MF_FRAMES),
    ("mf_to_biwi", "mf_ROM", cmp_mod.MF_TEST_IDX, "biwi", BIWI_MESH_IDX, 0, "mf_ROM", cmp_mod.MF_FRAMES),
    ("ict_to_coma_m00", "ict-cap", cmp_mod.ICT_ID_M00, "coma", COMA_MESH_IDX, cmp_mod.ICT_EXP_924, "ict-cap", cmp_mod.ICT_FRAMES_M00),
    ("ict_to_coma_m02", "ict-cap", cmp_mod.ICT_ID_M02, "coma", COMA_MESH_IDX, cmp_mod.ICT_EXP_922, "ict-cap", cmp_mod.ICT_FRAMES_M02),
    ("ict_to_biwi_m00", "ict-cap", cmp_mod.ICT_ID_M00, "biwi", BIWI_MESH_IDX, cmp_mod.ICT_EXP_924, "ict-cap", cmp_mod.ICT_FRAMES_M00),
    ("ict_to_biwi_m02", "ict-cap", cmp_mod.ICT_ID_M02, "biwi", BIWI_MESH_IDX, cmp_mod.ICT_EXP_922, "ict-cap", cmp_mod.ICT_FRAMES_M02),
]


EVAL_CYC_ROOT = REPO_ROOT / "eval_cyc"


def scenarios_for(ckpt_name):
    """coma/biwi scenarios, plus mf<->ict when the model has no eval_cyc results."""
    extra = [s for s in cmp_mod.SCENARIOS
             if not (EVAL_CYC_ROOT / ckpt_name / eval_cyc_name(*s[1:6])).exists()]
    return SCENARIOS + extra


def eval_cyc_name(src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num):
    if src_sel == "ict-cap":
        fn = f"ict-cap-ID_{src_mesh:03d}_test-to-{tgt_sel}-ID_{tgt_mesh:03d}_test_{exp_num:02d}"
    else:
        fn = f"{src_sel}-src{src_mesh:03d}_test-to-{tgt_sel}_test-ID_{tgt_mesh:03d}"
    return f"{fn}-masked-laplacian"


def done(out_dir, frames):
    return all((out_dir / f"frame_{i:06d}_{r}.npy").exists() for i in frames for r in ("cross", "cyclic"))


def main():
    for model_name, ckpt, version, override in cmp_mod.MODEL_SETS[cmp_mod.COMPARISON_SET]:
        print(f"\n========== {model_name} ({ckpt}) ==========")
        ckpt_name = Path(ckpt).name
        todo = [s for s in scenarios_for(ckpt_name) if not done(OUT_ROOT / ckpt_name / s[0], s[-1])]
        if not todo:
            print("  all cached")
            continue
        opts = cmp_mod.build_opts(ckpt, version, override)
        trainer = Trainer(opts)
        pipeline = Pipeline(opts)
        pipeline.model = trainer.model

        for key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, filter_dn, frames in todo:
            out_dir = OUT_ROOT / ckpt_name / key
            frames = [f for f in frames if not done(out_dir, [f])]  # only the missing frames
            print(f"  -- cross+cyclic {key}: {frames} --")
            eval_CBD_cyc.opts = pipeline.opts
            v_cross, v_cyc = cmp_mod.get_or_run(
                pipeline, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num,
                filter_dn, frames, need_cyc=True, ckpt=ckpt,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, frame_idx in enumerate(frames):
                np.save(out_dir / f"frame_{frame_idx:06d}_cross.npy", v_cross[k])
                np.save(out_dir / f"frame_{frame_idx:06d}_cyclic.npy", v_cyc[k])
            print(f"  [saved] {out_dir} ({len(frames)} frames)")

        del trainer, pipeline
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    print("\nALL DONE:", OUT_ROOT)


if __name__ == "__main__":
    main()
