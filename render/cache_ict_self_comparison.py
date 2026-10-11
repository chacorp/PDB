"""Self-retargeting outputs for the ict-cap sources (ict_to_mf_m00 / m02), for the
comparison baselines NC / NFS / NFR / PDB, saved as one .npy per frame under
vis_CBD/vis_comparison/self_ict/<ckpt>/<scenario>/frame_XXXXXX.npy so that
render/render_comparison_figure.py can read them without re-running inference.

Uses render_ablation_frames_comparison.get_or_run (filtered pass for just the
figure frames). Run from repo root:
    python render/cache_ict_self_comparison.py
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

OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "self_ict"
# mf self for models that have no eval_CBD self results (e.g. the continue set)
MF_OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "self_mf"
EVAL_CBD_ROOT = REPO_ROOT / "eval_CBD"

# (scenario key, ict id, exp_num, frames)
ICT_SELF_SCENARIOS = [
    ("ict_to_mf_m00", cmp_mod.ICT_ID_M00, cmp_mod.ICT_EXP_924, cmp_mod.ICT_FRAMES_M00),
    ("ict_to_mf_m02", cmp_mod.ICT_ID_M02, cmp_mod.ICT_EXP_922, cmp_mod.ICT_FRAMES_M02),
]


def main():
    for model_name, ckpt, version, override in cmp_mod.MODEL_SETS[cmp_mod.COMPARISON_SET]:
        print(f"\n========== {model_name} ({ckpt}) ==========")
        opts = cmp_mod.build_opts(ckpt, version, override)
        trainer = Trainer(opts)
        pipeline = Pipeline(opts)
        pipeline.model = trainer.model
        ckpt_name = Path(ckpt).name

        if not (EVAL_CBD_ROOT / f"{ckpt_name}-eval" / "mf_ROM-masked-laplacian").exists():
            out_dir = MF_OUT_ROOT / ckpt_name
            frames = [f for f in cmp_mod.MF_FRAMES if not (out_dir / f"frame_{f:06d}.npy").exists()]
            if frames:
                print(f"  -- self mf: {frames} --")
                eval_CBD_cyc.opts = pipeline.opts
                v_self, _ = cmp_mod.get_or_run(
                    pipeline, "mf_ROM", cmp_mod.MF_TEST_IDX, "mf_ROM", cmp_mod.MF_TEST_IDX, 0,
                    "mf_ROM", frames, need_cyc=False, ckpt=ckpt,
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                for k, frame_idx in enumerate(frames):
                    np.save(out_dir / f"frame_{frame_idx:06d}.npy", v_self[k])
                print(f"  [saved] {out_dir} ({len(frames)} frames)")

        for key, ict_id, exp_num, frames in ICT_SELF_SCENARIOS:
            out_dir = OUT_ROOT / ckpt_name / key
            frames = [f for f in frames if not (out_dir / f"frame_{f:06d}.npy").exists()]
            if not frames:
                print(f"  -- self {key}: all cached --")
                continue
            print(f"  -- self {key}: {frames} --")
            eval_CBD_cyc.opts = pipeline.opts
            v_self, _ = cmp_mod.get_or_run(
                pipeline, "ict-cap", ict_id, "ict-cap", ict_id, exp_num,
                "ict-cap", frames, need_cyc=False, ckpt=ckpt,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, frame_idx in enumerate(frames):
                np.save(out_dir / f"frame_{frame_idx:06d}.npy", v_self[k])
            print(f"  [saved] {out_dir} ({len(frames)} frames)")

    print("\nALL DONE:", OUT_ROOT)


if __name__ == "__main__":
    main()
