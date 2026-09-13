"""
Displacement-from-neutral heatmap gallery for the design_study ablation
checkpoints -- companion to render_ablation_frames_design_study.py, same 8
models / 9 scenarios / frames / identity meshes, but instead of shaded
renders, colors each of the 4 panels (original/self/cross/cyclic) by
per-vertex displacement magnitude from that mesh's own neutral template:

  - original, self, cyclic  -> diff against the SOURCE identity's neutral
    (all three live in source-identity space: the raw GT frame, the model's
    self-retarget, and the model's cross-retarget cycled back to source).
  - cross                   -> diff against the TARGET identity's neutral
    (the cross-retargeted output lives in target-identity space).

Color scale is a single FIXED (vmin=0, vmax=<global 99th percentile
displacement>) range shared across ALL 8 models/9 scenarios/4 frames/4 panels,
so redder = more displaced is directly comparable model-to-model -- unlike
utils.matplotlib_rnd.plot_image_array_diff's default per-panel auto-normalize.

Two-pass: pass 1 loads/reuses every (model, scenario, frame) mesh (via the
same eval_cyc/ reuse-or-filtered-run machinery as
render_ablation_frames_design_study.py) and the two neutral templates, and
keeps them in memory (some ~200MB of float arrays for the full grid -- cheap)
while computing the global displacement percentile; pass 2 renders every panel
with that fixed scale.

Run from repo root (after render_ablation_frames_design_study.py, so its
eval_cyc/ full-dataset outputs for mf_to_ict etc. are cached and reused here
too):
    python render/render_ablation_frames_design_study_diff.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

# Importing this module runs its top-level setup (chdir to repo root, monkey-
# patches for filtered/single-process dataset construction) but not main(),
# and gives us its MODELS/SCENARIOS/frame/identity constants, build_opts, and
# the eval_cyc/ reuse-or-filtered-run helpers, so none of that is duplicated.
import render_ablation_frames_design_study as base  # noqa: E402
from eval_CBD import Trainer  # noqa: E402
from utils.matplotlib_rnd import plot_image_array_diff  # noqa: E402

REPO_ROOT = base.REPO_ROOT
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_design_study_diff"
DISPLACEMENT_PERCENTILE = 99.0  # robust global vmax: ignore the top 1% of
# per-vertex displacements so one extreme outlier (e.g. a diverged model on
# one scenario) doesn't wash out the color scale for everyone else.


def neutral_for(pipeline, selection, mesh_idx):
    v, _f, _mesh_id = pipeline.get_mesh(selection, None, mesh_idx)
    return v


def collect_scenario(pipeline, ckpt, scenario):
    key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, filter_dn, idxs = scenario
    f_self_cyc, f_cross = base.FACES[src_sel], base.FACES[tgt_sel]

    print(f" -- self ({key}) --")
    v_self, _ = base.get_or_run(pipeline, src_sel, src_mesh, src_sel, src_mesh, exp_num,
                                 filter_dn, idxs, need_cyc=False, ckpt=ckpt)

    print(f" -- cross+cyclic ({key}) --")
    v_cross, v_cyclic = base.get_or_run(pipeline, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num,
                                         filter_dn, idxs, need_cyc=True, ckpt=ckpt)

    neutral_src = neutral_for(pipeline, src_sel, src_mesh)
    neutral_tgt = neutral_for(pipeline, tgt_sel, tgt_mesh)

    frames = []
    for k, frame_idx in enumerate(idxs):
        v_orig = base.get_raw_vertices(src_sel, src_mesh, exp_num, frame_idx)
        frames.append({
            "frame_idx": frame_idx,
            "v": [v_orig, v_self[k], v_cross[k], v_cyclic[k]],
            "f": [f_self_cyc, f_self_cyc, f_cross, f_self_cyc],
            "d": [neutral_src, neutral_src, neutral_tgt, neutral_src],
        })
    return frames


def displacement_mag(v, f, d):
    """Per-face-vertex displacement magnitude, matching
    plot_image_array_diff's own diff computation (abs(D-V) indexed by faces,
    then norm over xyz, then norm over the 3 face corners) so the percentile
    we compute lines up with what actually gets colored."""
    diff = np.abs(d - v)[f]           # [num_faces, 3, 3]
    diff = np.linalg.norm(diff, axis=1)  # [num_faces, 3]
    diff = np.linalg.norm(diff, axis=1)  # [num_faces]
    return diff


def main():
    all_data = {}  # (ckpt, key) -> list of per-frame dicts
    all_diffs = []

    for model_name, ckpt, override in base.MODELS:
        print(f"\n========== [collect] {model_name} ({ckpt}) ==========")
        opts = base.build_opts(ckpt, override)
        trainer = Trainer(opts)
        pipeline = base.Pipeline(opts)
        pipeline.model = trainer.model

        for scenario in base.SCENARIOS:
            key = scenario[0]
            frames = collect_scenario(pipeline, ckpt, scenario)
            all_data[(ckpt, key)] = frames
            for fr in frames:
                for v, f, d in zip(fr["v"], fr["f"], fr["d"]):
                    all_diffs.append(displacement_mag(v, f, d))

        del trainer, pipeline
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    pooled = np.concatenate(all_diffs)
    vmax = float(np.percentile(pooled, DISPLACEMENT_PERCENTILE))
    print(f"\nglobal displacement stats: min={pooled.min():.5f} max={pooled.max():.5f} "
          f"p{DISPLACEMENT_PERCENTILE}={vmax:.5f} (used as fixed vmax for all panels)")

    for model_name, ckpt, override in base.MODELS:
        print(f"\n========== [render] {model_name} ({ckpt}) ==========")
        for scenario in base.SCENARIOS:
            key = scenario[0]
            out_dir = OUT_ROOT / Path(ckpt).name / key
            out_dir.mkdir(parents=True, exist_ok=True)
            for fr in all_data[(ckpt, key)]:
                name = f"frame_{fr['frame_idx']:06d}"
                plot_image_array_diff(
                    fr["v"], fr["f"], fr["d"],
                    rot_list=[[0, 0, 0]] * 5,  # plot_image_array_diff indexes rot_list[idx+1]
                    size=8, bg_black=False, draw_base=False,
                    vmin=0.0, vmax=vmax,
                    logdir=str(out_dir), name=name, save=True,
                )
                print(f"  [saved] {out_dir / (name + '.png')}")

    (OUT_ROOT / "_vmax.txt").write_text(
        f"min={pooled.min():.6f} max={pooled.max():.6f} "
        f"p{DISPLACEMENT_PERCENTILE}={vmax:.6f}\n"
    )
    print("\nALL DONE:", OUT_ROOT)


if __name__ == "__main__":
    main()
