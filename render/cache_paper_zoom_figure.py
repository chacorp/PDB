"""Inference cache for the paper zoom-in comparison figures (Fig.13 / Fig.14 style).

  fig13 (cross): mf_ROM ID_012, frame 7623 -> high-res BIWI ID_012 (M5, 23,370 verts,
                 test-mesh/biwi_templates.pkl + utils/biwi/align.npy, as in vis_CBD.py /
                 render_mitsuba.py). eval_CBD_cyc.Pipeline.get_mesh loads the decimated
                 BIWI, so its biwi branch is patched here. The eval_cyc folder name does
                 not tell high-res from decimated, so fig13 never reuses an existing folder.
  fig14 (self):  ict-cap ID_009 / 924 motion / frame 843 (eyes closed) and
                 ict-cap ID_004 / 924 motion / frame 1117 (mouth open).

Models: NFR / NFS (render_ablation_frames_comparison.MODELS) + PDB = ckpts_CBD/2026-04-27-15-36-49-NGBCv5.
Output: vis_CBD/vis_comparison/paper_fig/cache/<ckpt>/<key>.npy

Run from repo root:
    python render/cache_paper_zoom_figure.py
"""
import os
import sys
import shutil
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

import render_ablation_frames_comparison as cmp_mod  # noqa: E402
import eval_CBD_cyc  # noqa: E402
from eval_CBD_cyc import Pipeline  # noqa: E402
from eval_CBD import Trainer  # noqa: E402

CACHE_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "paper_fig" / "cache"
# (name, ckpt_path, version, opts_override)
# NFR: public NFR checkpoint experiments/ICT_augment_cnn_ext_dfn4_grad/ICT_augment_cnn_ext_dfn4_grad_0.pth,
#      loaded by evaluation.Trainer (opts.NFR; opts.ckpt only names the cache dir). eval_CBD_cyc's opts.NFR
#      branch calls NFS-only methods (get_local_feature / encode_seg / encode_id) on NFR_helper, so
#      NFR_helper.inference (encode src expression -> calc_new_mesh on the target neutral) is called directly.
#      Its output is mean-centered (calc_new_mesh); it is cached as is.
# NFS: 2024-08-18-23-32-29-all (cmp_mod.MODELS) gave an almost static BIWI result -> 2024-07-08-06-27-12-all
NFR_CKPT = "./experiments/ICT_augment_cnn_ext_dfn4_grad"
NFS_CKPT = "./ckpt_stage1/2024-07-08-06-27-12-all"
PDB_CKPT = "./ckpts_CBD/2026-04-27-15-36-49-NGBCv5"
MODELS = ([("NFR", NFR_CKPT, 0, {"use_NFR": True, "NFR": True})]
          + [("NFS", NFS_CKPT, 0, {})]
          + [("PDB", PDB_CKPT, 5, {"align_latent": True})])  # yaml align_latent is overwritten by the argparse default

MF_ID = 12
MF_FRAME = 7623
BIWI_ID = 12  # M5, the BIWI identity used in the mitsuba renders
ICT_EXP_924 = cmp_mod.ICT_EXP_924
# (key, ict id, exp_num, frame)
FIG14 = [
    ("fig14_id009_924_f000843", 9, ICT_EXP_924, 843),
    ("fig14_id004_924_f001117", 4, ICT_EXP_924, 1117),
]
FIG13_KEY = f"fig13_mf{MF_ID:03d}_f{MF_FRAME:06d}_to_biwiHR{BIWI_ID:03d}"


def load_biwi_hr(select_mesh):
    """High-res BIWI template (aligned) + faces, same as vis_CBD.py get_mesh."""
    tri = trimesh.load(str(REPO_ROOT / "test-mesh" / "BIWI.ply"), process=False)
    with open(REPO_ROOT / "test-mesh" / "biwi_templates.pkl", "rb") as f:
        mesh = pickle.load(f)
    ids = [k for k in mesh if k != "face"]
    v = np.asarray(mesh[ids[select_mesh]])
    m_align = np.load(REPO_ROOT / "utils" / "biwi" / "align.npy")
    v = np.concatenate((v, np.ones((v.shape[0], 1))), axis=1) @ m_align.T
    return v, np.asarray(tri.faces), ids[select_mesh]


_orig_get_mesh = Pipeline.get_mesh


# eval_CBD_cyc decides self-retargeting by SRC_SELECT_MESH == TGT_SELECT_mesh only (dataset
# ignored), so mf ID_012 -> biwi ID_012 would take the self branch for NGBC (version 5).
# The biwi target index is passed with this offset and stripped again in get_mesh; for a biwi
# target the index is used nowhere else (tgt_dataset ignores it, the log dir name omits it).
BIWI_IDX_OFFSET = 100


def _patched_get_mesh(self, selection, dataset, SELECT_MESH):
    if selection == "biwi":
        return load_biwi_hr(SELECT_MESH % BIWI_IDX_OFFSET)
    return _orig_get_mesh(self, selection, dataset, SELECT_MESH)


Pipeline.get_mesh = _patched_get_mesh


def run_fresh(pipeline, ckpt, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, filter_dn, idxs):
    """Like cmp_mod.get_or_run but never reuses an existing eval_cyc folder."""
    d = cmp_mod.expected_dir(ckpt, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num)
    swapped = None
    if d.exists():
        swapped = d.with_name(d.name + "__swap_tmp")
        shutil.move(str(d), str(swapped))
    try:
        cmp_mod.set_filter(filter_dn, idxs)
        pipeline.save_vis_data(cmp_mod.DATA_IDX[src_sel], src_mesh,
                               cmp_mod.DATA_IDX[tgt_sel], tgt_mesh, exp_num)
        cmp_mod.clear_filter()
        assert d.exists(), f"expected output not created: {d}"
        v = [np.load(d / "verts" / f"{i:06d}.npy") for i in range(len(idxs))]
        shutil.rmtree(d)
    finally:
        cmp_mod.clear_filter()
        if d.exists():  # partial output of a failed run; drop it so the restore is not nested
            shutil.rmtree(d)
        if swapped is not None:
            shutil.move(str(swapped), str(d))
            print(f"  [restored] {d}")
    return v


def nfr_public_inference(nfr_helper, src_expr, src_neu, src_f, tgt_neu, tgt_f):
    """Public NFR retarget of one source frame onto the target neutral (process=False keeps vertex order)."""
    import torch
    src_m = trimesh.Trimesh(vertices=np.asarray(src_neu), faces=np.asarray(src_f), process=False)
    tgt_m = trimesh.Trimesh(vertices=np.asarray(tgt_neu), faces=np.asarray(tgt_f), process=False)
    out = nfr_helper.inference(torch.from_numpy(np.asarray(src_expr)).float()[None], src_m, tgt_m)
    out = out[0].detach().cpu().numpy()
    assert out.shape[0] == len(tgt_neu), f"vertex count changed: {out.shape} vs {len(tgt_neu)}"
    return out


def run_nfr_public(opts, out_dir):
    import dataloader_CBD
    from evaluation import Trainer as NFRTrainer
    from utils.remesh_utils import ICT_face_model
    nfr = NFRTrainer(opts).model  # NFR_helper with the public checkpoint
    nfr.model.eval()

    if not (out_dir / f"{FIG13_KEY}.npy").exists():
        print(f"  -- {FIG13_KEY} --")
        with open("/data/sihun/pca/multiface_align/mf_templates.pkl", "rb") as f:
            mf_f = np.asarray(pickle.load(f)["face"])
        item = dataloader_CBD.EvalDataset(data_name="mf_ROM", toggle=False)[MF_FRAME]
        biwi_v, biwi_f, _ = load_biwi_hr(BIWI_ID)
        v = nfr_public_inference(nfr, item[0].numpy(), item[1].numpy(), mf_f, biwi_v, biwi_f)
        np.save(out_dir / f"{FIG13_KEY}.npy", v)
        print(f"  [saved] {FIG13_KEY} {v.shape}")

    ict_f = np.asarray(ICT_face_model().faces)
    for key, ict_id, exp_num, frame in FIG14:
        if (out_dir / f"{key}.npy").exists():
            continue
        print(f"  -- {key} --")
        item = dataloader_CBD.EvalDataset(data_name="ict-cap", toggle=False,
                                          ict_cap_id_num=ict_id, ict_cap_exp_num=exp_num)[frame]
        neu = item[1].numpy()
        v = nfr_public_inference(nfr, item[0].numpy(), neu, ict_f, neu, ict_f)
        np.save(out_dir / f"{key}.npy", v)
        print(f"  [saved] {key} {v.shape}")


def main():
    for model_name, ckpt, version, override in MODELS:
        ckpt_name = Path(ckpt).name
        out_dir = CACHE_ROOT / ckpt_name
        out_dir.mkdir(parents=True, exist_ok=True)
        keys = [FIG13_KEY] + [k for k, *_ in FIG14]
        if all((out_dir / f"{k}.npy").exists() for k in keys):
            print(f"[{model_name}] all cached")
            continue
        print(f"\n========== {model_name} ({ckpt}) ==========")
        opts = cmp_mod.build_opts(ckpt, version, override)
        if override.get("NFR"):
            run_nfr_public(opts, out_dir)
            continue
        trainer = Trainer(opts)
        pipeline = Pipeline(opts)
        pipeline.model = trainer.model
        eval_CBD_cyc.opts = pipeline.opts

        if not (out_dir / f"{FIG13_KEY}.npy").exists():
            print(f"  -- {FIG13_KEY} --")
            v = run_fresh(pipeline, ckpt, "mf_ROM", MF_ID, "biwi", BIWI_ID + BIWI_IDX_OFFSET, 0,
                          "mf_ROM", [MF_FRAME])
            assert v[0].shape[0] == 23370, f"expected high-res BIWI output, got {v[0].shape}"
            np.save(out_dir / f"{FIG13_KEY}.npy", v[0])
            print(f"  [saved] {FIG13_KEY} {v[0].shape}")

        for key, ict_id, exp_num, frame in FIG14:
            if (out_dir / f"{key}.npy").exists():
                continue
            print(f"  -- {key} --")
            eval_CBD_cyc.opts = pipeline.opts
            v, _ = cmp_mod.get_or_run(pipeline, "ict-cap", ict_id, "ict-cap", ict_id, exp_num,
                                      "ict-cap", [frame], need_cyc=False, ckpt=ckpt)
            np.save(out_dir / f"{key}.npy", v[0])
            print(f"  [saved] {key} {v[0].shape}")

        del trainer, pipeline
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    print("\nALL DONE:", CACHE_ROOT)


if __name__ == "__main__":
    main()
