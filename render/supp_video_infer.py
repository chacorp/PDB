"""
Supplementary video, stage 1: per-frame self / cross / cyclic outputs (plus the
source expression and the neutrals) for each video, cached as .npy under
vis_CBD/supp_video/verts/{video}/{scenario}/...

  design_study  : mf -> coma, coma -> biwi, the 8 design-study variants
  comparison    : mf -> ict, ict m00 -> mf, ict m02 -> mf; NC / NFR / NFS / PDB
                  (PDB = ckpts_CBD8/2026-09-16-02-19-03-NGBCv5)
  loss_ablation : mf -> biwi, the 4 mask_cage_consist variants

Frames are dataloader_CBD indices: mf = EXP_free_face test clip (3773..4963),
coma = 00138 cheeks_in / eyebrow / high_smile / mouth_extreme / mouth_open (205..738,
1254..1446, 1647..1821),
ict m00 = 20240325_MySlate_924 rows (0..1146), ict m02 = 20240318_MySlate_922 rows
(0..182). Existing full eval results (eval_CBD / eval_cyc *-masked-laplacian) are
reused; missing frames are inferred with the module's get_or_run (filtered pass).
Only frames missing from the cache are computed.

Options are env vars (eval_CBD_cyc.Options() parses sys.argv):
    SUPP_VIDEO=comparison [SUPP_FRAMES=4300,4301] python render/supp_video_infer.py
    SUPP_SCENARIOS=mf_to_ict   only these scenarios (comma separated)
    SUPP_MODELS=PDB            only these model columns (comma separated)
    SUPP_CMP_PDB_CKPT=./ckpts_CBD8/2026-10-08-22-31-39-NGBCv5
                               comparison PDB checkpoint (the column is still called PDB)
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
import torch  # noqa: E402

import render_ablation_frames_comparison as cmp_mod  # noqa: E402
import render_ablation_frames_design_study as ds_mod  # noqa: E402
import render_ablation_frames_mask as mask_mod  # noqa: E402
import eval_CBD_cyc  # noqa: E402
from eval_CBD_cyc import Pipeline  # noqa: E402
from eval_CBD import Trainer  # noqa: E402

OUT_ROOT = REPO_ROOT / "vis_CBD" / "supp_video" / "verts"
from supp_video_frames import VIDEO_FRAMES  # noqa: E402
# ds_mod indexes coma frames inside _CLIP_SUBSTR["coma"]; widening it to the whole 00138
# identity block (which starts the test list) makes those indices the dataloader indices
ds_mod._CLIP_SUBSTR["coma"] = "/FaceTalk_170809_00138_TA/vertices_npy/"

_DS_LABEL = {  # design-study column key -> ds_mod.MODELS label (render_design_study_figure.COLUMN_MODEL)
    "ours": "relu (chosen)", "no_activation": "no activation", "softplus": "softplus", "elu": "elu",
    "soft_pou": "soft POU", "wo_mask": "w/o mask", "eq15_delta": "delta (displacement)",
    "eq16_matrix": "matrix (transform)",
}
_MASK_LABEL = {"baseline": "baseline", "w_consist": "cage-consistency+mask",
               "w_cyclic": "cyclic+mask", "w_both": "cage+cyclic+mask"}


# design-study "ours" column for the video (instead of ds_mod's relu-chosen checkpoint)
DS_OURS_CKPT = "./ckpts_CBD/2025-10-23-17-32-49-NGBCv5"


def _ds_models():
    by = {n: (c, ov) for n, c, ov in ds_mod.MODELS}
    by[_DS_LABEL["ours"]] = (DS_OURS_CKPT, {})
    return [(k, by[l][0], lambda c=by[l][0], ov=by[l][1]: ds_mod.build_opts(c, ov)) for k, l in _DS_LABEL.items()]


def _mask_models():
    by = {n: (c, ov) for n, c, ov in mask_mod.MODELS}
    return [(k, by[l][0], lambda c=by[l][0], ov=by[l][1]: mask_mod.build_opts(c, ov)) for k, l in _MASK_LABEL.items()]


def _cmp_models():
    specs = [("NC", "./ckpts_CBD/2025-10-20-00-15-40-CBD", 1, {}),
             ("NFR", "./ckpt_stage1/exp_019_ICT_MF-jacob_NFR", 0, {"use_NFR": True}),
             ("NFS", "./ckpt_stage1/2024-08-18-23-32-29-all", 0, {}),
             ("PDB", os.environ.get("SUPP_CMP_PDB_CKPT", "./ckpts_CBD8/2026-09-16-02-19-03-NGBCv5"), 5, {})]
    return [(n, c, lambda c=c, v=v, ov=ov: cmp_mod.build_opts(c, v, ov)) for n, c, v, ov in specs]


VIDEOS = {
    "design_study": dict(mod=ds_mod, models=_ds_models, scenarios=["coma_to_biwi", "mf_to_coma"]),
    "comparison": dict(mod=cmp_mod, models=_cmp_models, scenarios=["mf_to_ict", "ict_to_mf_m00", "ict_to_mf_m02"]),
    "loss_ablation": dict(mod=mask_mod, models=_mask_models, scenarios=["mf_to_biwi"]),
}


def scenario_tuple(mod, key):
    return next(s for s in mod.SCENARIOS if s[0] == key)


def template(mod, data_name, id_num, exp_num):
    """identity neutral: pkl template for coma/biwi, dataset template ds[i][1] otherwise"""
    if data_name in ("coma", "biwi"):
        with open(mod._FACE_PKL[data_name], "rb") as f:
            mesh = pickle.load(f)
        ids = [k for k in mesh.keys() if k != "face"]
        return np.asarray(mesh[ids[id_num]])
    key = (data_name, id_num, exp_num)
    if key not in mod._raw_ds_cache:
        mod.get_raw_vertices(data_name, id_num, exp_num, 0)
    return mod._raw_ds_cache[key][0][1].numpy()


def eval_full_dir(ckpt, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, kind):
    """existing full eval result dir (or None): kind 'self' (eval_CBD) / 'cyc' (eval_cyc).
    Both the '-masked-laplacian' and the '-masked' naming are searched."""
    name = Path(ckpt).name
    if kind == "self":
        sub = {"mf_ROM": "mf_ROM", "coma": "coma"}.get(src_sel)
        bases = [REPO_ROOT / "eval_CBD" / f"{name}-eval" / sub] if sub else []
    elif src_sel == "ict-cap":
        bases = [REPO_ROOT / "eval_cyc" / name / f"ict-cap-ID_{src_mesh:03d}_test-to-{tgt_sel}-ID_{tgt_mesh:03d}_test_{exp_num:02d}"]
    elif tgt_sel == "ict-cap":
        bases = [REPO_ROOT / "eval_cyc" / name / f"{src_sel}-src{src_mesh:03d}_test-to-{tgt_sel}_test-ID_{tgt_mesh:03d}"]
    else:
        bases = [REPO_ROOT / "eval_cyc" / name / f"{src_sel}_test-to-{tgt_sel}_test"]
    for b in bases:
        for suffix in ("-masked-laplacian", "-masked"):
            d = b.with_name(b.name + suffix)
            if (d / "verts").exists() and any((d / "verts").iterdir()):
                return d
    return None


def save_refs(mod, sc, frames, out):
    key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, _fdn, _ = sc
    ref = out / "ref"
    (ref / "source_expression").mkdir(parents=True, exist_ok=True)
    if not (ref / "target_neutral.npy").exists():
        np.save(ref / "source_neutral.npy", template(mod, src_sel, src_mesh, exp_num))
        np.save(ref / "target_neutral.npy", template(mod, tgt_sel, tgt_mesh, exp_num))
        np.save(ref / "faces_src.npy", np.asarray(mod.FACES[src_sel]))
        np.save(ref / "faces_tgt.npy", np.asarray(mod.FACES[tgt_sel]))
    for f in frames:
        p = ref / "source_expression" / f"frame_{f:06d}.npy"
        if not p.exists():
            np.save(p, mod.get_raw_vertices(src_sel, src_mesh, exp_num, f))


def main():
    video = os.environ["SUPP_VIDEO"]
    only = [int(x) for x in os.environ.get("SUPP_FRAMES", "").split(",") if x]
    scenarios = [x for x in os.environ.get("SUPP_SCENARIOS", "").split(",") if x]
    models = [x for x in os.environ.get("SUPP_MODELS", "").split(",") if x]
    cfg = VIDEOS[video]
    mod = cfg["mod"]
    for sc_key in cfg["scenarios"]:
        if scenarios and sc_key not in scenarios:
            continue
        sc = scenario_tuple(mod, sc_key)
        key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, fdn, _ = sc
        frames = VIDEO_FRAMES[video][key]
        if only:
            frames = [f for f in frames if f in only]
        if not frames:
            continue
        out = OUT_ROOT / video / key
        print(f"\n===== {video} / {key}: {len(frames)} frames =====", flush=True)
        save_refs(mod, sc, frames, out)
        for name, ckpt, make_opts in cfg["models"]():
            if models and name not in models:
                continue
            print(f"  {name}: {ckpt}", flush=True)
            todo = {row: [f for f in frames if not (out / name / row / f"frame_{f:06d}.npy").exists()]
                    for row in ("self", "cross", "cyclic")}
            if not any(todo.values()):
                print(f"  {name}: cached", flush=True)
                continue
            pipeline = None
            for row in ("self", "cross", "cyclic"):
                (out / name / row).mkdir(parents=True, exist_ok=True)
            # self (src -> src), then cross + cyclic (src -> tgt -> src)
            for rows, tgt in ((("self",), (src_sel, src_mesh)), (("cross", "cyclic"), (tgt_sel, tgt_mesh))):
                fs = sorted(set(sum((todo[r] for r in rows), [])))
                if not fs:
                    continue
                kind = "self" if rows == ("self",) else "cyc"
                d = eval_full_dir(ckpt, src_sel, src_mesh, tgt[0], tgt[1], exp_num, kind)
                if d is not None and all((d / "verts" / f"{f:06d}.npy").exists() for f in fs):
                    print(f"  {name} {rows}: reuse {d.name}", flush=True)
                    vs = [np.load(d / "verts" / f"{f:06d}.npy") for f in fs]
                    vc = [np.load(d / "verts_cyc" / f"{f:06d}.npy") for f in fs] if kind == "cyc" else None
                else:
                    if pipeline is None:
                        opts = make_opts()
                        pipeline = Pipeline(opts)
                        pipeline.model = Trainer(opts).model
                    eval_CBD_cyc.opts = pipeline.opts
                    print(f"  {name} {rows}: infer {len(fs)} frames", flush=True)
                    vs, vc = mod.get_or_run(pipeline, src_sel, src_mesh, tgt[0], tgt[1], exp_num, fdn, fs,
                                            need_cyc=(kind == "cyc"), ckpt=ckpt)
                for i, f in enumerate(fs):
                    np.save(out / name / rows[0] / f"frame_{f:06d}.npy", vs[i])
                    if kind == "cyc":
                        np.save(out / name / "cyclic" / f"frame_{f:06d}.npy", vc[i])
            del pipeline
            torch.cuda.empty_cache()
    print("\nALL DONE:", OUT_ROOT / video)


if __name__ == "__main__":
    main()
