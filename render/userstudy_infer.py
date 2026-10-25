"""
User study (userstudy-pdb), stage 1: per-frame self / cross / cyclic outputs of
NC / NFR / NFS / PDB for the study segments, cached as .npy under
vis_CBD/userstudy/verts/{source}/...

    {source}/ref/source_neutral.npy, faces_src.npy, source_expression/ -> supp_video
    {source}/self/{model}/                         -> supp_video (dataset PDB)
    {source}/{target}/ref/target_neutral.npy, faces_tgt.npy
    {source}/{target}/{model}/{cross|cyclic}/frame_XXXXXX.npy

Sources / segments (dataloader_CBD indices, userstudy-pdb/PLAN.md 2.1):
  mf      = mf_ROM test id 12, EXP_free_face: 3875..4174, 4226..4525, 4577..4876
  ict_m00 = ict-cap m00, 20240325_MySlate_924: 90..389, 480..779
  ict_m02 = ict-cap m02, 20240318_MySlate_922: 0..182
Targets and how their outputs are obtained:
  ict / mf (the other dataset)  : links to vis_CBD/supp_video/verts/comparison (no inference)
  coma / biwi                   : render_ablation_frames_comparison.get_or_run, as the
                                  coma / biwi cells of vis_comparison-2.html
  aura / piers / face-reference / head-reference
                                : render_stylized_retarget_comparison.run_model, as
                                  vis_comparison_stylize_retarget.html
PDB = ckpts_CBD8/2026-09-16-02-19-03-NGBCv5 for every target (stylized: as in
      vis_comparison_stylize_retarget.html).
PDB-TTT (step 'ttt', stylized targets only, for the supplementary video): that PDB after
      test-time training on the target (render_stylized_retarget_comparison.TTT_RUNS,
      model_000.pth), saved under {source}/{target}/PDB-TTT/.
Only frames missing from the cache are computed.

NFS / NFR spend most of a call on the neutral meshes' DiffusionNet operators
(nfr_utils.get_dfn_info) and mesh operators (get_mesh_operators). Both are wrapped
to save / load them as utils/data_prepare.py does ({key}_dfn_info.pkl,
{key}_operators.pkl), keyed by a hash of the mesh, under vis_CBD/userstudy/precomputes.
The 'precompute' step fills that cache by running every (source, target) once
through the real NFS / NFR inference paths (1 frame), before the parallel runs.

Options are env vars (eval_CBD_cyc.Options() parses sys.argv):
    US_STEPS=precompute,link,dataset,stylized,ttt  steps to run (default: link,dataset,stylized)
    US_MODELS=NFR,NFS               models (default: all)
    US_SOURCES=mf,ict_m00           sources (default: all)
    US_TARGETS=coma,aura            inferred targets (default: all)
    US_MAX_FRAMES=10                only the first N frames per source (timing test)
    US_OUT=<dir>                    output root (default: vis_CBD/userstudy/verts)
    python render/userstudy_infer.py
"""
import os
import sys
import time
import pickle
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import render_ablation_frames_comparison as cmp_mod  # noqa: E402
import cache_cross_cyc_comparison as cc_mod  # noqa: E402
import render_stylized_retarget_comparison as sty_mod  # noqa: E402
import eval_CBD_cyc  # noqa: E402
import utils.nfr_utils as nfr_utils  # noqa: E402
import utils.mesh_utils as mesh_utils  # noqa: E402
from eval_CBD_cyc import Pipeline  # noqa: E402
from eval_CBD import Trainer  # noqa: E402

OUT_ROOT = Path(os.environ.get("US_OUT", REPO_ROOT / "vis_CBD" / "userstudy" / "verts"))
PRE_ROOT = REPO_ROOT / "vis_CBD" / "userstudy" / "precomputes"
WARMUP_ROOT = REPO_ROOT / "_tmp" / "userstudy_warmup"
SUPP_ROOT = REPO_ROOT / "vis_CBD" / "supp_video" / "verts" / "comparison"
CHUNK = 100  # frames per inference call (resume granularity)

SEGMENTS = {
    "mf": [(3875, 4174), (4226, 4525), (4577, 4876)],
    "ict_m00": [(90, 389), (480, 779)],
    "ict_m02": [(0, 182)],
}
# source -> (supp_video scenario, the other dataset as target, cache_cross_cyc scenario prefix)
SOURCES = {
    "mf": ("mf_to_ict", "ict", "mf_to"),
    "ict_m00": ("ict_to_mf_m00", "mf", "ict_to"),
    "ict_m02": ("ict_to_mf_m02", "mf", "ict_to"),
}
DATASET_TARGETS = ["coma", "biwi"]
STYLIZED_TARGETS = ["aura", "piers", "face-reference", "head-reference"]
MODEL_NAMES = ["NC", "NFR", "NFS", "PDB"]

# (name, ckpt, version, cmp_mod override)
DATASET_MODELS = [
    ("NC", "./ckpts_CBD/2025-10-20-00-15-40-CBD", 1, {}),
    ("NFR", "./ckpt_stage1/exp_019_ICT_MF-jacob_NFR", 0, {"use_NFR": True}),
    ("NFS", "./ckpt_stage1/2024-08-18-23-32-29-all", 0, {}),
    ("PDB", "./ckpts_CBD8/2026-09-16-02-19-03-NGBCv5", 5, {}),
]
STYLIZED_MODELS = [  # same PDB as vis_comparison_stylize_retarget.html
    ("NC", "./ckpts_CBD/2025-10-20-00-15-40-CBD", 1, {}),
    ("NFR", "./ckpt_stage1/exp_019_ICT_MF-jacob_NFR", 0, {"use_NFR": True}),
    ("NFS", "./ckpt_stage1/2024-08-18-23-32-29-all", 0, {}),
    ("PDB", "./ckpts_CBD8/2026-09-16-02-19-03-NGBCv5", 5, {}),
]


def env_list(name, default):
    v = os.environ.get(name, "")
    return [x for x in v.split(",") if x] or list(default)


# ---------------------------------------------------------------------------
# precomputed neutral-mesh operators (utils/data_prepare.py file format)
# ---------------------------------------------------------------------------
def mesh_key(mesh):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def atomic_pickle(obj, path):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _to_cpu(x):
    return x.cpu() if torch.is_tensor(x) else x


def install_precompute_hooks():
    PRE_ROOT.mkdir(parents=True, exist_ok=True)
    orig_dfn, orig_ops = nfr_utils.get_dfn_info, mesh_utils.get_mesh_operators

    def get_dfn_info(mesh, cache_dir=None, map_location="cuda"):
        p = PRE_ROOT / f"{mesh_key(mesh)}_dfn_info.pkl"
        if p.exists():
            with open(p, "rb") as f:
                info = pickle.load(f)
        else:
            info = [_to_cpu(x) for x in orig_dfn(mesh, cache_dir=cache_dir, map_location="cpu")]
            atomic_pickle(info, p)
            print(f"[precompute] saved {p.name} (V={len(mesh.vertices)})", flush=True)
        return [x.to(map_location) if torch.is_tensor(x) else x for x in info]

    def get_mesh_operators(mesh, *args, **kwargs):
        p = PRE_ROOT / f"{mesh_key(mesh)}_operators.pkl"
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        ops = orig_ops(mesh, *args, **kwargs)
        atomic_pickle(ops, p)
        print(f"[precompute] saved {p.name} (V={len(mesh.vertices)})", flush=True)
        return ops

    # eval_CBD_cyc reaches both through the modules at call time; run_model calls
    # nfr_utils.get_dfn_info and imports get_mesh_operators from utils.mesh_utils
    nfr_utils.get_dfn_info = get_dfn_info
    mesh_utils.get_mesh_operators = get_mesh_operators


def install_synced_deformation_gradient():
    """NFR's jacob decoder hands tensors between PyTorch and CuPy (DLPack) without a
    sync; under parallel runs that race crashed with cudaErrorIllegalAddress, and the
    original retry cannot recover (the CUDA context is already broken). Always sync
    at both handoffs (same math as deformation_transfer.deformation_gradient.forward)."""
    import cupy
    import utils.deformation_transfer as dt

    def forward(ctx, input, solver, idxs, vals, shape):
        batch_size = input.shape[0]
        input = input.reshape(input.shape[0], -1, 3)
        input = input.transpose(1, 0)
        input = input.reshape(input.shape[0], -1)
        ctx.solver = solver
        ctx.idxs = idxs
        ctx.vals = vals
        ctx.shape = shape
        ctx.set_materialize_grads(False)
        b = dt.spmm(idxs, vals, m=shape[0], n=shape[1], matrix=input)
        torch.cuda.synchronize()
        cupy_output = ctx.solver.solve(cupy.from_dlpack(dt.to_dlpack(b)))
        cupy.cuda.get_current_stream().synchronize()
        output = torch.from_dlpack(cupy_output)
        output = output.reshape(-1, batch_size, 3)
        return output.transpose(0, 1)

    dt.deformation_gradient.forward = staticmethod(forward)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def source_frames(source, max_frames=0):
    frames = [f for lo, hi in SEGMENTS[source] for f in range(lo, hi + 1)]
    return frames[:max_frames] if max_frames else frames


def missing(out, frames):
    return [f for f in frames
            if not all((out / row / f"frame_{f:06d}.npy").exists() for row in ("cross", "cyclic"))]


def save(out, frames, v_cross, v_cyc):
    for row in ("cross", "cyclic"):
        (out / row).mkdir(parents=True, exist_ok=True)
    for f, vc, vy in zip(frames, v_cross, v_cyc):
        np.save(out / "cross" / f"frame_{f:06d}.npy", vc)
        np.save(out / "cyclic" / f"frame_{f:06d}.npy", vy)


def save_ref(ref, tv, tf):
    """target neutral / faces, written atomically (parallel runs share them)"""
    if (ref / "target_neutral.npy").exists() and (ref / "faces_tgt.npy").exists():
        return
    ref.mkdir(parents=True, exist_ok=True)
    for name, arr in (("faces_tgt.npy", tf), ("target_neutral.npy", tv)):
        tmp = ref / f"{name}.{os.getpid()}.tmp.npy"
        np.save(tmp, np.asarray(arr))
        os.replace(tmp, ref / name)


def link(dst, src):
    if dst.is_symlink() or dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    except FileExistsError:  # created by a parallel run
        pass


def step_link(root, sources):
    """supp_video results: source refs, self rows, and the other dataset as target."""
    for s in sources:
        scen, other, _ = SOURCES[s]
        sup = SUPP_ROOT / scen
        for name in ("source_neutral.npy", "faces_src.npy", "source_expression"):
            link(root / s / "ref" / name, sup / "ref" / name)
        link(root / s / other / "ref" / "target_neutral.npy", sup / "ref" / "target_neutral.npy")
        link(root / s / other / "ref" / "faces_tgt.npy", sup / "ref" / "faces_tgt.npy")
        for m in MODEL_NAMES:
            link(root / s / "self" / m, sup / m / "self")
            for row in ("cross", "cyclic"):
                link(root / s / other / m / row, sup / m / row)
        print(f"[link] {root} {s}: refs, self, {other}", flush=True)


def cc_scenario(source, target):
    prefix = SOURCES[source][2]
    suffix = "" if source == "mf" else "_" + source.split("_")[1]
    key = f"{prefix}_{target}{suffix}"
    return next(sc for sc in cc_mod.SCENARIOS if sc[0] == key)


def cc_template(sc):
    """coma / biwi identity neutral from the template pkl (as supp_video_infer.template)."""
    with open(cmp_mod._FACE_PKL[sc[3]], "rb") as f:
        mesh = pickle.load(f)
    ids = [k for k in mesh.keys() if k != "face"]
    return np.asarray(mesh[ids[sc[4]]])


# ---------------------------------------------------------------------------
# inference steps
# ---------------------------------------------------------------------------
def step_dataset(root, models, sources, targets, max_frames=0):
    """coma / biwi targets: filtered eval_CBD_cyc pass (get_or_run), CHUNK frames per call."""
    targets = [t for t in targets if t in DATASET_TARGETS]
    if not targets:
        return
    for name, ckpt, version, override in DATASET_MODELS:
        if name not in models:
            continue
        jobs = []
        for s in sources:
            frames = source_frames(s, max_frames)
            for t in targets:
                sc = cc_scenario(s, t)
                save_ref(root / s / t / "ref", cc_template(sc), cmp_mod.faces_for(sc[3]))
                out = root / s / t / name
                todo = missing(out, frames)
                if todo:
                    jobs.append((s, t, sc, out, todo))
        if not jobs:
            print(f"[dataset] {name}: cached", flush=True)
            continue
        opts = cmp_mod.build_opts(ckpt, version, override)
        pipeline = Pipeline(opts)
        pipeline.model = Trainer(opts).model
        for s, t, sc, out, todo in jobs:
            key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, fdn, _ = sc
            for i in range(0, len(todo), CHUNK):
                fs = todo[i:i + CHUNK]
                t0 = time.time()
                eval_CBD_cyc.opts = pipeline.opts
                v_cross, v_cyc = cmp_mod.get_or_run(pipeline, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num,
                                                    fdn, fs, need_cyc=True, ckpt=ckpt)
                save(out, fs, v_cross, v_cyc)
                dt = time.time() - t0
                print(f"[dataset] {name} {s}->{t}: {len(fs)} frames {dt:.1f}s ({dt / len(fs):.2f} s/frame)",
                      flush=True)
        del pipeline
        torch.cuda.empty_cache()


def step_stylized(root, models, sources, targets, max_frames=0):
    """stylized targets: run_model cross (src -> tgt) and cyclic (cross -> src)."""
    targets = [t for t in targets if t in STYLIZED_TARGETS]
    if not targets:
        return
    all_tgts = sty_mod.load_targets()
    cache = np.load(sty_mod.CACHE_PATH)
    for name, ckpt, version, _ in STYLIZED_MODELS:
        if name not in models:
            continue
        jobs = []
        for s in sources:
            frames = source_frames(s, max_frames)
            for t in targets:
                tv, tf = all_tgts[t]
                save_ref(root / s / t / "ref", tv, tf)
                out = root / s / t / name
                todo = missing(out, frames)
                if todo:
                    jobs.append((s, t, out, todo))
        if not jobs:
            print(f"[stylized] {name}: cached", flush=True)
            continue
        opts = sty_mod.build_opts(ckpt, version, use_nfr=name in sty_mod.USE_NFR)
        run_stylized_jobs(name, opts, version, jobs, root, all_tgts, cache)


def run_stylized_jobs(name, opts, version, jobs, root, all_tgts, cache):
    """jobs [(source, target, out dir, frames)]: cross / cyclic with one model"""
    trainer = Trainer(opts)
    pipe = sty_mod.Pipeline(opts)  # vis_CBD_retarget_fig.Pipeline, as the original script
    pipe.model = trainer.model
    for s, t, out, todo in jobs:
        sv, sf = cache[f"src/{s}/v"], cache[f"src/{s}/f"]
        tv, tf = all_tgts[t]
        expr = root / s / "ref" / "source_expression"
        for i in range(0, len(todo), CHUNK):
            fs = todo[i:i + CHUNK]
            t0 = time.time()
            src_frames = [np.load(expr / f"frame_{f:06d}.npy") for f in fs]
            v_cross = sty_mod.run_model(pipe, version, sv, sf, src_frames, tv, tf)
            v_cyc = sty_mod.run_model(pipe, version, tv, tf, v_cross, sv, sf)
            save(out, fs, v_cross, v_cyc)
            dt = time.time() - t0
            print(f"[stylized] {name} {s}->{t}: {len(fs)} frames {dt:.1f}s ({dt / len(fs):.2f} s/frame)",
                  flush=True)
    del trainer, pipe
    torch.cuda.empty_cache()


def step_ttt(root, sources, targets, max_frames=0):
    """stylized targets with PDB after test-time training on that target (one checkpoint
    per target, model_000.pth, as PDB-TTT-<target> of ttt_figure.html), saved as 'PDB-TTT'"""
    targets = [t for t in targets if t in STYLIZED_TARGETS]
    if not targets:
        return
    all_tgts = sty_mod.load_targets()
    cache = np.load(sty_mod.CACHE_PATH)
    for t in targets:
        tv, tf = all_tgts[t]
        jobs = []
        for s in sources:
            save_ref(root / s / t / "ref", tv, tf)
            out = root / s / t / "PDB-TTT"
            todo = missing(out, source_frames(s, max_frames))
            if todo:
                jobs.append((s, t, out, todo))
        if not jobs:
            print(f"[ttt] PDB-TTT {t}: cached", flush=True)
            continue
        name = f"PDB-TTT-{t}"
        opts = sty_mod.build_opts(f"./{sty_mod.TTT_RUNS[t]}", 5, sty_mod.MODEL_EPOCH[name])
        run_stylized_jobs(name, opts, 5, jobs, root, all_tgts, cache)


def main():
    steps = env_list("US_STEPS", ["link", "dataset", "stylized"])
    models = env_list("US_MODELS", MODEL_NAMES)
    sources = env_list("US_SOURCES", SOURCES)
    targets = env_list("US_TARGETS", DATASET_TARGETS + STYLIZED_TARGETS)
    max_frames = int(os.environ.get("US_MAX_FRAMES", 0))
    print(f"out={OUT_ROOT} steps={steps} models={models} sources={sources} targets={targets}", flush=True)
    install_precompute_hooks()
    install_synced_deformation_gradient()
    if "precompute" in steps:
        # every (source, target) once through the NFS / NFR paths (1 frame, outputs discarded)
        step_link(WARMUP_ROOT, sources)
        pre_models = [m for m in ("NFS", "NFR") if m in models]
        step_dataset(WARMUP_ROOT, pre_models, sources, targets, max_frames=1)
        step_stylized(WARMUP_ROOT, pre_models, sources, targets, max_frames=1)
        print("[precompute] files:", len(list(PRE_ROOT.glob("*.pkl"))), flush=True)
    # dataset / stylized steps read the source expression through the links
    step_link(OUT_ROOT, sources)
    if "dataset" in steps:
        step_dataset(OUT_ROOT, models, sources, targets, max_frames)
    if "stylized" in steps:
        step_stylized(OUT_ROOT, models, sources, targets, max_frames)
    if "ttt" in steps:
        step_ttt(OUT_ROOT, sources, targets, max_frames)
    print("ALL DONE:", OUT_ROOT, flush=True)


if __name__ == "__main__":
    main()
