"""
Original / self / cross / cyclic retargeting frame images for the
"comparison" baselines (NC / NFS / NFR / PDB), mf<->ict scenarios only
(coma/biwi scenarios to be added later once their eval data exists).

Copied from render_ablation_frames_design_study.py (8-model use_data2
ablation) with MODELS/SCENARIOS/OUT_ROOT changed, and build_opts() extended
to handle version==0 (NFS/NFR, which load a fixed config/train_NFS.yml base
instead of the checkpoint's own train_opts.yml, plus extra opts eval_CBD.py's
own __main__ sets for that path) and version==1 (NC, no extra opts needed)
in addition to the version==5 (NGBC) path the other design_study/mask
galleries already handled.

Three scenarios (SCENARIOS below), matching design_study/mask's own subset:
  - mf_to_ict:      mf_ROM (full test set, absolute frame index) -> ict-cap m00
  - ict_to_mf_m00:  ict-cap m00, 20240325_MySlate_924 (its own motion) -> mf_ROM
  - ict_to_mf_m02:  ict-cap m02, 20240318_MySlate_922 (its own motion) -> mf_ROM

For each (model, scenario, frame): renders one image with 4 panels
[original, self, cross, cyclic] using utils.matplotlib_rnd.plot_image_array
(size=8, mode='shade', bg_black=False, M_SCALE=0.6, no crop -- whole mesh,
frontal). "original" is the raw (non-model) ground-truth mesh for that frame.

Reuses already-computed full eval_cyc/<ckpt_name>/... results where available
(get_or_run checks disk dynamically per model/scenario) instead of re-running
inference; only runs a filtered (4-frame) save_vis_data pass for what's
missing. Existing eval_cyc/ folders are never modified in place -- any name
collision is swapped aside and restored after the filtered run.

Run from repo root:
    python render/render_ablation_frames_comparison.py
"""
import os
import sys
import shutil
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import torch.multiprocessing as mp  # noqa: E402
mp.set_start_method("spawn", force=True)

# save_vis_data hardcodes num_workers=4/persistent_workers=True, which deadlocks
# with our tiny filtered (4-frame) datasets under spawn. Force single-process.
import torch.utils.data as _tud  # noqa: E402
_orig_dl_init = _tud.DataLoader.__init__


def _patched_dl_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs.pop("persistent_workers", None)
    _orig_dl_init(self, *args, **kwargs)


_tud.DataLoader.__init__ = _patched_dl_init

import yaml  # noqa: E402
import argparse  # noqa: E402
import numpy as np  # noqa: E402

import dataloader_CBD  # noqa: E402
import eval_CBD_cyc  # noqa: E402
from eval_CBD_cyc import Pipeline, Options  # noqa: E402
from eval_CBD import Trainer  # noqa: E402
from utils.matplotlib_rnd import plot_image_array  # noqa: E402
from utils.remesh_utils import ICT_face_model  # noqa: E402

# (name, ckpt_path, version, opts_override)
# version 1 = NC, 0 = NFS/NFR (distinguished by override's use_NFR), 5 = NGBC (PDB)
MODELS = [
    ("NC",  "./ckpts_CBD/2025-10-20-00-15-40-CBD",          1, {}),
    ("NFS", "./ckpt_stage1/2024-08-18-23-32-29-all",         0, {}),
    ("NFR", "./ckpt_stage1/exp_019_ICT_MF-jacob_NFR",        0, {"use_NFR": True}),
    ("PDB", "./ckpts_CBD8/2026-09-16-02-19-03-NGBCv5",       5, {}),
]

MF_TEST_IDX = 12
ICT_ID_M00 = 0
ICT_ID_M02 = 2
ICT_EXP_922 = 0  # dataloader_CBD.py: exp_num==0 -> 20240318_MySlate_922 (m02's own motion)
ICT_EXP_924 = 1  # exp_num!=0 -> 20240325_MySlate_924 (m00's own motion)
# frames shared by vis_comparison.html / vis_comparison_stylize_retarget.html / continue_figure.html
from comparison_frames import MF_FRAMES, ICT_FRAMES_M00, ICT_FRAMES_M02  # noqa: E402

# (key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, filter_dn, frames)
SCENARIOS = [
    ("mf_to_ict", "mf_ROM", MF_TEST_IDX, "ict-cap", ICT_ID_M00, 0, "mf_ROM", MF_FRAMES),
    ("ict_to_mf_m00", "ict-cap", ICT_ID_M00, "mf_ROM", MF_TEST_IDX, ICT_EXP_924, "ict-cap", ICT_FRAMES_M00),
    ("ict_to_mf_m02", "ict-cap", ICT_ID_M02, "mf_ROM", MF_TEST_IDX, ICT_EXP_922, "ict-cap", ICT_FRAMES_M02),
]

DATA_IDX = {"mf_ROM": 4, "ict-cap": 6, "biwi": 1, "coma": 3}
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison"

_FACE_PKL = {
    "mf_ROM": "/data/sihun/pca/multiface_align/mf_templates.pkl",
    "biwi": "/data/sihun/pca/BIWI_align_deci/templates_align_deci.pkl",
    "coma": "/data/sihun/pca/VOCA-COMA/voca_templates.pkl",
}


def faces_for(selection):
    if selection == "ict-cap":
        return ICT_face_model().faces
    with open(_FACE_PKL[selection], "rb") as f:
        return pickle.load(f)["face"]


FACES = {sel: faces_for(sel) for sel in ["mf_ROM", "ict-cap"]}

# clip-substring restriction, applied before the index filter, for datasets with
# multiple identities/clips per identity (biwi, coma) so "index i" means "the i-th
# frame of this one clip" rather than an arbitrary offset into the whole test set.
_CLIP_SUBSTR = {}
_DATALIST_ATTR = {"mf_ROM": "mf_ROM_datalist", "biwi": "biwi_datalist", "coma": "coma_datalist"}

# ---------------------------------------------------------------------------
# frame filter: restricts a dataset construction matching the active data_name to
# an exact list of frames (by absolute index for mf_ROM/ict-cap identity-frame
# range, or by clip-relative index for biwi/coma after a clip-substring filter).
# Any OTHER construction (e.g. a template-only tgt_dataset built without frame
# iteration) is left untouched via the length guard.
# ---------------------------------------------------------------------------
_orig_eval_ds_init = dataloader_CBD.EvalDataset.__init__
_active_filter = {"data_name": None, "indices": None}


def _patched_init(self, *args, **kwargs):
    _orig_eval_ds_init(self, *args, **kwargs)
    data_name = kwargs.get("data_name")
    if data_name is None or data_name != _active_filter["data_name"]:
        return
    idxs = _active_filter["indices"]

    if data_name in _DATALIST_ATTR:
        attr = _DATALIST_ATTR[data_name]
        full = getattr(self, attr)
        if data_name in _CLIP_SUBSTR:
            substr = _CLIP_SUBSTR[data_name]
            full = [p for p in full if substr in p]
        if len(full) > 0 and max(idxs) < len(full):
            setattr(self, attr, [full[i] for i in idxs])
            self.len = len(idxs)
            print(f"  [index filter] {data_name}: kept {len(idxs)} frames {idxs} "
                  f"(clip-filtered pool size {len(full)})")
        else:
            print(f"  [index filter] {data_name}: skipped (pool size {len(full)} too small "
                  f"for max({idxs})) -- template-only construction, not the one being iterated")
    elif data_name == "ict-cap":
        if max(idxs) < len(self.expression_vecs):
            self.expression_vecs = self.expression_vecs[idxs]
            self.len = len(idxs)
            print(f"  [index filter] {data_name}: kept {len(idxs)} frames {idxs}")
        else:
            print(f"  [index filter] {data_name}: skipped (this construction's array is "
                  f"too short, {len(self.expression_vecs)} < max({idxs})) -- "
                  f"template-only construction, not the one being iterated")


dataloader_CBD.EvalDataset.__init__ = _patched_init


def set_filter(data_name, indices):
    _active_filter["data_name"] = data_name
    _active_filter["indices"] = indices


def clear_filter():
    _active_filter["data_name"] = None
    _active_filter["indices"] = None


def build_opts(ckpt, version, override):
    opts = Options()
    opts.ckpt = ckpt
    opts.version = version

    if version == 0:
        # NFS/NFR: fixed base config (not the checkpoint's own train_opts.yml),
        # plus the extra opts eval_CBD.py's/eval_CBD_cyc.py's own __main__ sets
        # for this path -- replicated here since render scripts bypass that
        # __main__ entirely and call build_opts()+Trainer()/Pipeline() directly.
        opts_yaml = yaml.load(open("config/train_NFS.yml"), Loader=yaml.FullLoader)
        opts_ = vars(opts)
        opts_yaml.update(opts_)
        opts = argparse.Namespace(**opts_yaml)
        opts.img_feat_dim = 128
        opts.feature_type = "cents&norms"
        opts.stage1 = True
        opts.scale_exp = 1.0
        opts.ict_face_only = False
        use_nfr = override.get("use_NFR", False)
        opts.use_NFR = use_nfr
        if use_nfr:
            opts.design = "nfr"
            opts.dec_type = "jacob"
        else:
            opts.design = "new2"
            opts.dec_type = "disp"
    else:
        opts_yaml = yaml.load(open(f"{ckpt}/train_opts.yml"), Loader=yaml.FullLoader)
        opts_ = vars(opts)
        opts_yaml.update(opts_)
        opts = argparse.Namespace(**opts_yaml)

    opts.use_t_mask = True
    opts.continue_ckpt = False
    opts.batch_size = 1
    for k, v in override.items():
        if k == "use_NFR":
            continue  # already applied above for version==0
        setattr(opts, k, v)
    eval_CBD_cyc.opts = opts  # save_vis_data references the bare module-level name
    return opts


def expected_dir(ckpt, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num):
    ckpt_name = Path(ckpt).name
    if src_sel == "ict-cap":
        fn = f"{src_sel}-ID_{src_mesh:03d}_test-to-{tgt_sel}-ID_{tgt_mesh:03d}_test_{exp_num:02d}"
    elif "ict" in tgt_sel:
        fn = f"{src_sel}-src{src_mesh:03d}_test-to-{tgt_sel}_test-ID_{tgt_mesh:03d}"
    else:
        fn = f"{src_sel}_test-to-{tgt_sel}_test"
    return REPO_ROOT / "eval_cyc" / ckpt_name / f"{fn}-masked"


def get_or_run(pipeline, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num,
                filter_dn, idxs, need_cyc, ckpt):
    d = expected_dir(ckpt, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num)
    verts_dir = d / "verts"

    if verts_dir.exists():
        n = len(list(verts_dir.glob("*.npy")))
        if n >= max(idxs) + 1:
            print(f"  [reuse] {d} (n={n})")
            v = [np.load(verts_dir / f"{i:06d}.npy") for i in idxs]
            vc = None
            if need_cyc:
                vc_dir = d / "verts_cyc"
                vc = [np.load(vc_dir / f"{i:06d}.npy") for i in idxs]
            return v, vc

    # not cached (or not enough frames) -- run a filtered pass just for these
    # exact frame indices. If a differently-shaped result already sits at the
    # expected path (e.g. a stale full run under an older naming quirk), swap
    # it aside so save_vis_data starts clean, then restore it afterward.
    swapped = None
    if d.exists():
        swapped = d.with_name(d.name + "__swap_tmp")
        shutil.move(str(d), str(swapped))

    set_filter(filter_dn, idxs)
    print(f"  [run, filtered] src={src_sel}#{src_mesh} tgt={tgt_sel}#{tgt_mesh} exp={exp_num}")
    pipeline.save_vis_data(
        DATA_IDX[src_sel], src_mesh, DATA_IDX[tgt_sel], tgt_mesh, exp_num,
    )
    clear_filter()

    assert d.exists(), f"expected output not created: {d}"
    v = [np.load(d / "verts" / f"{i:06d}.npy") for i in range(len(idxs))]
    vc = [np.load(d / "verts_cyc" / f"{i:06d}.npy") for i in range(len(idxs))] if need_cyc else None

    shutil.rmtree(d)
    if swapped is not None:
        shutil.move(str(swapped), str(d))
        print(f"  [restored] {swapped} -> {d}")

    return v, vc


_raw_ds_cache = {}


def get_raw_vertices(data_name, id_num, exp_num, frame_idx):
    key = (data_name, id_num, exp_num)
    if key not in _raw_ds_cache:
        if data_name == "mf_ROM":
            ds = dataloader_CBD.EvalDataset(data_name=data_name, toggle=False)
        else:
            ds = dataloader_CBD.EvalDataset(
                data_name="ict-cap", toggle=False,
                ict_cap_id_num=id_num, ict_cap_exp_num=exp_num,
            )
        _raw_ds_cache[key] = ds
    ds = _raw_ds_cache[key]
    return ds[frame_idx][0].numpy()


def render_frame(v_orig, f_orig, v_self, f_self, v_cross, f_cross, v_cyclic, f_cyclic, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    M_SCALE = 0.6
    v_list = [v_orig * M_SCALE, v_self * M_SCALE, v_cross * M_SCALE, v_cyclic * M_SCALE]
    f_list = [f_orig, f_self, f_cross, f_cyclic]
    plot_image_array(
        v_list, f_list,
        rot_list=[[0, 0, 0]] * 4, size=8, mode="shade",
        bg_black=False, logdir=str(out_dir), name=name, save=True,
    )
    print(f"  [saved] {out_dir / (name + '.png')}")


def run_scenario(pipeline, ckpt, model_name, scenario):
    key, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, filter_dn, idxs = scenario
    f_self_cyc, f_cross = FACES[src_sel], FACES[tgt_sel]

    print(f" -- self ({key}) --")
    v_self, _ = get_or_run(pipeline, src_sel, src_mesh, src_sel, src_mesh, exp_num,
                            filter_dn, idxs, need_cyc=False, ckpt=ckpt)

    print(f" -- cross+cyclic ({key}) --")
    v_cross, v_cyclic = get_or_run(pipeline, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num,
                                    filter_dn, idxs, need_cyc=True, ckpt=ckpt)

    out_dir = OUT_ROOT / Path(ckpt).name / key
    for k, frame_idx in enumerate(idxs):
        v_orig = get_raw_vertices(src_sel, src_mesh, exp_num, frame_idx)
        render_frame(
            v_orig, f_self_cyc,
            v_self[k], f_self_cyc,
            v_cross[k], f_cross,
            v_cyclic[k], f_self_cyc,
            out_dir, f"frame_{frame_idx:06d}",
        )


def main():
    for model_name, ckpt, version, override in MODELS:
        print(f"\n========== {model_name} ({ckpt}) ==========")
        try:
            opts = build_opts(ckpt, version, override)
            trainer = Trainer(opts)
            pipeline = Pipeline(opts)
            pipeline.model = trainer.model

            for scenario in SCENARIOS:
                try:
                    run_scenario(pipeline, ckpt, model_name, scenario)
                except Exception:
                    import traceback
                    print(f"!! FAILED scenario {scenario[0]} for {model_name}")
                    traceback.print_exc()

            del trainer, pipeline
        except Exception:
            import traceback
            print(f"!! FAILED to init {model_name}")
            traceback.print_exc()

        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    print("\nALL DONE:", OUT_ROOT)


if __name__ == "__main__":
    main()
