"""
Self / cross / cyclic retargeting videos for the 4 use_data1 ablation checkpoints
(baseline / cage-consistency / cyclic / cage+cyclic), on biwi / coma / multiface(mf_ROM).

Run from the render/ directory:
    cd render && python render_retarget_4model.py

For each (model, dataset) we pick 1 test identity (fixed) and either:
  - 2 random whole expression clips (biwi, coma), or
  - the single EXP_free_face clip (multiface, only one available)
and render:
  - self-retargeting   (source identity == target identity)
  - cross-retargeting  (source expression -> fixed ict-cap ID_000 identity)
  - cyclic retargeting (cross-retargeting result retargeted back to the source identity)
"""
import os
import sys
import shutil
import random
from pathlib import Path

# --- import render_trimesh with cwd=render/ so its module-level abs_path/sys.path setup is correct ---
RENDER_DIR = Path(__file__).resolve().parent
REPO_ROOT = RENDER_DIR.parent
_orig_cwd = os.getcwd()
os.chdir(RENDER_DIR)
import render_trimesh  # noqa: E402
os.chdir(_orig_cwd)

# --- now switch to repo root: eval_CBD_cyc.py / eval_CBD.py use repo-root-relative paths ---
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import torch.multiprocessing as mp  # noqa: E402
mp.set_start_method("spawn", force=True)

import dataloader_CBD  # noqa: E402
import eval_CBD_cyc  # noqa: E402
from eval_CBD_cyc import Pipeline, Options  # noqa: E402
from eval_CBD import Trainer  # noqa: E402

# save_vis_data hardcodes num_workers=4/persistent_workers=True, which deadlocks with our
# small single-clip datasets under spawn start method. Force single-process loading instead.
import torch.utils.data as _tud  # noqa: E402
_orig_dataloader_init = _tud.DataLoader.__init__


def _patched_dataloader_init(self, *args, **kwargs):
    kwargs["num_workers"] = 0
    kwargs.pop("persistent_workers", None)
    _orig_dataloader_init(self, *args, **kwargs)


_tud.DataLoader.__init__ = _patched_dataloader_init
import yaml
import argparse
import pickle
import numpy as np
import tempfile
from glob import glob as _glob
from subprocess import call as _call

VIDEO_OUT = REPO_ROOT / "video" / "retarget_4model"

# render_trimesh.py's own get_mesh() is stale for our datasets (wrong/missing template paths,
# and for biwi it mixes a decimated vertex template with the full-resolution BIWI.ply face
# topology, which crashes). Fetch faces directly from the same template pkls the model itself
# uses (via Pipeline.get_mesh), so topology always matches the predicted vertices exactly.
_FACE_PKL = {
    "biwi": "/data/sihun/pca/BIWI_align_deci/templates_align_deci.pkl",
    "coma": "/data/sihun/pca/VOCA-COMA/voca_templates.pkl",
    "mf_ROM": "/data/sihun/pca/multiface_align/mf_templates.pkl",
}


def _faces_for(selection):
    if selection == "ict-cap":
        return render_trimesh.ICT_face_model().faces
    with open(_FACE_PKL[selection], "rb") as f:
        mesh = pickle.load(f)
    return mesh["face"]


def render_verts_video(verts_dir, faces, out_dir, name, fps=30):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_files = sorted(_glob(str(verts_dir) + "/*.npy"))
    if not frame_files:
        print(f"  [warn] no frames in {verts_dir}, skipping")
        return
    tmp_vert = np.load(frame_files[0])
    center = np.mean(tmp_vert, axis=0)
    H, W = 800, 800
    camera_params = {
        "c": np.array([H // 2, W // 2]),
        "k": np.array([-0.19816071, 0.92822711, 0, 0, 0]),
        "f": np.array([4754.97941935 / 2, 4754.97941935 / 2]),
    }
    mp4_path = out_dir / f"{name}.mp4"
    tmp_video = tempfile.NamedTemporaryFile("w", suffix=".mp4", dir=str(out_dir))
    writer = render_trimesh.cv2.VideoWriter(
        tmp_video.name, render_trimesh.cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H), True
    )
    for f in render_trimesh.tqdm(frame_files, desc=name):
        v = np.load(f)
        mesh = render_trimesh.trimesh.Trimesh(vertices=v * 0.5, faces=faces)
        img = render_trimesh.render_mesh_helper(mesh, center, camera_params, z_offset=1.3, H=H, W=W)
        writer.write(img.astype(np.uint8))
    writer.release()
    _call(f"ffmpeg -y -i {tmp_video.name} -pix_fmt yuv420p -qscale 0 {mp4_path}".split())
    print(f"  [saved] {mp4_path}")

MODELS = [
    ("baseline", "./ckpts_CBD7/2026-08-03-09-09-59-NGBCv5"),
    ("cage-consistency", "./ckpts_CBD7/2026-08-02-19-04-17-NGBCv5"),
    ("cyclic", "./ckpts_CBD_cyclic/2026-08-04-08-48-34-NGBCv5"),
    ("cage+cyclic", "./ckpts_CBD7/2026-08-05-00-55-47-NGBCv5"),
]

# data_name_list index used by Pipeline.save_vis_data: ['voca','biwi','mf_SEN','coma','mf_ROM','ict','ict-cap']
DATA_IDX = {"biwi": 1, "coma": 3, "mf_ROM": 4, "ict-cap": 6}
TGT_ICT_CAP_ID = 0  # fixed cross-retargeting target identity

BIWI_ID_NAME, BIWI_MESH_IDX = "F2", 1
COMA_ID_NAME, COMA_MESH_IDX = "FaceTalk_170809_00138_TA", 6
MF_ID_NAME, MF_MESH_IDX = "m--20190828--1318--002645310--GHS", 12

BIWI_CLIPS = ["F2_e37", "F2_e38", "F2_e39", "F2_e40"]
COMA_CLIPS = [
    "bareteeth", "cheeks_in", "eyebrow", "high_smile", "lips_back", "lips_up",
    "mouth_down", "mouth_extreme", "mouth_middle", "mouth_open", "mouth_side", "mouth_up",
]

random.seed(42)
SELECTED_CLIPS = {
    "biwi": random.sample(BIWI_CLIPS, 2),
    "coma": random.sample(COMA_CLIPS, 2),
    "mf_ROM": ["EXP_free_face"],
}
print("selected clips:", SELECTED_CLIPS)

# ---------------------------------------------------------------------------
# monkeypatch EvalDataset so the driving/source dataset is restricted to one clip
# ---------------------------------------------------------------------------
_ATTR_BY_DATA = {"biwi": "biwi_datalist", "coma": "coma_datalist", "mf_ROM": "mf_ROM_datalist"}
_orig_eval_ds_init = dataloader_CBD.EvalDataset.__init__
_active_filter = {"data_name": None, "substr": None}


def _patched_init(self, *args, **kwargs):
    _orig_eval_ds_init(self, *args, **kwargs)
    data_name = kwargs.get("data_name")
    if data_name is not None and data_name == _active_filter["data_name"]:
        attr = _ATTR_BY_DATA[data_name]
        full_list = getattr(self, attr)
        filtered = [p for p in full_list if _active_filter["substr"] in p]
        assert len(filtered) > 0, f"clip filter matched 0 frames: {_active_filter}"
        setattr(self, attr, filtered)
        self.len = len(filtered)
        print(f"  [clip filter] {data_name}: {len(full_list)} -> {len(filtered)} frames "
              f"(substr={_active_filter['substr']!r})")


dataloader_CBD.EvalDataset.__init__ = _patched_init


def set_clip_filter(data_name, substr):
    _active_filter["data_name"] = data_name
    _active_filter["substr"] = substr


def clear_clip_filter():
    _active_filter["data_name"] = None
    _active_filter["substr"] = None


# ---------------------------------------------------------------------------
# replicate Pipeline.save_vis_data's file_name / log_dir derivation so we can
# pre-clean and then rename the (non-clip-specific) output folders per clip
# ---------------------------------------------------------------------------
def expected_file_name(src_selection, src_mesh, tgt_selection, tgt_mesh, exp_num):
    if src_selection == "ict-cap":
        return f"{src_selection}-ID_{src_mesh:03d}_test-to-{tgt_selection}-ID_{tgt_mesh:03d}_test_{exp_num:02d}"
    if "ict" in tgt_selection:
        return f"{src_selection}-src{src_mesh:03d}_test-to-{tgt_selection}_test-ID_{tgt_mesh:03d}"
    if "mf" in tgt_selection:
        if tgt_mesh == 12:
            return f"{src_selection}_test-to-{tgt_selection}_test"
        elif tgt_mesh == 11:
            return f"{src_selection}_test-to-{tgt_selection}_val"
        return f"{src_selection}_test-to-{tgt_selection}_train"
    return f"{src_selection}_test-to-{tgt_selection}_test"


def run_one(pipeline, ckpt, src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num, clip_id, tag):
    """Run save_vis_data once, then rename its output folder to include the clip id.
    Returns the renamed absolute folder path (containing verts/ [and verts_cyc/])."""
    ckpt_name = ckpt.split("/")[-1]
    base_name = expected_file_name(src_sel, src_mesh, tgt_sel, tgt_mesh, exp_num) + "-masked"  # use_t_mask forced True
    default_dir = REPO_ROOT / "eval_cyc" / ckpt_name / base_name
    final_dir = REPO_ROOT / "eval_cyc" / ckpt_name / f"{base_name}--{clip_id}--{tag}"

    if final_dir.exists():
        print(f"  [skip, already rendered-data present] {final_dir}")
        return final_dir

    if default_dir.exists():
        shutil.rmtree(default_dir)

    print(f"  [run] src={src_sel}#{src_mesh} tgt={tgt_sel}#{tgt_mesh} clip={clip_id} tag={tag}")
    pipeline.save_vis_data(DATA_IDX[src_sel], src_mesh, DATA_IDX[tgt_sel], tgt_mesh, exp_num)

    assert default_dir.exists(), f"expected output dir not created: {default_dir}"
    shutil.move(str(default_dir), str(final_dir))
    return final_dir


def build_opts(ckpt):
    class Opts:
        pass
    opts = Options()
    opts.ckpt = ckpt
    opts.version = 5
    config = f"{opts.ckpt}/train_opts.yml"
    opts_yaml = yaml.load(open(config), Loader=yaml.FullLoader)
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
    opts.use_t_mask = True
    opts.continue_ckpt = False
    opts.align_latent = True  # architecture flag: must be explicit (see earlier eval_CBD.py bug)
    # batch_size=1: self-retargeting caches key_weight from the first batch and reuses it
    # without re-checking batch size, so a smaller final partial batch (clip length not a
    # multiple of batch_size) crashes with a broadcast mismatch. 1 sidesteps that entirely.
    opts.batch_size = 1
    eval_CBD_cyc.opts = opts  # save_vis_data references the bare module-level `opts` name (pre-existing bug)
    return opts


def main():
    id_setup = {
        "biwi": (BIWI_ID_NAME, BIWI_MESH_IDX),
        "coma": (COMA_ID_NAME, COMA_MESH_IDX),
        "mf_ROM": (MF_ID_NAME, MF_MESH_IDX),
    }

    generated = []  # (ckpt_name, dataset, clip_id, tag, folder)

    for model_name, ckpt in MODELS:
        print(f"\n========== loading model: {model_name} ({ckpt}) ==========")
        opts = build_opts(ckpt)
        trainer = Trainer(opts)
        pipeline = Pipeline(opts)
        pipeline.model = trainer.model
        ckpt_name = ckpt.split("/")[-1]

        for dataset, (id_name, mesh_idx) in id_setup.items():
            for clip_id in SELECTED_CLIPS[dataset]:
                if dataset == "biwi":
                    substr = f"/vertices_npy/{clip_id}/"
                elif dataset == "coma":
                    substr = f"/{id_name}/vertices_npy/{clip_id}/"
                else:  # mf_ROM
                    substr = f"/vertices_npy/{id_name}/{clip_id}/"

                set_clip_filter(dataset, substr)

                # self-retargeting
                d_self = run_one(pipeline, ckpt, dataset, mesh_idx, dataset, mesh_idx, 0, clip_id, "self")
                generated.append((ckpt_name, dataset, clip_id, "self", d_self / "verts"))

                # cross + cyclic (single pass)
                d_cross = run_one(pipeline, ckpt, dataset, mesh_idx, "ict-cap", TGT_ICT_CAP_ID, 0, clip_id, "cross")
                generated.append((ckpt_name, dataset, clip_id, "cross", d_cross / "verts"))
                generated.append((ckpt_name, dataset, clip_id, "cyclic", d_cross / "verts_cyc"))

                clear_clip_filter()

        del trainer, pipeline
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n========== inference done, {len(generated)} sequences. now rendering videos ==========")

    dataset_faces = {"biwi": _faces_for("biwi"), "coma": _faces_for("coma"), "mf_ROM": _faces_for("mf_ROM")}
    ict_faces = _faces_for("ict-cap")

    for ckpt_name, dataset, clip_id, tag, verts_dir in generated:
        faces = ict_faces if tag == "cross" else dataset_faces[dataset]
        out_dir = VIDEO_OUT / ckpt_name / dataset / tag
        print(f"[render] {ckpt_name}/{dataset}/{clip_id}/{tag} -> {out_dir}")
        render_verts_video(verts_dir, faces, out_dir, clip_id)

    print("\nALL RENDERING DONE")
    print("output root:", VIDEO_OUT)


if __name__ == "__main__":
    main()
