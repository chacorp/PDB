"""
Train-set counterpart of render_ablation_frames.py: original / self / cross /
cyclic retargeting frame images for the 4 use_data1 ablation checkpoints,
using TRAIN-split identities for mf_ROM, coma, biwi, ict.

Unlike the test set, none of these datasets have literal per-frame train
files in the format eval_CBD_cyc.py's EvalDataset reads -- per
dataloader_CBD.py's CBDDataset (the actual training dataloader, confirmed via
train_opts.yml: data_toggle=true, use_data1=true for all 4 checkpoints):
  - mf_ROM / coma / biwi train samples are generated on the fly from a
    per-identity PCA model (/data/sihun/pca/<dataset>/train/<id>_pca.npz),
    via PCA_holder.sample_from_pca(scale=1.5) -- there is no fixed "frame index"
    for train, so we fix a numpy seed before each sample_from_pca() call for
    reproducibility across reruns.
  - ict train samples come from ict_face_pt/random_identity_vecs.npy (111 synth
    identities) + random_expression_vecs.npy (2360 synth expression coeffs),
    decoded via ICT_face_model.apply_coeffs -- fully deterministic by index.

Because these are not EvalDataset-compatible, self/cross/cyclic retargeting is
implemented directly here via model.predict_coordinate / model.retarget_animation
(mirroring eval_CBD_cyc.py's version==5 self-/cross-retargeting branches
exactly), instead of reusing save_vis_data.

Run from repo root:
    python render/render_ablation_frames_train.py
"""
import os
import sys
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402
import argparse  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import igl  # noqa: E402

from eval_CBD import Trainer  # noqa: E402
from utils.exp_utils import PCA_holder  # noqa: E402
from utils.remesh_utils import ICT_face_model, calc_norm_torch  # noqa: E402
from utils.matplotlib_rnd import plot_image_array  # noqa: E402

MODELS = [
    ("baseline", "./ckpts_CBD7/2026-08-03-09-09-59-NGBCv5"),
    ("cage-consistency", "./ckpts_CBD7/2026-08-02-19-04-17-NGBCv5"),
    ("cyclic", "./ckpts_CBD_cyclic/2026-08-04-08-48-34-NGBCv5"),
    ("cage+cyclic", "./ckpts_CBD7/2026-08-05-00-55-47-NGBCv5"),
]

PCA_BASE = "/data/sihun/pca"
TRAIN_SCALE = 1.5  # matches CBDDataset: self.scale = 1.5 when is_train=True

MF_TRAIN_ID = "m--20171024--0000--002757580--GHS"
MF_TRAIN_SEEDS = [18, 0, 3, 11]
COMA_TRAIN_ID = "FaceTalk_170728_03272_TA"
COMA_TRAIN_SEEDS = [0, 11, 3, 7]
BIWI_TRAIN_ID = "F2"
BIWI_TRAIN_SEEDS = [0, 1, 7, 3]
ICT_TRAIN_ID_IDX = 10
ICT_TRAIN_EXP_IDX = [433, 689, 1224, 1788]

OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_ablation_train"

with open(f"{PCA_BASE}/multiface_align/mf_templates.pkl", "rb") as f:
    MF_MESH = pickle.load(f)
with open(f"{PCA_BASE}/VOCA-COMA/voca_templates.pkl", "rb") as f:
    COMA_MESH = pickle.load(f)
with open(f"{PCA_BASE}/BIWI_align_deci/templates_align_deci.pkl", "rb") as f:
    BIWI_MESH = pickle.load(f)

_MESH_BY_DATASET = {"mf_ROM": MF_MESH, "coma": COMA_MESH, "biwi": BIWI_MESH}
_PCA_NPZ_FMT = {
    "mf_ROM": f"{PCA_BASE}/multiface_align/ROM/train/{{id}}_pca.npz",
    "coma": f"{PCA_BASE}/VOCA-COMA/COMA/train/{{id}}_pca.npz",
    "biwi": f"{PCA_BASE}/BIWI_align_deci/train/{{id}}_pca.npz",
}

_pca_holder_cache = {}


def get_pca_holder(dataset, id_name):
    key = (dataset, id_name)
    if key not in _pca_holder_cache:
        _pca_holder_cache[key] = PCA_holder(_PCA_NPZ_FMT[dataset].format(id=id_name))
    return _pca_holder_cache[key]


def decode_pca_frame(dataset, id_name, seed):
    """Returns (template, deformed, faces) as float32 np arrays -- one
    reproducible PCA sample (fixed numpy seed) for the given train identity."""
    mesh = _MESH_BY_DATASET[dataset]
    template, faces = mesh[id_name].astype(np.float32), mesh["face"]
    holder = get_pca_holder(dataset, id_name)
    np.random.seed(seed)
    deformed = holder.sample_from_pca(scale=TRAIN_SCALE).astype(np.float32)
    return template, deformed, faces


ICT = ICT_face_model()
ID_VECS = np.load(f"{REPO_ROOT}/ict_face_pt/random_identity_vecs.npy")[:111]
EXP_VECS = np.load(f"{REPO_ROOT}/ict_face_pt/random_expression_vecs.npy")


def decode_ict_frame(id_idx, exp_idx):
    id_coeff = ID_VECS[id_idx]
    exp_coeff = EXP_VECS[exp_idx]
    vertices, template, _ = ICT.apply_coeffs(id_coeff, exp_coeff, return_all=True)
    return template[0].astype(np.float32), vertices[0].astype(np.float32), ICT.faces


DECODERS = {
    "mf_ROM": lambda i: decode_pca_frame("mf_ROM", MF_TRAIN_ID, MF_TRAIN_SEEDS[i]),
    "coma": lambda i: decode_pca_frame("coma", COMA_TRAIN_ID, COMA_TRAIN_SEEDS[i]),
    "biwi": lambda i: decode_pca_frame("biwi", BIWI_TRAIN_ID, BIWI_TRAIN_SEEDS[i]),
    "ict": lambda i: decode_ict_frame(ICT_TRAIN_ID_IDX, ICT_TRAIN_EXP_IDX[i]),
}

# (key, src_dataset, tgt_dataset)
SCENARIOS = [
    ("mf_to_ict", "mf_ROM", "ict"),
    ("ict_to_mf", "ict", "mf_ROM"),
    ("coma_to_mf", "coma", "mf_ROM"),
    ("mf_to_coma", "mf_ROM", "coma"),
    ("biwi_to_mf", "biwi", "mf_ROM"),
    ("mf_to_biwi", "mf_ROM", "biwi"),
    ("coma_to_biwi", "coma", "biwi"),
    ("biwi_to_coma", "biwi", "coma"),
]

N_FRAMES = 4


def build_opts(ckpt):
    opts_yaml = yaml.load(open(f"{ckpt}/train_opts.yml"), Loader=yaml.FullLoader)
    opts_yaml.update(
        ckpt=ckpt, version=5, device="cuda:0", continue_ckpt=True, start_epoch=200,
        align_latent=True, tb=False, seed=42,
    )
    return argparse.Namespace(**opts_yaml)


def to_th(v, device):
    return torch.tensor(v).float()[None].to(device)


def normals(v, f):
    return igl.per_vertex_normals(v.astype(np.float64), f).astype(np.float32)


@torch.no_grad()
def self_retarget(model, template, deformed, faces, device):
    tmpl_th, tmpl_n_th = to_th(template, device), to_th(normals(template, faces), device)
    def_th, def_n_th = to_th(deformed, device), to_th(normals(deformed, faces), device)
    key_weight = model.predict_coordinate(tmpl_th, tmpl_n_th)
    pred, _ = model.retarget_animation(tmpl_th, tmpl_n_th, def_th, def_n_th, key_weight, tmpl_th)
    return pred[0].cpu().numpy()


@torch.no_grad()
def cross_cyclic_retarget(model, src_template, src_deformed, src_faces,
                           tgt_template, tgt_faces, device):
    src_tmpl_th = to_th(src_template, device)
    src_tmpl_n_th = to_th(normals(src_template, src_faces), device)
    src_def_th = to_th(src_deformed, device)
    src_def_n_th = to_th(normals(src_deformed, src_faces), device)
    tgt_tmpl_th = to_th(tgt_template, device)
    tgt_tmpl_n_th = to_th(normals(tgt_template, tgt_faces), device)
    tgt_faces_th = torch.from_numpy(tgt_faces).long().to(device)
    src_faces_th = torch.from_numpy(src_faces).long().to(device)

    key_weight = model.predict_coordinate(tgt_tmpl_th, tgt_tmpl_n_th)
    src_key_weight = model.predict_coordinate(src_tmpl_th, src_tmpl_n_th)

    pred_cross, _ = model.retarget_animation(
        src_tmpl_th, src_tmpl_n_th, src_def_th, src_def_n_th, key_weight, tgt_tmpl_th
    )
    pred_cross_n = calc_norm_torch(pred_cross, tgt_faces_th, at="verts")
    pred_cyclic, _ = model.retarget_animation(
        tgt_tmpl_th, tgt_tmpl_n_th, pred_cross, pred_cross_n, src_key_weight, src_tmpl_th
    )
    return pred_cross[0].cpu().numpy(), pred_cyclic[0].cpu().numpy()


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


def run_scenario(model, device, ckpt, scenario):
    key, src_ds, tgt_ds = scenario
    print(f" -- {key} --")
    out_dir = OUT_ROOT / Path(ckpt).name / key
    for i in range(N_FRAMES):
        src_template, src_deformed, src_faces = DECODERS[src_ds](i)
        tgt_template, _, tgt_faces = DECODERS[tgt_ds](i)

        v_self = self_retarget(model, src_template, src_deformed, src_faces, device)
        v_cross, v_cyclic = cross_cyclic_retarget(
            model, src_template, src_deformed, src_faces, tgt_template, tgt_faces, device
        )
        render_frame(
            src_deformed, src_faces,
            v_self, src_faces,
            v_cross, tgt_faces,
            v_cyclic, src_faces,
            out_dir, f"frame_{i:02d}",
        )


def main():
    for model_name, ckpt in MODELS:
        print(f"\n========== {model_name} ({ckpt}) ==========")
        opts = build_opts(ckpt)
        trainer = Trainer(opts)
        model = trainer.model
        device = trainer.device

        for scenario in SCENARIOS:
            run_scenario(model, device, ckpt, scenario)

        del trainer, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("\nALL DONE:", OUT_ROOT)


if __name__ == "__main__":
    main()
