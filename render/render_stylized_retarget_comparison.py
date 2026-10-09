"""
Self / cross / cyclic retargeting of the MF test identity and ICT m00
animations with the aligned stylized meshes (test-mesh/, see TARGETS) as
targets, for the NC / NFS / PDB (/ PDB-cont / PDBplus) models:
  self   : source expression -> source neutral (reconstruction)
  cross  : source expression -> target neutral
  cyclic : cross output -> back onto source neutral (source -> target -> source)
Stage 1 (inference): writes the predicted vertices to
vis_CBD/vis_comparison/stylize_retarget/preds_cache.npz (keys
self/{model}/{src}, pred/{model}/{tgt}/{src} (cross), cyc/{model}/{tgt}/{src}).
Only keys missing from an existing cache are inferred; delete it to rerun all.
Rendering and the gallery HTML are done by render/render_stylized_retarget_mpr.py.

Run from repo root:
    python render/render_stylized_retarget_comparison.py
    cd _tmp && python -I ../render/render_stylized_retarget_mpr.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.argv = [sys.argv[0]]

import argparse  # noqa: E402
import yaml  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import igl  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

import dataloader_CBD  # noqa: E402
import utils.nfr_utils as nfr_utils  # noqa: E402
from vis_CBD_retarget_fig import Pipeline  # noqa: E402
from eval_CBD_cyc import Options  # noqa: E402
from eval_CBD import Trainer  # noqa: E402
from utils.matplotlib_rnd import fix_triangle_widning  # noqa: E402

# target name -> aligned mesh under test-mesh/
TARGETS = {
    'aura': 'test-aura-aligned.obj',
    'bowen': 'test-bowen-aligned.obj',
    'jupiter': 'test-jupiter-aligned.obj',
    'proteus': 'test-proteus-aligned.obj',
    'bonnie': 'bonnie-bald-align.obj',
    'malcolm': 'malcolm-align.obj',
    'mary': 'mary-align.obj',
    'morphy': 'morphy-bald-align.obj',
    'piers': 'piers-align.obj',
    'girl': 'girl-align.obj',
}
MF_TEST_IDX = 12
ICT_ID_M00 = 0
ICT_EXP_924 = 1
MF_FRAMES = [1039, 5044, 5112, 7623, 9013]
ICT_FRAMES = [107, 602, 843, 873, 1117]

# (name, ckpt, version)  version 1 = NC, 0 = NFS, 5 = PDB, 6 = PDBplus
MODELS = [
    ('NC', './ckpts_CBD/2025-10-20-00-15-40-CBD', 1),
    ('NFS', './ckpt_stage1/2024-08-18-23-32-29-all', 0),
    # NFR = version 0 with use_NFR (design nfr, jacob decoder), as eval_CBD_cyc.sh
    ('NFR', './ckpt_stage1/exp_019_ICT_MF-jacob_NFR', 0),
    ('PDB', './ckpts_CBD8/2026-09-16-02-19-03-NGBCv5', 5),
    # PDB continued 1000 -> 2000 (train_continue.sh, use_data3); stopped at
    # epoch 1221, model_best.pth = epoch 1220. Shown in continue_figure.html.
    ('PDB-cont', './ckpts_CBD8/2026-10-07-00-58-27-NGBCv5', 5),
    # PDBplus (models/PDBplus.py, train_PDBplus.sh, use_data8); stopped at
    # epoch 463, model_best.pth = epoch 430. Shown in continue_figure.html.
    ('PDBplus', './ckpts_CBD/2026-10-06-01-00-32-PDBplusv6', 6),
    # PDB after test-time training on bowen (ttt_CBD.py, 10 epochs from PDB
    # model_best.pth). Shown in ttt_figure.html. (set once the TTT run finishes)
    # ('PDB-TTT-bowen', './ttt_CBD/<run>', 5),
]

# per-model overrides: checkpoint epoch to load instead of model_best.pth, and the
# only targets to infer (a TTT model is adapted to a single target)
MODEL_EPOCH = {'PDB-TTT-bowen': 10}
MODEL_TARGETS = {'PDB-TTT-bowen': ['bowen']}
USE_NFR = {'NFR'}

OUT_ROOT = REPO_ROOT / 'vis_CBD' / 'vis_comparison'
IMG_ROOT = OUT_ROOT / 'stylize_retarget'
CACHE_PATH = IMG_ROOT / 'preds_cache.npz'


def build_opts(ckpt, version, epoch=None, use_nfr=False):
    opts = Options()
    opts.ckpt = ckpt
    opts.version = version
    if version == 0:
        opts_yaml = yaml.load(open('config/train_NFS.yml'), Loader=yaml.FullLoader)
        opts_ = vars(opts)
        opts_yaml.update(opts_)
        opts = argparse.Namespace(**opts_yaml)
        opts.img_feat_dim = 128
        opts.feature_type = 'cents&norms'
        opts.stage1 = True
        opts.scale_exp = 1.0
        opts.ict_face_only = False
        opts.use_NFR = use_nfr
        # same switch as eval_CBD_cyc.py: NFR -> design nfr + jacob decoder
        opts.design = 'nfr' if use_nfr else 'new2'
        opts.dec_type = 'jacob' if use_nfr else 'disp'
    else:
        opts_yaml = yaml.load(open(f'{ckpt}/train_opts.yml'), Loader=yaml.FullLoader)
        opts_ = vars(opts)
        opts_yaml.update(opts_)
        opts = argparse.Namespace(**opts_yaml)
    opts.use_t_mask = True
    # eval_CBD.Trainer loads model_{start_epoch}.pth when continue_ckpt, else model_best.pth
    opts.continue_ckpt = epoch is not None
    if epoch is not None:
        opts.start_epoch = epoch
    opts.batch_size = 1
    return opts


def build_pdbplus(opts):
    """eval_CBD.Trainer has no version 6; build PDBplus as train_CBD.py does and
    load model_best.pth (continue_ckpt is False)."""
    from models.PDBplus import PDBplus
    acts = [opts.last_activation == a for a in ['relu', 'elu', 'softmax', 'softplus', 'none']]
    model = PDBplus(
        opts, num_layers=4, num_cage_vertices=opts.num_cage_v,
        use_exp_recon=False, use_shp_recon=False, use_shp=False,
        use_relu=acts[0], use_elu=acts[1], use_softmax=acts[2], use_softplus=acts[3],
        no_activation=acts[4], use_least_N_on_V=False, is_train=True,
        use_pou=not opts.no_pou, device=opts.device,
        hid_dim=128 if opts.align_latent else 256,
    )
    ckpt_dict = torch.load(os.path.join(opts.ckpt, 'model_best.pth'), map_location=opts.device)
    model.load_state_dict(ckpt_dict['model'] if 'model' in ckpt_dict else ckpt_dict)
    print(f"Loaded! {opts.ckpt}/model_best.pth (epoch {ckpt_dict.get('epoch')})", flush=True)
    return model


def load_targets():
    out = {}
    for n, fn in TARGETS.items():
        m = trimesh.load(f'test-mesh/{fn}', process=False, maintain_order=True)
        v = np.asarray(m.vertices, float)
        # outward winding (bowen/proteus are inverted): the vertex normals fed to the
        # models (PDB in_type 1, NFS mesh features) follow the face winding
        out[n] = (v, fix_triangle_widning(v, np.asarray(m.faces, np.int64)))
    return out


def load_sources(pipe):
    mf_ds = dataloader_CBD.EvalDataset(data_name='mf_ROM', toggle=False)
    ict_ds = dataloader_CBD.EvalDataset(
        data_name='ict-cap', toggle=False, ict_cap_id_num=ICT_ID_M00, ict_cap_exp_num=ICT_EXP_924,
    )
    srcs = {}
    for key, ds, sel, mid, frames in [
        ('mf', mf_ds, 'mf_ROM', MF_TEST_IDX, MF_FRAMES),
        ('ict', ict_ds, 'ict-cap', ICT_ID_M00, ICT_FRAMES),
    ]:
        src_v, src_f, _ = pipe.get_mesh(sel, None, mid)
        frame_vs = [ds[i][0].numpy() for i in frames]
        srcs[key] = dict(neutral_v=np.asarray(src_v, float), faces=np.asarray(src_f, np.int64),
                         frames=frame_vs, idx=frames)
    return srcs


def run_model(pipe, version, src_v, src_f, frames, tgt_v, tgt_f):
    """Cross-retarget a list of source expression frames onto (tgt_v, tgt_f)."""
    device = pipe.device
    model = pipe.model
    model.eval()
    outs = []
    if version == 1:
        src_v_th = torch.tensor(src_v).float()[None].to(device)
        tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)
        for fv in frames:
            fv_th = torch.tensor(fv).float()[None].to(device)
            with torch.no_grad():
                pred, _ = model.retarget(src_v_th, fv_th, tgt_v_th)
            outs.append(pred[0].detach().cpu().numpy())
    elif version == 0:
        model.eval()
        tgt_m = trimesh.Trimesh(vertices=tgt_v, faces=tgt_f)
        tgt_verts = torch.tensor(tgt_v).float().to(device)
        tgt_faces = torch.from_numpy(tgt_f).to(device)
        tgt_dfn_info = nfr_utils.get_dfn_info(tgt_m, map_location=device)
        tgt_operators = nfr_utils_get_ops(tgt_m)
        tgt_img = model.renderer.render_img(tgt_m).float().to(device)
        tgt_img_feat = model.get_img_feat(tgt_img)
        tgt_vert_feat = model.get_local_feature(tgt_verts[None], tgt_faces, tgt_img_feat).float()
        tgt_tri_feat = model.get_local_feature(tgt_verts[None], tgt_faces, tgt_img_feat, at='faces').float()
        with torch.no_grad():
            pred_seg_coeff = model.encode_seg(tgt_vert_feat, tgt_dfn_info)
            pred_id_coeff = model.encode_id(tgt_vert_feat, tgt_dfn_info)

        src_m = trimesh.Trimesh(vertices=src_v, faces=src_f)
        src_faces_th = torch.from_numpy(src_f).to(device)
        src_dfn_info = nfr_utils.get_dfn_info(src_m, map_location=device)
        src_img = model.renderer.render_img(src_m).float().to(device)
        src_img_feat = model.get_img_feat(src_img)
        for fv in frames:
            fv_th = torch.tensor(fv).float()[None].to(device)
            with torch.no_grad():
                # expression feature scale 1.3 for NFS, 1.0 for the jacob decoder (NFR), as eval_CBD_cyc.py
                exp_scale = 1.0 if pipe.opts.dec_type == 'jacob' else 1.3
                vert_feat_exp = model.get_local_feature(fv_th, src_faces_th, src_img_feat).float() * exp_scale
                pred_exp_coeff = model.encode_exp(vert_feat_exp, src_dfn_info, batch_process=True, verbose=False)
                inputs = (
                    tgt_tri_feat if pipe.opts.dec_type == 'jacob' else tgt_vert_feat,
                    pred_exp_coeff, pred_id_coeff, pred_seg_coeff,
                    None, tgt_verts[None], tgt_faces, tgt_operators
                )
                decode_out = model.decode(inputs, batch_process=True)
                pred = decode_out[0] if isinstance(decode_out, tuple) else decode_out
            pred = pred[0].detach().cpu().numpy()
            if pipe.opts.dec_type == 'jacob':
                # jacob decoder output is zero-centered (no translation, evaluation.py
                # calc_new_mesh); place it at the mean of the mesh it is decoded on, which
                # equals evaluation.py's zero-mean comparison
                pred = pred - pred.mean(0) + tgt_v.mean(0)
            outs.append(pred)
    else:
        src_n = igl.per_vertex_normals(src_v, src_f)
        tgt_n = igl.per_vertex_normals(tgt_v, tgt_f)
        src_v_th = torch.tensor(src_v).float()[None].to(device)
        src_n_th = torch.tensor(src_n).float()[None].to(device)
        tgt_v_th = torch.tensor(tgt_v).float()[None].to(device)
        tgt_n_th = torch.tensor(tgt_n).float()[None].to(device)
        with torch.no_grad():
            key_weight = model.predict_coordinate(tgt_v_th, tgt_n_th)
        # PDBplus also needs the target's own neutral normal (as in eval_CBD_cyc.py)
        tgt_kwargs = {'tgt_neu_norm': tgt_n_th} if version == 6 else {}
        for fv in frames:
            fn = igl.per_vertex_normals(fv, src_f)
            fv_th = torch.tensor(fv).float()[None].to(device)
            fn_th = torch.tensor(fn).float()[None].to(device)
            with torch.no_grad():
                pred, _ = model.retarget_animation(src_v_th, src_n_th, fv_th, fn_th, key_weight, tgt_v_th,
                                                   **tgt_kwargs)
            outs.append(pred[0].detach().cpu().numpy())
    return outs


def nfr_utils_get_ops(mesh):
    from utils.mesh_utils import get_mesh_operators
    return get_mesh_operators(mesh)


def needed_keys(model_name, targets, sources):
    keys = [f'self/{model_name}/{sn}' for sn in sources]
    for tn in targets:
        for sn in sources:
            keys += [f'pred/{model_name}/{tn}/{sn}', f'cyc/{model_name}/{tn}/{sn}']
    return keys


def main():
    IMG_ROOT.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    if CACHE_PATH.exists():
        cache = dict(np.load(CACHE_PATH))
        srcs = {sn: dict(neutral_v=cache[f'src/{sn}/v'], faces=cache[f'src/{sn}/f'],
                         frames=list(cache[f'src/{sn}/frames']), idx=list(cache[f'src/{sn}/idx']))
                for sn in cache['sources']}
    else:
        cache = {}
        # source frames and neutral templates depend only on the dataset, not the model
        base_pipe = Pipeline(build_opts(MODELS[0][1], MODELS[0][2]))
        srcs = load_sources(base_pipe)

    for model_name, ckpt, version in MODELS:
        m_targets = {t: targets[t] for t in MODEL_TARGETS.get(model_name, targets)}
        missing = [k for k in needed_keys(model_name, m_targets, srcs) if k not in cache]
        if not missing:
            print(f'===== {model_name}: cached, skip =====', flush=True)
            continue
        print(f'===== {model_name} ({ckpt}): {len(missing)} missing =====', flush=True)
        opts = build_opts(ckpt, version, MODEL_EPOCH.get(model_name), use_nfr=model_name in USE_NFR)
        trainer = None if version == 6 else Trainer(opts)
        pipe = Pipeline(opts)
        pipe.model = build_pdbplus(opts) if version == 6 else trainer.model
        for sname, s in srcs.items():
            sv, sf = s['neutral_v'], s['faces']
            k = f'self/{model_name}/{sname}'
            if k not in cache:
                cache[k] = np.stack(run_model(pipe, version, sv, sf, s['frames'], sv, sf))
                print(f'  self {sname}', flush=True)
            for tname, (tv, tf) in m_targets.items():
                k = f'pred/{model_name}/{tname}/{sname}'
                if k not in cache:
                    cache[k] = np.stack(run_model(pipe, version, sv, sf, s['frames'], tv, tf))
                    print(f'  cross {tname} <- {sname}', flush=True)
                k = f'cyc/{model_name}/{tname}/{sname}'
                if k not in cache:
                    cross = list(cache[f'pred/{model_name}/{tname}/{sname}'])
                    cache[k] = np.stack(run_model(pipe, version, tv, tf, cross, sv, sf))
                    print(f'  cyclic {sname} -> {tname} -> {sname}', flush=True)
        del trainer, pipe
        torch.cuda.empty_cache()

    save_cache(cache, targets, srcs)
    print('wrote', CACHE_PATH)


def save_cache(cache, targets, srcs):
    """Flat npz: target/source meshes, source frames and per-model predictions."""
    d = dict(cache)
    d.update({'targets': np.array(list(targets)), 'sources': np.array(list(srcs)),
              'models': np.array([m for m, _, _ in MODELS])})
    for tname, (tv, tf) in targets.items():
        d[f'tgt/{tname}/v'] = tv
        d[f'tgt/{tname}/f'] = tf
        d[f'tgt/{tname}/file'] = np.array(TARGETS[tname])
    for sname, s in srcs.items():
        d[f'src/{sname}/v'] = s['neutral_v']
        d[f'src/{sname}/f'] = s['faces']
        d[f'src/{sname}/idx'] = np.array(s['idx'])
        d[f'src/{sname}/frames'] = np.stack(s['frames'])
    np.savez_compressed(CACHE_PATH, **d)


if __name__ == '__main__':
    main()
