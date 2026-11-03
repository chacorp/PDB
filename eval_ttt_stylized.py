"""
Quantitative evaluation of test-time training (TTT) on the aligned stylized meshes:
PDB (before TTT) vs PDB-TTT-<target> (ttt_CBD.py, one model per target).

  self   : source expression reconstructed on the source neutral (model.forward,
           as eval_CBD.py --realtest --use_t_mask)
  cyclic : source -> stylized target -> source (retarget_animation, as
           eval_CBD_cyc.py cross-retargeting), compared with the source expression
Metrics per source: MSE (whole mesh), MSE-in (vs GT expression, inner mask),
MSE-out (vs source neutral, outer mask), averaged over frames. Masks follow the
reference scripts: self uses plateau_hat_points(r0=1.0, r1=2.25) (eval_CBD.py),
cyclic uses the defaults r0=1.25, r1=2.5 (eval_CBD_cyc.py).
Sources: mf test id 12 (mf_SEN, mf_ROM; reported as MF = frame-count weighted
mean) and ICT capture m05 with the 20240318_MySlate_922 motion.
Writes _tmp/ttt_eval/metrics_ttt.json (raw values, not scaled).

Run from repo root:
    python eval_ttt_stylized.py
"""
import json
from functools import partial
from pathlib import Path

import igl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import dataloader_CBD
from dataloader_CBD import CBD_collate_wrapper_eval
from utils.exp_utils import plateau_hat_points
from utils.remesh_utils import calc_norm_torch
from render.render_stylized_retarget_comparison import (
    TARGETS, TTT_RUNS, build_opts, load_targets,
)
from eval_CBD import Trainer

PDB_CKPT = './ckpts_CBD8/2026-09-16-02-19-03-NGBCv5'
TTT_EPOCH = 0  # model_000.pth, same checkpoint as ttt_figure.html
BS = 16
OUT_PATH = Path('_tmp/ttt_eval/metrics_ttt.json')

# source key -> EvalDataset kwargs (mf test split holds only id 12)
SOURCES = {
    'mf_SEN': dict(data_name='mf_SEN'),
    'mf_ROM': dict(data_name='mf_ROM'),
    'ict_m05': dict(data_name='ict-cap', ict_cap_id_num=5, ict_cap_exp_num=0),  # 20240318_MySlate_922
}


def masked_mse(gt, template, pred, r0=None, r1=None):
    """Batch-summed per-frame MSE / MSE-in / MSE-out (as eval_CBD*.py). All frames of a
    source share one template, so the mask is computed once and the batch mean x B
    equals the sum of the per-frame MSEs."""
    kw = {} if r0 is None else dict(r0=r0, r1=r1)
    B = gt.shape[0]
    inner = plateau_hat_points(template[:1], **kw)
    outer = 1 - inner
    return {'MSE': F.mse_loss(gt, pred).item() * B,
            'MSE-in': F.mse_loss(gt * inner, pred * inner).item() * B,
            'MSE-out': F.mse_loss(template * outer, pred * outer).item() * B}


def evaluate(model, loader, n, targets, device):
    """Self metrics on the source and cyclic metrics through every target in `targets`."""
    acc = {'self': dict(MSE=0.0, **{'MSE-in': 0.0, 'MSE-out': 0.0})}
    tgt = {}
    with torch.no_grad():
        for t, (tv, tf) in targets.items():
            tv_th = torch.tensor(tv).float()[None].to(device)
            tn_th = torch.tensor(igl.per_vertex_normals(tv, tf)).float()[None].to(device)
            tgt[t] = (tv_th, tn_th, torch.from_numpy(tf).to(device), model.predict_coordinate(tv_th, tn_th))
            acc[t] = dict(MSE=0.0, **{'MSE-in': 0.0, 'MSE-out': 0.0})

        src_key_weight = None
        for batch in tqdm(loader, ncols=100, leave=False):
            B = batch.vertices.shape[0]
            pred_self, *_ = model(batch.template, batch.vertices, batch.template_normal,
                                  batch.vertices_normal, mesh_data=batch.mesh_data, epoch=0)
            for k, v in masked_mse(batch.vertices, batch.template, pred_self, 1.0, 2.25).items():
                acc['self'][k] += v
            if src_key_weight is None:
                src_key_weight = model.predict_coordinate(batch.template[:1], batch.template_normal[:1])
            for t, (tv_th, tn_th, tf_th, key_weight) in tgt.items():
                cross, _ = model.retarget_animation(batch.template, batch.template_normal,
                                                    batch.vertices, batch.vertices_normal,
                                                    key_weight, tv_th)
                cross_n = calc_norm_torch(cross, tf_th, at='verts')
                cyc, _ = model.retarget_animation(tv_th.expand(B, -1, -1), tn_th.expand(B, -1, -1),
                                                  cross, cross_n, src_key_weight.expand(B, -1, -1),
                                                  batch.template)
                for k, v in masked_mse(batch.vertices, batch.template, cyc).items():
                    acc[t][k] += v
    return {k: {m: v / n for m, v in d.items()} for k, d in acc.items()}


def target_self(model, targets, device):
    """Target self retargeting T -> T on each stylized target neutral, the ttt-self
    term of NGBC.ttt_loss (key_weight(T) * key_s(T) vs the target neutral)."""
    out = {}
    with torch.no_grad():
        for t, (tv, tf) in targets.items():
            tv_th = torch.tensor(tv).float()[None].to(device)
            tn_th = torch.tensor(igl.per_vertex_normals(tv, tf)).float()[None].to(device)
            dummy_cage = torch.zeros(1, model.num_cage_vertices, 3, device=device)  # cross term, unused
            _, _, pred_self, _ = model.ttt_loss(tv_th, tn_th, dummy_cage)
            out[t] = masked_mse(tv_th, tv_th, pred_self)
    return out


def load_model(ckpt, epoch):
    opts = build_opts(ckpt, 5, epoch)
    trainer = Trainer(opts)
    trainer.model.eval()
    return trainer.model, trainer.device


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}
    targets = load_targets()
    models = [('PDB', PDB_CKPT, None, list(TARGETS))]
    models += [(f'PDB-TTT-{t}', f'./{run}', TTT_EPOCH, [t]) for t, run in TTT_RUNS.items()]

    batch_cache = {}  # ict-cap frames are built on the CPU by the ICT face model (slow): build once
    for name, ckpt, epoch, m_targets in models:
        if name in results and 'tself' in results[name]:
            print(f'===== {name}: done, skip =====', flush=True)
            continue
        print(f'===== {name} ({ckpt}) =====', flush=True)
        model, device = load_model(ckpt, epoch)
        if name in results:  # source metrics done, only the target self retargeting is missing
            results[name]['tself'] = target_self(model, {t: targets[t] for t in m_targets}, device)
            OUT_PATH.write_text(json.dumps(results, indent=1))
            del model
            torch.cuda.empty_cache()
            continue
        res = {'tself': target_self(model, {t: targets[t] for t in m_targets}, device)}
        for sname, kw in SOURCES.items():
            ds = dataloader_CBD.EvalDataset(toggle=False, **kw)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=BS, collate_fn=partial(CBD_collate_wrapper_eval, device=device))  # collate moves to CUDA: no forked workers
            if kw['data_name'] == 'ict-cap':
                if sname not in batch_cache:
                    batch_cache[sname] = list(loader)
                loader = batch_cache[sname]
            res[sname] = dict(n=len(ds), **evaluate(model, loader, len(ds), {t: targets[t] for t in m_targets}, device))
            print(f'  {sname} (n={len(ds)}) self MSE {res[sname]["self"]["MSE"]:.4e}', flush=True)
        results[name] = res
        OUT_PATH.write_text(json.dumps(results, indent=1))  # checkpoint after every model
        del model
        torch.cuda.empty_cache()
    print('wrote', OUT_PATH)


if __name__ == '__main__':
    main()
