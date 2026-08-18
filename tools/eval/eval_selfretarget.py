# tools/eval/eval_selfretarget.py — E1: self-retargeting quantitative eval (v2).
#
# v2 changes (2026-08-18, protocol audit):
#  * Clip policy: --clips 0 (default) = ALL held-out clips (following the NFS
#    lineage protocol: full sorted split). Per-dataset caps: BIWI capped at
#    --biwi-clips (default 12) by top mean-motion (its clips are homogeneous
#    spoken sentences); explicitly-listed seen-id reference targets capped at
#    --seen-clips (12) — pass e.g. --seen mf:0.
#  * Similarity-Procrustes alignment of EVERY method's prediction to GT before
#    scoring (rotation+translation+scale), following the fairness protocol of
#    Cha et al. 2025 (NFR's Poisson solve shifts/rescales the mesh).
#    Disable with --align none.
#  * ict_real targets (ds 'ict_real'): driven by the ICT-live test coefficient
#    sequence (parametric self-retargeting; source-capable ids only).
import sys, os, json, argparse
import numpy as np
import torch
from pathlib import Path

_REPO = '/source/inyup/NeuralFacialAnimation'
os.chdir(_REPO)
sys.path.insert(0, _REPO)
sys.path.insert(0, f'{_REPO}/tools/vis')
sys.path.insert(0, f'{_REPO}/tools/eval')
import viser_debug as vd
from scipy.spatial import cKDTree


def lip_set(neu, faces, jaw_disp):
    dy = jaw_disp[:, 1]
    lower = np.where(dy < np.quantile(dy, 0.005))[0]
    d_all, _ = cKDTree(neu[lower]).query(neu)
    diag = np.linalg.norm(neu.max(0) - neu.min(0))
    return np.where(d_all < 0.02 * diag)[0]


def procrustes_align(pred, gt):
    """Similarity Procrustes: align pred to gt (rot+trans+scale). [V,3]->[V,3]"""
    P = pred - pred.mean(0); G = gt - gt.mean(0)
    H = P.T @ G
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = (S * np.array([1, 1, d])).sum() / (P ** 2).sum()
    return (s * (R @ P.T)).T + gt.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpts_hlbs/combined500_import')
    ap.add_argument('--targets', nargs='+', required=True)
    ap.add_argument('--methods', nargs='+', default=['hlbs'],
                    choices=['hlbs', 'nfr', 'nfs', 'dt'])
    ap.add_argument('--clips', type=int, default=0, help='global cap (0=all)')
    ap.add_argument('--biwi-clips', type=int, default=12,
                    help='BIWI cap, top mean-motion (homogeneous sentences)')
    ap.add_argument('--seen', nargs='*', default=[],
                    help='seen-id reference targets (capped at --seen-clips)')
    ap.add_argument('--seen-clips', type=int, default=12)
    ap.add_argument('--frames-per-clip', type=int, default=8)
    ap.add_argument('--ictlive-frames', type=int, default=16,
                    help='frames sampled from the ICT-live test sequence')
    ap.add_argument('--align', choices=['procrustes', 'none'], default='procrustes')
    ap.add_argument('--out', default='eval_out/selfretarget_v2')
    args = ap.parse_args()

    dev = 'cuda'
    model, rig, opts, nfs_dir, helper_idx, geo_per_topo = vd._build_model(Path(args.ckpt), dev)
    needs_nfs = getattr(opts, 'nfs_feat_dir', None) is not None
    from utils.remesh_utils import ICT_face_model
    ict = ICT_face_model(base_dir=_REPO)
    topos = vd._build_topos(ict, nfs_dir, geo_per_topo)
    cache = vd.IdentityCache(model, rig, dev, model_needs_nfs=needs_nfs)
    baseline = vd.BaselineRunner(dev, Path(_REPO))
    ictlive = np.load('data/ICT_live_100/expression_vecs_test.npy').astype(np.float32)
    # make ALL 20 ict_real ids source-capable: ict_id_vecs_test.pt rows map 1:1
    # onto sorted real-id names (verified: parametric neutral == template scan).
    if 'ict_real' in topos:
        _v20 = torch.load('ict_face_pt/ict_id_vecs_test.pt').numpy()
        _tr = topos['ict_real']
        _tr._ict_id_vecs = {nm: _v20[i] for i, nm in enumerate(_tr.id_names)}


    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def eval_frame(td, idx, exp, spec, clip_label, fi, lips, fmask, diag, prev):
        o = cache.retarget(td, idx, exp, td, idx)
        if not o['model_ran']:
            return
        gt = o['src_def_v']
        preds = {}
        if 'hlbs' in args.methods:
            preds['hlbs'] = o['tgt_pred_v']
        if 'dt' in args.methods:
            try:
                from baseline_dt import dt_self_transfer
                preds['dt'] = dt_self_transfer((td.name, idx), o['src_neu_v'], gt,
                                               o['tgt_neu_v'], td.faces)
            except Exception as ex:
                print('dt EXC', str(ex)[:120]); preds['dt'] = None
        for meth in ('nfr', 'nfs'):
            if meth in args.methods:
                try:
                    preds[meth] = baseline.retarget(
                        method=meth, src_td=td, src_idx=idx,
                        src_neu_v=o['src_neu_v'], src_def_v=gt,
                        tgt_td=td, tgt_idx=idx, tgt_neu_v=o['tgt_neu_v'])
                except Exception as ex:
                    print(meth, 'EXC', str(ex)[:120]); preds[meth] = None
        for meth, p in preds.items():
            if p is None:
                continue
            if args.align == 'procrustes':
                p = procrustes_align(np.asarray(p, np.float64),
                                     np.asarray(gt, np.float64))
            e = np.linalg.norm(p - gt, axis=1)
            rows.append(dict(target=spec, clip=clip_label, frame=int(fi), method=meth,
                             v_l2_all=float(e.mean() / diag),
                             v_l2_face=float(e[fmask].mean() / diag),
                             lve=float(e[lips].max() / diag)))

    for spec in args.targets:
        ds, idn = spec.split(':', 1)
        td = topos[ds]
        idx = int(idn) if idn.isdigit() else td.id_names.index(idn)
        id_name = td.id_names[idx]
        base = cache.get(idx, td)
        neu = base['neu_v']; faces = td.faces.astype(np.int64)
        diag = float(np.linalg.norm(neu.max(0) - neu.min(0)))
        try:
            fmask = vd._face_vert_mask(neu)
            if fmask is None or fmask.dtype != bool or fmask.sum() == 0:
                fmask = np.ones(len(neu), bool)
        except Exception:
            fmask = np.ones(len(neu), bool)

        if ds == 'ict_real':
            # parametric self-retarget via ICT-live test coefficients
            step = max(1, len(ictlive) // args.ictlive_frames)
            fidx = list(range(0, len(ictlive), step))[:args.ictlive_frames]
            probe = td.apply_exp(id_name, ictlive[max(fidx)])
            if probe is None:
                print(f'[skip] {spec}: not source-capable'); continue
            lips = lip_set(neu, faces, probe - neu)
            for fi in fidx:
                eval_frame(td, idx, ictlive[fi], spec, 'ICTlive_test', fi,
                           lips, fmask, diag, None)
            print(f'[{spec}] ICTlive_test: {len(fidx)} frames done', flush=True)
            continue

        clips = td.clips_for(id_name)
        if not clips:
            print(f'[skip] {spec}: no clips'); continue

        # clip policy
        cap = args.clips
        if spec in args.seen:
            cap = args.seen_clips
        if ds == 'biwi':
            cap = args.biwi_clips if cap == 0 else min(cap, args.biwi_clips)
        if cap and len(clips) > cap:
            mot = []
            for cl in clips:
                n = td.n_frames(id_name, cl)
                v = td.apply_exp(id_name, (cl, n // 2))
                mot.append((float(np.linalg.norm(v - neu, axis=1).mean()) if v is not None else 0, cl))
            mot.sort(reverse=True)
            clips = [cl for _, cl in mot[:cap]]
        print(f'[{spec}] using {len(clips)} clips', flush=True)

        probe = None; bestm = -1
        for cl in clips[:6]:
            n = td.n_frames(id_name, cl)
            for fi in range(0, n, max(1, n // 6)):
                v = td.apply_exp(id_name, (cl, fi))
                if v is None: continue
                m = float(np.linalg.norm(v - neu, axis=1).mean())
                if m > bestm: bestm, probe = m, (v - neu)
        lips = lip_set(neu, faces, probe)

        for cl in clips:
            n = td.n_frames(id_name, cl)
            if n < 2: continue
            step = max(1, n // args.frames_per_clip)
            fidx = list(range(0, n, step))[:args.frames_per_clip]
            for fi in fidx:
                eval_frame(td, idx, (cl, fi), spec, cl, fi, lips, fmask, diag, None)
            print(f'[{spec}] {cl}: {len(fidx)} frames done', flush=True)

    json.dump(rows, open(out_dir / 'per_frame.json', 'w'), indent=0, default=float)
    agg = {}
    for r in rows:
        agg.setdefault((r['target'], r['method']), []).append(r)
    print('\n== summary (mean, diag-normalized, %s-aligned) ==' % args.align)
    print(f"{'target':<24} {'method':<6} {'n':>4} {'v_l2_face':>10} {'v_l2_all':>10} {'LVE':>10}")
    summary = []
    for (t, m), rs in sorted(agg.items()):
        s = dict(target=t, method=m, n=len(rs), align=args.align,
                 v_l2_face=float(np.mean([r['v_l2_face'] for r in rs])),
                 v_l2_all=float(np.mean([r['v_l2_all'] for r in rs])),
                 lve=float(np.mean([r['lve'] for r in rs])))
        summary.append(s)
        print(f"{t:<24} {m:<6} {s['n']:>4} {s['v_l2_face']:>10.6f} {s['v_l2_all']:>10.6f} {s['lve']:>10.6f}")
    json.dump(summary, open(out_dir / 'summary.json', 'w'), indent=1, default=float)
    print('EVAL_DONE ->', out_dir, flush=True)


if __name__ == '__main__':
    main()
