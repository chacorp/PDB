# tools/eval/eval_selfretarget.py — E1: self-retargeting quantitative eval.
#
# Protocol: for each target (ds, id), drive the model with the target's own
# held-out clip frames (deformed mesh as source) and retarget onto ITSELF;
# prediction is compared to the driving frame (= GT).
#
# Methods: ours (hlbs), nfr, nfs — via viser_debug's IdentityCache /
# BaselineRunner (same code path as the interactive tool → no eval drift).
#
# Metrics (per frame, aggregated per clip/target):
#   v_l2_face : mean vertex L2 inside the face mask (plateau r0/r1, model's own)
#   v_l2_all  : mean vertex L2 (all verts)
#   LVE       : max lip-vertex L2 (lip set derived data-driven from jaw-open
#               motion — same derivation as teaser descriptor sets)
#   jitter    : mean ||(pred_t - pred_{t-1}) - (gt_t - gt_{t-1})|| on consecutive
#               sampled frames (velocity error; only for stride-1 pairs)
#
# Usage:
#   python tools/eval/eval_selfretarget.py --targets mf:12 biwi:M3 coma:FaceTalk_...
#       --methods hlbs nfr nfs --frames-per-clip 10 --out eval_out/selfretarget
#   Smoke: python tools/eval/eval_selfretarget.py --targets mf:12 --methods hlbs \
#       --clips 2 --frames-per-clip 3
import sys, os, json, argparse, time
import numpy as np
import torch
from pathlib import Path

_REPO = '/source/inyup/NeuralFacialAnimation'
os.chdir(_REPO)
sys.path.insert(0, _REPO)
sys.path.insert(0, f'{_REPO}/tools/vis')
import viser_debug as vd
from scipy.spatial import cKDTree


def lip_set(neu, faces, jaw_disp):
    """Lip vertex set: strong downward movers under jaw-open + neutral-proximal
    upper counterpart (same derivation family as the teaser descriptors)."""
    dy = jaw_disp[:, 1]
    lower = np.where(dy < np.quantile(dy, 0.005))[0]
    d_all, _ = cKDTree(neu[lower]).query(neu)
    diag = np.linalg.norm(neu.max(0) - neu.min(0))
    return np.where(d_all < 0.02 * diag)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='ckpts_hlbs/combined500_import')
    ap.add_argument('--targets', nargs='+', required=True,
                    help='ds:id  (mf:12 | biwi:M3 | coma:FaceTalk_...)')
    ap.add_argument('--methods', nargs='+', default=['hlbs'],
                    choices=['hlbs', 'nfr', 'nfs'])
    ap.add_argument('--clips', type=int, default=0, help='max clips per target (0=all)')
    ap.add_argument('--frames-per-clip', type=int, default=10)
    ap.add_argument('--out', default='eval_out/selfretarget')
    args = ap.parse_args()

    dev = 'cuda'
    model, rig, opts, nfs_dir, helper_idx, geo_per_topo = vd._build_model(Path(args.ckpt), dev)
    needs_nfs = getattr(opts, 'nfs_feat_dir', None) is not None
    from utils.remesh_utils import ICT_face_model
    ict = ICT_face_model(base_dir=_REPO)
    topos = vd._build_topos(ict, nfs_dir, geo_per_topo)
    cache = vd.IdentityCache(model, rig, dev, model_needs_nfs=needs_nfs)
    baseline = vd.BaselineRunner(dev, Path(_REPO))

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in args.targets:
        ds, idn = spec.split(':', 1)
        td = topos[ds]
        idx = int(idn) if idn.isdigit() else td.id_names.index(idn)
        id_name = td.id_names[idx]
        base = cache.get(idx, td)
        neu = base['neu_v']
        faces = td.faces.astype(np.int64)
        diag = float(np.linalg.norm(neu.max(0) - neu.min(0)))

        # face mask (model's own plateau mask if available)
        try:
            fmask = vd._face_vert_mask(neu)
        except Exception:
            fmask = np.ones(len(neu), bool)

        clips = td.clips_for(id_name)
        if not clips:
            print(f'[skip] {spec}: no clips'); continue
        if args.clips:
            clips = clips[:args.clips]

        # lip set from the clip frame with max motion (proxy for jaw-open)
        probe = None; bestm = -1
        for cl in clips[:4]:
            n = td.n_frames(id_name, cl)
            for fi in range(0, n, max(1, n // 6)):
                v = td.apply_exp(id_name, (cl, fi))
                m = float(np.linalg.norm(v - neu, axis=1).mean())
                if m > bestm: bestm, probe = m, (v - neu)
        lips = lip_set(neu, faces, probe)

        for cl in clips:
            n = td.n_frames(id_name, cl)
            if n < 2: continue
            step = max(1, n // args.frames_per_clip)
            fidx = list(range(0, n, step))[:args.frames_per_clip]
            prev = {}   # method -> (pred, gt) of previous consecutive frame
            for fi in fidx:
                o = cache.retarget(td, idx, (cl, fi), td, idx)
                if not o['model_ran']:
                    continue
                gt = o['src_def_v']
                preds = {}
                if 'hlbs' in args.methods:
                    preds['hlbs'] = o['tgt_pred_v']
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
                    if p is None: continue
                    e = np.linalg.norm(p - gt, axis=1)
                    row = dict(target=spec, id=id_name, clip=cl, frame=fi, method=meth,
                               v_l2_all=float(e.mean() / diag),
                               v_l2_face=float(e[fmask].mean() / diag),
                               lve=float(e[lips].max() / diag))
                    if meth in prev and prev[meth][2] == fi - step and step == 1:
                        dp = (p - prev[meth][0]) - (gt - prev[meth][1])
                        row['jitter'] = float(np.linalg.norm(dp, axis=1).mean() / diag)
                    rows.append(row)
                    prev[meth] = (p, gt, fi)
            print(f'[{spec}] {cl}: {len(fidx)} frames done', flush=True)

    json.dump(rows, open(out_dir / 'per_frame.json', 'w'), indent=0, default=float)
    # aggregate
    agg = {}
    for r in rows:
        k = (r['target'], r['method'])
        agg.setdefault(k, []).append(r)
    print('\n== summary (mean over frames, diag-normalized) ==')
    print(f"{'target':<28} {'method':<6} {'n':>4} {'v_l2_face':>10} {'v_l2_all':>10} {'LVE':>10}")
    summary = []
    for (t, m), rs in sorted(agg.items()):
        s = dict(target=t, method=m, n=len(rs),
                 v_l2_face=float(np.mean([r['v_l2_face'] for r in rs])),
                 v_l2_all=float(np.mean([r['v_l2_all'] for r in rs])),
                 lve=float(np.mean([r['lve'] for r in rs])))
        summary.append(s)
        print(f"{t:<28} {m:<6} {s['n']:>4} {s['v_l2_face']:>10.6f} {s['v_l2_all']:>10.6f} {s['lve']:>10.6f}")
    json.dump(summary, open(out_dir / 'summary.json', 'w'), indent=1, default=float)
    print('EVAL_DONE ->', out_dir, flush=True)


if __name__ == '__main__':
    main()
