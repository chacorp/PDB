# E2b v2: cycle-consistency — 3 methods x diversified chains + ICT calibration.
#
# Modes:
#   --mode chains : hop curves (1..4) for {hlbs, nfr, nfs} on multiple chains
#                   whose pairs were NEVER used by the xcycle training loss.
#   --mode calib  : ICT parametric pairs — per-sample (cycle_err, true_err)
#                   correlation (Pearson/Spearman) to calibrate the GT-free metric.
import sys, os, json, argparse, time
import numpy as np, torch
from pathlib import Path
_REPO = '/source/inyup/NeuralFacialAnimation'
os.chdir(_REPO); sys.path.insert(0, _REPO); sys.path.insert(0, f'{_REPO}/tools/vis')
import viser_debug as vd

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['chains', 'calib'], default='chains')
ap.add_argument('--methods', nargs='+', default=['hlbs', 'nfr', 'nfs'])
ap.add_argument('--frames', type=int, default=6, help='frames per chain source')
ap.add_argument('--pairs', type=int, default=12, help='calib: number of ict id pairs')
ap.add_argument('--calib-frames', type=int, default=8, help='calib: frames per pair')
ap.add_argument('--out', default='eval_out')
args = ap.parse_args()

dev = 'cuda'
model, rig, opts, nfs_dir, helper_idx, geo_per_topo = vd._build_model(
    Path('ckpts_hlbs/combined500_import'), dev)
needs_nfs = getattr(opts, 'nfs_feat_dir', None) is not None
from utils.remesh_utils import ICT_face_model
ict = ICT_face_model(base_dir=_REPO)
topos = vd._build_topos(ict, nfs_dir, geo_per_topo)
cache = vd.IdentityCache(model, rig, dev, model_needs_nfs=needs_nfs)
baseline = vd.BaselineRunner(dev, Path(_REPO))
os.makedirs(args.out, exist_ok=True)

def _b(a):
    return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).to(dev).float()

@torch.no_grad()
def hop_ours(td_s, si, src_def, td_t, ti):
    sb = cache.get(si, td_s); tb = cache.get(ti, td_t)
    sn = vd._per_vertex_normal(sb['neu_v'], td_s.faces.astype(np.int64))
    dn = vd._per_vertex_normal(src_def, td_s.faces.astype(np.int64))
    tn = vd._per_vertex_normal(tb['neu_v'], td_t.faces.astype(np.int64))
    pred, _, _ = model.retarget(
        src_neu_vert=_b(sb['neu_v']), src_neu_norm=_b(sn),
        src_def_vert=_b(src_def), src_def_norm=_b(dn),
        tgt_neu_vert=_b(tb['neu_v']), tgt_neu_norm=_b(tn),
        tgt_nfs_feat=tb['nfs'], tgt_dist_sq_geo=tb['dist_sq_geo'],
        return_joints=True)
    return pred[0].cpu().numpy().astype(np.float32)

def hop_base(meth, td_s, si, src_def, td_t, ti):
    sb = cache.get(si, td_s); tb = cache.get(ti, td_t)
    return baseline.retarget(method=meth, src_td=td_s, src_idx=si,
                             src_neu_v=sb['neu_v'], src_def_v=src_def,
                             tgt_td=td_t, tgt_idx=ti, tgt_neu_v=tb['neu_v'])

HELPER_SET = set(helper_idx) if helper_idx else set()

def _markers(td, i):
    jp = cache.get(i, td)['joint_pos_pred']
    keep = [j for j in range(jp.shape[0]) if j not in HELPER_SET]
    return jp[keep]

def hop_dt(td_s, si, src_def, td_t, ti):
    import sys as _s
    if 'tools/eval' not in _s.path:
        _s.path.insert(0, 'tools/eval')
    from baseline_dt_cross import dt_cross_transfer
    sb = cache.get(si, td_s); tb = cache.get(ti, td_t)
    return dt_cross_transfer((td_s.name, si, td_t.name, ti),
                             sb['neu_v'], src_def, td_s.faces, _markers(td_s, si),
                             tb['neu_v'], td_t.faces, _markers(td_t, ti))

def hop(meth, td_s, si, src_def, td_t, ti):
    if meth == 'hlbs':
        return hop_ours(td_s, si, src_def, td_t, ti)
    if meth == 'dt':
        return hop_dt(td_s, si, src_def, td_t, ti)
    return hop_base(meth, td_s, si, src_def, td_t, ti)

def resolve(spec):
    ds, idn = spec.split(':', 1)
    td = topos[ds]
    return td, (int(idn) if idn.isdigit() else td.id_names.index(idn))

def top_motion_frames(td, idx, k):
    """k frames with max mean displacement, temporally deduped (>=15f apart)."""
    id_name = td.id_names[idx]; neu = cache.get(idx, td)['neu_v']
    cand = []
    for cl in td.clips_for(id_name)[:12]:
        n = td.n_frames(id_name, cl)
        for fi in range(0, n, max(1, n // 20)):
            v = td.apply_exp(id_name, (cl, fi))
            if v is None: continue
            cand.append((float(np.linalg.norm(v - neu, axis=1).mean()), cl, fi))
    cand.sort(reverse=True)
    out = []
    for m, cl, fi in cand:
        if all(c != cl or abs(f - fi) >= 15 for _, c, f in out):
            out.append((m, cl, fi))
        if len(out) == k: break
    return [(cl, fi) for _, cl, fi in out]

# ─────────────────────────── chains mode ───────────────────────────
CHAINS = [
    ['mf:12', 'biwi:M4', 'coma:FaceTalk_170809_00138_TA', 'ict_real:m00'],
    ['mf:0', 'coma:FaceTalk_170908_03277_TA', 'biwi:F3', 'ict_real:m02'],
    ['biwi:M5', 'mf:12', 'coma:FaceTalk_170725_00137_TA', 'ict_real:m01'],
    ['coma:FaceTalk_170908_03277_TA', 'biwi:F2', 'mf:0', 'ict_real:m03'],
]

if args.mode == 'chains':
    rows = []
    for chain in CHAINS:
        nodes = [resolve(s) for s in chain]
        A = nodes[0]
        neuA = cache.get(A[1], A[0])['neu_v']
        maskA = vd._face_vert_mask(neuA)
        diagA = float(np.linalg.norm(neuA.max(0) - neuA.min(0)))
        frames = top_motion_frames(A[0], A[1], args.frames)
        print('== chain', ' -> '.join(chain), f'({len(frames)} frames)', flush=True)
        for cl, fi in frames:
            gtA = A[0].apply_exp(A[0].id_names[A[1]], (cl, fi))
            for meth in args.methods:
                try:
                    r = {'chain': ' > '.join(chain), 'clip': cl, 'frame': fi, 'method': meth}
                    # walk forward: A->n1, n1->n2, n2->n3; return legs at each depth
                    cur = gtA; src = A
                    fwd = []
                    for d in range(1, len(nodes)):
                        cur = hop(meth, src[0], src[1], cur, nodes[d][0], nodes[d][1])
                        if cur is None: raise RuntimeError('hop none')
                        fwd.append(cur); src = nodes[d]
                    for d in range(1, len(nodes)):
                        back = hop(meth, nodes[d][0], nodes[d][1], fwd[d - 1], A[0], A[1])
                        e = np.linalg.norm(back - gtA, axis=1)
                        r[f'h{d + 1}_face'] = float(e[maskA].mean() / diagA)
                    # self (h1)
                    s1 = hop(meth, A[0], A[1], gtA, A[0], A[1])
                    e = np.linalg.norm(s1 - gtA, axis=1)
                    r['h1_face'] = float(e[maskA].mean() / diagA)
                    rows.append(r)
                    print('  %-5s %-24s f%03d  h1 %.5f h2 %.5f h3 %.5f h4 %.5f' %
                          (meth, cl, fi, r['h1_face'], r['h2_face'], r['h3_face'], r['h4_face']),
                          flush=True)
                except Exception as ex:
                    print('  EXC', meth, cl, fi, str(ex)[:120], flush=True)
    json.dump(rows, open(f'{args.out}/cycle_chains_' + '_'.join(args.methods) + '.json', 'w'), indent=0, default=float)
    print('\n== mean face err by method x cycle length ==')
    for meth in args.methods:
        rs = [r for r in rows if r['method'] == meth]
        if not rs: continue
        m = {k: float(np.mean([r[f'{k}_face'] for r in rs if f'{k}_face' in r]))
             for k in ('h1', 'h2', 'h3', 'h4')}
        print('%-5s n=%d  h1 %.5f  h2 %.5f  h3 %.5f  h4 %.5f  (h4/h2 %.2f)' %
              (meth, len(rs), m['h1'], m['h2'], m['h3'], m['h4'], m['h4'] / max(m['h2'], 1e-9)))
    print('CHAINS_DONE', flush=True)

# ─────────────────────────── calib mode ───────────────────────────
else:
    ictt = topos['ict_train']
    rng = np.random.RandomState(7)
    ids = rng.choice(len(ictt.id_names), size=(args.pairs, 2), replace=True)
    ids = [(int(a), int(b)) for a, b in ids if a != b][:args.pairs]
    samples = []
    for (i, j) in ids:
        neuA = cache.get(i, ictt)['neu_v']
        maskA = vd._face_vert_mask(neuA)
        diagA = float(np.linalg.norm(neuA.max(0) - neuA.min(0)))
        nmB = ictt.id_names[j]
        neuB = cache.get(j, ictt)['neu_v']
        maskB = vd._face_vert_mask(neuB)
        diagB = float(np.linalg.norm(neuB.max(0) - neuB.min(0)))
        for f in range(args.calib_frames):
            exp = np.zeros(53, dtype=np.float32)
            act = rng.choice(53, size=rng.randint(2, 5), replace=False)
            exp[act] = rng.uniform(0.3, 1.0, size=len(act)).astype(np.float32)
            o = cache.retarget(ictt, i, exp, ictt, j)
            if not o['model_ran']: continue
            gtA = o['src_def_v']
            gtB = ictt.apply_exp(nmB, exp)
            predB = o['tgt_pred_v']
            true_err = float(np.linalg.norm(predB - gtB, axis=1)[maskB].mean() / diagB)
            back = hop_ours(ictt, j, predB, ictt, i)
            cyc_err = float(np.linalg.norm(back - gtA, axis=1)[maskA].mean() / diagA)
            samples.append({'i': i, 'j': j, 'f': f, 'true': true_err, 'cycle': cyc_err})
        print(f'pair ict_{i:03d}->ict_{j:03d} done ({len(samples)} samples)', flush=True)
    x = np.array([s['cycle'] for s in samples]); y = np.array([s['true'] for s in samples])
    from scipy.stats import pearsonr, spearmanr
    pr = pearsonr(x, y); sr = spearmanr(x, y)
    json.dump({'samples': samples, 'pearson_r': float(pr[0]), 'pearson_p': float(pr[1]),
               'spearman_r': float(sr[0]), 'spearman_p': float(sr[1])},
              open(f'{args.out}/cycle_calib.json', 'w'), indent=0, default=float)
    print(f'\n== calibration: n={len(samples)}  Pearson r={pr[0]:.3f} (p={pr[1]:.1e})  '
          f'Spearman r={sr[0]:.3f} ==')
    print('CALIB_DONE', flush=True)
