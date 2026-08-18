# tools/eval/eval_cycle_v3.py — E2b expanded: permutation-complete cycles.
#
# v3 (2026-08-18, protocol audit):
#  * 2-hop: ALL 12 ordered pairs among the 4 topology representatives.
#  * 3-hop / 4-hop: N=8 seeded random ordered permutations each.
#  * ict_real joins as a SOURCE (driven by ICT-live test coefficients; all 20
#    ids source-capable via ict_id_vecs_test.pt row<->sorted-name mapping).
#  * Similarity-Procrustes alignment of the returned mesh to the driving frame
#    before scoring (uniform across methods, following Cha et al. 2025).
# Reps: mf:12, biwi:M5, coma:FaceTalk_170809_00138_TA, ict_real:m00 (unseen).
import sys, os, json, argparse, itertools
import numpy as np, torch
from pathlib import Path
_REPO = '/source/inyup/NeuralFacialAnimation'
os.chdir(_REPO); sys.path.insert(0, _REPO)
sys.path.insert(0, f'{_REPO}/tools/vis'); sys.path.insert(0, f'{_REPO}/tools/eval')
import viser_debug as vd

ap = argparse.ArgumentParser()
ap.add_argument('--methods', nargs='+', default=['hlbs', 'nfr', 'nfs', 'dt'])
ap.add_argument('--frames', type=int, default=6)
ap.add_argument('--n-perm', type=int, default=8, help='random 3/4-hop chains each')
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--out', default='eval_out/cycle_v3')
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
if 'ict_real' in topos:
    _v20 = torch.load('ict_face_pt/ict_id_vecs_test.pt').numpy()
    _tr = topos['ict_real']
    _tr._ict_id_vecs = {nm: _v20[i] for i, nm in enumerate(_tr.id_names)}
ictlive = np.load('data/ICT_live_100/expression_vecs_test.npy').astype(np.float32)
os.makedirs(args.out, exist_ok=True)

REPS = ['mf:12', 'biwi:M5', 'coma:FaceTalk_170809_00138_TA', 'ict_real:m00']
HELPER_SET = set(helper_idx) if helper_idx else set()


def _b(a):
    return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).to(dev).float()


def procrustes_align(pred, gt):
    P = pred - pred.mean(0); G = gt - gt.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ G)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    s = (S * np.array([1, 1, d])).sum() / (P ** 2).sum()
    return (s * (R @ P.T)).T + gt.mean(0)


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


def _markers(td, i):
    jp = cache.get(i, td)['joint_pos_pred']
    return jp[[j for j in range(jp.shape[0]) if j not in HELPER_SET]]


def hop(meth, td_s, si, src_def, td_t, ti):
    if meth == 'hlbs':
        return hop_ours(td_s, si, src_def, td_t, ti)
    if meth == 'dt':
        from baseline_dt_cross import dt_cross_transfer
        sb = cache.get(si, td_s); tb = cache.get(ti, td_t)
        return dt_cross_transfer((td_s.name, si, td_t.name, ti),
                                 sb['neu_v'], src_def, td_s.faces, _markers(td_s, si),
                                 tb['neu_v'], td_t.faces, _markers(td_t, ti))
    sb = cache.get(si, td_s); tb = cache.get(ti, td_t)
    return baseline.retarget(method=meth, src_td=td_s, src_idx=si,
                             src_neu_v=sb['neu_v'], src_def_v=src_def,
                             tgt_td=td_t, tgt_idx=ti, tgt_neu_v=tb['neu_v'])


def resolve(spec):
    ds, idn = spec.split(':', 1)
    td = topos[ds]
    return td, (int(idn) if idn.isdigit() else td.id_names.index(idn))


def source_frames(spec, k):
    td, idx = resolve(spec)
    id_name = td.id_names[idx]
    neu = cache.get(idx, td)['neu_v']
    if spec.startswith('ict_real'):
        step = max(1, len(ictlive) // 20)
        cand = [(float(np.linalg.norm(td.apply_exp(id_name, ictlive[f]) - neu, axis=1).mean()),
                 ('ICTlive', f)) for f in range(0, len(ictlive), step)]
        cand.sort(reverse=True)
        return [c for _, c in cand[:k]]
    cand = []
    for cl in td.clips_for(id_name)[:14]:
        n = td.n_frames(id_name, cl)
        for fi in range(0, n, max(1, n // 12)):
            v = td.apply_exp(id_name, (cl, fi))
            if v is None: continue
            cand.append((float(np.linalg.norm(v - neu, axis=1).mean()), (cl, fi)))
    cand.sort(reverse=True)
    out = []
    for m, (cl, fi) in cand:
        if all(c != cl or abs(f - fi) >= 15 for c, f in out):
            out.append((cl, fi))
        if len(out) == k: break
    return out


def get_def(spec, key):
    td, idx = resolve(spec)
    id_name = td.id_names[idx]
    if key[0] == 'ICTlive':
        return td.apply_exp(id_name, ictlive[key[1]])
    return td.apply_exp(id_name, key)


rng = np.random.RandomState(args.seed)
chains2 = [[a, b] for a, b in itertools.permutations(REPS, 2)]
p3 = list(itertools.permutations(REPS, 3)); rng.shuffle(p3)
p4 = list(itertools.permutations(REPS, 4)); rng.shuffle(p4)
chains3 = [list(c) for c in p3[:args.n_perm]]
chains4 = [list(c) for c in p4[:args.n_perm]]
print(f'2-hop pairs {len(chains2)} | 3-hop {len(chains3)} | 4-hop {len(chains4)}', flush=True)

FRAMES = {r: source_frames(r, args.frames) for r in REPS}
for r in REPS:
    print(r, '->', FRAMES[r], flush=True)

rows = []
for depth, chains in ((2, chains2), (3, chains3), (4, chains4)):
    for chain in chains:
        A = resolve(chain[0])
        neuA = cache.get(A[1], A[0])['neu_v']
        maskA = vd._face_vert_mask(neuA)
        if maskA is None or maskA.dtype != bool or maskA.sum() == 0:
            maskA = np.ones(len(neuA), bool)
        diagA = float(np.linalg.norm(neuA.max(0) - neuA.min(0)))
        nodes = [resolve(s) for s in chain]
        for key in FRAMES[chain[0]]:
            gtA = get_def(chain[0], key)
            for meth in args.methods:
                try:
                    cur = gtA; src = nodes[0]
                    for d in range(1, len(nodes)):
                        cur = hop(meth, src[0], src[1], cur, nodes[d][0], nodes[d][1])
                        if cur is None: raise RuntimeError('hop none')
                        src = nodes[d]
                    back = hop(meth, nodes[-1][0], nodes[-1][1], cur, A[0], A[1])
                    back = procrustes_align(np.asarray(back, np.float64),
                                            np.asarray(gtA, np.float64))
                    e = np.linalg.norm(back - gtA, axis=1)
                    rows.append(dict(k=depth + 0, chain=' > '.join(chain),
                                     clip=str(key[0]), frame=int(key[1]), method=meth,
                                     err_face=float(e[maskA].mean() / diagA)))
                except Exception as ex:
                    print('EXC', depth, chain[0], meth, str(ex)[:100], flush=True)
        print(f'[k={depth}] {" > ".join(chain)} done', flush=True)
        json.dump(rows, open(f'{args.out}/cycle_v3_rows.json', 'w'), indent=0, default=float)

print('\n== mean err by method x k ==')
for meth in args.methods:
    line = f'{meth:<5}'
    for k in (2, 3, 4):
        rs = [r['err_face'] for r in rows if r['method'] == meth and r['k'] == k]
        line += f'  k{k} {np.mean(rs):.5f}(n={len(rs)})' if rs else f'  k{k} -'
    print(line)
print('CYCLE_V3_DONE', flush=True)
