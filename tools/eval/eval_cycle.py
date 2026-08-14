# E2b pilot: cycle-consistency curve (ours only).
# Chain A=mf12 -> B=biwi M4 -> C=coma 00138 -> D=ict_real m00 -> back to A.
# Cycle lengths: 1(self A->A), 2(A->B->A), 3(A->B->C->A), 4(A->B->C->D->A).
# Error = diag-normalized vertex L2 vs A's driving frame (face mask + all).
import sys, os, json
import numpy as np, torch
from pathlib import Path
_REPO = '/source/inyup/NeuralFacialAnimation'
os.chdir(_REPO); sys.path.insert(0, _REPO); sys.path.insert(0, f'{_REPO}/tools/vis')
import viser_debug as vd

dev = 'cuda'
model, rig, opts, nfs_dir, helper_idx, geo_per_topo = vd._build_model(
    Path('ckpts_hlbs/combined500_import'), dev)
needs_nfs = getattr(opts, 'nfs_feat_dir', None) is not None
from utils.remesh_utils import ICT_face_model
ict = ICT_face_model(base_dir=_REPO)
topos = vd._build_topos(ict, nfs_dir, geo_per_topo)
cache = vd.IdentityCache(model, rig, dev, model_needs_nfs=needs_nfs)

mf = topos['mf']; biwi = topos['biwi']; coma = topos['coma']; ictr = topos['ict_real']
A = (mf, 12)
B = (biwi, biwi.id_names.index('M4'))
C = (coma, coma.id_names.index('FaceTalk_170809_00138_TA'))
D = (ictr, 0)   # m00
print('chain:', [t[0].name + ':' + t[0].id_names[t[1]] for t in (A, B, C, D)], flush=True)

def _b(a):
    return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).to(dev).float()

@torch.no_grad()
def hop(src, src_neu, src_def, tgt):
    """One retarget hop with raw arrays. Returns tgt pred verts [V,3] np."""
    td_s, _ = src; td_t, ti = tgt
    tb = cache.get(ti, td_t)
    sn = vd._per_vertex_normal(src_neu, td_s.faces.astype(np.int64))
    dn = vd._per_vertex_normal(src_def, td_s.faces.astype(np.int64))
    tn = vd._per_vertex_normal(tb['neu_v'], td_t.faces.astype(np.int64))
    pred, _, _ = model.retarget(
        src_neu_vert=_b(src_neu), src_neu_norm=_b(sn),
        src_def_vert=_b(src_def), src_def_norm=_b(dn),
        tgt_neu_vert=_b(tb['neu_v']), tgt_neu_norm=_b(tn),
        tgt_nfs_feat=tb['nfs'], tgt_dist_sq_geo=tb['dist_sq_geo'],
        return_joints=True)
    return pred[0].cpu().numpy().astype(np.float32)

# A frames: curated extremes + mid-motion
FRAMES = [('EXP_jaw005', 357), ('EXP_lip001', 306), ('EXP_lip003', 164),
          ('EXP_cheek001', 286), ('EXP_jaw003', 420), ('EXP_free_face', 146),
          ('EXP_lip002', 513), ('EXP_cheek002', 266)]
neuA = cache.get(A[1], A[0])['neu_v']
maskA = vd._face_vert_mask(neuA)
diagA = float(np.linalg.norm(neuA.max(0) - neuA.min(0)))

rows = []
for clip, pos in FRAMES:
    o = cache.retarget(A[0], A[1], (f'ROM/test/{clip}', pos), A[0], A[1])
    if not o['model_ran']:
        print('SKIP', clip, pos); continue
    gtA = o['src_def_v']
    # 1-hop: A->A (self)
    e1 = o['tgt_pred_v']
    # 2-hop: A->B->A
    pB = hop(A, o['src_neu_v'], gtA, B)
    e2 = hop(B, cache.get(B[1], B[0])['neu_v'], pB, A)
    # 3-hop: A->B->C->A
    pC = hop(B, cache.get(B[1], B[0])['neu_v'], pB, C)
    e3 = hop(C, cache.get(C[1], C[0])['neu_v'], pC, A)
    # 4-hop: A->B->C->D->A
    pD = hop(C, cache.get(C[1], C[0])['neu_v'], pC, D)
    e4 = hop(D, cache.get(D[1], D[0])['neu_v'], pD, A)
    def err(p):
        e = np.linalg.norm(p - gtA, axis=1)
        return float(e[maskA].mean() / diagA), float(e.mean() / diagA)
    r = {'clip': clip, 'frame': pos}
    for k, p in (('h1', e1), ('h2', e2), ('h3', e3), ('h4', e4)):
        r[k + '_face'], r[k + '_all'] = err(p)
    # motion amplitude sanity: how much motion survives at each stage (vs A disp)
    dispA = float(np.linalg.norm(gtA - neuA, axis=1)[maskA].mean())
    r['ampA'] = dispA / diagA
    rows.append(r)
    print('%-16s f%03d  h1 %.5f  h2 %.5f  h3 %.5f  h4 %.5f (face, diag-norm)' %
          (clip, pos, r['h1_face'], r['h2_face'], r['h3_face'], r['h4_face']), flush=True)

os.makedirs('eval_out', exist_ok=True)
json.dump(rows, open('eval_out/pilot_cycle.json', 'w'), indent=1)
m = {k: float(np.mean([r[k + '_face'] for r in rows])) for k in ('h1', 'h2', 'h3', 'h4')}
print('\n== mean face err by cycle length ==')
print('  1-hop(self) %.5f\n  2-hop %.5f (x%.2f)\n  3-hop %.5f (x%.2f)\n  4-hop %.5f (x%.2f)'
      % (m['h1'], m['h2'], m['h2'] / m['h1'], m['h3'], m['h3'] / m['h1'], m['h4'], m['h4'] / m['h1']))
print('PILOT_DONE', flush=True)
