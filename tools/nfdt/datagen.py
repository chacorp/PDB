# tools/nfdt/datagen.py — NFDT training data (original-paper recipe on our mf).
#
# Template = mf-mean (same topology as all mf ids -> delta transfer is exact).
# Cleanup projection (patch-solver substitute): global mf expression PCA built
# from train-id ROM/SEN deltas; template expression = mf_mean + PCA_proj(delta).
# Pair: input (template deformed [pseudo], target id neutral) -> output real frame.
#
# Split: train ids (11, not test id12 / val id) for training pairs; held-out
# clips of train ids + all clips of test/val ids reserved for evaluation (E2-B).
import os, sys, glob, pickle, json
import numpy as np

R = '/source/inyup/NeuralFacialAnimation'
os.chdir(R); sys.path.insert(0, R)
OUT = 'nfdt_data'
os.makedirs(OUT, exist_ok=True)

tpl = pickle.load(open('utils/templates/mf_templates.pkl', 'rb'))
ids = [k for k in tpl.keys() if k != 'face']
F = np.asarray(tpl['face'], np.int64)
neu = {k: np.asarray(tpl[k], np.float32) for k in ids}
mf_mean = np.mean([neu[k] for k in ids], 0).astype(np.float32)
np.save(f'{OUT}/template_mean.npy', mf_mean)
np.save(f'{OUT}/faces.npy', F)

TEST_ID = ids[12]                     # m--20190828 (test)
VAL_ID = [k for k in ids if '20190529-1300' in k.replace('--', '-')] or [ids[11]]
VAL_ID = VAL_ID[0]
train_ids = [k for k in ids if k not in (TEST_ID, VAL_ID)]
print('train ids', len(train_ids), '| heldout:', TEST_ID[:20], VAL_ID[:20])

BASES = ['/data/sihun/multiface_align', '/data/inyup/multiface_align']

def clips_of(idn):
    out = []
    for b in BASES:
        for cat in ('ROM', 'SEN'):
            for mode in ('train', 'test', 'val'):
                p = f'{b}/{cat}/{mode}/vertices_npy/{idn}'
                if os.path.isdir(p):
                    for c in sorted(glob.glob(p + '/*')):
                        if os.path.isdir(c):
                            fr = sorted(glob.glob(c + '/*.npy'))
                            if len(fr) > 5:
                                out.append((f'{cat}/{mode}/{os.path.basename(c)}', fr))
    # dedup by clip name across bases
    seen = {}
    for nm, fr in out:
        seen.setdefault(nm, fr)
    return list(seen.items())

# ── pass 1: expression PCA from train-id deltas (subsampled frames) ────
from sklearn.decomposition import PCA
samples = []
for idn in train_ids:
    for nm, fr in clips_of(idn):
        for f in fr[::6]:
            v = np.load(f).astype(np.float32)
            samples.append((v - neu[idn]).reshape(-1))
X = np.stack(samples)
print('PCA samples', X.shape)
K = 150
pca = PCA(n_components=K, svd_solver='randomized', random_state=0)
pca.fit(X)
np.savez(f'{OUT}/exp_pca.npz', mean=pca.mean_.astype(np.float32),
         comps=pca.components_.astype(np.float32),
         var=pca.explained_variance_ratio_.astype(np.float32))
print('PCA var captured: %.4f' % pca.explained_variance_ratio_.sum())

def project(delta):
    d = delta.reshape(1, -1)
    z = (d - pca.mean_) @ pca.components_.T
    return (pca.mean_ + z @ pca.components_).reshape(-1, 3).astype(np.float32)

# ── pass 2: emit pair shards (train split; every 2nd frame) ────────────
meta = {'template': 'mf_mean', 'K': K, 'train_ids': train_ids,
        'test_id': TEST_ID, 'val_id': VAL_ID, 'pairs': []}
sid = 0
for idn in train_ids:
    cl = clips_of(idn)
    for ci, (nm, fr) in enumerate(cl):
        tgt_frames, tpl_defs = [], []
        for f in fr[::2]:
            v = np.load(f).astype(np.float32)
            delta = v - neu[idn]
            tpl_defs.append(mf_mean + project(delta))
            tgt_frames.append(v)
        arr_t = np.stack(tpl_defs); arr_g = np.stack(tgt_frames)
        np.savez_compressed(f'{OUT}/shard_{sid:04d}.npz',
                            tpl_def=arr_t, tgt_gt=arr_g, tgt_id=idn, clip=nm)
        meta['pairs'].append({'shard': sid, 'id': idn, 'clip': nm, 'n': len(arr_t)})
        sid += 1
    print(f'{idn[:24]}: {len(cl)} clips done', flush=True)
json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=1)
n_pairs = sum(p['n'] for p in meta['pairs'])
print('DATAGEN_DONE shards', sid, 'pairs', n_pairs, flush=True)
