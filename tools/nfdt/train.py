# tools/nfdt/train.py — NFDT reimplementation (Chandran et al., EG'25 short).
#
# Decoder-only shape transformer: per-vertex displacement tokens
#   template token: [T_def - T_0, T_0]   (template MLP)
#   target   token: [S_0  - T_0, T_0]    (target MLP)
# -> 4 XCA (cross-covariance attention, XCiT-style; LPI omitted) blocks, d=256
# -> output MLP on target tokens -> displacement added to S_0.  L2 loss, Adam.
#
# Same-topology (mf) data from tools/nfdt/datagen.py shards. Last clip of each
# train id held out for validation.
import os, sys, json, glob, time, argparse
import numpy as np
import torch
import torch.nn as nn

R = '/source/inyup/NeuralFacialAnimation'
os.chdir(R)
ap = argparse.ArgumentParser()
ap.add_argument('--data', default='nfdt_data')
ap.add_argument('--out', default='ckpts_nfdt')
ap.add_argument('--dim', type=int, default=256)
ap.add_argument('--blocks', type=int, default=4)
ap.add_argument('--heads', type=int, default=8)
ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--val-every', type=int, default=2)
args = ap.parse_args()
dev = 'cuda'
os.makedirs(args.out, exist_ok=True)

T0 = torch.tensor(np.load(f'{args.data}/template_mean.npy'), device=dev)  # [V,3]
V = T0.shape[0]
meta = json.load(open(f'{args.data}/meta.json'))

# split: last clip per id -> val
by_id = {}
for p in meta['pairs']:
    by_id.setdefault(p['id'], []).append(p)
train_shards, val_shards = [], []
for idn, ps in by_id.items():
    val_shards.append(ps[-1]['shard'])
    train_shards += [p['shard'] for p in ps[:-1]]
print(f'shards: train {len(train_shards)} val {len(val_shards)} V={V}')

import pickle
tplm = pickle.load(open('utils/templates/mf_templates.pkl', 'rb'))
NEU = {k: torch.tensor(np.asarray(v, np.float32), device=dev)
       for k, v in tplm.items() if k != 'face'}


class Shard:
    def __init__(self, sid):
        z = np.load(f'{args.data}/shard_{sid:04d}.npz')
        self.tpl = torch.tensor(z['tpl_def'])
        self.gt = torch.tensor(z['tgt_gt'])
        self.idn = str(z['tgt_id'])

_CACHE = {}
def shard(sid):
    if sid not in _CACHE:
        if len(_CACHE) > 40:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[sid] = Shard(sid)
    return _CACHE[sid]


class XCA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.tau = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x):                       # [B,N,d]
        B, N, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        def sp(t):
            return t.view(B, N, self.h, d // self.h).permute(0, 2, 3, 1)  # [B,h,dh,N]
        q, k, v = sp(q), sp(k), sp(v)
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.tau          # [B,h,dh,dh]
        attn = attn.softmax(-1)
        out = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.att = XCA(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.att(self.n1(x))
        return x + self.ff(self.n2(x))


class NFDT(nn.Module):
    def __init__(self, d, blocks, heads):
        super().__init__()
        def mlp():
            return nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.emb_t = mlp(); self.emb_s = mlp()
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(blocks)])
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, tpl_def, tgt_neu):        # [B,V,3],[B,V,3]
        t_tok = self.emb_t(torch.cat([tpl_def - T0, T0.expand_as(tpl_def)], -1))
        s_tok = self.emb_s(torch.cat([tgt_neu - T0, T0.expand_as(tgt_neu)], -1))
        x = torch.cat([t_tok, s_tok], 1)
        for b in self.blocks:
            x = b(x)
        disp = self.out(x[:, t_tok.shape[1]:])
        return tgt_neu + disp


model = NFDT(args.dim, args.blocks, args.heads).to(dev)
n_par = sum(p.numel() for p in model.parameters())
print(f'NFDT params {n_par/1e6:.2f}M')
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
rng = np.random.RandomState(0)


def batch_from(shards, bs):
    tp, gt, ne = [], [], []
    for _ in range(bs):
        s = shard(int(rng.choice(shards)))
        f = rng.randint(len(s.tpl))
        tp.append(s.tpl[f]); gt.append(s.gt[f]); ne.append(NEU[s.idn].cpu())
    return (torch.stack(tp).to(dev), torch.stack(gt).to(dev),
            torch.stack([n for n in ne]).to(dev))


@torch.no_grad()
def validate():
    model.eval(); errs = []
    for sid in val_shards:
        s = shard(sid)
        for f in range(0, len(s.tpl), 4):
            pred = model(s.tpl[f:f+1].to(dev), NEU[s.idn][None])
            errs.append(torch.norm(pred - s.gt[f:f+1].to(dev), dim=-1).mean().item())
    model.train()
    return float(np.mean(errs))


steps_per_ep = max(1, sum(1 for _ in train_shards) * 40 // args.batch)
best = 1e9
log = open(f'{args.out}/log.txt', 'a')
for ep in range(args.epochs):
    t0 = time.time(); tot = 0.0
    for it in range(steps_per_ep):
        tp, gt, ne = batch_from(train_shards, args.batch)
        pred = model(tp, ne)
        loss = ((pred - gt) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    msg = f'[ep {ep:03d}] train {tot/steps_per_ep:.6e}  ({time.time()-t0:.0f}s)'
    if ep % args.val_every == 0:
        v = validate()
        msg += f'  val_vert_l2 {v:.6f}'
        if v < best:
            best = v
            torch.save({'model': model.state_dict(), 'ep': ep, 'val': v,
                        'args': vars(args)}, f'{args.out}/nfdt_best.pth')
            msg += '  *best*'
    print(msg, flush=True); log.write(msg + '\n'); log.flush()
    torch.save({'model': model.state_dict(), 'ep': ep}, f'{args.out}/nfdt_last.pth')
print('TRAIN_DONE best', best, flush=True)
