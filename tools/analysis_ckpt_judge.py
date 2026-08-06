"""wref@200 5-metric interim judgment vs combo700 / xcycle670.
All three models measured with IDENTICAL code -> relative comparison authoritative.
Outputs: wref_ep200_report.txt + wref_ep200_ab.png in repo root.
"""
import sys, os, json, pickle
import numpy as np, torch

sys.path.insert(0, "."); sys.path.insert(0, "tools/vis")
import viser_debug as VD
from pathlib import Path

dev = torch.device("cuda")
PFX = os.environ.get("JUDGE_PREFIX", "wref_ep200")
REP = open(f"{PFX}_report.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); REP.write(s + "\n"); REP.flush()

# args: tag:dir[:pth_override] triplets, e.g. combined500:ckpts_hlbs/2026-...:model_hlbs_500.pth
CKPTS = []
for spec in sys.argv[1:]:
    parts = spec.split(":")
    CKPTS.append((parts[0], parts[1], parts[2] if len(parts) > 2 else None))

from utils.remesh_utils import ICT_face_model
ict = ICT_face_model(base_dir=".")
eb = np.asarray(ict.exp_basis)
jaw = int(np.argmin(eb[:, :, 1].min(axis=1)))          # jawOpen
vecs = np.load("ict_face_pt/random_identity_vecs.npy").astype(np.float32)
coeff = np.zeros(eb.shape[0], dtype=np.float32); coeff[jaw] = 1.0
src_neu = np.asarray(ict.apply_coeffs(vecs[0], exp_coeffs=None)[0], dtype=np.float32)
src_def = np.asarray(ict.apply_coeffs(vecs[0], exp_coeffs=coeff)[0], dtype=np.float32)
F_ict = ict.faces.astype(np.int64)
ict_neu_mean = np.asarray(ict.neutral_verts, dtype=np.float32)   # mean identity

tpl = pickle.load(open("utils/templates/mf_templates.pkl", "rb"))
ids = [k for k in tpl.keys() if k != "face"]
F_mf = np.asarray(tpl["face"], dtype=np.int64)
mf0_v = np.asarray(tpl[ids[0]], dtype=np.float32)
mf12_v = np.asarray(tpl[ids[12]], dtype=np.float32)
mf0_f = np.load(f"diff3f_feat_raw/{ids[0]}_nfs_feat.npy").astype(np.float32)
mf12_f = np.load(f"diff3f_feat_raw/{ids[12]}_nfs_feat.npy").astype(np.float32)
ict0_feat = np.load("diff3f_feat_raw/ict_000_nfs_feat.npy").astype(np.float32)
ict0_neu = np.asarray(ict.apply_coeffs(vecs[0], exp_coeffs=None)[0], dtype=np.float32)

pvn = VD._per_vertex_normal
def b(a): return torch.from_numpy(a).unsqueeze(0).to(dev)

# Diff3F cosine NN match: mf12 verts -> ict0 verts (fixed across models)
with torch.no_grad():
    A = torch.tensor(mf12_f, device=dev); A = A / (A.norm(dim=-1, keepdim=True) + 1e-8)
    Bt = torch.tensor(ict0_feat, device=dev); Bt = Bt / (Bt.norm(dim=-1, keepdim=True) + 1e-8)
    nn_idx = []
    for i in range(0, A.shape[0], 2048):
        nn_idx.append((A[i:i+2048] @ Bt.T).argmax(dim=-1))
    nn_idx = torch.cat(nn_idx).cpu().numpy()
# face subset: frontal ict verts (z above 60th pct) & above chin (y > 20th pct)
zthr = np.quantile(ict0_neu[:, 2], 0.60); ythr = np.quantile(ict0_neu[:, 1], 0.20)
face_pair = (ict0_neu[nn_idx, 2] > zthr) & (ict0_neu[nn_idx, 1] > ythr)

# nose region on mf0 (for entropy)
cx = (mf0_v[:, 0] > -0.15) & (mf0_v[:, 0] < 0.15)
nose_tip = mf0_v[cx][np.argmax(mf0_v[cx][:, 2])]
nreg = np.where(np.linalg.norm(mf0_v - nose_tip, axis=1) < 0.35)[0]

results = {}
renders = []
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader,
    DirectionalLights, TexturesVertex)
import imageio.v2 as imageio
Rr, Tt = look_at_view_transform(dist=2.4, elev=0, azim=0)
cams = FoVPerspectiveCameras(device=dev, R=Rr, T=Tt, fov=30)
lights = DirectionalLights(device=dev, direction=[[0, 0.2, 1]],
    ambient_color=[[0.45, 0.45, 0.48]], diffuse_color=[[0.55, 0.55, 0.6]],
    specular_color=[[0.05, 0.05, 0.05]])
rast = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)
rend = MeshRenderer(rasterizer=MeshRasterizer(cameras=cams, raster_settings=rast),
                    shader=HardPhongShader(device=dev, cameras=cams, lights=lights))
def rd(v, F):
    c = v.mean(0); s = float(np.abs(v - c).max())
    vv = torch.tensor((v - c) / s, device=dev, dtype=torch.float32)
    mesh = Meshes(verts=[vv], faces=[torch.tensor(F, device=dev)],
                  textures=TexturesVertex(torch.tensor([[0.78, 0.85, 0.95]], device=dev).expand(1, len(vv), 3)))
    return (rend(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

for tag, ck, pth_override in CKPTS:
    ckd = Path(ck)
    model, rig, opts, nfs_dir, helper_idx, geo_per_topo = VD._build_model(ckd, dev)
    if pth_override:
        sd = torch.load(ckd / pth_override, map_location="cpu", weights_only=False)
        model.load_state_dict(sd); model.eval()
        P(f"[{tag}] loaded override {pth_override}")
    jn = rig.joint_names
    helpers = set(helper_idx or [])
    gd_mf = geo_per_topo.get("mf")
    gd_ict = geo_per_topo.get("ict")

    def dsg_for(v, gd):
        if gd is None: return None
        g = gd.t()
        if g.shape[0] != v.shape[0]: g = g[: v.shape[0]]
        return (g.unsqueeze(0)) ** 2

    def pose_R(sneu, sdef):
        with torch.no_grad():
            delta = b(sdef) - b(sneu)
            si = torch.cat([b(sneu), b(pvn(sneu, F_ict))], dim=-1)
            di = torch.cat([delta, b(pvn(sdef, F_ict)), si], dim=-1)
            z = model.lbs_exp_z_model(di); zf = z.squeeze(1) if z.dim() == 3 else z
            po = model.lbs_pose_model(zf.unsqueeze(1)).squeeze(1)
            J = model.num_joints
            return model._rot6d(po[:, :J*6].reshape(J, 6)).reshape(J, 3, 3)

    def W_of(v, feat, F, gd):
        with torch.no_grad():
            tsi, tad = model._prepare_feat(b(v), b(pvn(v, F)), b(feat))
            Binv, jp = model._get_bind_pose(tsi, adain_input=tad)
            W, _ = model._get_skinning_weights(tsi, adain_input=tad, source_vert=b(v),
                                               joint_pos=jp, dist_sq_override=dsg_for(v, gd))
            return W[0].cpu().numpy()

    # ── ① rotation profile (neutral-relative) ─────────────────────────
    Rn = pose_R(src_neu, src_neu); Re = pose_R(src_neu, src_def)
    Rel = torch.bmm(Re, Rn.transpose(1, 2))
    ang = torch.rad2deg(torch.acos(((Rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)))
    top = torch.argsort(-ang)[:6]
    nosej = [i for i, n in enumerate(jn) if "Nose" in n]
    nose_max = float(ang[nosej].max()) if nosej else 0.0
    P(f"[{tag}] ① jawOpen top rotations:", [(jn[i], round(float(ang[i]), 1)) for i in top.tolist()])
    P(f"[{tag}] ① nose-joint max rotation: {nose_max:.1f} deg")

    # ── ② nose W entropy on mf0 ───────────────────────────────────────
    W0 = W_of(mf0_v, mf0_f, F_mf, gd_mf)
    Wn = W0[nreg]
    ent = float((-Wn * np.log(Wn + 1e-9)).sum(-1).mean())
    tj = np.argsort(-Wn.mean(0))[:4]
    P(f"[{tag}] ② nose W entropy: {ent:.3f} | top:", [(jn[i], round(float(Wn.mean(0)[i]), 3)) for i in tj.tolist()])

    # ── ③ extreme A/B renders (src fixed) ─────────────────────────────
    with torch.no_grad():
        row = []
        for tv, tf in ((mf0_v, mf0_f), (mf12_v, mf12_f)):
            rv, _, _ = model.retarget(
                src_neu_vert=b(src_neu), src_neu_norm=b(pvn(src_neu, F_ict)),
                src_def_vert=b(src_def), src_def_norm=b(pvn(src_def, F_ict)),
                tgt_neu_vert=b(tv), tgt_neu_norm=b(pvn(tv, F_mf)),
                tgt_nfs_feat=b(tf), tgt_dist_sq_geo=dsg_for(tv, gd_mf), return_joints=True)
            row.append(rd(rv[0].cpu().numpy(), F_mf))
    renders.append((tag, row))

    # ── ⑤ agreement mf12<->ict0 (same-code reimpl; relative comparison) ──
    W_mf = W_of(mf12_v, mf12_f, F_mf, gd_mf)
    W_ic = W_of(ict0_neu, ict0_feat, F_ict, gd_ict)
    am = W_mf.argmax(-1); ai = W_ic.argmax(-1)[nn_idx]
    agree_all = float((am == ai).mean()) * 100
    agree_face = float((am[face_pair] == ai[face_pair]).mean()) * 100
    hmask = np.array([(a in helpers) or (c in helpers) for a, c in zip(am, ai)])
    agree_h = float((am[hmask] == ai[hmask]).mean()) * 100 if hmask.any() else float("nan")
    l1 = float(np.abs(W_mf - W_ic[nn_idx]).sum(-1).mean())
    P(f"[{tag}] ⑤ agreement all {agree_all:.1f}% | face {agree_face:.1f}% | helper-involved {agree_h:.1f}% | W-L1 {l1:.3f}")

    del model; torch.cuda.empty_cache()

# grid: rows per model [src | mf0 | mf12]
src_img = rd(src_def, F_ict)
rows = [np.concatenate([src_img] + row, 1) for _, row in renders]
imageio.imwrite(f"{PFX}_ab.png", np.concatenate(rows, 0))
P(f"③ A/B grid saved: {PFX}_ab.png (rows in CKPTS order)")
P("④ val: see log grep below")
REP.close()
print("JUDGE_DONE", flush=True)
