"""GNM FULL HEAD (teeth/tongue/mouth-sock included) target + v4b-style clips.
- All 17,821 verts, all faces. Normalization reuses the skin-version constants
  (c0/sg from skin_exterior) so gnm_full co-locates with existing gnm_mean.
- Clips: same jaw-dvec recipe as v4b + mild tongue activation (basis moves
  lower teeth 14mm / tongue 11mm per 2cm jaw drop -> valid interior GT).
Outputs: export/gnm_full.obj + export/gnm_full_clips/synth00..04 + validate.png
"""
import numpy as np, os, sys, shutil

d = np.load("GNM/gnm/shape/data/versions/v3_0/gnm_head.npz", allow_pickle=True)
tpl = d["template_vertex_positions"].astype(np.float64)
eb = d["expression_basis"].astype(np.float64)
tris = d["triangles"].astype(np.int64)
en = [str(x) for x in d["expression_names"]]
gnames = [str(x) for x in d["vertex_group_names"]]; vg = d["vertex_groups"]
gm = lambda n: vg[gnames.index(n)] > 0.5
gidx = lambda *ns: np.where(np.any(np.stack([gm(n) for n in ns]), 0))[0]
NOSEI = gidx("nose_region", "left_infraorbital_region", "right_infraorbital_region")
reg = {"left_eye_region": gidx("expression_basis_left_eye"),
       "right_eye_region": gidx("expression_basis_right_eye"),
       "lower_face_region": gidx("expression_basis_mouth_nose_ears", "lower_teeth_and_gums", "tongue")}
sig = np.zeros(len(en)); wt = np.zeros(len(en))
for i, n in enumerate(en):
    for rname, ridx in reg.items():
        if n.startswith(rname):
            sig[i] = np.linalg.norm(eb[i][ridx])
            wt[i] = 1.0 if rname == "lower_face_region" else 0.3
    if n.startswith("tongue") and n != "tongue_mean":
        sig[i] = np.linalg.norm(eb[i]); wt[i] = 0.25       # mild tongue motion
nose_frac = np.zeros(len(en))
for i in np.where(wt > 0)[0]:
    nose_frac[i] = np.linalg.norm(eb[i][NOSEI]) / (sig[i] + 1e-12)
nosey = nose_frac > 0.25

# jaw direction (precomputed on skin; coefficient-space -> valid for full head)
dvec = np.load("export/jawdir_test/dvec.npy"); lfidx = np.load("export/jawdir_test/lf_idx.npy")
cj = np.zeros(len(en)); cj[lfidx] = dvec
keepS = np.where(gm("skin_exterior"))[0]
chinS = (gm("chin_region") | gm("lower_lip_region"))[keepS]
D = np.einsum("i,ivk->vk", cj, eb)
A1CM = 0.01 / (-(D[keepS][chinS, 1].mean()))

# normalization constants from SKIN version (co-locate with gnm_mean)
sys.path.insert(0, "/source/inyup/NeuralFacialAnimation")
from utils.remesh_utils import ICT_face_model
ict = ICT_face_model(); Vi = np.asarray(ict.neutral_verts)
V0S = tpl[keepS]
sg = (Vi[:, 1].max() - Vi[:, 1].min()) / (V0S[:, 1].max() - V0S[:, 1].min())
c0 = V0S.mean(0); ci = Vi.mean(0)
A = lambda v: (v - c0) * sg + ci

# export full-head template obj
def write_obj(p, V, F):
    with open(p, "w") as f:
        for v in V: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in F: f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
write_obj("export/gnm_full.obj", A(tpl), tris)
print(f"gnm_full.obj V={len(tpl)} F={len(tris)}")

G = 8.0
rng = np.random.RandomState(7); T = 300; KEY = 12
nk = T // KEY + 2
shutil.rmtree("export/gnm_full_clips", ignore_errors=True)
os.makedirs("export/gnm_full_clips")
V0 = tpl

def decode(coef):
    return tpl + np.einsum("i,ivk->vk", coef, eb)

def write_clip(name, raws):
    outdir = f"export/gnm_full_clips/{name}"; os.makedirs(outdir)
    md = []
    for t in range(T):
        f = t / KEY; k0 = int(f); a = (1 - np.cos((f - k0) * np.pi)) / 2
        coef = raws[k0] * (1 - a) + raws[min(k0 + 1, nk - 1)] * a
        V = decode(coef)
        np.save(f"{outdir}/{t:04d}.npy", A(V).astype(np.float32))
        md.append(float(np.abs(V - V0).max()) * sg)
    print(f"{name} peak-disp {max(md):.3f}")

def base_keys(scale=1.0, clip=2.0):
    keys = np.clip(rng.randn(nk, len(en)), -clip, clip) * wt[None, :] * scale
    keys[:, nosey] = np.clip(keys[:, nosey], -1.0, 1.0)
    keys[0] = 0; keys[-1] = 0
    return keys

for c in range(3):
    keys = base_keys()
    raw = keys * sig[None, :] * G
    for k in range(1, nk - 1, 3):
        raw[k] += cj * (rng.uniform(0.8, 2.8) * A1CM)
    write_clip(f"synth{c:02d}", raw)

keys = base_keys(scale=0.3)
raw = keys * sig[None, :] * G
for k in range(1, nk - 1):
    phase = (k - 1) % 4
    amp_cm = 0.5 + 2.5 * (k / (nk - 1))
    raw[k] += cj * (amp_cm * A1CM) * (0.0, 0.6, 1.0, 0.6)[phase]
write_clip("synth03", raw)

keys = base_keys(scale=0.5, clip=1.2)
raw = keys * sig[None, :] * G
jaw_seq = [1.4, 0.0, 2.2, 0.8, 2.6, 0.0, 1.8, 1.0]
for k in range(1, nk - 1):
    raw[k] += cj * (jaw_seq[(k - 1) % len(jaw_seq)] * A1CM)
write_clip("synth04", raw)
print("GEN_DONE")
