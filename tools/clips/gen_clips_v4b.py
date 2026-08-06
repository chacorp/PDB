"""GNM synthetic clips v4b: mouth-extreme via solved jaw-open direction.
jaw dvec = argmax chin-drop with nose-deformation penalty (regularized LS),
validated clean up to ~3cm mean chin drop (no nose crease).
- synth00-02: v3h random recipe + jaw quota (every 3rd key: chin drop 0.8-2.8cm)
- synth03: jaw gym — open/close cycles ramping 0.5 -> 3.0cm
- synth04: jaw x background combos (jaw 1.0-2.6cm + 0.5-sigma random face)
"""
import numpy as np, os, sys, shutil

d = np.load("GNM/gnm/shape/data/versions/v3_0/gnm_head.npz", allow_pickle=True)
tpl = d["template_vertex_positions"].astype(np.float64)
eb = d["expression_basis"].astype(np.float64)
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
nose_frac = np.zeros(len(en))
for i in np.where(wt > 0)[0]:
    nose_frac[i] = np.linalg.norm(eb[i][NOSEI]) / (sig[i] + 1e-12)
nosey = nose_frac > 0.25

mask = gm("skin_exterior"); keep = np.where(mask)[0]
V0 = tpl[keep]
chin = (gm("chin_region") | gm("lower_lip_region"))[keep]
NOSE = (gm("nose_region") | gm("left_infraorbital_region") | gm("right_infraorbital_region"))[keep]
isLF = np.array([n.startswith("lower_face_region") for n in en])
lf = np.where(isLF)[0]
E = eb[lf][:, keep]
g = -(E[:, chin, 1].mean(axis=1))
Nn = np.einsum("ivk,jvk->ij", E[:, NOSE], E[:, NOSE])
lam = np.trace(Nn) / len(lf) * 0.5
dvec = np.linalg.solve(Nn + lam * np.eye(len(lf)), g)
cj = np.zeros(len(en)); cj[lf] = dvec
drop1 = -(np.einsum("i,ivk->vk", cj, eb)[keep][chin, 1].mean())  # chin drop per unit alpha
A1CM = 0.01 / drop1                                              # alpha for 1cm mean chin drop
print(f"jaw dvec ready: alpha(1cm) = {A1CM:.3f}")

sys.path.insert(0, "/source/inyup/NeuralFacialAnimation")
from utils.remesh_utils import ICT_face_model
ict = ICT_face_model(); Vi = np.asarray(ict.neutral_verts)
sg = (Vi[:, 1].max() - Vi[:, 1].min()) / (V0[:, 1].max() - V0[:, 1].min())
c0 = V0.mean(0); ci = Vi.mean(0)
A = lambda v: (v - c0) * sg + ci
G = 8.0
rng = np.random.RandomState(7); T = 300; KEY = 12
nk = T // KEY + 2

shutil.rmtree("export/gnm_clips_v4", ignore_errors=True)
os.makedirs("export/gnm_clips_v4")

def decode(coef):
    return (tpl + np.einsum("i,ivk->vk", coef, eb))[keep]

def write_clip(name, raws):
    outdir = f"export/gnm_clips_v4/{name}"; os.makedirs(outdir)
    md = []; cd = []
    for t in range(T):
        f = t / KEY; k0 = int(f); a = (1 - np.cos((f - k0) * np.pi)) / 2
        coef = raws[k0] * (1 - a) + raws[min(k0 + 1, nk - 1)] * a
        V = decode(coef)
        np.save(f"{outdir}/{t:04d}.npy", A(V).astype(np.float32))
        md.append(float(np.abs(V - V0).max()) * sg)
        cd.append(float(-(V[chin, 1] - V0[chin, 1]).mean()) * 100)
    md = np.array(md); cd = np.array(cd)
    print(f"{name} peak-disp {md.max():.3f} | chin-drop max {cd.max():.1f}mm mean {cd.mean():.1f}mm")

def base_keys():
    keys = np.clip(rng.randn(nk, len(en)), -2, 2) * wt[None, :]
    keys[:, nosey] = np.clip(keys[:, nosey], -1.0, 1.0)
    keys[0] = 0; keys[-1] = 0
    return keys

# ── synth00-02: random + jaw quota ────────────────────────────────────
for c in range(3):
    keys = base_keys()
    raw = keys * sig[None, :] * G
    for k in range(1, nk - 1, 3):
        drop_cm = rng.uniform(0.8, 2.8)
        raw[k] += cj * (drop_cm * A1CM * 100 * 0.01)   # alpha = drop_cm[cm] -> A1CM per cm
    write_clip(f"synth{c:02d}", raw)

# ── synth03: jaw gym (ramp cycles 0.5 -> 3.0cm) ───────────────────────
keys = np.clip(rng.randn(nk, len(en)), -2, 2) * wt[None, :] * 0.3
keys[:, nosey] = np.clip(keys[:, nosey], -0.5, 0.5)
keys[0] = 0; keys[-1] = 0
raw = keys * sig[None, :] * G
for k in range(1, nk - 1):
    phase = (k - 1) % 4
    amp_cm = 0.5 + 2.5 * (k / (nk - 1))
    raw[k] += cj * (amp_cm * A1CM) * (0.0, 0.6, 1.0, 0.6)[phase]
write_clip("synth03", raw)

# ── synth04: jaw x face-background combos ─────────────────────────────
keys = np.clip(rng.randn(nk, len(en)), -1.2, 1.2) * wt[None, :] * 0.5
keys[:, nosey] = np.clip(keys[:, nosey], -0.5, 0.5)
keys[0] = 0; keys[-1] = 0
raw = keys * sig[None, :] * G
jaw_seq = [1.4, 0.0, 2.2, 0.8, 2.6, 0.0, 1.8, 1.0]
for k in range(1, nk - 1):
    raw[k] += cj * (jaw_seq[(k - 1) % len(jaw_seq)] * A1CM)
write_clip("synth04", raw)
print("GEN_DONE")
