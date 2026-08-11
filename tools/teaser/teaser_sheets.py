# Stage 3b: from saved /tmp/teaser_runs npz -> descriptor diagnostics + full contact sheets.
import sys, os, json, glob
import numpy as np, torch
from pathlib import Path
from scipy.spatial import cKDTree
R = '/source/inyup/NeuralFacialAnimation'
os.chdir(R); sys.path.insert(0, R); sys.path.insert(0, f'{R}/tools/vis')
from utils.remesh_utils import ICT_face_model
import pickle
ict = ICT_face_model(base_dir=R)
F_ict = ict.faces.astype(np.int64)
tpl = pickle.load(open('utils/templates/mf_templates.pkl', 'rb'))
F_mf = np.asarray(tpl['face'], dtype=np.int64)

# ICT lip sets (same derivation as scan)
z = np.zeros(100, dtype=np.float32)
neu_i = np.asarray(ict.apply_coeffs(z, exp_coeffs=None)[0], dtype=np.float64)
diag_i = np.linalg.norm(neu_i.max(0) - neu_i.min(0))
morphs = []
for e in range(53):
    ec = np.zeros((1, 53), dtype=np.float32); ec[0, e] = 1.0
    morphs.append(np.asarray(ict.apply_coeffs(z, exp_coeffs=ec)[0], dtype=np.float64))
tot = [np.linalg.norm(v - neu_i, axis=1).sum() for v in morphs]
jaw_e = int(np.argmax(tot))
vj = morphs[jaw_e]; dyj = (vj - neu_i)[:, 1]
lower_i = np.where(dyj < np.quantile(dyj, 0.005))[0]
d_all, _ = cKDTree(neu_i[lower_i]).query(neu_i)
mouth_i = np.where(d_all < 0.03 * diag_i)[0]
dy_m = dyj[mouth_i]
lo = mouth_i[dy_m < np.quantile(dy_m, 0.35)]
up = mouth_i[dy_m > np.quantile(dy_m, 0.65)]
xs = neu_i[mouth_i, 0]; xr = xs.max() - xs.min(); xc = np.median(xs)
lc_i = lo[np.abs(neu_i[lo, 0] - xc) < 0.25 * xr]
uc_i = up[np.abs(neu_i[up, 0] - xc) < 0.25 * xr]
Lc_i = mouth_i[xs < xs.min() + 0.08 * xr]
Rc_i = mouth_i[xs > xs.max() - 0.08 * xr]
print('sets mouth %d lc %d uc %d jaw_e %d' % (len(mouth_i), len(lc_i), len(uc_i), jaw_e))

def desc(v):
    v = np.asarray(v, dtype=np.float64)
    g = np.quantile(cKDTree(v[uc_i]).query(v[lc_i])[0], 0.1)
    w = np.linalg.norm(v[Lc_i].mean(0) - v[Rc_i].mean(0))
    return w, g

OUT = Path('/tmp/teaser_runs')
recs = []
for p in sorted(OUT.glob('*.npz')):
    zf = np.load(p)
    name = p.stem  # {tgt}_{clip}_{framestem}
    tgt = name.split('_')[0]
    tn = zf['tgt_neu']; diag_t = float(np.linalg.norm(tn.max(0) - tn.min(0)))
    w0, g0 = desc(tn)
    row = {'name': name, 'tgt': tgt}
    for m in ('ours', 'nfr', 'nfs'):
        if zf[m].size <= 3:
            row[m] = None; continue
        w, g = desc(zf[m])
        row[m] = {'dw': float((w - w0) / diag_t), 'dg': float((g - g0) / diag_t)}
    recs.append(row)
    def _f(m, k):
        return row[m][k] if row[m] else float('nan')
    print('%-46s ours dw %+0.4f dg %+0.4f | nfr dw %+0.4f dg %+0.4f | nfs dw %+0.4f dg %+0.4f' %
          (name, _f('ours','dw'), _f('ours','dg'), _f('nfr','dw'), _f('nfr','dg'),
           _f('nfs','dw'), _f('nfs','dg')), flush=True)
json.dump(recs, open(OUT / 'desc_diag.json', 'w'), indent=1, default=float)

# score: want dw large (stretch preserved), dg ~ 0 (mouth stays closed).
# quality q = dw - max(0,dg)*2 ; score = q_ours - max(q_nfr, q_nfs)
scored = []
for r in recs:
    if not (r['ours'] and r['nfr'] and r['nfs']):
        r['score'] = -9e9; continue
    def q(m):
        return r[m]['dw'] - 2.0 * max(0.0, r[m]['dg'])
    r['score'] = q('ours') - max(q('nfr'), q('nfs'))
    scored.append(r)
json.dump(recs, open(OUT / 'scores2.json', 'w'), indent=1, default=float)

# ---- render all rows per target, sorted by score desc ----
dev = 'cuda'
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader,
    DirectionalLights, TexturesVertex, Materials, BlendParams)
import imageio.v2 as imageio
key = np.array([0.0, 0.7, 0.9]); key = key / np.linalg.norm(key)
lights = DirectionalLights(device=dev, direction=[tuple(key)],
    ambient_color=[[0.42, 0.43, 0.46]], diffuse_color=[[0.62, 0.61, 0.58]],
    specular_color=[[0.10, 0.10, 0.11]])
mats = Materials(device=dev, shininess=28.0)
COL = np.array([89, 128, 212], dtype=np.float32) / 255.0
IMS = 400

def render_mesh(v, f):
    v = np.asarray(v, dtype=np.float32); c = v.mean(0)
    half = float(np.abs(v - c).max()) * 1.10; fov = 10.0
    dist = half / np.tan(np.radians(fov / 2))
    Rr, Tt = look_at_view_transform(dist=dist, elev=3, azim=0,
                                    at=((float(c[0]), float(c[1]), float(c[2])),))
    cams = FoVPerspectiveCameras(device=dev, R=Rr, T=Tt, fov=fov)
    rast = RasterizationSettings(image_size=IMS, blur_radius=0.0, faces_per_pixel=2)
    rend = MeshRenderer(rasterizer=MeshRasterizer(cameras=cams, raster_settings=rast),
        shader=HardPhongShader(device=dev, cameras=cams, lights=lights, materials=mats,
                               blend_params=BlendParams(background_color=(1., 1., 1.))))
    vv = torch.tensor(v, device=dev)
    mesh = Meshes(verts=[vv], faces=[torch.tensor(np.asarray(f, dtype=np.int64), device=dev)],
                  textures=TexturesVertex(torch.tensor(COL, device=dev).expand(1, len(v), 3)))
    return (rend(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

from PIL import Image, ImageDraw
os.makedirs('report_assets/teaser_scan', exist_ok=True)
for tgt in ('m00', 'm02'):
    rs = [r for r in recs if r['tgt'] == tgt and r['score'] > -1e8]
    rs.sort(key=lambda r: -r['score'])
    rows = []
    for r in rs:
        zf = np.load(OUT / (r['name'] + '.npz'))
        row = [render_mesh(zf['src_def'], F_mf)]
        for m in ('nfr', 'nfs', 'ours'):
            row.append(render_mesh(zf[m], F_ict) if zf[m].size > 3
                       else np.full((IMS, IMS, 3), 255, np.uint8))
        img = np.concatenate(row, 1)
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        dr.text((6, 6), '%s  score %+.4f' % (r['name'], r['score']), fill=(30, 30, 30))
        for j, lab in enumerate(['src mf12', 'NFR', 'NFS', 'Ours']):
            dr.text((j * IMS + 6, IMS - 18), lab, fill=(30, 30, 30))
        rows.append(np.array(im))
    if rows:
        sheet = np.concatenate(rows, 0)
        p = 'report_assets/teaser_scan/sheet_%s.png' % tgt
        imageio.imwrite(p, sheet)
        print('SHEET', p, sheet.shape, flush=True)
print('ALL_DONE', flush=True)
