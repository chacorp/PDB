# Stage 2+3: mf12 ROM grin candidates -> {ours, NFR, NFS} x {ict_real m00, m02}
# -> descriptor scoring (mouth-open error vs corner-stretch damping) -> contact sheets.
import sys, os, json, glob, pickle
import numpy as np, torch
from pathlib import Path
from scipy.spatial import cKDTree
R = '/source/inyup/NeuralFacialAnimation'
os.chdir(R); sys.path.insert(0, R); sys.path.insert(0, f'{R}/tools/vis')
import viser_debug as vd
dev = 'cuda'
model, rig, opts, nfs_dir, helper_idx, geo_per_topo = vd._build_model(Path("ckpts_hlbs/combined500_import"), dev)
needs_nfs = getattr(opts, 'nfs_feat_dir', None) is not None
from utils.remesh_utils import ICT_face_model
ict = ICT_face_model(base_dir=R)
topos = vd._build_topos(ict, nfs_dir, geo_per_topo)
mf = topos['mf']; ictr = topos['ict_real']
SRC = 12
print('src:', mf.id_names[SRC], ' tgts:', ictr.id_names[0], ictr.id_names[2], flush=True)
cache = vd.IdentityCache(model, rig, dev, model_needs_nfs=needs_nfs)
baseline = vd.BaselineRunner(dev, Path(R))

# ---- ICT lip descriptor sets (data-driven from blendshape morphs) ----
z = np.zeros(100, dtype=np.float32)
neu_i = np.asarray(ict.apply_coeffs(z, exp_coeffs=None)[0], dtype=np.float64)
diag_i = np.linalg.norm(neu_i.max(0) - neu_i.min(0))
morphs = []
for e in range(53):
    ec = np.zeros((1, 53), dtype=np.float32); ec[0, e] = 1.0
    ve = np.asarray(ict.apply_coeffs(z, exp_coeffs=ec)[0], dtype=np.float64)
    morphs.append(ve)
tot = [np.linalg.norm(v - neu_i, axis=1).sum() for v in morphs]
jaw_e = int(np.argmax(tot))          # jawOpen dominates total displacement
vj = morphs[jaw_e]; dyj = (vj - neu_i)[:, 1]
lower_i = np.where(dyj < np.quantile(dyj, 0.005))[0]      # strong downward movers
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
print('ict sets: mouth %d lc %d uc %d corners %d/%d jaw_e=%d' %
      (len(mouth_i), len(lc_i), len(uc_i), len(Lc_i), len(Rc_i), jaw_e), flush=True)

def desc_ict(v):
    v = np.asarray(v, dtype=np.float64)
    g = np.quantile(cKDTree(v[uc_i]).query(v[lc_i])[0], 0.1)
    w = np.linalg.norm(v[Lc_i].mean(0) - v[Rc_i].mean(0))
    return w, g

# ---- candidates ----
cands = json.load(open('/tmp/mf12_candidates.json'))['A']
mfid = mf.id_names[SRC]

def frame_pos(clip, fname):
    fr = sorted(Path(f'/data/sihun/multiface_align/ROM/test/vertices_npy/{mfid}/{clip}').glob('*.npy'))
    fr = [f for f in fr if f.stat().st_size > 1024]
    return [f.name for f in fr].index(fname)

OUT = Path('/tmp/teaser_runs'); OUT.mkdir(exist_ok=True)
results = []
for ti in (0, 2):
    tname = ictr.id_names[ti]
    tb = cache.get(ti, ictr)
    w_neu_t, g_neu_t = desc_ict(tb['neu_v'])
    diag_t = np.linalg.norm(tb['neu_v'].max(0) - tb['neu_v'].min(0))
    for (clip, fname, dw_s, g_s, dj_s) in cands:
        pos = frame_pos(clip, fname)
        key = f'ROM/test/{clip}'
        o = cache.retarget(mf, SRC, (key, pos), ictr, ti)
        if not o['model_ran']:
            print('SKIP (ours failed)', tname, clip, fname, flush=True); continue
        rec = {'tgt': tname, 'clip': clip, 'frame': fname, 'dw_src': dw_s, 'g_src': g_s}
        preds = {'ours': o['tgt_pred_v']}
        for meth in ('nfr', 'nfs'):
            try:
                p = baseline.retarget(method=meth, src_td=mf, src_idx=SRC,
                                      src_neu_v=o['src_neu_v'], src_def_v=o['src_def_v'],
                                      tgt_td=ictr, tgt_idx=ti, tgt_neu_v=o['tgt_neu_v'])
            except Exception as ex:
                print(meth, 'EXC', type(ex).__name__, str(ex)[:150], flush=True); p = None
            preds[meth] = p
        for meth, p in preds.items():
            if p is None:
                rec[meth] = None; continue
            w, g = desc_ict(p)
            open_err = max(0.0, (g - g_neu_t)) / diag_t
            stretch = (w - w_neu_t) / diag_t
            rec[meth] = {'open_err': open_err, 'stretch': stretch, 'e': open_err - stretch}
        if rec.get('nfr') and rec.get('nfs') and rec.get('ours'):
            rec['score'] = min(rec['nfr']['e'], rec['nfs']['e']) - rec['ours']['e']
        else:
            rec['score'] = -9e9
        stem = fname.replace('.npy', '')
        np.savez_compressed(OUT / ('%s_%s_%s.npz' % (tname, clip, stem)),
                            src_def=o['src_def_v'], src_neu=o['src_neu_v'], tgt_neu=o['tgt_neu_v'],
                            ours=preds['ours'],
                            nfr=(preds['nfr'] if preds['nfr'] is not None else np.zeros(1)),
                            nfs=(preds['nfs'] if preds['nfs'] is not None else np.zeros(1)))
        results.append(rec)
        def _e(m):
            return rec[m]['e'] if rec.get(m) else 9.0
        print('DONE %s %s %s score %.4f (ours e %.4f nfr %.4f nfs %.4f)' %
              (tname, clip, fname, rec['score'], _e('ours'), _e('nfr'), _e('nfs')), flush=True)
json.dump(results, open(OUT / 'scores.json', 'w'), indent=1)
print('SCAN_COMPLETE n=', len(results), flush=True)

# ---- contact sheets ----
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
IMS = 420

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

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except Exception:
    HAS_PIL = False
os.makedirs('report_assets/teaser_scan', exist_ok=True)
for tname in (ictr.id_names[0], ictr.id_names[2]):
    rs = [r for r in results if r['tgt'] == tname and r['score'] > -1e8]
    rs.sort(key=lambda r: -r['score'])
    rows = []
    for r in rs[:8]:
        stem = r['frame'].replace('.npy', '')
        zf = np.load(OUT / ('%s_%s_%s.npz' % (tname, r['clip'], stem)))
        row = [render_mesh(zf['src_def'], mf.faces)]
        for m in ('nfr', 'nfs', 'ours'):
            row.append(render_mesh(zf[m], ictr.faces) if zf[m].size > 3
                       else np.full((IMS, IMS, 3), 255, np.uint8))
        img = np.concatenate(row, 1)
        if HAS_PIL:
            im = Image.fromarray(img); dr = ImageDraw.Draw(im)
            dr.text((6, 6), '%s %s  score %.4f' % (r['clip'], r['frame'], r['score']),
                    fill=(30, 30, 30))
            for j, lab in enumerate(['src mf12', 'NFR', 'NFS', 'Ours']):
                dr.text((j * IMS + 6, IMS - 18), lab, fill=(30, 30, 30))
            img = np.array(im)
        rows.append(img)
    if rows:
        sheet = np.concatenate(rows, 0)
        p = 'report_assets/teaser_scan/sheet_%s.png' % tname
        imageio.imwrite(p, sheet); print('SHEET', p, sheet.shape, flush=True)
print('ALL_DONE', flush=True)
