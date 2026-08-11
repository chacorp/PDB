# Round 2: curated extreme mf12 ROM frames -> {NFR, NFS, ours} x {m00, m02}.
# No metric gating — render rows immediately; sheets per target.
import sys, os, json, glob
import numpy as np, torch
from pathlib import Path
R = '/source/inyup/NeuralFacialAnimation'
os.chdir(R); sys.path.insert(0, R); sys.path.insert(0, f'{R}/tools/vis')
import viser_debug as vd
dev = 'cuda'
model, rig, opts, nfs_dir, helper_idx, geo_per_topo = vd._build_model(
    Path('ckpts_hlbs/combined500_import'), dev)
needs_nfs = getattr(opts, 'nfs_feat_dir', None) is not None
from utils.remesh_utils import ICT_face_model
ict = ICT_face_model(base_dir=R)
topos = vd._build_topos(ict, nfs_dir, geo_per_topo)
mf = topos['mf']; ictr = topos['ict_real']
SRC = 12
cache = vd.IdentityCache(model, rig, dev, model_needs_nfs=needs_nfs)
baseline = vd.BaselineRunner(dev, Path(R))
F_ict = ictr.faces.astype(np.int64); F_mf = mf.faces.astype(np.int64)

CASES = [
    ('EXP_lip001', '30fps-0306.npy'),
    ('EXP_lip001', '30fps-0279.npy'),
    ('EXP_lip002', '30fps-0513.npy'),
    ('EXP_lip003', '30fps-0164.npy'),
    ('EXP_lip003', '30fps-0182.npy'),
    ('EXP_cheek002', '30fps-0266.npy'),
    ('EXP_jaw005', '30fps-0357.npy'),
    ('EXP_jaw003', '30fps-0420.npy'),
    ('EXP_free_face', '30fps-0146.npy'),
    ('EXP_free_face', '30fps-0120.npy'),
]
mfid = mf.id_names[SRC]

def frame_pos(clip, fname):
    fr = sorted(Path(f'/data/sihun/multiface_align/ROM/test/vertices_npy/{mfid}/{clip}').glob('*.npy'))
    fr = [f for f in fr if f.stat().st_size > 1024]
    return [f.name for f in fr].index(fname)

# renderer
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
IMS = 440

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
OUT = Path('/tmp/teaser_runs2'); OUT.mkdir(exist_ok=True)
sheets = {0: [], 2: []}
for ti in (0, 2):
    tname = ictr.id_names[ti]
    for clip, fname in CASES:
        pos = frame_pos(clip, fname)
        o = cache.retarget(mf, SRC, (f'ROM/test/{clip}', pos), ictr, ti)
        if not o['model_ran']:
            print('SKIP', tname, clip, fname, flush=True); continue
        preds = {'ours': o['tgt_pred_v']}
        for meth in ('nfr', 'nfs'):
            try:
                preds[meth] = baseline.retarget(method=meth, src_td=mf, src_idx=SRC,
                    src_neu_v=o['src_neu_v'], src_def_v=o['src_def_v'],
                    tgt_td=ictr, tgt_idx=ti, tgt_neu_v=o['tgt_neu_v'])
            except Exception as ex:
                print(meth, 'EXC', type(ex).__name__, str(ex)[:150], flush=True)
                preds[meth] = None
        stem = fname.replace('.npy', '')
        np.savez_compressed(OUT / ('%s_%s_%s.npz' % (tname, clip, stem)),
            src_def=o['src_def_v'], tgt_neu=o['tgt_neu_v'], ours=preds['ours'],
            nfr=(preds['nfr'] if preds['nfr'] is not None else np.zeros(1)),
            nfs=(preds['nfs'] if preds['nfs'] is not None else np.zeros(1)))
        row = [render_mesh(o['src_def_v'], F_mf)]
        for m in ('nfr', 'nfs', 'ours'):
            row.append(render_mesh(preds[m], F_ict) if preds[m] is not None
                       else np.full((IMS, IMS, 3), 255, np.uint8))
        img = np.concatenate(row, 1)
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        dr.text((6, 6), '%s %s -> %s' % (clip, stem, tname), fill=(30, 30, 30))
        for j, lab in enumerate(['src mf12', 'NFR', 'NFS', 'Ours']):
            dr.text((j * IMS + 6, IMS - 18), lab, fill=(30, 30, 30))
        sheets[ti].append(np.array(im))
        print('DONE', tname, clip, fname, flush=True)
    if sheets[ti]:
        sheet = np.concatenate(sheets[ti], 0)
        p = 'report_assets/teaser_scan/sheet2_%s.png' % tname
        imageio.imwrite(p, sheet)
        print('SHEET', p, sheet.shape, flush=True)
print('ALL_DONE', flush=True)
