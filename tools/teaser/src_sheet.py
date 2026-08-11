# Source-only contact sheet: per mf12 ROM clip, top-2 frames by face motion
# (temporal dedupe >=15f), rendered frontal, labeled. For visual candidate picking.
import numpy as np, glob, os, pickle, torch, sys
sys.path.insert(0, '/source/inyup/NeuralFacialAnimation')
os.chdir('/source/inyup/NeuralFacialAnimation')
R = '/data/sihun/multiface_align/ROM/test/vertices_npy/m--20190828--1318--002645310--GHS'
tpl = pickle.load(open('utils/templates/mf_templates.pkl', 'rb'))
ids = [k for k in tpl.keys() if k != 'face']
neu = np.asarray(tpl[ids[12]], dtype=np.float64)
F = np.asarray(tpl['face'], dtype=np.int64)

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
IMS = 380

def render_mesh(v):
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
    mesh = Meshes(verts=[vv], faces=[torch.tensor(F, device=dev)],
                  textures=TexturesVertex(torch.tensor(COL, device=dev).expand(1, len(v), 3)))
    return (rend(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

from PIL import Image, ImageDraw
cells = []
for clip in sorted(os.listdir(R)):
    cd = os.path.join(R, clip)
    if not os.path.isdir(cd):
        continue
    fr = sorted(glob.glob(os.path.join(cd, '*.npy')))
    if not fr:
        continue
    mots = []
    for f in fr:
        v = np.load(f)
        mots.append(np.linalg.norm(v - neu, axis=1).mean())
    mots = np.array(mots)
    order = np.argsort(-mots)
    picked = []
    for i in order:
        if all(abs(i - j) >= 15 for j in picked):
            picked.append(i)
        if len(picked) == 2:
            break
    for i in picked:
        img = render_mesh(np.load(fr[i]))
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        dr.text((5, 5), '%s' % clip.replace('EXP_', ''), fill=(20, 20, 20))
        dr.text((5, 20), '%s  mot %.3f' % (os.path.basename(fr[i]).replace('.npy',''), mots[i]),
                fill=(20, 20, 20))
        cells.append(np.array(im))
    print(clip, [os.path.basename(fr[i]) for i in picked], flush=True)
ncol = 6
nrow = (len(cells) + ncol - 1) // ncol
pad = np.full((IMS, IMS, 3), 255, np.uint8)
while len(cells) < nrow * ncol:
    cells.append(pad)
rows = [np.concatenate(cells[r*ncol:(r+1)*ncol], 1) for r in range(nrow)]
sheet = np.concatenate(rows, 0)
imageio.imwrite('/tmp/src_sheet_mf12.png', sheet)
print('SHEET /tmp/src_sheet_mf12.png', sheet.shape, flush=True)
