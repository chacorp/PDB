"""
Weight-visualization supplementary render.

Timeline (video frames):
  [0,  29]  Close-up single face — same camera/framing as render_dynamic_6faces frame 0
  [30, 49]  20-frame cosine zoom-out + weight face / ctrl-pts fade in
  [50, 139] Wide 3-face view: left=skin | middle=weight colormap | right=ctrl spheres
  [140,159] Simultaneous: weight face + ctrl-pts fade out AND camera cosine zoom back in;
            mesh vertices lerp data[140]→data[0]
  [160+]    Hold at frame 0 close-up (35 extra frames)

Pre-requisite:
    python save_weight_vis_data.py   # saves key_weight.npy + key_d/*.npy

Usage:
    cd render
    python render_weight_vis.py           # full render  (~200 frames)
    python render_weight_vis.py --debug   # key frames only
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
from tqdm import tqdm

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

import mitsuba as mi
from render_mitsuba import (
    _init_mitsuba, _lookat, _translate, _rotate, _scale,
    _sphere_light, _skin_bsdf, _cam_pos, _rotate_xz,
    _tonemap, _bilateral, _denoise,
    RenderConfig,
)

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(ROOT, 'vis_CBD', '2026-04-27-15-36-49-NGBCv5',
                        'ict-cap-ID_000_test-to-ict-neutral_01-masked')
OUT_DIR  = os.path.join(ROOT, 'video_mi')

# ── Timeline ────────────────────────────────────────────────────────────────────
TRANS_IN_START = 30    # start zoom-out + weight/ctrl fade-in
TRANS_IN_END   = 50    # 20-frame transition ends
FADE_OUT_START = 140   # weight + ctrl start fading out (simultaneous with zoom-in)
FADE_OUT_END   = 160   # fully gone (vanished by this frame)
RETURN_START   = 140   # camera zoom-in + mesh-lerp begins (simultaneous with fade-out)
RETURN_END     = 160   # at frame-0 position
TOTAL_FRAMES   = 195   # 5 hold frames after return
MAX_DATA       = 180   # last available data frame

# ── Scene layout (matching dynamic_6faces X spacing) ───────────────────────────
FACE_SPACING = 0.65
X_LEFT   = -1.625              # original skin face (same as dynamic_6faces)
X_MIDDLE = X_LEFT + FACE_SPACING        # -0.975  weight-colormap face
X_RIGHT  = X_LEFT + 2 * FACE_SPACING   # -0.325  control-point spheres
X_CENTER = X_MIDDLE            # wide-view camera horizontal target

# ── Camera ──────────────────────────────────────────────────────────────────────
CAM_CLOSE = dict(dist=5.0,  target_x=X_LEFT)
CAM_WIDE  = dict(dist=6.5,  target_x=X_CENTER)
CAM_Y     = -0.05

# ── Render config ───────────────────────────────────────────────────────────────
W, H   = 1920, 1080
SPP    = 256
FPS    = 30
FOV    = 20.0
SCALE  = 0.225
CTRL_SPHERE_RADIUS = 0.035  # pre-scale units (~1% of face height)
CTRL_SPHERE_SUBS   = 1      # icosphere subdivisions

CFG = RenderConfig(
    width=W, height=H, spp=SPP, max_depth=6,
    fov=FOV,
    cam_distance=CAM_CLOSE['dist'],
    cam_elevation=8.0, cam_azimuth=0.0,
    cam_target=(CAM_CLOSE['target_x'], CAM_Y, 0.0),
    mesh_scale=SCALE,
    ground_plane=True,
    bg_color=(1.0, 1.0, 1.0),
    ground_color=(0.65, 0.65, 0.65),
    ground_y=-0.5, flatness=0.0, fps=FPS, aperture_radius=0.0,
    key_pos=(-3.5,  2.5,  0.8), key_radius=0.6,  key_intensity=55.0,
    fill_pos=( 3.5,  1.0,  0.8), fill_radius=0.8, fill_intensity=12.0,
    rim_pos=(  0.2,  4.0, -3.5), rim_radius=0.4,  rim_intensity=20.0,
)

DEBUG_FRAMES = [0, 30, 49, 90, 140, 150, 159, 160, 194]


# ── Easing / timeline helpers ───────────────────────────────────────────────────
def _ease(t: float) -> float:
    return (1.0 - math.cos(math.pi * t)) / 2.0


def camera_at(frame: int):
    """Return (dist, target_x) with cosine easing."""
    if frame < TRANS_IN_START:
        return CAM_CLOSE['dist'], CAM_CLOSE['target_x']
    if frame < TRANS_IN_END:
        t = (frame - TRANS_IN_START) / (TRANS_IN_END - TRANS_IN_START)
        e = _ease(t)
        d  = CAM_CLOSE['dist']     + (CAM_WIDE['dist']     - CAM_CLOSE['dist'])     * e
        tx = CAM_CLOSE['target_x'] + (CAM_WIDE['target_x'] - CAM_CLOSE['target_x']) * e
        return d, tx
    if frame < RETURN_START:
        return CAM_WIDE['dist'], CAM_WIDE['target_x']
    if frame < RETURN_END:
        t = (frame - RETURN_START) / (RETURN_END - RETURN_START)
        e = _ease(t)
        d  = CAM_WIDE['dist']     + (CAM_CLOSE['dist']     - CAM_WIDE['dist'])     * e
        tx = CAM_WIDE['target_x'] + (CAM_CLOSE['target_x'] - CAM_WIDE['target_x']) * e
        return d, tx
    return CAM_CLOSE['dist'], CAM_CLOSE['target_x']


def wc_alpha_at(frame: int) -> float:
    """Opacity of the weight face + ctrl-pt meshes."""
    if frame < TRANS_IN_START:
        return 0.0
    if frame < TRANS_IN_END:
        t = (frame - TRANS_IN_START) / (TRANS_IN_END - TRANS_IN_START)
        return _ease(t)
    if frame < FADE_OUT_START:
        return 1.0
    if frame < FADE_OUT_END:
        t = (frame - FADE_OUT_START) / (FADE_OUT_END - FADE_OUT_START)
        return 1.0 - _ease(t)
    return 0.0


def mesh_verts_at(frame: int, verts_all: list) -> np.ndarray:
    """Vertex positions with cosine lerp back to frame 0 during return phase."""
    if frame < RETURN_START:
        return verts_all[min(frame, MAX_DATA)]
    if frame < RETURN_END:
        t = (frame - RETURN_START) / (RETURN_END - RETURN_START)
        e = _ease(t)
        return verts_all[RETURN_START] * (1.0 - e) + verts_all[0] * e
    return verts_all[0]


# ── Color maps ──────────────────────────────────────────────────────────────────
def build_colormaps(key_weight: np.ndarray):
    """
    Build per-vertex and per-cage colors using nipy_spectral + blend mode,
    matching vis_mesh_all_cage_weights(mode='blend'):
      rgb = key_weight @ cage_rgb  then per-channel min-max normalize.

    Returns:
        vertex_rgb  (N, 3) float32  — per-vertex weight color (constant across frames)
        cage_rgb    (M, 3) float32  — color of each control point sphere
    """
    N, M = key_weight.shape
    cmap = plt.get_cmap('nipy_spectral')
    cage_rgb = np.array([cmap(i / (M - 1))[:3] for i in range(M)], dtype=np.float32)

    # nipy_spectral starts at pure black (t=0) — ensure every cage color has
    # a max channel >= 0.25 to prevent black spheres in the render.
    # For truly-zero colors, replace with a neutral light-grey.
    max_ch = cage_rgb.max(axis=1)                          # (M,)
    is_zero = max_ch < 1e-9
    # scale dark-but-nonzero colors up to min brightness 0.25
    scale = np.where(
        (~is_zero) & (max_ch < 0.25),
        0.25 / (max_ch + 1e-12),
        1.0,
    )
    cage_rgb = cage_rgb * scale[:, None]
    # replace pure-zero entries with light-grey so they remain visible
    cage_rgb[is_zero] = 0.5
    cage_rgb = np.clip(cage_rgb, 0.0, 1.0).astype(np.float32)

    # blend: weighted sum then per-channel min-max normalize
    rgb = key_weight @ cage_rgb                           # (N, 3)
    rgb_min, rgb_max = rgb.min(axis=0), rgb.max(axis=0)
    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-12)
    rgb = np.clip(rgb, 0.0, 1.0)

    # boost saturation to compensate for desaturation from multi-cage blending
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0.0, 1.0)
    vertex_rgb = hsv_to_rgb(hsv).astype(np.float32)
    return vertex_rgb, cage_rgb


# ── PLY export with per-vertex colors ──────────────────────────────────────────
def export_colored_ply(path: str, verts: np.ndarray, faces: np.ndarray,
                       vertex_rgb: np.ndarray) -> None:
    """Write PLY with float32 per-vertex RGB in [0,1] for Mitsuba mesh_attribute."""
    N = len(verts)
    F = len(faces)
    rgb = np.clip(vertex_rgb, 0.0, 1.0).astype(np.float32)

    v_arr = np.empty(N, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'),
    ])
    v_arr['x'], v_arr['y'], v_arr['z'] = verts[:, 0], verts[:, 1], verts[:, 2]
    v_arr['red'],  v_arr['green'], v_arr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    f_arr = np.empty(F, dtype=[('vertex_indices', 'O')])
    for i, tri in enumerate(faces):
        f_arr['vertex_indices'][i] = tri.astype(np.int32)

    PlyData([
        PlyElement.describe(v_arr, 'vertex'),
        PlyElement.describe(f_arr, 'face'),
    ]).write(path)


# ── Control-point sphere mesh ───────────────────────────────────────────────────
def build_ctrl_spheres(key_d: np.ndarray, cage_rgb: np.ndarray) -> tuple:
    """
    Merge M icospheres, one per control point, into a single (verts, faces, colors).
    Positions are in pre-scale space; caller must multiply by SCALE before export.
    """
    tmpl = trimesh.creation.icosphere(subdivisions=CTRL_SPHERE_SUBS,
                                      radius=CTRL_SPHERE_RADIUS)
    Nv, Nf = len(tmpl.vertices), len(tmpl.faces)
    M = len(key_d)

    all_v = np.empty((M * Nv, 3), dtype=np.float32)
    all_f = np.empty((M * Nf, 3), dtype=np.int32)
    all_c = np.empty((M * Nv, 3), dtype=np.float32)

    for i in range(M):
        vi, fi = i * Nv, i * Nf
        all_v[vi:vi + Nv] = tmpl.vertices + key_d[i]
        all_f[fi:fi + Nf] = tmpl.faces + vi
        all_c[vi:vi + Nv] = cage_rgb[i]

    return all_v, all_f, all_c


# ── Mitsuba BSDF helpers ────────────────────────────────────────────────────────
def _vc_bsdf(alpha: float = 1.0) -> dict:
    """Diffuse BSDF reading vertex_color attribute, with optional blend for fade."""
    inner = {
        'type': 'diffuse',
        'reflectance': {
            'type': 'mesh_attribute',
            'name': 'vertex_color',
        },
    }
    if alpha >= 1.0:
        return inner
    return {
        'type': 'blendbsdf',
        'weight': float(alpha),
        'bsdf_0': {'type': 'null'},
        'bsdf_1': inner,
    }


# ── Scene builder ───────────────────────────────────────────────────────────────
def build_scene(
    orig_obj:   str,
    weight_ply: str,
    ctrl_ply:   str,
    cam_dist:   float,
    cam_tx:     float,
    alpha_wc:   float,
    cfg:        RenderConfig,
) -> dict:
    target  = (cam_tx, CAM_Y, 0.0)
    az      = cfg.cam_azimuth
    cx, cy, cz = _cam_pos(cam_dist, cfg.cam_elevation, az)
    cam_pos = (cx + cam_tx, cy, cz)

    kp = tuple(p + (cam_tx if i == 0 else 0)
               for i, p in enumerate(_rotate_xz(cfg.key_pos,  az)))
    fp = tuple(p + (cam_tx if i == 0 else 0)
               for i, p in enumerate(_rotate_xz(cfg.fill_pos, az)))
    rp = tuple(p + (cam_tx if i == 0 else 0)
               for i, p in enumerate(_rotate_xz(cfg.rim_pos,  az)))

    ground_T = _translate([0, cfg.ground_y, 0]) @ _rotate([1, 0, 0], -90) @ _scale(12)

    scene = {
        'type': 'scene',
        'integrator': {
            'type': 'aov',
            'aovs': 'albedo:albedo,normals:sh_normal',
            'nested': {
                'type': 'path',
                'max_depth': cfg.max_depth,
                'hide_emitters': False,
            },
        },
        'sensor': {
            'type': 'perspective', 'fov': cfg.fov,
            'to_world': _lookat(cam_pos, target),
            'film': {
                'type': 'hdrfilm',
                'width': W, 'height': H,
                'pixel_filter': {'type': 'box'},
            },
            'sampler': {'type': 'multijitter', 'sample_count': cfg.spp},
        },
        'env': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': [cfg.env_intensity * c for c in cfg.bg_color],
            },
        },
        'key_light':  _sphere_light(kp, cfg.key_radius,  cfg.key_intensity,  cfg.key_color),
        'fill_light': _sphere_light(fp, cfg.fill_radius, cfg.fill_intensity, cfg.fill_color),
        'rim_light':  _sphere_light(rp, cfg.rim_radius,  cfg.rim_intensity,  cfg.rim_color),
        'ground': {
            'type': 'rectangle', 'to_world': ground_T,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': list(cfg.ground_color)},
            },
        },
        'orig_face': {
            'type': 'obj',
            'filename': orig_obj,
            'to_world': _translate([X_LEFT, 0.0, 0.0]),
            'bsdf': _skin_bsdf(cfg),
        },
    }

    # both weight_face and ctrl geometry are skipped below 3% alpha:
    # near-transparent meshes cast residual shadow artifacts
    if alpha_wc >= 0.03:
        scene['weight_face'] = {
            'type': 'ply',
            'filename': weight_ply,
            'to_world': _translate([X_MIDDLE, 0.0, 0.0]),
            'bsdf': _vc_bsdf(alpha_wc),
        }
        if True:
            scene['ctrl_pts'] = {
                'type': 'ply',
                'filename': ctrl_ply,
                'to_world': _translate([X_RIGHT, 0.0, 0.0]),
                'bsdf': _vc_bsdf(alpha_wc),
            }
            # single point light inside the cluster; scaled by alpha_wc so it
            # vanishes with the spheres and causes no residual artifacts
            _ci = 8.0 * alpha_wc
            scene['ctrl_fill'] = {
                'type': 'point',
                'position': [X_RIGHT, 0.0, 0.0],
                'intensity': {'type': 'rgb', 'value': [_ci, _ci, _ci]},
            }

    return scene


# ── Per-frame render ────────────────────────────────────────────────────────────
def render_frame(
    verts:          np.ndarray,
    faces:          np.ndarray,
    vertex_rgb:     np.ndarray,   # (N, 3) constant weight colors
    key_d:          np.ndarray,   # (M, 3) control point positions this frame
    cage_rgb:       np.ndarray,   # (M, 3) constant cage colors
    cam_dist:       float,
    cam_tx:         float,
    alpha_wc:       float,
    cfg:            RenderConfig,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        v = verts * cfg.mesh_scale

        # Original face — OBJ with skin BSDF
        orig_obj = os.path.join(tmpdir, 'orig.obj')
        trimesh.Trimesh(vertices=v, faces=faces, process=False).export(orig_obj)

        # Weight-colored face — PLY with vertex_color BSDF (same topology, same verts)
        weight_ply = os.path.join(tmpdir, 'weight.ply')
        export_colored_ply(weight_ply, v, faces, vertex_rgb)

        ctrl_ply = os.path.join(tmpdir, 'ctrl.ply')
        cv, cf, cc = build_ctrl_spheres(key_d, cage_rgb)
        export_colored_ply(ctrl_ply, cv * cfg.mesh_scale, cf, cc)

        scene  = mi.load_dict(build_scene(
            orig_obj, weight_ply, ctrl_ply,
            cam_dist, cam_tx, alpha_wc, cfg,
        ))
        image  = mi.render(scene, spp=cfg.spp)

    arr     = np.array(image)
    rgb_t   = mi.TensorXf(arr[..., :3])
    alb_t   = mi.TensorXf(arr[..., 3:6]) if arr.shape[-1] >= 6 else None
    nrm_t   = mi.TensorXf(arr[..., 6:9]) if arr.shape[-1] >= 9 else None
    denoised = _denoise(rgb_t, alb_t, nrm_t)
    img      = _tonemap(np.array(denoised))
    return _bilateral(img)


# ── Main ────────────────────────────────────────────────────────────────────────
def render_all(debug: bool = False):
    from PIL import Image

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading vertex data...")
    vert_paths = sorted(glob(os.path.join(DATA_DIR, 'verts', '*.npy')))
    if not vert_paths:
        raise FileNotFoundError(f"No verts in {DATA_DIR}/verts/")

    # Load only the frames we need (0 .. MAX_DATA)
    verts_all = [np.load(p) for p in vert_paths[:MAX_DATA + 1]]
    print(f"  Loaded {len(verts_all)} vert frames")

    key_weight_path = os.path.join(DATA_DIR, 'key_weight.npy')
    key_d_dir       = os.path.join(DATA_DIR, 'key_d')
    if not os.path.exists(key_weight_path):
        raise FileNotFoundError(
            f"key_weight.npy not found.\nRun:  python save_weight_vis_data.py"
        )
    key_weight = np.load(key_weight_path)            # (N, M)
    key_d_paths = sorted(glob(os.path.join(key_d_dir, '*.npy')))
    key_d_all   = [np.load(p) for p in key_d_paths[:MAX_DATA + 1]]
    print(f"  key_weight shape={key_weight.shape},  key_d frames={len(key_d_all)}")

    # ── Mesh topology ────────────────────────────────────────────────────────
    from utils.remesh_utils import ICT_face_model
    faces = ICT_face_model().faces
    print(f"  faces shape={faces.shape}")

    # ── Colormaps (constant across frames) ───────────────────────────────────
    vertex_rgb, cage_rgb = build_colormaps(key_weight)
    print(f"  vertex_rgb={vertex_rgb.shape}  cage_rgb={cage_rgb.shape}")

    # ── Output dirs ──────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    frame_dir = os.path.join(OUT_DIR, 'weight_vis_frames')
    os.makedirs(frame_dir, exist_ok=True)

    frame_indices = DEBUG_FRAMES if debug else list(range(TOTAL_FRAMES))
    print(f"\nRendering {len(frame_indices)} frames  ({W}×{H}, SPP={SPP})")

    for i in tqdm(frame_indices, desc='frames'):
        cam_dist, cam_tx = camera_at(i)
        alpha_wc         = wc_alpha_at(i)
        verts            = mesh_verts_at(i, verts_all)
        kd               = key_d_all[min(i, MAX_DATA)]

        img = render_frame(
            verts, faces, vertex_rgb, kd, cage_rgb,
            cam_dist, cam_tx, alpha_wc, CFG,
        )

        out_path = os.path.join(frame_dir, f'{i:06d}.png')
        Image.fromarray(img).save(out_path)

        if debug:
            print(f"  frame {i:4d}: dist={cam_dist:.2f} tx={cam_tx:.3f} "
                  f"alpha_wc={alpha_wc:.2f}  shape={img.shape}")

    if debug:
        print("\n[Debug] Skipping MP4 assembly.")
        return

    mp4_path = os.path.join(OUT_DIR, 'weight_vis.mp4')
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', os.path.join(frame_dir, '%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        mp4_path,
    ], check=True)
    print(f"\n[Done] {mp4_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Render key frames only (no MP4)')
    parser.add_argument('--out-dir', default=None,
                        help='Override output directory (default: video_mi/)')
    args = parser.parse_args()
    if args.out_dir:
        OUT_DIR = args.out_dir
    render_all(debug=args.debug)
