"""
Dynamic 6-face supplementary video renderer.

Timeline:
  0   - 199  : ict_neutral close-up, facial animation plays
  200 - 289  : camera zoom-out + pan (cosine easing, ~90 frames)
  290 - 1146 : wide view, all 6 faces, animation continues

Single Mitsuba scene — all 6 meshes share lighting, shadows, ground.

Output: video_mi/dynamic_6faces.mp4  (1920x1080, 30fps)

Usage:
    cd render
    python render_dynamic_6faces.py           # full render
    python render_dynamic_6faces.py --debug   # 6 key frames only
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

import mitsuba as mi
from render_mitsuba import (
    _init_mitsuba, _lookat, _translate, _rotate, _scale,
    _sphere_light, _skin_bsdf, _cam_pos, _rotate_xz,
    _tonemap, _bilateral, _denoise,
    get_mesh, RenderConfig,
)

# ── paths ──────────────────────────────────────────────────────────────────────
VIS_DIR = os.path.join(ROOT, 'vis_CBD', '2026-04-27-15-36-49-NGBCv5-best')
OUT_DIR  = os.path.join(ROOT, 'video_mi')

SEQUENCES = [
    ('ict-cap-ID_000_test-to-ict-neutral_01-masked',         'ict',    -1.625),
    ('ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked', 'ict',    -0.975),
    ('ict-cap-ID_000_test-to-biwi-ID_012_test_01-masked',    'biwi',   -0.325),
    ('ict-cap-ID_000_test-to-coma-ID_006_test_01-masked',    'coma',    0.325),
    ('ict-cap-ID_000_test-to-coma-ID_008_test_01-masked',    'coma',    0.975),
    ('ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01-masked',  'mf_ROM',  1.625),
]

# ── camera / visibility timeline ──────────────────────────────────────────────
CAM_CLOSEUP = dict(dist=5.0,  target_x=-1.625)  # ict_neutral X position (spacing 0.65)
CAM_WIDE    = dict(dist=12.0, target_x= 0.0)
TRANS_START = 200
TRANS_END   = 290
CAM_Y       = -0.05  # face center Y (measured: -0.052)

# ── render config ──────────────────────────────────────────────────────────────
W, H   = 1920, 1080
SPP    = 256
FPS    = 30
FOV    = 20.0
SCALE  = 0.225

# Lights moved far enough out of FOV for both close-up and wide shots.
# wide-view horizontal half-FOV ≈ 17.4°; at z-dist ~9 units,
# lights need |x| > 9*tan(17.4°) ≈ 2.8 from target to stay out of frame.
CFG = RenderConfig(
    width=W, height=H, spp=SPP, max_depth=6,
    fov=FOV,
    cam_distance=CAM_CLOSEUP['dist'],
    cam_elevation=8.0,
    cam_azimuth=0.0,
    cam_target=(CAM_CLOSEUP['target_x'], CAM_Y, 0.0),
    mesh_scale=SCALE,
    ground_plane=True,
    bg_color=(1.0, 1.0, 1.0),
    ground_color=(0.65, 0.65, 0.65),
    ground_y=-0.5,
    flatness=0.0,
    fps=FPS,
    aperture_radius=0.0,
    # 3-point lights repositioned outside camera FOV
    key_pos=(-3.5,  2.5,  0.8), key_radius=0.6,  key_intensity=55.0,
    fill_pos=( 3.5,  1.0,  0.8), fill_radius=0.8, fill_intensity=12.0,
    rim_pos=(  0.2,  4.0, -3.5), rim_radius=0.4,  rim_intensity=20.0,
)

# DEBUG frames: closeup, mid-closeup, trans-start, mid-trans, trans-end, wide
DEBUG_FRAMES = [0, 100, 200, 245, 289, 300]


# ── camera interpolation ───────────────────────────────────────────────────────
def _cosine_ease(t: float) -> float:
    return (1.0 - math.cos(math.pi * t)) / 2.0


def camera_interp(frame_idx: int):
    """Return (cam_dist, cam_target_x) for the given frame."""
    if frame_idx < TRANS_START:
        return CAM_CLOSEUP['dist'], CAM_CLOSEUP['target_x']
    if frame_idx >= TRANS_END:
        return CAM_WIDE['dist'], CAM_WIDE['target_x']
    t  = (frame_idx - TRANS_START) / (TRANS_END - TRANS_START)
    e  = _cosine_ease(t)
    d  = CAM_CLOSEUP['dist']    + (CAM_WIDE['dist']     - CAM_CLOSEUP['dist'])    * e
    tx = CAM_CLOSEUP['target_x']+ (CAM_WIDE['target_x'] - CAM_CLOSEUP['target_x'])* e
    return d, tx


def face_alphas_at(frame_idx: int) -> list:
    """Per-face opacity: 0=invisible, 1=fully opaque.
    face_0 (ict_neutral) is always visible.
    Others fade in during the transition.
    """
    if frame_idx < TRANS_START:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if frame_idx >= TRANS_END:
        return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    t = (frame_idx - TRANS_START) / (TRANS_END - TRANS_START)
    e = _cosine_ease(t)
    return [1.0, e, e, e, e, e]


# ── multi-mesh scene builder ───────────────────────────────────────────────────
def _build_multi_scene(
    obj_x_list,    # [(obj_path, x_offset), ...]
    cam_dist:      float,
    cam_target_x:  float,
    cfg:           RenderConfig,
    face_alphas:   list = None,  # per-face opacity [0..1], None = all opaque
) -> dict:
    target = (cam_target_x, CAM_Y, 0.0)
    az = cfg.cam_azimuth
    cx, cy, cz = _cam_pos(cam_dist, cfg.cam_elevation, az)
    # offset camera X so it's always directly in front of the target
    cam_pos = (cx + cam_target_x, cy, cz)

    key_pos  = tuple(p + (cam_target_x if i == 0 else 0)
                     for i, p in enumerate(_rotate_xz(cfg.key_pos,  az)))
    fill_pos = tuple(p + (cam_target_x if i == 0 else 0)
                     for i, p in enumerate(_rotate_xz(cfg.fill_pos, az)))
    rim_pos  = tuple(p + (cam_target_x if i == 0 else 0)
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
            'type': 'perspective',
            'fov': cfg.fov,
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
        'key_light':  _sphere_light(key_pos,  cfg.key_radius,  cfg.key_intensity,  cfg.key_color),
        'fill_light': _sphere_light(fill_pos, cfg.fill_radius, cfg.fill_intensity, cfg.fill_color),
        'rim_light':  _sphere_light(rim_pos,  cfg.rim_radius,  cfg.rim_intensity,  cfg.rim_color),
        'ground': {
            'type': 'rectangle',
            'to_world': ground_T,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': list(cfg.ground_color)},
            },
        },
    }

    alphas = face_alphas if face_alphas is not None else [1.0] * len(obj_x_list)

    for idx, ((obj_path, x_off), alpha) in enumerate(zip(obj_x_list, alphas)):
        if alpha <= 0.0:
            continue  # completely absent — no shadow, no geometry
        elif alpha >= 1.0:
            bsdf = _skin_bsdf(cfg)
        else:
            bsdf = {
                'type': 'blendbsdf',
                'weight': float(alpha),
                'bsdf_0': {'type': 'null'},
                'bsdf_1': _skin_bsdf(cfg),
            }
        scene[f'face_{idx}'] = {
            'type': 'obj',
            'filename': obj_path,
            'to_world': _translate([x_off, 0.0, 0.0]),
            'bsdf': bsdf,
        }

    return scene


# ── per-frame render ───────────────────────────────────────────────────────────
def render_frame_multi(
    verts_list,    # [(verts_np, faces_np, x_offset), ...]
    cam_dist:      float,
    cam_target_x:  float,
    cfg:           RenderConfig,
    face_alphas:   list = None,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_x_list = []
        for i, (verts, faces, x_off) in enumerate(verts_list):
            v = verts * cfg.mesh_scale
            mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
            obj_path = os.path.join(tmpdir, f'face_{i}.obj')
            mesh.export(obj_path)
            obj_x_list.append((obj_path, x_off))

        scene = mi.load_dict(
            _build_multi_scene(obj_x_list, cam_dist, cam_target_x, cfg, face_alphas)
        )
        image = mi.render(scene, spp=cfg.spp)

    image_np = np.array(image)
    rgb_t    = mi.TensorXf(image_np[..., :3])
    alb_t    = mi.TensorXf(image_np[..., 3:6]) if image_np.shape[-1] >= 6 else None
    nrm_t    = mi.TensorXf(image_np[..., 6:9]) if image_np.shape[-1] >= 9 else None
    denoised = _denoise(rgb_t, alb_t, nrm_t)
    img      = _tonemap(np.array(denoised))
    return _bilateral(img)


# ── main render loop ───────────────────────────────────────────────────────────
def render_all(debug: bool = False):
    from PIL import Image

    os.makedirs(OUT_DIR, exist_ok=True)
    frame_dir = os.path.join(OUT_DIR, 'dynamic_6faces_frames')
    os.makedirs(frame_dir, exist_ok=True)

    print("Loading mesh topologies...")
    faces_cache = {}
    for _, mesh_type, _ in SEQUENCES:
        if mesh_type not in faces_cache:
            _, f = get_mesh(mesh_type)
            faces_cache[mesh_type] = f
            print(f"  {mesh_type}: faces={f.shape}")

    print("Loading npy paths...")
    seq_data = []
    for seq_name, mesh_type, x_off in SEQUENCES:
        npy_dir = os.path.join(VIS_DIR, seq_name, 'verts')
        paths   = sorted(glob(os.path.join(npy_dir, '*.npy')))
        if not paths:
            raise FileNotFoundError(f"No .npy in {npy_dir}")
        seq_data.append((paths, mesh_type, x_off))
        print(f"  {seq_name}: {len(paths)} frames, x={x_off}")

    n_frames = min(len(p) for p, _, _ in seq_data)
    frame_indices = DEBUG_FRAMES if debug else list(range(n_frames))
    print(f"\nRendering {len(frame_indices)} frames  ({W}x{H})")

    for i in tqdm(frame_indices, desc='frames'):
        cam_dist, cam_tx = camera_interp(i)

        verts_list = []
        for (paths, mesh_type, x_off) in seq_data:
            idx   = min(i, len(paths) - 1)
            verts = np.load(paths[idx])
            faces = faces_cache[mesh_type]
            verts_list.append((verts, faces, x_off))

        alphas = face_alphas_at(i)
        img = render_frame_multi(verts_list, cam_dist, cam_tx, CFG, alphas)

        out_path = os.path.join(frame_dir, f'{i:06d}.png')
        Image.fromarray(img).save(out_path)

        if debug:
            print(f"  frame {i:4d}: dist={cam_dist:.2f} target_x={cam_tx:.2f}  shape={img.shape}")

    if debug:
        print("\n[Debug] Skipping MP4 assembly.")
        sample = img
        assert sample.shape == (H, W, 3), f"Shape mismatch: {sample.shape}"
        print(f"[Verify] shape {sample.shape}  PASS")
        return

    mp4_path = os.path.join(OUT_DIR, 'dynamic_6faces.mp4')
    png_paths = sorted(glob(os.path.join(frame_dir, '*.png')))
    # write file list for ffmpeg (handles non-contiguous debug frames safely)
    list_path = os.path.join(frame_dir, 'frames.txt')
    with open(list_path, 'w') as fh:
        for p in png_paths:
            fh.write(f"file '{p}'\n")
            fh.write(f"duration {1/FPS:.6f}\n")

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
                        help='Render 6 key frames only (no MP4)')
    args = parser.parse_args()
    render_all(debug=args.debug)
