"""
Pyrender pseudo-renderer for NeuralFacialAnimation.
Approximates render_mitsuba.py with rasterization + 3-point lighting.
Drop-in API replacement: RenderConfig, render_frame, render_figure,
render_multiview, render_sequence_mi, get_mesh.

Install:
    pip install pyrender pyopengl==3.1.4
    apt install libosmesa6-dev

Usage:
    from render.render_trimesh_pseudo import render_frame, render_figure, PAPER_CFG
    render_figure([(src_v, f), (pred_v, f), (tgt_v, f)], PAPER_CFG, 'fig.png')

    python render_trimesh-pseudo.py --mesh_type 'mf' --npy_dir './vis_CBD/2026-04-02-02-04-44-NGBCv5-dist/ict-cap-ID_002_test-to-mf_ROM-ID_012_test_00-masked/verts'
"""

import os
import sys
import math
import pickle
import argparse
import subprocess
from glob import glob
from pathlib import Path
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from tqdm import tqdm

os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
import pyrender


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class RenderConfig:
    # Image
    width:     int   = 1024
    height:    int   = 1024

    # Camera (spherical coords, same convention as render_mitsuba)
    fov:           float = 20.0   # vertical FOV, degrees
    cam_distance:  float = 2.5
    cam_elevation: float = 8.0    # degrees above horizon
    cam_azimuth:   float = 0.0    # 0=front, 90=left, -90=right
    cam_target:    Tuple = (0.0, 0.0, 0.0)

    # Geometry
    mesh_scale: float = 0.25

    # Skin material (MetallicRoughness approx of Principled BSDF)
    base_color:  Tuple = (0.80, 0.64, 0.52)
    roughness:   float = 0.55
    metallic:    float = 0.0

    # 3-point directional lights (intensities tuned for pyrender lux scale)
    key_pos:        Tuple = (-1.4,  1.8,  0.8)
    key_intensity:  float = 5.0
    key_color:      Tuple = (1.00, 0.95, 0.88)   # warm

    fill_pos:       Tuple = ( 1.6,  0.5,  0.8)
    fill_intensity: float = 2.0
    fill_color:     Tuple = (0.82, 0.88, 1.00)   # cool

    rim_pos:        Tuple = ( 0.2,  2.2, -2.5)
    rim_intensity:  float = 3.0
    rim_color:      Tuple = (1.00, 1.00, 0.96)   # neutral

    env_intensity:  float = 0.28   # ambient

    # Scene
    bg_color:    Tuple = (1.0, 1.0, 1.0)
    ground_plane: bool  = True
    ground_color: Tuple = (0.90, 0.90, 0.90)
    ground_y:    float  = -0.45

    # Video
    fps:          int = 30
    video_width:  int = 800
    video_height: int = 800


# ── Presets ───────────────────────────────────────────────────────────────────
PAPER_CFG = RenderConfig(width=1024, height=1024)

TEASER_CFG = RenderConfig(
    width=1200, height=900,
    bg_color=(0.08, 0.08, 0.10),
    env_intensity=0.10,
    ground_color=(0.10, 0.10, 0.13),
    ground_y=-0.48,
    key_intensity=8.0,
    rim_intensity=5.0,
)

VIDEO_CFG = RenderConfig(width=800, height=800)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cam_pos(dist: float, elev_deg: float, azim_deg: float) -> tuple:
    el = math.radians(elev_deg)
    az = math.radians(azim_deg)
    return (
        dist * math.cos(el) * math.sin(az),
        dist * math.sin(el),
        dist * math.cos(el) * math.cos(az),
    )


def _look_at_pose(eye, target=(0, 0, 0), up=(0, 1, 0)) -> np.ndarray:
    """4x4 camera-to-world pose matrix. Camera looks along its local -Z axis."""
    eye    = np.array(eye,    dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up     = np.array(up,     dtype=np.float64)
    z = eye - target;  z /= np.linalg.norm(z)   # backward (away from scene)
    x = np.cross(up, z);  x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def _rotate_xz(pos: tuple, azim_deg: float) -> tuple:
    """Rotate a point around Y-axis so lights co-rotate with the camera."""
    a = math.radians(azim_deg)
    x, y, z = pos
    return (x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a))


def _skin_material(cfg: RenderConfig) -> pyrender.MetallicRoughnessMaterial:
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*cfg.base_color, 1.0],
        metallicFactor=cfg.metallic,
        roughnessFactor=cfg.roughness,
    )


def _ground_mesh(cfg: RenderConfig) -> pyrender.Mesh:
    s, y = 8.0, cfg.ground_y
    verts = np.array([
        [-s, y, -s], [s, y, -s], [s, y, s], [-s, y, s],
    ], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[*cfg.ground_color, 1.0],
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False)


def _save_img(img: np.ndarray, path: str) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    try:
        import cv2
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(path)


# ── Core rasterizer ───────────────────────────────────────────────────────────
def _render(vertices: np.ndarray, faces: np.ndarray, cfg: RenderConfig, W: int, H: int) -> np.ndarray:
    az = cfg.cam_azimuth
    cam_origin = _cam_pos(cfg.cam_distance, cfg.cam_elevation, az)
    cam_pose   = _look_at_pose(cam_origin, cfg.cam_target)

    verts = vertices * cfg.mesh_scale
    tm    = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh  = pyrender.Mesh.from_trimesh(tm, material=_skin_material(cfg), smooth=True)

    ambient = [cfg.env_intensity * c for c in cfg.bg_color]
    bg      = [*cfg.bg_color, 1.0]
    scene   = pyrender.Scene(ambient_light=ambient, bg_color=bg)
    scene.add(mesh, pose=np.eye(4))

    if cfg.ground_plane:
        scene.add(_ground_mesh(cfg), pose=np.eye(4))

    camera = pyrender.PerspectiveCamera(yfov=math.radians(cfg.fov), znear=0.001, zfar=20.0)
    scene.add(camera, pose=cam_pose)

    # 3-point directional lights co-rotate with camera
    for pos, intensity, color in [
        (_rotate_xz(cfg.key_pos,  az), cfg.key_intensity,  cfg.key_color),
        (_rotate_xz(cfg.fill_pos, az), cfg.fill_intensity, cfg.fill_color),
        (_rotate_xz(cfg.rim_pos,  az), cfg.rim_intensity,  cfg.rim_color),
    ]:
        light = pyrender.DirectionalLight(color=list(color), intensity=intensity)
        scene.add(light, pose=_look_at_pose(pos))

    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    r.delete()
    return color  # uint8 RGB


# ── Public API (mirrors render_mitsuba) ───────────────────────────────────────
def render_frame(
    vertices:   np.ndarray,
    faces:      np.ndarray,
    cfg:        RenderConfig = None,
    save_path:  str = None,
    return_img: bool = True,
) -> Optional[np.ndarray]:
    """
    Render a single mesh to an image.

    Args:
        vertices:   (N, 3) float vertex positions
        faces:      (F, 3) int face indices
        cfg:        RenderConfig; defaults to RenderConfig()
        save_path:  optional PNG/JPG output path
        return_img: if True, returns uint8 (H, W, 3) array

    Returns:
        uint8 RGB numpy array or None
    """
    cfg = cfg or RenderConfig()
    img = _render(vertices, faces, cfg, cfg.width, cfg.height)
    if save_path:
        _save_img(img, save_path)
    return img if return_img else None


def render_figure(
    mesh_list:  List[Tuple[np.ndarray, np.ndarray]],
    cfg:        RenderConfig = None,
    save_path:  str = None,
    gap:        int = 8,
    gap_color:  int = 255,
) -> np.ndarray:
    """
    Render N meshes side-by-side for paper comparison figures.

    Args:
        mesh_list:  list of (vertices, faces) tuples
        cfg:        RenderConfig; defaults to PAPER_CFG
        save_path:  optional output PNG path
        gap:        pixel gap between columns
        gap_color:  gap fill (0=black, 255=white)

    Returns:
        uint8 RGB (H, N*W + (N-1)*gap, 3)

    Example:
        render_figure([(src_v, f), (pred_v, f), (tgt_v, f)], PAPER_CFG, 'fig.png')
    """
    cfg = cfg or PAPER_CFG
    images = [render_frame(v, f, cfg) for v, f in mesh_list]
    gap_col = np.full((cfg.height, gap, 3), gap_color, dtype=np.uint8)
    strips = []
    for i, img in enumerate(images):
        strips.append(img)
        if i < len(images) - 1:
            strips.append(gap_col)
    combined = np.concatenate(strips, axis=1)
    if save_path:
        _save_img(combined, save_path)
    return combined


def render_multiview(
    vertices:  np.ndarray,
    faces:     np.ndarray,
    azimuths:  List[float] = [0, 30, -30, 90],
    cfg:       RenderConfig = None,
    save_path: str = None,
) -> np.ndarray:
    """
    Render one mesh from multiple azimuths and return side-by-side.

    Example:
        img = render_multiview(verts, faces, [0, 45, -45], PAPER_CFG, 'mv.png')
    """
    cfg = cfg or PAPER_CFG
    images = [render_frame(vertices, faces, replace(cfg, cam_azimuth=az)) for az in azimuths]
    combined = np.concatenate(images, axis=1)
    if save_path:
        _save_img(combined, save_path)
    return combined


def render_sequence_mi(
    npy_dir:     str,
    faces:       np.ndarray,
    output_path: str,
    filename:    str = None,
    cfg:         RenderConfig = None,
    fps:         int = None,
    debug:       bool = False,
) -> str:
    """
    Render per-frame .npy vertex files → MP4 video.

    Args:
        npy_dir:     directory with 000000.npy, 000001.npy, ...
        faces:       template face indices (F, 3)
        output_path: directory to save video
        filename:    video name without .mp4; defaults to npy_dir basename
        cfg:         RenderConfig; uses video_width / video_height
        fps:         overrides cfg.fps
        debug:       render only first 8 frames

    Returns:
        path to .mp4 file

    Example:
        _, faces = get_mesh('ict')
        render_sequence_mi('./eval_CBD/.../verts', faces, './video_pseudo/')
    """
    cfg = cfg or VIDEO_CFG
    fps = fps or cfg.fps
    os.makedirs(output_path, exist_ok=True)

    frame_files = sorted(glob(os.path.join(npy_dir, '*.npy')))
    if not frame_files:
        raise FileNotFoundError(f"No .npy files in {npy_dir}")
    if debug:
        frame_files = frame_files[:8]

    filename   = filename or Path(npy_dir).parent.name
    video_path = os.path.join(output_path, f'{filename}.mp4')
    frame_dir  = os.path.join(output_path, f'{filename}_frames')
    os.makedirs(frame_dir, exist_ok=True)

    vcfg = replace(cfg, width=cfg.video_width, height=cfg.video_height)

    for i, npy_path in enumerate(tqdm(frame_files, desc=filename)):
        verts = np.load(npy_path)
        render_frame(
            verts, faces, vcfg,
            save_path=os.path.join(frame_dir, f'{i:06d}.png'),
            return_img=False,
        )

    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(frame_dir, '%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        video_path,
    ], check=True)

    print(f"[pyrender] Saved: {video_path}")
    return video_path


# ── Mesh loaders (same as render_mitsuba) ─────────────────────────────────────
def get_mesh(selection: str, SELECT_MESH: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load template (vertices, faces) for a given dataset.
    selection: 'ict' | 'ict-cap' | 'voca' | 'coma' | 'biwi' | 'mf' | 'mf_ROM' | 'mf_SEN'
    """
    abs_path = str(Path(__file__).parent.parent.absolute())
    sys.path.insert(0, abs_path)

    if selection in ('ict', 'ict-cap'):
        import torch
        from utils.remesh_utils import ICT_face_model
        ict = ICT_face_model()
        id_vecs = torch.load(f'{abs_path}/ict_face_pt/ict_id_vecs_test.pt').numpy()
        id_disps = ict.get_id_disp(id_vecs[SELECT_MESH])
        verts = ict.neutral_verts.squeeze() + id_disps.squeeze()
        return verts, ict.faces

    if selection in ('voca', 'coma'):
        with open('/data/sihun/VOCA-COMA/voca_templates.pkl', 'rb') as f:
            mesh = pickle.load(f)
    elif selection == 'biwi':
        biwi_tri = trimesh.load(f'{abs_path}/test-mesh/BIWI.ply')
        with open(f'{abs_path}/test-mesh/biwi_templates.pkl', 'rb') as f:
            mesh = pickle.load(f)
        mesh['face'] = biwi_tri.faces
    elif selection in ('mf', 'mf_ROM', 'mf_SEN'):
        with open('/data/sihun/pca/multiface_align/mf_templates.pkl', 'rb') as f:
            mesh = pickle.load(f)
    else:
        raise ValueError(f"Unknown selection: {selection}")

    keys  = [k for k in mesh if k != 'face']
    verts = mesh[keys[SELECT_MESH]]

    if selection == 'biwi':
        m_align = np.load(f'{abs_path}/utils/biwi/align.npy')
        verts = np.concatenate((verts, np.ones((verts.shape[0], 1))), axis=1) @ m_align.T

    return verts, mesh['face']


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pyrender pseudo face renderer')
    parser.add_argument('--npy_dir',   type=str, required=True)
    parser.add_argument('--mesh_type', type=str, default='ict',
                        choices=['ict', 'ict-cap', 'voca', 'coma', 'biwi', 'mf', 'mf_ROM', 'mf_SEN'])
    parser.add_argument('--output',    type=str, default='./video_pseudo/')
    parser.add_argument('--filename',  type=str, default=None)
    parser.add_argument('--fps',       type=int, default=30)
    parser.add_argument('--width',     type=int, default=800)
    parser.add_argument('--height',    type=int, default=800)
    parser.add_argument('--scale',     type=float, default=0.25)
    parser.add_argument('--no_ground', action='store_true')
    parser.add_argument('--paper',     action='store_true')
    parser.add_argument('--teaser',    action='store_true')
    parser.add_argument('--debug',     action='store_true')
    args = parser.parse_args()
    print("[Args]")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    if args.paper:
        cfg = PAPER_CFG
    elif args.teaser:
        cfg = TEASER_CFG
    else:
        cfg = RenderConfig(
            width=args.width, height=args.height,
            video_width=args.width, video_height=args.height,
            mesh_scale=args.scale,
            ground_plane=not args.no_ground,
        )

    _, faces = get_mesh(args.mesh_type)
    render_sequence_mi(
        npy_dir=args.npy_dir, faces=faces,
        output_path=args.output, filename=args.filename,
        cfg=cfg, fps=args.fps, debug=args.debug,
    )
