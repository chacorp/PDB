"""
Mitsuba 3 high-quality renderer for NeuralFacialAnimation.
Features: path-traced soft shadows, reflections, skin material.
Headless server compatible (no display required).

SSS note: Mitsuba 3.6.1 principled BSDF approximates SSS via `flatness`
          (Disney diffuse → flat diffuse transition, mimics forward-scattering).

Install:
    # use docker jeolpyeoni0/gltorch:cu124-vessl
    # may need to run: apt-get update -y && apt-get install -y libnvidia-gl-535 ffmpeg && pip install ffmpeg-python mitsuba numpy==1.26.4

    ##### deprecated ########################################
    # may need to run: apt-get update && apt-get install libnvidia-gl-{*} 
    # NOTE: you need to change {*} with your driver number (ex: libnvidia-gl-470)
    # apt-get install nvidia-driver-470 libnvidia-gl-470 libnvidia-compute-470
    #########################################################
Usage:
    from render.render_mitsuba import render_frame, render_figure, PAPER_CFG
    render_figure([(src_v, f), (pred_v, f), (tgt_v, f)], PAPER_CFG, 'fig.png')
    
    python render_mitsuba.py --mesh_type 'mf' --npy_dir '../vis_CBD/2026-04-02-02-04-44-NGBCv5-dist/ict-cap-ID_002_test-to-mf_ROM-ID_012_test_00-masked/verts'
"""

import os
import sys
import math
import pickle
import tempfile
import argparse
import subprocess
from glob import glob
from pathlib import Path
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from tqdm import tqdm

# ── Mitsuba init ──────────────────────────────────────────────────────────────
import mitsuba as mi

def _init_mitsuba() -> str:
    for variant in ['cuda_ad_rgb', 'llvm_ad_rgb']:
        try:
            mi.set_variant(variant)
            # Actually test the variant by rendering a tiny scene
            t = mi.ScalarTransform4f()
            test_scene = mi.load_dict({
                'type': 'scene',
                'integrator': {'type': 'path', 'max_depth': 1},
                'sensor': {
                    'type': 'perspective', 'fov': 45,
                    'to_world': t.look_at(
                        mi.ScalarPoint3f(0, 0, 2),
                        mi.ScalarPoint3f(0, 0, 0),
                        mi.ScalarPoint3f(0, 1, 0),
                    ),
                    'film': {'type': 'hdrfilm', 'width': 4, 'height': 4},
                    'sampler': {'type': 'independent', 'sample_count': 1},
                },
                'sphere': {'type': 'sphere', 'center': [0, 0, 0], 'radius': 0.5},
                'env': {'type': 'constant'},
            })
            mi.render(test_scene, spp=1)
            print(f"[Mitsuba] variant: {variant}")
            return variant
        except Exception:
            continue
    raise RuntimeError("No Mitsuba variant found. pip install mitsuba")

VARIANT = _init_mitsuba()
_T = mi.ScalarTransform4f()   # used as factory for transforms


def _lookat(origin, target, up=(0, 1, 0)):
    return _T.look_at(
        mi.ScalarPoint3f(*origin),
        mi.ScalarPoint3f(*target),
        mi.ScalarPoint3f(*up),
    )

def _translate(v):
    return _T.translate(mi.ScalarVector3f(*v))

def _rotate(axis, angle_deg):
    return _T.rotate(mi.ScalarVector3f(*axis), float(angle_deg))

def _scale(s):
    if isinstance(s, (int, float)):
        return _T.scale(mi.ScalarVector3f(s, s, s))
    return _T.scale(mi.ScalarVector3f(*s))


# ── Configuration ─────────────────────────────────────────────────────────────
@dataclass
class RenderConfig:
    # Image
    width:     int   = 1024
    height:    int   = 1024
    spp:       int   = 132
    max_depth: int   = 8

    # Camera (spherical coords around origin)
    fov:           float = 20.0   # vertical FOV, degrees
    cam_distance:  float = 2.5
    cam_elevation: float = 8.0    # degrees above horizon
    cam_azimuth:   float = 0.0    # 0=front, 90=left, -90=right
    cam_target:    Tuple = (0.0, -0.02, 0.0)

    # Geometry
    mesh_scale: float = 0.225     # 0.25 * 0.9, suits ICT/MF world-space coords

    # Skin BSDF (Disney Principled)
    base_color:  Tuple = (0.80, 0.64, 0.52)
    roughness:   float = 0.55
    metallic:    float = 0.0
    sheen:       float = 0.12
    spec_tint:   float = 0.25
    # SSS approximation via Disney flatness (0=standard diffuse, 1=flat/forward)
    flatness:    float = 0.30

    # 3-point lighting (sphere area lights, positioned outside camera FOV)
    key_pos:        Tuple = (-1.4,  1.8,  0.8)
    key_radius:     float = 0.40
    key_intensity:  float = 30.0
    key_color:      Tuple = (1.00, 0.95, 0.88)   # warm

    fill_pos:       Tuple = ( 1.6,  0.5,  0.8)
    fill_radius:    float = 0.60
    fill_intensity: float = 10.0
    fill_color:     Tuple = (0.82, 0.88, 1.00)   # cool

    rim_pos:        Tuple = ( 0.2,  2.2, -2.5)
    rim_radius:     float = 0.30
    rim_intensity:  float = 16.0
    rim_color:      Tuple = (1.00, 1.00, 0.96)   # neutral

    env_intensity:  float = 0.28   # constant ambient

    # Depth of field (thinlens; 0.0 = pinhole / no DOF)
    aperture_radius: float = 0.02
    focus_distance:  float = 2.5

    # Scene
    bg_color:    Tuple = (1.0, 1.0, 1.0)   # white
    ground_plane: bool  = True
    ground_color: Tuple = (0.65, 0.65, 0.65)
    ground_y:    float  = -0.5

    # Video (overrides width/height/spp in render_sequence_mi)
    fps:          int = 30
    video_spp:    int = 400
    video_width:  int = 800
    video_height: int = 800


# ── Presets ───────────────────────────────────────────────────────────────────
PAPER_CFG = RenderConfig(
    width=1024, height=1024, spp=512, max_depth=10,
)

TEASER_CFG = RenderConfig(
    width=1200, height=900, spp=512, max_depth=10,
    bg_color=(0.08, 0.08, 0.10),
    env_intensity=0.10,
    ground_color=(0.10, 0.10, 0.13),
    ground_y=-0.48,
    key_intensity=52.0,
    rim_intensity=32.0,
)

VIDEO_CFG = RenderConfig(
    width=800, height=800, spp=256, max_depth=6,
    flatness=0.0,   # faster without SSS approximation
)


# ── Internal helpers ──────────────────────────────────────────────────────────
def _cam_pos(dist: float, elev_deg: float, azim_deg: float) -> tuple:
    el, az = math.radians(elev_deg), math.radians(azim_deg)
    return (
        dist * math.cos(el) * math.sin(az),
        dist * math.sin(el),
        dist * math.cos(el) * math.cos(az),
    )


def _prepare_mesh(vertices: np.ndarray, faces: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    verts = vertices * scale
    return verts, faces


def _export_obj(vertices: np.ndarray, faces: np.ndarray, path: str) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(path)


def _sphere_light(pos: tuple, radius: float, intensity: float, color: tuple) -> dict:
    return {
        'type': 'sphere',
        'center': list(pos),
        'radius': radius,
        'emitter': {
            'type': 'area',
            'radiance': {'type': 'rgb', 'value': [intensity * c for c in color]},
        },
    }


def _skin_bsdf(cfg: RenderConfig) -> dict:
    return  {
        'type': 'twosided',
        'material': {
            'type': 'principled',
            'base_color': {'type': 'rgb', 'value': list(cfg.base_color)},
            'roughness':  cfg.roughness,
            'metallic':   cfg.metallic,
            'sheen':      cfg.sheen,
            'spec_tint':  cfg.spec_tint,
            'flatness':   cfg.flatness,
        },
    }


def _rotate_xz(pos: tuple, azim_deg: float) -> tuple:
    """Rotate a point around the Y-axis by azim_deg (for camera-relative lights)."""
    a = math.radians(azim_deg)
    x, y, z = pos
    return (x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a))


def _build_scene(obj_path: str, cfg: RenderConfig, W: int, H: int, spp: int) -> dict:
    cam_pos = _cam_pos(cfg.cam_distance, cfg.cam_elevation, cfg.cam_azimuth)
    target  = cfg.cam_target
    az      = cfg.cam_azimuth

    # Lights rotate with the camera so they're never in frame
    key_pos  = _rotate_xz(cfg.key_pos,  az)
    fill_pos = _rotate_xz(cfg.fill_pos, az)
    rim_pos  = _rotate_xz(cfg.rim_pos,  az)

    # Ground: XY rect → rotate to horizontal → move to ground_y
    ground_T = _translate([0, cfg.ground_y, 0]) @ _rotate([1, 0, 0], -90) @ _scale(8)

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
            'type': 'thinlens' if cfg.aperture_radius > 0 else 'perspective',
            'fov': cfg.fov,
            'to_world': _lookat(cam_pos, target),
            **({'aperture_radius': cfg.aperture_radius,
                'focus_distance':  cfg.focus_distance}
               if cfg.aperture_radius > 0 else {}),
            'film': {
                'type': 'hdrfilm',
                'width': W, 'height': H,
                'pixel_filter': {'type': 'box'},
            },
            'sampler': {'type': 'multijitter', 'sample_count': spp},
        },
        # Constant ambient + background color
        'env': {
            'type': 'constant',
            'radiance': {
                'type': 'rgb',
                'value': [cfg.env_intensity * c for c in cfg.bg_color],
            },
        },
        # Face mesh
        'face': {
            'type': 'obj',
            'filename': obj_path,
            'bsdf': _skin_bsdf(cfg),
        },
        # 3-point area lights — co-rotate with camera so always off-frame
        'key_light':  _sphere_light(key_pos,  cfg.key_radius,  cfg.key_intensity,  cfg.key_color),
        'fill_light': _sphere_light(fill_pos, cfg.fill_radius, cfg.fill_intensity, cfg.fill_color),
        'rim_light':  _sphere_light(rim_pos,  cfg.rim_radius,  cfg.rim_intensity,  cfg.rim_color),
    }

    if cfg.ground_plane:
        scene['ground'] = {
            'type': 'rectangle',
            'to_world': ground_T,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {'type': 'rgb', 'value': list(cfg.ground_color)},
            },
        }

    return scene


def _tonemap(image_np: np.ndarray) -> np.ndarray:
    """Reinhard tonemap + gamma correction → uint8 RGB."""
    img = np.clip(image_np[..., :3], 0, None)
    img = img / (1.0 + img)                    # Reinhard
    img = np.clip(img ** (1.0 / 2.2), 0, 1)
    return (img * 255).astype(np.uint8)


def _bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 15, sigma_space: float = 15) -> np.ndarray:
    """Bilateral filter post-process (edge-preserving smoothing)."""
    try:
        import cv2
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    except ImportError:
        return img


def _background_blur(img: np.ndarray, albedo_raw: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Blur only background pixels using albedo-based face mask.

    Face skin has color variance (R≠G≠B); background/ground is neutral gray (R=G=B).
    """
    import cv2

    # std across RGB channels: high for colored face skin, ~0 for gray background
    std = albedo_raw.std(axis=-1).astype(np.float32)   # (H, W)
    face_mask = (std > 0.05).astype(np.uint8)

    # dilate slightly to include face edges cleanly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    face_mask = cv2.dilate(face_mask, kernel)

    # smooth mask edges for natural blending
    soft_mask = cv2.GaussianBlur(face_mask.astype(np.float32), (51, 51), 15)[..., np.newaxis]

    ksize = int(sigma * 6) | 1   # odd kernel size ≥ 6σ
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    result = img.astype(np.float32) * soft_mask + blurred.astype(np.float32) * (1 - soft_mask)
    return result.clip(0, 255).astype(np.uint8)


def _denoise(image: "mi.TensorXf", albedo: "mi.TensorXf" = None, normals: "mi.TensorXf" = None) -> "mi.TensorXf":
    """Apply mi.OptixDenoiser if available (cuda_ad_rgb variant only)."""
    if not hasattr(mi, 'OptixDenoiser'):
        return image
    try:
        use_albedo  = albedo is not None
        use_normals = normals is not None
        H, W = image.shape[:2]
        denoiser = mi.OptixDenoiser(
            input_size=[H, W],
            albedo=use_albedo,
            normals=use_normals,
            temporal=False,
        )
        kwargs = {}
        if use_albedo:
            kwargs['albedo'] = albedo
        if use_normals:
            kwargs['normals'] = normals
        return denoiser(image, **kwargs)
    except Exception as e:
        print(f"[Mitsuba] OptixDenoiser skipped: {e}")
        return image


def _save_img(img: np.ndarray, path: str) -> None:
    os.makedirs(Path(path).parent, exist_ok=True)
    try:
        import cv2
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(path)


# ── Public API ────────────────────────────────────────────────────────────────
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
    verts, faces_out = _prepare_mesh(vertices, faces, cfg.mesh_scale)

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path = os.path.join(tmpdir, 'mesh.obj')
        _export_obj(verts, faces_out, obj_path)
        scene = mi.load_dict(
            _build_scene(obj_path, cfg, cfg.width, cfg.height, cfg.spp)
        )
        image = mi.render(scene, spp=cfg.spp)

    # aov integrator outputs: [rgb(3), albedo(3), normals(3)] = 9 channels
    image_np_raw = np.array(image)
    albedo_raw = image_np_raw[..., 3:6] if image_np_raw.shape[-1] >= 6 else None
    image_t  = mi.TensorXf(image_np_raw[..., :3])
    albedo_t = mi.TensorXf(image_np_raw[..., 3:6]) if albedo_raw is not None else None
    normals_t= mi.TensorXf(image_np_raw[..., 6:9]) if image_np_raw.shape[-1] >= 9 else None
    denoised = _denoise(image_t, albedo_t, normals_t)
    img_np = _tonemap(np.array(denoised))
    img_np = _bilateral(img_np)

    if save_path:
        _save_img(img_np, save_path)
    return img_np if return_img else None


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
        cfg:         uses cfg.video_spp / video_width / video_height
        fps:         overrides cfg.fps
        debug:       render only first 8 frames

    Returns:
        path to .mp4 file

    Example:
        _, faces = get_mesh('ict')
        render_sequence_mi('./eval_CBD/.../verts', faces, './video_mi/')
    """
    cfg = cfg or VIDEO_CFG
    fps = fps or cfg.fps
    os.makedirs(output_path, exist_ok=True)

    frame_files = sorted(glob(os.path.join(npy_dir, '*.npy')))
    if not frame_files:
        raise FileNotFoundError(f"No .npy files in {npy_dir}")
    if debug:
        frame_files = frame_files[:8]

    filename  = filename or Path(npy_dir).parent.name
    video_path = os.path.join(output_path, f'{filename}.mp4')
    frame_dir  = os.path.join(output_path, f'{filename}_frames')
    os.makedirs(frame_dir, exist_ok=True)

    vcfg = replace(cfg, width=cfg.video_width, height=cfg.video_height, spp=cfg.video_spp)

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

    print(f"[Mitsuba] Saved: {video_path}")
    return video_path


# ── Mesh loaders ──────────────────────────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description='Mitsuba 3 face renderer')
    parser.add_argument('--npy_dir',   type=str, required=True)
    parser.add_argument('--mesh_type', type=str, default='ict',
                        choices=['ict', 'ict-cap', 'voca', 'coma', 'biwi', 'mf', 'mf_ROM', 'mf_SEN'])
    parser.add_argument('--output',    type=str, default='./video_mi/')
    parser.add_argument('--filename',  type=str, default=None)
    parser.add_argument('--fps',       type=int, default=30)
    parser.add_argument('--spp',       type=int, default=528)
    parser.add_argument('--width',     type=int, default=600)
    parser.add_argument('--height',    type=int, default=600)
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
            spp=args.spp, video_spp=args.spp,
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
    # render_sequence(
    #     output_path = output_path,
    #     npy_file = "./vis_CBD/2025-10-22-22-30-30-NGBCv5/ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked/verts",
    #     use_seg_color=use_seg_color, mesh_type='ict',
    # )
