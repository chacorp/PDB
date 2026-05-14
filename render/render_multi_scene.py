"""
Multi-scene 6-face renderer for supplementary video.

Layout: 6 faces in a single horizontal row  →  1050 × 350 px
        (each face rendered at 350×700, then scaled ×0.5 → 175×350)

Output:
  video_mi/multi_scene_6faces_new.mp4
  video_mi/multi_scene_6faces_new.gif

Usage:
    cd render
    python render_multi_scene.py          # full render (1147 frames)
    python render_multi_scene.py --debug  # first 4 frames only
"""

import argparse
import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, ROOT)

VIS_DIR = os.path.join(ROOT, 'vis_CBD', '2026-04-27-15-36-49-NGBCv5')
OUT_DIR  = os.path.join(ROOT, 'video_mi')

SEQUENCES = [
    ('ict-cap-ID_000_test-to-ict-neutral_01-masked',        'ict'),
    ('ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked','ict'),
    ('ict-cap-ID_000_test-to-biwi-ID_012_test_01-masked',   'biwi'),
    ('ict-cap-ID_000_test-to-coma-ID_006_test_01-masked',   'coma'),
    ('ict-cap-ID_000_test-to-coma-ID_008_test_01-masked',   'coma'),
    ('ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01-masked', 'mf_ROM'),
]

# ── render config ──────────────────────────────────────────────────────────────
# Portrait render per face: 350×700  →  stack 6  →  2100×700  →  scale 0.5  →  1050×350
FACE_W, FACE_H = 350, 700   # per-face render size
OUT_W,  OUT_H  = 1050, 350  # final output size
SPP   = 256
FPS   = 30


def build_cfg(mesh_scale: float = 0.225):
    from render_mitsuba import RenderConfig
    return RenderConfig(
        width=FACE_W, height=FACE_H,
        video_width=FACE_W, video_height=FACE_H,
        spp=SPP, video_spp=SPP,
        max_depth=6,
        fov=20.0,
        cam_distance=2.5,
        cam_elevation=8.0,
        cam_azimuth=0.0,
        cam_target=(0.0, -0.02, 0.0),
        mesh_scale=mesh_scale,
        ground_plane=True,
        bg_color=(1.0, 1.0, 1.0),
        ground_color=(0.65, 0.65, 0.65),
        ground_y=-0.5,
        flatness=0.0,   # faster
        fps=FPS,
    )


def render_all(debug: bool = False):
    import cv2
    from PIL import Image
    from render_mitsuba import render_frame, get_mesh

    os.makedirs(OUT_DIR, exist_ok=True)
    frame_dir = os.path.join(OUT_DIR, 'multi_scene_6faces_new_frames')
    os.makedirs(frame_dir, exist_ok=True)

    # ── load face topologies once ──────────────────────────────────────────────
    print("Loading mesh templates...")
    faces_cache = {}
    for _, mesh_type in SEQUENCES:
        if mesh_type not in faces_cache:
            _, f = get_mesh(mesh_type)
            faces_cache[mesh_type] = f
            print(f"  loaded: {mesh_type}  faces={f.shape}")

    # ── load per-sequence npy paths ────────────────────────────────────────────
    seq_frames = []
    for seq_name, mesh_type in SEQUENCES:
        npy_dir = os.path.join(VIS_DIR, seq_name, 'verts')
        paths   = sorted(glob(os.path.join(npy_dir, '*.npy')))
        if not paths:
            raise FileNotFoundError(f"No .npy files in {npy_dir}")
        seq_frames.append((paths, mesh_type))
        print(f"  {seq_name}: {len(paths)} frames")

    n_frames = min(len(p) for p, _ in seq_frames)
    if debug:
        n_frames = 4
    print(f"\nRendering {n_frames} frames × 6 faces  →  {OUT_W}×{OUT_H}")

    cfg = build_cfg()

    # ── frame loop ────────────────────────────────────────────────────────────
    for i in tqdm(range(n_frames), desc='frames'):
        panels = []
        for (paths, mesh_type) in seq_frames:
            verts = np.load(paths[i])
            faces = faces_cache[mesh_type]
            img   = render_frame(verts, faces, cfg)   # (FACE_H, FACE_W, 3) uint8
            panels.append(img)

        # hstack → 2100×700, scale → 1050×350
        composite = np.concatenate(panels, axis=1)                           # (700, 2100, 3)
        composite = cv2.resize(composite, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)  # (350, 1050, 3)

        out_path = os.path.join(frame_dir, f'{i:06d}.png')
        cv2.imwrite(out_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    # ── assemble MP4 ──────────────────────────────────────────────────────────
    mp4_path = os.path.join(OUT_DIR, 'multi_scene_6faces_new.mp4')
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', os.path.join(frame_dir, '%06d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        mp4_path,
    ], check=True)
    print(f"\n[MP4] {mp4_path}")

    # ── assemble GIF ──────────────────────────────────────────────────────────
    gif_path = os.path.join(OUT_DIR, 'multi_scene_6faces_new.gif')
    png_paths = sorted(glob(os.path.join(frame_dir, '*.png')))
    frames_pil = [Image.open(p).convert('P', palette=Image.ADAPTIVE, colors=256)
                  for p in png_paths]
    frames_pil[0].save(
        gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        loop=0,
        duration=int(1000 / FPS),
        optimize=False,
    )
    print(f"[GIF] {gif_path}")

    # ── verify ────────────────────────────────────────────────────────────────
    sample = np.array(Image.open(png_paths[0]).convert('RGB'))
    print(f"\n[Verify] composite frame shape: {sample.shape}  expected: ({OUT_H}, {OUT_W}, 3)")
    assert sample.shape == (OUT_H, OUT_W, 3), f"Shape mismatch: {sample.shape}"
    print("[Verify] PASS")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Render only first 4 frames')
    args = parser.parse_args()
    render_all(debug=args.debug)
