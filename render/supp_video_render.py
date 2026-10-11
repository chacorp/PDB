"""
Supplementary video, stage 2: render each output separately as a PNG sequence
with pyrender (OSMesa, plain shading -- no diff heatmap) from
the per-frame cache of render/supp_video_infer.py:
    vis_CBD/supp_video/cells/{video}/{scenario}/ref/source_neutral.png, target_neutral.png
    vis_CBD/supp_video/cells/{video}/{scenario}/source_expression/frame_XXXXXX.png
    vis_CBD/supp_video/cells/{video}/{scenario}/{model}/{self|cross|cyclic}/frame_XXXXXX.png
Render settings follow render_trimesh_gpu_pyrender.render_sequence (raw mesh
coordinates, same camera / lights). Existing PNGs are skipped. Needs render/set_render.sh.

    SUPP_VIDEO=comparison [SUPP_FRAMES=4300] [SUPP_CELL=400] python render/supp_video_render.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import pyrender  # noqa: E402
import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from utils.matplotlib_rnd import fix_triangle_widning  # noqa: E402

VERTS_ROOT = REPO_ROOT / "vis_CBD" / "supp_video" / "verts"
CELLS_ROOT = REPO_ROOT / "vis_CBD" / "supp_video" / "cells"
CELL_PX = {"design_study": 256, "comparison": 400, "loss_ablation": 400}


class CellRenderer:
    """render_trimesh_gpu_pyrender.render_sequence settings: IntrinsicsCamera
    f = 4754.98/2 at 800 px (scaled with the cell size), camera at z=1, vertices
    * 0.5 - [0, 0, 3.6], five directional lights (intensity 1.0), ambient 0.2,
    black background, default material, smooth shading, SKIP_CULL_FACES."""

    def __init__(self, size):
        self.size = size
        k = size / 800.0
        cam = pyrender.IntrinsicsCamera(fx=4754.97941935 / 2 * k, fy=4754.97941935 / 2 * k,
                                        cx=size / 2, cy=size / 2, znear=0.001, zfar=10.0)
        self.scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
        self.scene.add(cam, pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        light = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]), intensity=1.0)
        angle, camera_pos = np.pi / 4.0, np.array([0.0, 0.0, 1.0 - 1.3])
        for lrot in [None, np.array([angle, 0, 0]), np.array([-angle, 0, 0]),
                     np.array([0, -angle, 0]), np.array([0, angle, 0])]:
            lp = np.eye(4)
            lp[:3, 3] = cv2.Rodrigues(lrot)[0].dot(camera_pos) if lrot is not None else camera_pos
            self.scene.add(light, pose=lp)
        self.r = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size)
        self.node = None

    def render(self, V, F, out_path):
        verts = np.asarray(V, float) * 0.5 - np.array([0.0, 0.0, 3.6])
        mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=verts, faces=F, process=False),
                                          material=None, smooth=True)
        if self.node is not None:
            self.scene.remove_node(self.node)
        self.node = self.scene.add(mesh, pose=np.eye(4))
        color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color).save(out_path)

    def close(self):
        self.r.delete()


def main():
    video = os.environ["SUPP_VIDEO"]
    only = [int(x) for x in os.environ.get("SUPP_FRAMES", "").split(",") if x]
    size = int(os.environ.get("SUPP_CELL", CELL_PX[video]))
    rnd = CellRenderer(size)
    for sc_dir in sorted((VERTS_ROOT / video).iterdir()):
        ref = sc_dir / "ref"
        out = CELLS_ROOT / video / sc_dir.name
        src_n, tgt_n = np.load(ref / "source_neutral.npy"), np.load(ref / "target_neutral.npy")
        f_src = fix_triangle_widning(src_n, np.load(ref / "faces_src.npy"))
        f_tgt = fix_triangle_widning(tgt_n, np.load(ref / "faces_tgt.npy"))
        for name, V, F in [("source_neutral", src_n, f_src), ("target_neutral", tgt_n, f_tgt)]:
            p = out / "ref" / f"{name}.png"
            if not p.exists():
                rnd.render(V, F, p)
        # (output folder, faces): source expression + every model / row
        jobs = [(ref / "source_expression", out / "source_expression", f_src)]
        for mdir in sorted(p for p in sc_dir.iterdir() if p.is_dir() and p.name != "ref"):
            for row, F in [("self", f_src), ("cross", f_tgt), ("cyclic", f_src)]:
                jobs.append((mdir / row, out / mdir.name / row, F))
        n = 0
        for vdir, cdir, F in jobs:
            for vp in sorted(vdir.glob("frame_*.npy")):
                if only and int(vp.stem.split("_")[1]) not in only:
                    continue
                cp = cdir / f"{vp.stem}.png"
                if not cp.exists():
                    rnd.render(np.load(vp), F, cp)
                    n += 1
        print(f"{video}/{sc_dir.name}: rendered {n} cells", flush=True)
    rnd.close()
    print("ALL DONE:", CELLS_ROOT / video)


if __name__ == "__main__":
    main()
