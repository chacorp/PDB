"""
Supplementary video, stage 2: render each output separately as a PNG sequence
with pyrender (OSMesa, black background, plain shading -- no diff heatmap) from
the per-frame cache of render/supp_video_infer.py:
    vis_CBD/supp_video/cells/{video}/{scenario}/ref/source_neutral.png, target_neutral.png
    vis_CBD/supp_video/cells/{video}/{scenario}/source_expression/frame_XXXXXX.png
    vis_CBD/supp_video/cells/{video}/{scenario}/{model}/{self|cross|cyclic}/frame_XXXXXX.png
Meshes of one identity share the framing of its neutral (center / scale), so the
panels are comparable. Existing PNGs are skipped. Needs render/set_render.sh.

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
from PIL import Image  # noqa: E402

from utils.matplotlib_rnd import fix_triangle_widning  # noqa: E402

VERTS_ROOT = REPO_ROOT / "vis_CBD" / "supp_video" / "verts"
CELLS_ROOT = REPO_ROOT / "vis_CBD" / "supp_video" / "cells"
CELL_PX = {"design_study": 256, "comparison": 400, "loss_ablation": 400}
FIT = 1.8  # largest neutral extent -> this width of the [-1, 1] ortho view


class CellRenderer:
    def __init__(self, size):
        self.size = size
        self.r = pyrender.OffscreenRenderer(size, size)
        self.material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.6, 0.6, 0.6, 1.0], metallicFactor=0.0, roughnessFactor=0.8)

    def render(self, V, F, center, scale, out_path):
        V = (np.asarray(V, float) - center) * scale
        mesh = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(vertices=V, faces=F, process=False), material=self.material, smooth=True)
        scene = pyrender.Scene(bg_color=[0, 0, 0, 1], ambient_light=[0.05, 0.05, 0.05])
        scene.add(mesh)
        cam_pose = np.eye(4)
        cam_pose[2, 3] = 5.0
        scene.add(pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.1, zfar=20.0), pose=cam_pose)
        # key light from upper left, weaker fill from lower right and a side light
        for rx, ry, inten in [(-0.3, 0.3, 2.2), (0.3, -0.5, 0.5), (0, 0.8, 0.3)]:
            pose = trimesh.transformations.euler_matrix(rx, ry, 0)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=inten), pose=pose)
        color, _ = self.r.render(scene)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color).save(out_path)

    def close(self):
        self.r.delete()


def fit_of(neutral):
    center = (neutral.max(0) + neutral.min(0)) * 0.5
    return center, FIT / (neutral.max(0) - neutral.min(0)).max()


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
        src_fit, tgt_fit = fit_of(src_n), fit_of(tgt_n)
        for name, V, F, fit in [("source_neutral", src_n, f_src, src_fit), ("target_neutral", tgt_n, f_tgt, tgt_fit)]:
            p = out / "ref" / f"{name}.png"
            if not p.exists():
                rnd.render(V, F, *fit, p)
        # (output folder, faces, framing): source expression + every model / row
        jobs = [(ref / "source_expression", out / "source_expression", f_src, src_fit)]
        for mdir in sorted(p for p in sc_dir.iterdir() if p.is_dir() and p.name != "ref"):
            for row, F, fit in [("self", f_src, src_fit), ("cross", f_tgt, tgt_fit), ("cyclic", f_src, src_fit)]:
                jobs.append((mdir / row, out / mdir.name / row, F, fit))
        n = 0
        for vdir, cdir, F, fit in jobs:
            for vp in sorted(vdir.glob("frame_*.npy")):
                if only and int(vp.stem.split("_")[1]) not in only:
                    continue
                cp = cdir / f"{vp.stem}.png"
                if not cp.exists():
                    rnd.render(np.load(vp), F, *fit, cp)
                    n += 1
        print(f"{video}/{sc_dir.name}: rendered {n} cells", flush=True)
    rnd.close()
    print("ALL DONE:", CELLS_ROOT / video)


if __name__ == "__main__":
    main()
