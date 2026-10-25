"""
Assets for the PDB method video (render/method_video.py, run in the Manim venv).
Runs in the main env, no inference: everything comes from the stored results of
ckpts_CBD/2026-04-27-15-36-49-NGBCv5 (render/save_dynamic_6faces_data.py, the same data
as the weight_vis_combined.mp4 overview):
  source = ict-cap ID_000 exp 1 (20240325_MySlate_924), target = mf_ROM ID_012,
  output = ...-to-mf_ROM-ID_012_test_01-masked/verts/, points v = key_d/ of the ICT-neutral
  target run (v depends on the source only; output = w @ v), w = key_weight.npy (not stored
  by save_dynamic_6faces_data.py for this target: one psi pass computes it once and caches it).
Frames 1057-1146 (the last 90 frames, containing frame 1117 of vis_comparison.html);
the still images use frame 1117.

Writes vis_CBD/method_video/assets/:
  face/      : normal-colored inputs, w (nipy_spectral blend colors, as render_weight_vis.py),
               output, the expression sequences (source / target), and the weight maps of
               3 picked points v (PICKS) on mf, ict m02 and coma (each in its own point color)
  w.npy      : target weights w (Nt, 512)
  points.npz : the points v used by the target (orthographic x / y, normalized to [-1, 1])
               per frame, their colors, the still frame index, the 3 picked points, the mesh names, and
               vec: phi's flat output (all K points, 3K values) per frame as colors

    python render/method_video_assets.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))
sys.argv = [sys.argv[0]]

import numpy as np  # noqa: E402
import igl  # noqa: E402
import trimesh  # noqa: E402
import pyrender  # noqa: E402
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from dataloader_CBD import EvalDataset  # noqa: E402
from vis_CBD_retarget_fig import Pipeline  # noqa: E402
from utils.matplotlib_rnd import fix_triangle_widning  # noqa: E402

MF_DATA = (REPO_ROOT / "vis_CBD" / "2026-04-27-15-36-49-NGBCv5-best"
           / "ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01-masked")
KEYD_DATA = (REPO_ROOT / "vis_CBD" / "2026-04-27-15-36-49-NGBCv5-best"
        / "ict-cap-ID_000_test-to-ict-neutral_01-masked")
OUT = REPO_ROOT / "vis_CBD" / "method_video" / "assets"
FRAMES = list(range(1057, 1147))  # last 90 frames of the sequence, containing frame 1117
STILL = 1117                      # frame of the still images (vis_comparison.html frame)
SIZE = 720
# the 3 points v whose weight maps are shown (brow, cheek, nose regions)
PICKS = [325, 413, 509]
# point colors replaced in the weight maps / pick labels only (nipy_spectral v509 is near white)
PICK_COLOR = {509: (1.0, 0.25, 0.25)}
# other neutrals for the same weight maps (as vis_comparison-2.html): ict-cap m02 and coma
US_VERTS = REPO_ROOT / "vis_CBD" / "userstudy" / "verts"
EXTRA_TARGETS = {"ict": (US_VERTS / "ict_m02/ref/source_neutral.npy", US_VERTS / "ict_m02/ref/faces_src.npy"),
                 "coma": (US_VERTS / "mf/coma/ref/target_neutral.npy", US_VERTS / "mf/coma/ref/faces_tgt.npy")}
# camera (render_trimesh_gpu_pyrender layout: vertices * 0.5 - [0, 0, 3.6], camera on +z)
CAM_DIST = 18.0
F_FACE = 4754.97941935 / 2 * (SIZE / 800.0) * 15.0 / 4.6   # supplementary-video framing
HALF_WIDE = 2.0                                            # half view width (scaled units) for wide
F_WIDE = (SIZE / 2) * CAM_DIST / HALF_WIDE


class Renderer:
    def __init__(self, f):
        self.f = f
        cam = pyrender.IntrinsicsCamera(fx=f, fy=f, cx=SIZE / 2, cy=SIZE / 2,
                                        znear=CAM_DIST - 8.0, zfar=CAM_DIST + 8.0)
        self.scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0, 0])  # transparent bg
        self.cam_z = 1.0 + (CAM_DIST - 4.6)
        self.scene.add(cam, pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.cam_z], [0, 0, 0, 1]])
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        angle, camera_pos = np.pi / 4.0, np.array([0.0, 0.0, 1.0 - 1.3])
        for lrot in [None, np.array([angle, 0, 0]), np.array([-angle, 0, 0]),
                     np.array([0, -angle, 0]), np.array([0, angle, 0])]:
            lp = np.eye(4)
            lp[:3, 3] = cv2.Rodrigues(lrot)[0].dot(camera_pos) if lrot is not None else camera_pos
            self.scene.add(light, pose=lp)
        self.r = pyrender.OffscreenRenderer(SIZE, SIZE)
        self.node = None

    @staticmethod
    def world(V):
        return np.asarray(V, float) * 0.5 - np.array([0.0, 0.0, 3.6])

    def project(self, P):
        """3D mesh-space points -> pixel coords (x right, y down) of this camera"""
        W = self.world(P)
        z = self.cam_z - W[:, 2]
        return np.stack([self.f * W[:, 0] / z + SIZE / 2, SIZE / 2 - self.f * W[:, 1] / z], 1)

    def render(self, V, F, path, colors=None, wire=False, flat=False):
        """RGBA PNG with a transparent background (so neighbouring images don't cover each
        other in the video); flat=True shows vertex colors unlit (normal maps)"""
        tm = trimesh.Trimesh(vertices=self.world(V), faces=F, process=False)
        if colors is not None:
            tm.visual.vertex_colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        mesh = pyrender.Mesh.from_trimesh(tm, smooth=True, wireframe=wire)
        if self.node is not None:
            self.scene.remove_node(self.node)
        self.node = self.scene.add(mesh)
        flags = pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.RGBA
        if flat:
            flags |= pyrender.RenderFlags.FLAT
        color, _ = self.r.render(self.scene, flags=flags)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(color).save(path)


def build_colormaps(key_weight):
    """same as render_weight_vis.build_colormaps (copied: that module imports mitsuba):
    one nipy_spectral color per point v, vertex color = w @ point colors, per-channel
    min-max normalized, saturation x1.25"""
    N, M = key_weight.shape
    cmap = plt.get_cmap('nipy_spectral')
    cage_rgb = np.array([cmap(i / (M - 1))[:3] for i in range(M)], dtype=np.float32)
    max_ch = cage_rgb.max(axis=1)
    is_zero = max_ch < 1e-9
    scale = np.where((~is_zero) & (max_ch < 0.25), 0.25 / (max_ch + 1e-12), 1.0)
    cage_rgb = cage_rgb * scale[:, None]
    cage_rgb[is_zero] = 0.5
    cage_rgb = np.clip(cage_rgb, 0.0, 1.0).astype(np.float32)
    rgb = key_weight @ cage_rgb
    rgb_min, rgb_max = rgb.min(axis=0), rgb.max(axis=0)
    rgb = np.clip((rgb - rgb_min) / (rgb_max - rgb_min + 1e-12), 0.0, 1.0)
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0.0, 1.0)
    return hsv_to_rgb(hsv).astype(np.float32), cage_rgb


def mf_target_weights(T):
    """w of the mf_ROM ID_012 target: cached key_weight.npy, else one psi pass with the
    save_dynamic_6faces_data.py model (same options as its stored outputs)"""
    path = MF_DATA / "key_weight.npy"
    if path.exists():
        return np.load(path)
    import torch
    import save_dynamic_6faces_data as sd
    model, _ = sd.load_model()
    tv, tn = sd.th(T, np.load(MF_DATA / "faces.npy"))
    with torch.no_grad():
        w = model.predict_coordinate(tv, tn)[0].cpu().numpy()
    np.save(path, w)
    return w


def extra_target_weights():
    """{name: (V, F, w)} for EXTRA_TARGETS: one psi pass each with the same model as
    mf_target_weights, cached as OUT/w_{name}.npy"""
    out, model = {}, None
    for name, (vp, fp) in EXTRA_TARGETS.items():
        V, F = np.load(vp).astype(np.float64), np.load(fp)
        path = OUT / f"w_{name}.npy"
        if not path.exists():
            import torch
            import save_dynamic_6faces_data as sd
            if model is None:
                model, _ = sd.load_model()
            tv, tn = sd.th(V, F)
            with torch.no_grad():
                np.save(path, model.predict_coordinate(tv, tn)[0].cpu().numpy())
        out[name] = (V, fix_triangle_widning(V, F), np.load(path))
    return out


def main():
    ds = EvalDataset(data_name="ict-cap", toggle=False, ict_cap_id_num=0, ict_cap_exp_num=1)
    S = ds[0][1].numpy()
    Sf = fix_triangle_widning(S, np.asarray(ds[0][4]))
    E = [ds[i][0].numpy() for i in FRAMES]
    T = np.asarray(Pipeline.get_mesh(None, "mf_ROM", None, 12)[0], np.float64)
    Tf = fix_triangle_widning(T, np.load(MF_DATA / "faces.npy"))
    X = [np.load(MF_DATA / f"verts/{i:06d}.npy") for i in FRAMES]
    w = mf_target_weights(T)                                              # (Nt, 512)
    v_src = np.stack([np.load(KEYD_DATA / f"key_d/{i:06d}.npy") for i in FRAMES])
    used = np.where((w > 0).sum(0) > 0)[0]                                # points the target uses
    still = FRAMES.index(STILL)
    print(f"w {w.shape}, points used by target {len(used)}, check retarget err "
          f"{np.abs(w @ v_src[still] - X[still]).mean():.2e}", flush=True)

    face = Renderer(F_FACE)
    # --- face framing: normal-colored inputs, output, sequences ---
    nT = igl.per_vertex_normals(T, Tf) * 0.5 + 0.5
    nS = igl.per_vertex_normals(S, Sf) * 0.5 + 0.5
    nE = igl.per_vertex_normals(E[still], Sf) * 0.5 + 0.5
    face.render(T, Tf, OUT / "face/target_normal.png", colors=nT, flat=True)
    face.render(S, Sf, OUT / "face/source_normal.png", colors=nS, flat=True)
    face.render(E[still], Sf, OUT / "face/source_expression_normal.png", colors=nE, flat=True)
    face.render(X[still], Tf, OUT / "face/target_output.png")
    for k, (e, x) in enumerate(zip(E, X)):
        face.render(e, Sf, OUT / f"face/source_seq/{k:04d}.png")
        face.render(x, Tf, OUT / f"face/target_seq/{k:04d}.png")
        face.render(e, Sf, OUT / f"face/source_normal_seq/{k:04d}.png",
                    colors=igl.per_vertex_normals(e, Sf) * 0.5 + 0.5, flat=True)
    # --- w: nipy_spectral blend colors (render_weight_vis.py) ---
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "w.npy", w)
    wcol, point_rgb = build_colormaps(w)
    face.render(T, Tf, OUT / "face/target_w.png", colors=wcol)
    # --- weight maps of the picked points on mf and the other neutrals (point colors) ---
    picks = PICKS
    pick_rgb = np.array([PICK_COLOR.get(p, point_rgb[p]) for p in picks], np.float32)
    meshes = {"mf": (T, Tf, w), **extra_target_weights()}
    for name, (V, F, wm) in meshes.items():
        for k, pidx in enumerate(picks):
            wc = (wm[:, pidx] / max(wm[:, pidx].max(), 1e-8)) ** 0.6
            col = (1 - wc[:, None]) * 0.35 + wc[:, None] * pick_rgb[k]
            face.render(V, F, OUT / f"face/weight_{name}_{k}.png", colors=col)
    # --- points v alone (no face): orthographic x / y of the used points, normalized ---
    allp = v_src[:, used].reshape(-1, 3)
    c = (allp.max(0) + allp.min(0)) / 2
    r = np.abs(allp - c)[:, :2].max()
    # --- phi's flat output (3K values, before Reshape) per frame: each value's change from
    #     the sequence mean, as coolwarm colors ---
    flat = v_src.reshape(len(FRAMES), -1)
    dev = flat - flat.mean(0)
    vec = plt.get_cmap("coolwarm")(np.clip(dev / np.percentile(np.abs(dev), 99) * 0.5 + 0.5, 0, 1))
    np.savez(OUT / "points.npz", used=used, picks=np.array(picks), pick_colors=pick_rgb,
             colors=point_rgb[used], meshes=np.array(list(meshes)),
             seq=(v_src[:, used, :2] - c[:2]) / r, frames=np.array(FRAMES), still=still,
             vec=(vec[..., :3] * 255).astype(np.uint8))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
