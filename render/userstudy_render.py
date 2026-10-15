"""
User study (userstudy-pdb), stage 2: render the cached outputs of
render/userstudy_infer.py as PNG sequences with pyrender (EGL, plain shading, black
background, supp_video_render.CellRenderer settings) and encode one mp4 per segment.

Output (same file name in every method folder, as userstudy-pdf):
    vis_CBD/userstudy/videos/{NC,NFR,NFS,PDB}/{seg}-self.mp4
    vis_CBD/userstudy/videos/{NC,NFR,NFS,PDB}/{seg}-{cross|cyclic}-{target}.mp4
    vis_CBD/userstudy/videos/ref/{seg}-source_expression.mp4
    vis_CBD/userstudy/videos/ref/{source}_neutral.png, {target}_neutral.png
seg = mf_S1..S3, ict_m00_S1..S2, ict_m02_S1 (userstudy-pdb/PLAN.md 2.1).
No method name or frame number is drawn on the videos.

Reads only vis_CBD/userstudy (verts/ and cells/; no supp_video). PNG cells go to
vis_CBD/userstudy/cells/{source}/ref/source_expression, {source}/self/{model}/ and
{source}/{target}/{model}/{cross|cyclic}/. Existing PNGs / mp4s are skipped.
Clips with missing vertices are skipped and listed.

    python render/userstudy_render.py
    US_RENDER_SEGS=mf_S1 US_RENDER_TARGETS=coma US_RENDER_MAX=30 US_RENDER_OUT=_tmp/userstudy_render_test \
        python render/userstudy_render.py  # test (short clips: keep them out of the real video folder)
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # GPU offscreen
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import numpy as np  # noqa: E402

from supp_video_render import CellRenderer, clip_box  # noqa: E402
from utils.matplotlib_rnd import fix_triangle_widning  # noqa: E402

# same as render/userstudy_infer.py (not imported: it loads the models' modules)
SEGMENTS = {
    "mf": [(3875, 4174), (4226, 4525), (4577, 4876)],
    "ict_m00": [(90, 389), (480, 779)],
    "ict_m02": [(0, 182)],
}
SOURCES = {"mf": "ict", "ict_m00": "mf", "ict_m02": "mf"}  # source -> the other dataset as target
DATASET_TARGETS = ["coma", "biwi"]
STYLIZED_TARGETS = ["aura", "piers", "face-reference", "head-reference"]
MODEL_NAMES = ["NC", "NFR", "NFS", "PDB"]

VERTS_ROOT = REPO_ROOT / "vis_CBD" / "userstudy" / "verts"
CELLS_ROOT = REPO_ROOT / "vis_CBD" / "userstudy" / "cells"
VIDEO_ROOT = Path(os.environ.get("US_RENDER_OUT", REPO_ROOT / "vis_CBD" / "userstudy" / "videos"))
FFMPEG = "/usr/bin/ffmpeg"  # system ffmpeg has libx264 (the conda one does not)
CELL_PX = 400
FPS = 30


def env_list(name):
    return [x for x in os.environ.get(name, "").split(",") if x]


def segments():
    """[(seg name, source, frames)]"""
    out = []
    for s, ranges in SEGMENTS.items():
        for k, (lo, hi) in enumerate(ranges, 1):
            out.append((f"{s}_S{k}", s, list(range(lo, hi + 1))))
    return out


class Renders:
    """PNG cell lookup / rendering with one shared CellRenderer"""

    def __init__(self):
        self.rnd = None
        self.n = 0

    def cell(self, vp, cp, F, box):
        if not cp.exists():
            if self.rnd is None:
                self.rnd = CellRenderer(CELL_PX)
            self.rnd.render(np.clip(np.load(vp), *box), F, cp)
            self.n += 1
        return cp

    def close(self):
        if self.rnd is not None:
            self.rnd.close()


def encode(pngs, mp4):
    """mp4 from a list of PNG paths (numbered symlinks in a scratch dir)"""
    tmp = REPO_ROOT / "_tmp" / "userstudy_ffmpeg" / f"{mp4.parent.name}__{mp4.stem}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for i, p in enumerate(pngs):
        (tmp / f"{i:06d}.png").symlink_to(p.resolve())
    mp4.parent.mkdir(parents=True, exist_ok=True)
    part = mp4.with_name(mp4.stem + ".part.mp4")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                    str(part)], check=True)
    os.replace(part, mp4)
    shutil.rmtree(tmp)


def neutral_png(rd, name, V, F):
    p = VIDEO_ROOT / "ref" / f"{name}_neutral.png"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        if rd.rnd is None:
            rd.rnd = CellRenderer(CELL_PX)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.png")  # parallel runs share the neutrals
        rd.rnd.render(V, F, tmp)
        os.replace(tmp, p)


def main():
    only_segs = env_list("US_RENDER_SEGS")
    only_tgts = env_list("US_RENDER_TARGETS")
    max_frames = int(os.environ.get("US_RENDER_MAX", 0))
    rd = Renders()
    skipped, made = [], 0

    for seg, s, frames in segments():
        if only_segs and seg not in only_segs:
            continue
        if max_frames:
            frames = frames[:max_frames]
        other = SOURCES[s]
        src_ref = VERTS_ROOT / s / "ref"
        src_n = np.load(src_ref / "source_neutral.npy")
        f_src = fix_triangle_widning(src_n, np.load(src_ref / "faces_src.npy"))
        src_box = clip_box(src_n)
        neutral_png(rd, s, src_n, f_src)

        # (mp4, vertex dir, cell dir, faces, clip box)
        clips = [(VIDEO_ROOT / "ref" / f"{seg}-source_expression.mp4", src_ref / "source_expression",
                  CELLS_ROOT / s / "ref" / "source_expression", f_src, src_box)]
        for m in MODEL_NAMES:
            clips.append((VIDEO_ROOT / m / f"{seg}-self.mp4", VERTS_ROOT / s / "self" / m,
                          CELLS_ROOT / s / "self" / m, f_src, src_box))

        for t in [other] + DATASET_TARGETS + STYLIZED_TARGETS:
            if only_tgts and t not in only_tgts:
                continue
            tgt_ref = VERTS_ROOT / s / t / "ref"
            if not (tgt_ref / "target_neutral.npy").exists():
                skipped.append(f"{seg} {t}: no target_neutral")
                continue
            tgt_n = np.load(tgt_ref / "target_neutral.npy")
            f_tgt = fix_triangle_widning(tgt_n, np.load(tgt_ref / "faces_tgt.npy"))
            neutral_png(rd, t, tgt_n, f_tgt)
            for m in MODEL_NAMES:
                for row, F, box in (("cross", f_tgt, clip_box(tgt_n)), ("cyclic", f_src, src_box)):
                    clips.append((VIDEO_ROOT / m / f"{seg}-{row}-{t}.mp4", VERTS_ROOT / s / t / m / row,
                                  CELLS_ROOT / s / t / m / row, F, box))

        for mp4, vdir, cdir, F, box in clips:
            if mp4.exists():
                continue
            miss = [f for f in frames if not (vdir / f"frame_{f:06d}.npy").exists()]
            if miss:
                skipped.append(f"{mp4.parent.name}/{mp4.name}: {len(miss)} frames missing")
                continue
            cdir.mkdir(parents=True, exist_ok=True)
            pngs = [rd.cell(vdir / f"frame_{f:06d}.npy", cdir / f"frame_{f:06d}.png", F, box) for f in frames]
            encode(pngs, mp4)
            made += 1
        print(f"[render] {seg}: rendered cells so far {rd.n}, mp4 written so far {made}", flush=True)

    rd.close()
    print(f"ALL DONE: {made} mp4 written, {rd.n} cells rendered, {len(skipped)} skipped", flush=True)
    for x in skipped:
        print("  [skipped]", x, flush=True)


if __name__ == "__main__":
    main()
