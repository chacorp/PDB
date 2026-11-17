"""Paper zoom-in comparison figures (Fig.13 / Fig.14 style) as one HTML each.

  fig13 (cross): source neutral / source expressed (mf_ROM ID_012, frame 7623) /
                 target neutral (high-res BIWI ID_012 = M5) + eye / mouth zoom,
                 then "Retargeted results": NFR / NFS / PDB, each with eye / mouth zoom.
  fig14 (self):  row 1 ict-cap ID_009 / 924 / f843 (eyes closed) with eye zoom,
                 row 2 ict-cap ID_004 / 924 / f1117 (mouth open) with mouth zoom;
                 columns Ground Truth / PDB / NFS / NFR.

Model outputs come from render/cache_paper_zoom_figure.py. Meshes are rendered with
matplotrender plot_mesh_image (flat shading); every zoom panel is a crop of the same
high-res render, and images sharing a camera group are trimmed to one common bbox.

Run from repo root:
    python render/render_paper_zoom_figure.py           # render (cached) + HTML
    python render/render_paper_zoom_figure.py --force   # re-render meshes
"""
import os
import sys
import pickle
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
from matplotrender import plot_mesh_image  # noqa: E402  (before the repo's utils/ is on sys.path)
# matplotrender pulls in the site-packages utils.py as `utils`; drop it so the repo's utils/ loads
for _m in [m for m in sys.modules if m == "utils" or m.startswith("utils.")]:
    del sys.modules[_m]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

import dataloader_CBD  # noqa: E402
from utils.remesh_utils import ICT_face_model  # noqa: E402

OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "paper_fig"
CACHE_ROOT = OUT_ROOT / "cache"
RAW_DIR = OUT_ROOT / "raw"      # full-res renders (untrimmed)
IMG_DIR = OUT_ROOT / "img"      # trimmed full images + zoom crops used by the HTML

CKPT = {"NFR": "ICT_augment_cnn_ext_dfn4_grad",  # public NFR checkpoint
        "NFS": "2024-07-08-06-27-12-all",
        "PDB": "2026-04-27-15-36-49-NGBCv5"}
FIG13_KEY = "fig13_mf012_f007623_to_biwiHR012"
FIG13_COLS = ["NFR", "NFS", "PDB"]
FIG14_COLS = ["GT", "PDB", "NFS", "NFR"]
FIG14_LABEL = {"GT": "Ground<br>Truth", "PDB": "PDB", "NFS": "NFS", "NFR": "NFR"}
# (key, ict id, exp_num, frame, zoom name)
FIG14_ROWS = [("fig14_id009_924_f000843", 9, 1, 843, "eye"),
              ("fig14_id004_924_f001117", 4, 1, 1117, "mouth")]

M_SCALE = {"mf": 0.6, "biwi": 0.6, "ict": 0.6}
M_TRNS = np.array([0, 0, 0])
SIZE = 16

# zoom boxes: (center x / W, center y / H, side / W) on the trimmed image of the group;
# square boxes -> square zoom panels
# as in fig_example: fig13 eye box on the (viewer's) right eye, mouth box on the left mouth
# corner; fig14 eye box = nose + right eye + upper lip, mouth box = nose to chin
BOX13 = {"eye": (0.69, 0.34, 0.24), "mouth": (0.40, 0.68, 0.32)}
BOX14 = {"eye": (0.60, 0.43, 0.36), "mouth": (0.50, 0.62, 0.46)}
LINEWIDTH = 2  # polygon edge width; 1 leaves pinholes between the small high-res BIWI faces


# ---------------------------------------------------------------------------- meshes
def load_biwi_hr(select_mesh):
    """High-res BIWI template (aligned) + faces, same as vis_CBD.py / cache_paper_zoom_figure.py."""
    tri = trimesh.load(str(REPO_ROOT / "test-mesh" / "BIWI.ply"), process=False)
    with open(REPO_ROOT / "test-mesh" / "biwi_templates.pkl", "rb") as f:
        mesh = pickle.load(f)
    ids = [k for k in mesh if k != "face"]
    v = np.asarray(mesh[ids[select_mesh]])
    m_align = np.load(REPO_ROOT / "utils" / "biwi" / "align.npy")
    v = np.concatenate((v, np.ones((v.shape[0], 1))), axis=1) @ m_align.T
    return v, np.asarray(tri.faces)


def load_cache(model, key, tgt_neutral):
    v = np.load(CACHE_ROOT / CKPT[model] / f"{key}.npy")
    if model == "NFR":
        # NFR's Poisson solve returns a mean-centered mesh; put it back at the target neutral's
        # centroid so it shares the camera of its group (translation only, shape unchanged)
        v = v - v.mean(axis=0) + np.asarray(tgt_neutral).mean(axis=0)
    return v


def collect_meshes():
    """dict name -> (verts, faces, scale group); name is also the PNG stem."""
    meshes = {}
    with open("/data/sihun/pca/multiface_align/mf_templates.pkl", "rb") as f:
        mf_faces = np.asarray(pickle.load(f)["face"])
    mf_ds = dataloader_CBD.EvalDataset(data_name="mf_ROM", toggle=False)
    item = mf_ds[7623]
    meshes["fig13_src_neutral"] = (item[1].numpy(), mf_faces, "mf")
    meshes["fig13_src_expr"] = (item[0].numpy(), mf_faces, "mf")
    biwi_v, biwi_f = load_biwi_hr(12)
    meshes["fig13_tgt_neutral"] = (biwi_v, biwi_f, "biwi")
    for m in FIG13_COLS:
        meshes[f"fig13_{m}"] = (load_cache(m, FIG13_KEY, biwi_v), biwi_f, "biwi")

    ict_f = np.asarray(ICT_face_model().faces)
    for key, ict_id, exp_num, frame, _ in FIG14_ROWS:
        ds = dataloader_CBD.EvalDataset(data_name="ict-cap", toggle=False,
                                        ict_cap_id_num=ict_id, ict_cap_exp_num=exp_num)
        item = ds[frame]
        meshes[f"{key}_GT"] = (item[0].numpy(), ict_f, "ict")
        for m in FIG14_COLS[1:]:
            meshes[f"{key}_{m}"] = (load_cache(m, key, item[1].numpy()), ict_f, "ict")
    return meshes


def render_all(force):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meshes = collect_meshes()
    for name, (v, f, grp) in meshes.items():
        out = RAW_DIR / f"{name}.png"
        if out.exists() and not force:
            continue
        # flat shading: one color per face (face normal . light), depth-sorted, backface-culled
        plot_mesh_image([v * M_SCALE[grp] + M_TRNS], [f], rot_list=[[0, 0, 0]],
                        size=SIZE, mode="shade", bg_black=False, linewidth=LINEWIDTH,
                        savedir=str(RAW_DIR), name=name, save=True)
        print("rendered", name, v.shape)


# ---------------------------------------------------------------------------- images
def content_bbox(paths, pad=12, thr=250):
    """Union bbox of non-white pixels over images rendered with the same camera."""
    x0 = y0 = 10 ** 9
    x1 = y1 = 0
    for p in paths:
        a = np.asarray(Image.open(p).convert("RGB"))
        ys, xs = np.where(a.min(axis=-1) < thr)
        x0, y0 = min(x0, xs.min()), min(y0, ys.min())
        x1, y1 = max(x1, xs.max()), max(y1, ys.max())
        W, H = a.shape[1], a.shape[0]
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def trim_group(names):
    """Crop every image of a camera group to the group's bbox; returns the trimmed size."""
    bbox = content_bbox([RAW_DIR / f"{n}.png" for n in names])
    for n in names:
        Image.open(RAW_DIR / f"{n}.png").convert("RGB").crop(bbox).save(IMG_DIR / f"{n}.png")
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def box_px(box, W, H):
    cx, cy, s = box
    side = s * W
    return cx * W - side / 2, cy * H - side / 2, side


def save_zoom(name, zname, box, W, H, out_px=800):
    x, y, side = box_px(box, W, H)
    img = Image.open(IMG_DIR / f"{name}.png").crop((round(x), round(y), round(x + side), round(y + side)))
    img.resize((out_px, out_px), Image.LANCZOS).save(IMG_DIR / f"{name}_zoom_{zname}.png")


# ---------------------------------------------------------------------------- html
CSS = """
body { background:#fff; color:#111; font-family:'Linux Libertine','Times New Roman',serif; margin:24px; }
h2 { font-family:sans-serif; font-size:15px; color:#555; font-weight:normal; margin:40px 0 8px; }
.fig { display:inline-block; padding:12px 16px; border:1px solid #eee; }
.row { display:flex; align-items:flex-end; justify-content:center; gap:14px; }
.cap { text-align:center; font-size:22px; line-height:1.15; margin:4px 0; }
.unit { position:relative; }
.unit img { position:absolute; display:block; }
.unit img.zoom { outline:1.5px solid #000; outline-offset:-0.75px; }
.unit svg { position:absolute; left:0; top:0; overflow:visible; pointer-events:none; }
.sep { border-top:1.5px solid #000; margin:10px 0 4px; }
.col { display:flex; flex-direction:column; align-items:center; }
"""
# zoom-box stroke: vertical gradient (top, bottom) as in fig_example
BOX_GRAD = {"fig13_eye": ("#f4f4f4", "#c4c4c4"), "fig13_mouth": ("#e6e6e6", "#7c7c7c"),
            "fig14": ("#ffffff", "#bcbcbc")}


def grad_rect(gid, colors, x, y, s, width):
    top, bot = colors
    return (f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{top}"/><stop offset="1" stop-color="{bot}"/></linearGradient></defs>'
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{s:.1f}" height="{s:.1f}" fill="none" '
            f'stroke="url(#{gid})" stroke-width="{width}"/>')


def unit_side(name, W, H, boxes, img_w=150, panel=165, gap=6, hgap=20):
    """fig13 unit: mesh on the left, zoom panels stacked on the right, box->panel connector lines."""
    sc = img_w / W
    img_h = H * sc
    n = len(boxes)
    col_h = n * panel + (n - 1) * gap
    h = max(img_h, col_h)
    img_y = (h - img_h) / 2
    zx = img_w + hgap
    parts, svg = [], []
    parts.append(f'<img src="img/{name}.png" style="left:0;top:{img_y:.1f}px;width:{img_w}px">')
    for k, (zname, box) in enumerate(boxes.items()):
        bx, by, bs = (v * sc for v in box_px(box, W, H))
        by += img_y
        py = h - col_h + k * (panel + gap)
        parts.append(f'<img class="zoom" src="img/{name}_zoom_{zname}.png" '
                     f'style="left:{zx}px;top:{py:.1f}px;width:{panel}px;height:{panel}px">')
        svg.append(f'<line x1="{bx + bs:.1f}" y1="{by:.1f}" x2="{zx}" y2="{py:.1f}" stroke="#000" stroke-width="1"/>')
        svg.append(f'<line x1="{bx + bs:.1f}" y1="{by + bs:.1f}" x2="{zx}" y2="{py + panel:.1f}" '
                   f'stroke="#000" stroke-width="1"/>')
        svg.append(grad_rect(f"g_{name}_{zname}", BOX_GRAD[f"fig13_{zname}"], bx, by, bs, 2.5))
    w = zx + panel
    return (f'<div class="unit" style="width:{w}px;height:{h:.1f}px">{"".join(parts)}'
            f'<svg width="{w}" height="{h:.1f}">{"".join(svg)}</svg></div>')


def unit_below(name, W, H, zname, box, panel=245, head_frac=0.62, vgap=22):
    """fig14 unit: smaller mesh centered on top, one square zoom panel below, trapezoid connector."""
    img_w = panel * head_frac
    img_x = (panel - img_w) / 2
    sc = img_w / W
    img_h = H * sc
    py = img_h + vgap
    h = py + panel
    bx, by, bs = (v * sc for v in box_px(box, W, H))
    bx += img_x
    svg = (f'<line x1="{bx:.1f}" y1="{by + bs:.1f}" x2="0" y2="{py:.1f}" stroke="#000" stroke-width="1.2"/>'
           f'<line x1="{bx + bs:.1f}" y1="{by + bs:.1f}" x2="{panel}" y2="{py:.1f}" stroke="#000" stroke-width="1.2"/>'
           + grad_rect(f"g_{name}_{zname}", BOX_GRAD["fig14"], bx, by, bs, 3.5))
    return (f'<div class="unit" style="width:{panel}px;height:{h:.1f}px">'
            f'<img src="img/{name}.png" style="left:{img_x:.1f}px;top:0;width:{img_w:.1f}px">'
            f'<img class="zoom" src="img/{name}_zoom_{zname}.png" '
            f'style="left:0;top:{py:.1f}px;width:{panel}px;height:{panel}px">'
            f'<svg width="{panel}" height="{h:.1f}">{svg}</svg></div>')


def plain(name, W, H, img_w=200):
    return f'<img src="img/{name}.png" style="width:{img_w}px;display:block">'


def page(title, body):
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f'<style>{CSS}</style></head><body>{body}</body></html>')


def build_fig13():
    src = ["fig13_src_neutral", "fig13_src_expr"]
    tgt = ["fig13_tgt_neutral"] + [f"fig13_{m}" for m in FIG13_COLS]
    sW, sH = trim_group(src)
    tW, tH = trim_group(tgt)
    for n in tgt:
        for z, b in BOX13.items():
            save_zoom(n, z, b, tW, tH)
    top = (f'<div class="row">'
           f'<div class="col"><div class="cap">Source mesh<br>(Neutral)</div>{plain(src[0], sW, sH)}</div>'
           f'<div class="col"><div class="cap">Source mesh<br>(Expressed)</div>{plain(src[1], sW, sH)}</div>'
           f'<div class="col"><div class="cap">Target mesh<br>(Neutral)</div>{unit_side(tgt[0], tW, tH, BOX13)}</div>'
           f'</div>')
    bottom = '<div class="row">' + "".join(
        f'<div class="col">{unit_side(f"fig13_{m}", tW, tH, BOX13)}<div class="cap">{m}</div></div>'
        for m in FIG13_COLS) + '</div>'
    body = (f'<h2>Fig.13 &mdash; cross retargeting: mf_ROM ID_012 f7623 &rarr; BIWI ID_012 (M5, 23,370 verts)</h2>'
            f'<div class="fig">{top}<div class="sep"></div><div class="cap">Retargeted results</div>{bottom}</div>')
    (OUT_ROOT / "fig13.html").write_text(page("Fig13 cross zoom", body))


def build_fig14():
    rows = []
    for key, ict_id, _, frame, zname in FIG14_ROWS:
        names = [f"{key}_{m}" for m in FIG14_COLS]
        W, H = trim_group(names)
        for n in names:
            save_zoom(n, zname, BOX14[zname], W, H)
        rows.append('<div class="row" style="margin-bottom:18px">' +
                    "".join(unit_below(n, W, H, zname, BOX14[zname]) for n in names) + '</div>')
    caps = '<div class="row">' + "".join(
        f'<div class="cap" style="width:245px">{FIG14_LABEL[m]}</div>' for m in FIG14_COLS) + '</div>'
    body = (f'<h2>Fig.14 &mdash; self retargeting on ICT test: ID_009 924 f843 (eyes) / ID_004 924 f1117 (mouth)</h2>'
            f'<div class="fig">{"".join(rows)}{caps}</div>')
    (OUT_ROOT / "fig14.html").write_text(page("Fig14 self zoom", body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-render meshes")
    ap.add_argument("--render_only", action="store_true")
    args = ap.parse_args()
    render_all(args.force)
    if args.render_only:
        return
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    build_fig13()
    build_fig14()
    print("saved", OUT_ROOT / "fig13.html", OUT_ROOT / "fig14.html")


if __name__ == "__main__":
    main()
