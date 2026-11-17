"""
Figure cells and the HTML page for render/sparse_weight_infer.py (run that first).

For each figure frame and each mode (renorm / raw), rows self / cross / cyclic, columns
dense and each threshold: the output mesh, and below it a heatmap of each vertex's distance
to the dense output. Heatmaps of one (frame, mode, row) share one scale, shown in the row
header: the largest 99th-percentile distance among its thresholds.

Writes vis_CBD/sparse_weight_test/figure/{frame}/*.png, figure/scales.json,
figure/montage/frame{frame}_{mode}.png, sparse_weight_test.html.
SPARSE_FORM=delta uses vis_CBD/sparse_weight_test_delta/ and adds a table against the absolute form.

    python render/sparse_weight_report.py [--html-only]
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotrender import plot_mesh_gouraud  # noqa: E402
from matplotrender.utils import calc_face_norm  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FORM = os.environ.get("SPARSE_FORM", "absolute")  # absolute | delta
ABS_OUT = ROOT / "vis_CBD" / "sparse_weight_test"
OUT = ROOT / "vis_CBD" / "sparse_weight_test_delta" if FORM == "delta" else ABS_OUT
FIG = OUT / "figure"
FRAMES = [1117, 1139, 843, 873, 336]  # comparison_frames.ICT_FRAMES_M00
THRESHOLDS = [0.001, 0.005, 0.01, 0.02]
MODES = ["renorm", "raw"]
ROWS = ["self", "cross", "cyclic"]
M_SCALE, SIZE, WORKERS = 0.6, 8, 8
M_TRNS = np.array([0, 0, 0])
LIGHT_DIR = np.array([0, 0, 1])
SCALE = 1e4  # metric tables show value x 1e-4


def faces_of(row):
    return np.load(OUT / "ref" / ("tgt_faces.npy" if row == "cross" else "src_faces.npy"))


def verts(variant, row, frame):
    return np.load(OUT / "verts" / variant / row / f"{frame:06d}.npy")


def save_plain(args):
    V, F, path = args
    plot_mesh_gouraud([V], [F], mesh_scale=M_SCALE, mesh_trans=M_TRNS, is_diff=False,
                      backface_culling=True, rot_list=[[0, 0, 0]], size=SIZE, mode="shade", bg_black=False,
                      logdir=str(path.parent), name=path.stem, save=True, show=False)
    plt.close("all")


def save_err(args):
    """shaded mesh blended toward YlOrRd by the distance to the dense output"""
    V, F, dist, cmax, path = args
    t = np.clip(dist / cmax, 0, 1) if cmax > 0 else np.zeros_like(dist)
    vn = calc_face_norm(V * M_SCALE + M_TRNS, F, mode="v")
    base = np.clip(vn @ LIGHT_DIR, 0, 1)[..., None].repeat(3, axis=-1) * 0.6 + 0.3
    color = base * (1 - t[:, None]) + plt.get_cmap("YlOrRd")(t)[..., :3] * t[:, None]
    plot_mesh_gouraud([V], [F], Cs=[color], mesh_scale=M_SCALE, mesh_trans=M_TRNS, is_color=True,
                      backface_culling=True, rot_list=[[0, 0, 0]], size=SIZE, mode="shade", bg_black=False,
                      logdir=str(path.parent), name=path.stem, save=True, show=False)
    plt.close("all")


def render_cells():
    plain, err, scales = [], [], {}
    src_f, tgt_f = faces_of("self"), faces_of("cross")
    src_n, tgt_n = np.load(OUT / "ref" / "src_neutral.npy"), np.load(OUT / "ref" / "tgt_neutral.npy")
    plain += [(src_n, src_f, FIG / "ref_src_neutral.png"), (tgt_n, tgt_f, FIG / "ref_tgt_neutral.png")]
    for fr in FRAMES:
        d = FIG / str(fr)
        d.mkdir(parents=True, exist_ok=True)
        plain.append((np.load(OUT / "ref" / "source_expression" / f"{fr:06d}.npy"), src_f, d / "ref_gt.png"))
        for row in ROWS:
            F = faces_of(row)
            dense = verts("dense", row, fr)
            plain.append((dense, F, d / f"{row}_dense.png"))
            for mode in MODES:
                ds = {t: np.linalg.norm(verts(f"t{t}_{mode}", row, fr) - dense, axis=-1) for t in THRESHOLDS}
                cmax = max(float(np.percentile(x, 99)) for x in ds.values())
                scales[f"{fr}/{mode}/{row}"] = cmax
                for t in THRESHOLDS:
                    v = verts(f"t{t}_{mode}", row, fr)
                    plain.append((v, F, d / f"{row}_t{t}_{mode}.png"))
                    err.append((v, F, ds[t], cmax, d / f"{row}_t{t}_{mode}_err.png"))
    (FIG / "scales.json").write_text(json.dumps(scales, indent=1))
    print(f"rendering {len(plain)} meshes, {len(err)} heatmaps", flush=True)
    with Pool(WORKERS) as pool:
        pool.map(save_plain, plain, chunksize=4)
        pool.map(save_err, err, chunksize=4)


CSS = """
body { font-family:-apple-system,"Segoe UI",sans-serif; background:#fff; color:#111; margin:0; padding:24px; }
h1 { font-size:20px; } h2 { font-size:16px; margin-top:36px; border-bottom:1px solid #ddd; padding-bottom:6px; }
h3 { font-size:14px; margin:24px 0 6px; }
.legend { font-size:12px; color:#555; max-width:900px; line-height:1.5; }
.wrap { overflow-x:auto; }
table.m { border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; margin-top:6px; }
table.m th, table.m td { padding:5px 11px; text-align:right; white-space:nowrap; border-bottom:1px solid #eee; }
table.m thead th { font-size:11px; color:#777; font-weight:500; border-bottom:2px solid #ddd; }
table.m th.v { text-align:left; font-weight:600; }
table.m tr.sep td, table.m tr.sep th { border-top:2px solid #ddd; }
.row { display:flex; gap:12px; align-items:flex-start; }
table.g { border-collapse:collapse; }
table.g th { font-size:11px; font-weight:500; color:#444; padding:2px; }
table.g th.rh { white-space:nowrap; text-align:right; padding-right:6px; }
table.g td { padding:0; }
table.g img { width:118px; display:block; }
.refs { border-right:1px solid #ddd; padding-right:12px; }
.refs img { width:118px; display:block; }
.refs div { font-size:11px; color:#444; text-align:center; margin-bottom:6px; }
"""


def build_montages():
    """one labeled image per (frame, mode), same layout as the HTML grids"""
    scales = json.loads((FIG / "scales.json").read_text())
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    cell, left, head = 260, 190, 44
    (FIG / "montage").mkdir(exist_ok=True)
    for fr in FRAMES:
        for mode in MODES:
            page = Image.new("RGB", (left + cell * 5, head * 2 + cell * 6), "white")
            draw = ImageDraw.Draw(page)
            draw.text((10, 10), f"frame {fr} | {FORM} | {mode}", fill="black", font=font)
            for j, label in enumerate(["dense"] + [f"t={t}" for t in THRESHOLDS]):
                draw.text((left + j * cell + 100, head + 10), label, fill="black", font=font)
            y = head * 2
            for row in ROWS:
                draw.text((10, y + cell // 2 - 10), row, fill="black", font=font)
                for j, name in enumerate(["dense"] + [f"t{t}_{mode}" for t in THRESHOLDS]):
                    img = Image.open(FIG / str(fr) / f"{row}_{name}.png").convert("RGB").resize((cell, cell))
                    page.paste(img, (left + j * cell, y))
                y += cell
                draw.text((10, y + cell // 2 - 30), f"{row}: distance\nto dense\nred = {scales[f'{fr}/{mode}/{row}']:.3g}",
                          fill="black", font=font)
                for j, t in enumerate(THRESHOLDS):
                    img = Image.open(FIG / str(fr) / f"{row}_t{t}_{mode}_err.png").convert("RGB").resize((cell, cell))
                    page.paste(img, (left + (j + 1) * cell, y))
                y += cell
            page.save(FIG / "montage" / f"frame{fr}_{mode}.png")


def form_table(r):
    """absolute (w @ v) against delta (neutral + w @ (v - rest)): GT MSE of self / cyclic, Lap of self"""
    a = json.loads((ABS_OUT / "metrics.json").read_text())["metrics"]
    head = ("<th></th><th>self MSE abs</th><th>self MSE delta</th><th>self Lap abs</th><th>self Lap delta</th>"
            "<th>cyclic MSE abs</th><th>cyclic MSE delta</th><th>cross to own dense abs</th><th>cross to own dense delta</th>")
    body = []
    for v, label in variants():
        d = r["metrics"][v]
        cells = (num(a[v]["self"]["MSE"]), num(d["self"]["MSE"]), num(a[v]["self"]["Lap"]), num(d["self"]["Lap"]),
                 num(a[v]["cyclic"]["MSE"]), num(d["cyclic"]["MSE"]),
                 num(a[v]["cross"]["to_dense_MSE"]), num(d["cross"]["to_dense_MSE"]))
        body.append(f'<tr><th class="v">{label}</th>' + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return f'<div class="wrap"><table class="m"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def num(x):
    return f"{x * SCALE:.3f}" if x * SCALE < 1000 else f"{x * SCALE:.1f}"


def variants():
    yield "dense", "dense"
    for mode in MODES:
        for t in THRESHOLDS:
            yield f"t{t}_{mode}", f"t={t} {mode}"


def metric_table(r, row, cols):
    head = "".join(f"<th>{lbl}</th>" for lbl, _ in cols)
    body = []
    for v, label in variants():
        m = r["metrics"][v][row]
        cls = ' class="sep"' if v in ("t0.001_renorm", "t0.001_raw") else ""
        cells = "".join(f"<td>{num(m[k])}</td>" for _, k in cols)
        body.append(f'<tr{cls}><th class="v">{label}</th>{cells}</tr>')
    return f'<div class="wrap"><table class="m"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def sparsity_table(r):
    head = ("<th></th><th>src nnz/vertex</th><th>src density</th><th>src kept sum</th><th>src empty rows</th>"
            "<th>tgt nnz/vertex</th><th>tgt density</th><th>tgt kept sum</th><th>tgt empty rows</th>")
    body = []
    for t in THRESHOLDS:
        for mode in MODES:
            s = r["sparsity"][f"t{t}_{mode}"]
            cells = "".join(
                f'<td>{s[k]["nnz_per_vertex_mean"]:.1f}</td><td>{s[k]["density"] * 100:.2f}%</td>'
                f'<td>{s[k]["kept_mass_before_renorm"]:.3f}</td><td>{s[k]["empty_rows"]}</td>' for k in ("src", "tgt"))
            body.append(f'<tr><th class="v">t={t} {mode}</th>{cells}</tr>')
    return f'<div class="wrap"><table class="m"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def grid(fr, mode, scales):
    head = "<th></th><th>dense</th>" + "".join(f"<th>t={t}</th>" for t in THRESHOLDS)
    trs = []
    for row in ROWS:
        cells = "".join(f'<td><img loading="lazy" src="figure/{fr}/{row}_t{t}_{mode}.png"></td>' for t in THRESHOLDS)
        trs.append(f'<tr><th class="rh">{row}</th><td><img loading="lazy" src="figure/{fr}/{row}_dense.png"></td>{cells}</tr>')
        errs = "".join(f'<td><img loading="lazy" src="figure/{fr}/{row}_t{t}_{mode}_err.png"></td>' for t in THRESHOLDS)
        trs.append(f'<tr><th class="rh">distance to dense<br>red = {scales[f"{fr}/{mode}/{row}"]:.3g}</th><td></td>{errs}</tr>')
    return f'<table class="g"><thead><tr>{head}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def build_html():
    r = json.loads((OUT / "metrics.json").read_text())
    scales = json.loads((FIG / "scales.json").read_text())
    c = r["checks"]
    if FORM == "delta":
        formula = ("The output adds only the control point motion to the neutral: self = source neutral + w_src @ (v &minus; v_src), "
                   "cross = target neutral + w_tgt @ (v &minus; v_src), cyclic = source neutral + w_src @ (v&prime; &minus; v_tgt). "
                   "v_src / v_tgt = 512 control points of that neutral (zero-deformation input), v = from the source expression, "
                   "v&prime; = re-estimated from the cross output, w = per-vertex blend ratios (vertices &times; 512).")
        check = (f'Dense delta cross against the original output w_tgt @ v: MSE {num(r["metrics"]["dense"]["cross"]["to_abs_dense_MSE"])}.')
        compare = f"<h2>Absolute against delta (&times;1e-4)</h2>{form_table(r)}"
    else:
        formula = ("The output is w @ v: w = per-vertex blend ratios (vertices &times; 512), v = 512 control point positions from the "
                   "source expression. self = w_src @ v, cross = w_tgt @ v, cyclic = w_src @ v&prime; with v&prime; re-estimated from the cross output.")
        check = (f'Check: largest difference between the dense outputs and the previous run of this checkpoint on the figure frames, '
                 f'self {c["max_abs_dense_minus_previous_run"]["self"]:.2g}, cross {c["max_abs_dense_minus_previous_run"]["cross"]:.2g}.')
        compare = ""
    gt_cols = [("MSE", "MSE"), ("MSE-in", "MSE-in"), ("MSE-out", "MSE-out"), ("Lap", "Lap"),
               ("MSE to dense", "to_dense_MSE"), ("Lap to dense", "to_dense_Lap")]
    cross_cols = [("MSE to dense", "to_dense_MSE"), ("MSE-in to dense", "to_dense_MSE-in"), ("Lap to dense", "to_dense_Lap")]
    parts = [f"""<!doctype html><html><head><meta charset="utf-8"><title>Sparse weight retargeting ({FORM})</title>
<style>{CSS}</style></head><body>
<h1>Sparse weight retargeting test ({FORM})</h1>
<p class="legend">Model: ckpts_CBD/2026-04-27-15-36-49-NGBCv5/model_best.pth. Source: ict-cap ID_000, 20240325_MySlate_924
({r["n_vertices"]["src"]} vertices). Target: mf_ROM test ID_012 ({r["n_vertices"]["tgt"]} vertices). {formula}</p>
<p class="legend">Each threshold t zeroes entries below t on both w_src and w_tgt. <b>renorm</b>: each row is divided by its
sum so it sums to 1, and a row left empty keeps its largest weight. <b>raw</b>: only zeroed, so rows sum to less than 1.</p>
<h2>Sparsity</h2>
<p class="legend">kept sum = mean per-vertex sum of the entries that survive, before renorm. empty rows = vertices with no
entry left.</p>
{sparsity_table(r)}
<h2>Metrics, {r["frames"]} frames (&times;1e-4)</h2>
<p class="legend">Frame means. MSE / MSE-in / MSE-out / Lap against the source expression, same definition as eval_CBD_cyc.py
(in / out = plateau_hat_points regions; Lap in the in region). Cross has no ground truth, so it is compared with the dense
output only. {check}</p>
{compare}
<h3>Self</h3>{metric_table(r, "self", gt_cols)}
<h3>Cross</h3>{metric_table(r, "cross", cross_cols)}
<h3>Cyclic</h3>{metric_table(r, "cyclic", gt_cols)}
<h2>Qualitative</h2>
<p class="legend">Below each output: distance of every vertex to the dense output. Within a (frame, mode, row), red means the
distance in the row header (the largest 99th-percentile distance among its thresholds), so colors can be compared across
thresholds but not across rows.</p>"""]
    for fr in FRAMES:
        parts.append(f"<h3>frame {fr}</h3>")
        refs = (f'<div class="refs"><img src="figure/ref_src_neutral.png"><div>source neutral</div>'
                f'<img src="figure/{fr}/ref_gt.png"><div>source expression</div>'
                f'<img src="figure/ref_tgt_neutral.png"><div>target neutral</div></div>')
        for mode in MODES:
            parts.append(f'<h3>frame {fr}, {mode}</h3><div class="row">{refs}<div class="wrap">{grid(fr, mode, scales)}</div></div>')
    parts.append("</body></html>")
    path = OUT / "sparse_weight_test.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    if "--html-only" not in sys.argv:
        render_cells()
        build_montages()
    build_html()
