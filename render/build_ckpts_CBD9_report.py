"""Build vis_CBD/vis_ckpts_CBD9/ckpts_CBD9_eval.html: quantitative metrics tables
(self-retarget / cyclic MSE, MSE-in, MSE-out) + qualitative figure for the 4
ckpts_CBD9 models (num_cage_v 512 / 256 / 128 / 64, best checkpoints).

Layout follows build_ablation_gallery_design_study.py, with only the metrics
tables and the figure images (no description text).
Metrics: _tmp/ckpts_CBD9/metrics_<key>.json (aggregate_cbd8.py <ckpt> <json>).
Figure: vis_CBD/vis_ckpts_CBD9/figure/ (render_ckpts_CBD9_figure.py).
"""
import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_ckpts_CBD9"
FIG_ROOT = OUT_ROOT / "figure"
METRIC_ROOT = REPO_ROOT / "_tmp" / "ckpts_CBD9"

MODELS = [
    ("cage_v512", "512"),
    ("cage_v256", "256"),
    ("cage_v128", "128"),
    ("cage_v64", "64"),
]

SCENARIOS = [
    ("mf_to_ict", [8880, 5840, 8404, 6638]),
    ("ict_to_mf_m00", [14, 1139, 336, 1116]),
    ("ict_to_mf_m02", [39, 75, 92, 158]),
    ("coma_to_mf", [47, 74, 89, 104]),
    ("mf_to_coma", [8880, 5840, 8404, 6638]),
    ("biwi_to_mf", [9, 73, 92, 121]),
    ("mf_to_biwi", [8880, 5840, 8404, 6638]),
    ("coma_to_biwi", [47, 74, 89, 104]),
    ("biwi_to_coma", [9, 73, 92, 121]),
]

REF_ROWS = [
    ("source_neutral", "Source Neutral Mesh", ""),
    ("source_expression", "Source Expression Mesh", "frame"),
    ("target_neutral", "Target Neutral Mesh", ""),
]

FIG_ROWS = [
    ("self", "Self retargeting"),
    ("cross", "Cross retargeting"),
    ("cyclic", "Cyclic retargeting"),
]


def fig_img_cell(path: Path, caption: str = "") -> str:
    if not path.exists():
        return f'<td class="missing">missing<br>{path.name}</td>'
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    cap = f'<div class="frame-tag">{caption}</div>' if caption else ""
    img = f'<div class="clip"><img src="data:image/png;base64,{b64}" loading="lazy"></div>'
    return f'<td class="imgcell">{img}{cap}</td>'


def build_fig_ref_table(frame_dir: Path, frame_idx: int) -> str:
    out = ['<table class="grid ref-grid">', "<tbody>"]
    for row_key, row_label, tag in REF_ROWS:
        caption = f"Frame {frame_idx}" if tag == "frame" else ""
        cell = fig_img_cell(frame_dir / f"{row_key}.png", caption)
        out.append(f'<tr><th class="rowhead">{row_label}</th>{cell}</tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def build_fig_result_table(frame_dir: Path) -> str:
    out = ['<table class="grid">', "<tbody>"]
    for row_key, row_label in FIG_ROWS:
        out.append(f'<tr><th class="rowhead">{row_label}</th>')
        for col_key, _ in MODELS:
            out.append(fig_img_cell(frame_dir / f"{row_key}_{col_key}.png"))
        out.append("</tr>")
    out.append("</tbody>")
    out.append("<tfoot><tr>")
    out.append('<th class="rowhead-col"></th>')
    for _, label in MODELS:
        out.append(f"<th>{label}</th>")
    out.append("</tr></tfoot>")
    out.append("</table>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# quantitative metrics: rows = 4 models, columns = biwi / coma / MF
# (n-weighted mf_SEN+mf_ROM) x {self-retarget, cyclic}. Values are MSE x1e4.
# ---------------------------------------------------------------------------
METRIC_DATASETS = ["biwi", "coma", "MF"]
METRIC_VARIANTS = []
for key, label in MODELS:
    m = json.load(open(METRIC_ROOT / f"metrics_{key}.json"))
    METRIC_VARIANTS.append({"name": label, **{k: m[k] for k in
                            ("self", "cyc", "self_in", "cyc_in", "self_out", "cyc_out")}})


def _lerp(a, b, t):
    return a + (b - a) * t


def _seq_color(t):
    lo = (0xf3, 0xf8, 0xfe)
    hi = (0x0d, 0x36, 0x6b)
    t = max(0.0, min(1.0, t))
    r = round(_lerp(lo[0], hi[0], t))
    g = round(_lerp(lo[1], hi[1], t))
    b = round(_lerp(lo[2], hi[2], t))
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return f"rgb({r},{g},{b})", lum < 0.55  # (css, use_light_text)


def _fmt(v):
    return f"{v:.4f}"


def _build_metric_table(self_key, cyc_key):
    n = len(METRIC_DATASETS)
    out = ['<table class="mtable">']
    out.append(
        '<thead><tr class="grp"><th></th>'
        f'<th class="self" colspan="{n}">self-retarget</th><th class="gapcol"></th>'
        f'<th class="cyc" colspan="{n}">cyclic</th></tr>'
    )
    data_ths = ''.join(f'<th>{d}</th>' for d in METRIC_DATASETS)
    out.append(f'<tr class="cols"><th class="rowhead">num_cage_v</th>{data_ths}'
               f'<th class="gapcol"></th>{data_ths}</tr></thead>')

    out.append('<tbody>')
    for v in METRIC_VARIANTS:
        out.append(f'<tr><th class="rowhead">{v["name"]}</th>')
        for i, metric in enumerate((self_key, cyc_key)):
            if i == 1:
                out.append('<td class="gapcol"></td>')
            for c in METRIC_DATASETS:
                vals = [w[metric][c] for w in METRIC_VARIANTS]
                lo, hi = min(vals), max(vals)
                val = v[metric][c]
                t = (val - lo) / (hi - lo) if hi > lo else 0.0
                css, use_light_text = _seq_color(t)
                extra = ' mf' if c == 'MF' else ''
                best = ' best' if val == lo else ''
                text_cls = ' light-text' if use_light_text else ''
                out.append(f'<td class="cell{extra}{best}{text_cls}" style="background:{css}">{_fmt(val)}</td>')
        out.append('</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


METRICS_SECTION = f"""
<section class="metrics">
  <h2>Quantitative <span class="unit">&middot; MSE &times;10&#8315;&#8308;</span></h2>
  <div class="mblock">
    <h3 class="mlabel">MSE</h3>
    <div class="table-scroll">{_build_metric_table('self', 'cyc')}</div>
  </div>
  <div class="mblock">
    <h3 class="mlabel">MSE-in</h3>
    <div class="table-scroll">{_build_metric_table('self_in', 'cyc_in')}</div>
  </div>
  <div class="mblock">
    <h3 class="mlabel">MSE-out</h3>
    <div class="table-scroll">{_build_metric_table('self_out', 'cyc_out')}</div>
  </div>
</section>
"""

HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>ckpts_CBD9 evaluation</title>
<style>
  body { font-family: -apple-system, sans-serif; background:#fff; color:#111; margin:0; padding:24px; }
  h2 { font-size:16px; margin-top:40px; border-bottom:1px solid #ddd; padding-bottom:6px; }
  h3 { font-size:15px; color:#333; margin-top:28px; }

  .figure-section { font-family: "Times New Roman", Times, "Liberation Serif", serif; }
  .figure-row { display:flex; align-items:flex-start; gap:14px; margin-top:10px; margin-bottom:24px; }
  table.grid { border-collapse:collapse; }
  table.grid th, table.grid td { border:none; padding:0 1px; text-align:center; vertical-align:middle; }
  table.grid tfoot th { font-size:12px; color:#333; padding:6px 1px; font-weight:normal; }
  table.grid tfoot th.rowhead-col { border:none; }
  table.grid tbody th.rowhead { font-size:12px; color:#333; font-weight:normal; white-space:nowrap;
    writing-mode:vertical-rl; transform:rotate(180deg); text-align:center; padding:0 4px; }
  table.grid td .clip { width:100px; overflow:hidden; margin:0 auto; }
  table.grid td .clip img { width:120px; height:auto; display:block; margin-left:-10px; background:#fff; }
  table.grid td.missing { color:#bbb; font-size:11px; }
  table.grid td .frame-tag { font-size:11px; color:#333; }
  table.ref-grid { border-right:1px solid #ddd; padding-right:14px; }

  .metrics { margin-bottom:32px; }
  .metrics .unit { font-size:12px; color:#777; font-weight:normal; }
  .mblock { margin-top:18px; }
  .mlabel { font-size:12px; font-weight:600; color:#444; margin:0 0 6px; }
  table.mtable { border-collapse:collapse; width:100%; max-width:760px; font-size:13px; font-variant-numeric:tabular-nums; }
  table.mtable th, table.mtable td { padding:6px 12px; text-align:right; white-space:nowrap; }
  table.mtable thead tr.grp th { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#333; border-bottom:1px solid #ddd; padding-bottom:4px; font-weight:600; }
  table.mtable thead tr.grp th.self { color:#1565c0; }
  table.mtable thead tr.grp th.cyc { color:#7b3fa0; }
  table.mtable thead tr.cols th { font-size:10.5px; text-transform:uppercase; letter-spacing:.03em; color:#888; border-bottom:2px solid #ddd; padding-bottom:6px; font-weight:500; }
  table.mtable th.rowhead { text-align:left; color:#111; font-weight:500; }
  table.mtable tbody tr td, table.mtable tbody tr th.rowhead { border-bottom:1px solid #eee; }
  table.mtable tbody tr:last-child td, table.mtable tbody tr:last-child th { border-bottom:none; }
  table.mtable td.cell { color:#0c0c0c; font-weight:500; }
  table.mtable td.cell.light-text { color:#f5f4ef; }
  table.mtable td.cell.mf { font-weight:700; }
  table.mtable td.cell.best { font-weight:800; }
  table.mtable td.gapcol, table.mtable th.gapcol { border:none; padding:0; width:14px; }
</style>
</head><body>
""" + METRICS_SECTION + """
<section class="figure-section">
<h2>Qualitative</h2>
"""

rows = [HTML_HEAD]
for direction, frames in SCENARIOS:
    scenario_dir = FIG_ROOT / direction
    rows.append(f'<h2>{direction}</h2>')
    for fr in frames:
        frame_dir = scenario_dir / f"frame_{fr:06d}"
        rows.append('<div class="figure-row">')
        rows.append(build_fig_ref_table(frame_dir, fr))
        rows.append(build_fig_result_table(frame_dir))
        rows.append("</div>")

rows.append("</section>")
rows.append("</body></html>")

OUT_ROOT.mkdir(parents=True, exist_ok=True)
out_path = OUT_ROOT / "ckpts_CBD9_eval.html"
out_path.write_text("\n".join(rows))
print(f"wrote {out_path}")
