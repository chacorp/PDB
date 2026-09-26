"""Build vis_CBD/vis_ablation_mask/vis_ablation_mask.html gallery for
render_ablation_frames_mask.py output (mask_cage_consist ablation).

Copied from build_ablation_gallery.py (4-model use_data1 ablation) with only
MODELS and OUT_ROOT changed, plus a masked-MSE metrics section (MSE-in /
MSE-out, self-retarget vs cyclic) prepended above the image gallery, styled
like vis_design_study/index.html's metrics tables (same table CSS/color
logic, wrapped in a white card since this page is dark-themed).

Metric source: self-retarget (eval_CBD.py --use_t_mask) values for the 3
retrained ablation checkpoints are parsed from
_tmp/eval_ablation_logs/*__cbd_*.log; cyclic (eval_CBD_cyc.py) values come
from _tmp/cyc_results.json. baseline (not retrained) values for both are
copied from report/three_way_baseline_cage_cyclic.md. MF = mf_SEN/mf_ROM
weighted average by test-frame count (n=6309 / 11649), same convention as
report/ablation_mask_cage_consist.md. cyclic has no ict column: ict-cap is
the cyclic pivot identity, not a source dataset, in eval_CBD_cyc.py.

Grouped by scenario -> frame, with all 4 models' images laid out side by side
so the ablation effect is visible at a glance for each frame.
"""
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_ablation_mask"

MODELS = [
    ("baseline", "2026-08-03-09-09-59-NGBCv5"),
    ("cage-consistency+mask", "2026-09-06-16-56-01-NGBCv5"),
    ("cyclic+mask", "2026-09-06-23-43-19-NGBCv5"),
    ("cage+cyclic+mask", "2026-09-07-08-23-51-NGBCv5"),
]

SCENARIOS = [
    ("mf_to_ict", "mf -> ict m00 (source: mf_ROM test id, absolute frame idx)",
     [8880, 5840, 8404, 6638]),
    ("ict_to_mf_m00", "ict m00 -> mf (source: ict-cap m00, 20240325_MySlate_924, its own motion)",
     [14, 1139, 336, 1116]),
    ("ict_to_mf_m02", "ict m02 -> mf (source: ict-cap m02, 20240318_MySlate_922, its own motion)",
     [39, 75, 92, 158]),
    ("coma_to_mf", "coma -> mf (source: FaceTalk_170809_00138_TA, clip mouth_extreme)",
     [47, 74, 89, 104]),
    ("mf_to_coma", "mf -> coma (source: mf_ROM test id, absolute frame idx)",
     [8880, 5840, 8404, 6638]),
    ("biwi_to_mf", "biwi -> mf (source: F2, clip F2_e38)",
     [9, 73, 92, 121]),
    ("mf_to_biwi", "mf -> biwi (source: mf_ROM test id, absolute frame idx)",
     [8880, 5840, 8404, 6638]),
    ("coma_to_biwi", "coma -> biwi (source: FaceTalk_170809_00138_TA, clip mouth_extreme)",
     [47, 74, 89, 104]),
    ("biwi_to_coma", "biwi -> coma (source: F2, clip F2_e38)",
     [9, 73, 92, 121]),
]

# ---------------------------------------------------------------------------
# masked-MSE metrics section: rows = 4 models, columns = biwi / coma / MF
# (self-retarget also has ict; cyclic does not -- see module docstring), x
# {self-retarget, cyclic}, x {MSE-in, MSE-out}. All values already x1e-4.
# ---------------------------------------------------------------------------
METRIC_SELF_DATASETS = ["biwi", "coma", "MF", "ict"]
METRIC_CYC_DATASETS = ["biwi", "coma", "MF"]

METRIC_VARIANTS = [
    {"name": "baseline", "chosen": False,
     "self_in": {"biwi": 0.5421, "coma": 2.1830, "MF": 2.2456, "ict": 7.5970},
     "self_out": {"biwi": 0.0000, "coma": 0.1233, "MF": 0.2299, "ict": 1.1790},
     "cyc_in": {"biwi": 3.7061, "coma": 5.6452, "MF": 0.7198},
     "cyc_out": {"biwi": 0.0000, "coma": 0.7043, "MF": 0.0092}},
    {"name": "cage-consistency+mask", "chosen": False,
     "self_in": {"biwi": 0.7885, "coma": 3.3205, "MF": 3.4750, "ict": 9.7013},
     "self_out": {"biwi": 0.0000, "coma": 0.1494, "MF": 0.2500, "ict": 0.4523},
     "cyc_in": {"biwi": 6.0803, "coma": 11.5269, "MF": 1.0628},
     "cyc_out": {"biwi": 0.0000, "coma": 0.4193, "MF": 0.0199}},
    {"name": "cyclic+mask", "chosen": False,
     "self_in": {"biwi": 1.2138, "coma": 2.9644, "MF": 3.6524, "ict": 5.0227},
     "self_out": {"biwi": 0.0000, "coma": 0.1216, "MF": 0.2026, "ict": 0.4697},
     "cyc_in": {"biwi": 2.6479, "coma": 6.7268, "MF": 0.6168},
     "cyc_out": {"biwi": 0.0000, "coma": 0.3693, "MF": 0.0091}},
    {"name": "cage+cyclic+mask", "chosen": False,
     "self_in": {"biwi": 1.2750, "coma": 2.5567, "MF": 3.2512, "ict": 4.5455},
     "self_out": {"biwi": 0.0000, "coma": 0.1563, "MF": 0.1154, "ict": 0.4175},
     "cyc_in": {"biwi": 2.7688, "coma": 6.1925, "MF": 0.5522},
     "cyc_out": {"biwi": 0.0000, "coma": 0.3673, "MF": 0.0049}},
]


def _lerp(a, b, t):
    return a + (b - a) * t


def _seq_color(t):
    """light-surface sequential blue ramp: low ~ near-white, high = deep blue,
    so magnitude reads as light->dark on the white metrics card."""
    lo = (0xf3, 0xf8, 0xfe)
    hi = (0x0d, 0x36, 0x6b)
    t = max(0.0, min(1.0, t))
    r = round(_lerp(lo[0], hi[0], t))
    g = round(_lerp(lo[1], hi[1], t))
    b = round(_lerp(lo[2], hi[2], t))
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return f"rgb({r},{g},{b})", lum < 0.55  # (css, use_light_text)


def _col_min_max(metric, c):
    vals = [v[metric][c] for v in METRIC_VARIANTS]
    return min(vals), max(vals)


def _fmt(v):
    return f"{v:.4f}"


def _build_metric_table(self_key, cyc_key):
    mins, maxs = {}, {}
    for metric, cols in ((self_key, METRIC_SELF_DATASETS), (cyc_key, METRIC_CYC_DATASETS)):
        mins[metric], maxs[metric] = {}, {}
        for c in cols:
            lo, hi = _col_min_max(metric, c)
            mins[metric][c], maxs[metric][c] = lo, hi

    out = ['<table class="mtable">']
    out.append(
        '<thead><tr class="grp"><th></th>'
        f'<th class="self" colspan="{len(METRIC_SELF_DATASETS)}">self-retarget</th><th class="gapcol"></th>'
        f'<th class="cyc" colspan="{len(METRIC_CYC_DATASETS)}">cyclic</th></tr>'
    )
    rowhead_th = '<th class="rowhead">variant</th>'
    self_ths = ''.join(f'<th>{d}</th>' for d in METRIC_SELF_DATASETS)
    cyc_ths = ''.join(f'<th>{d}</th>' for d in METRIC_CYC_DATASETS)
    out.append(f'<tr class="cols">{rowhead_th}{self_ths}<th class="gapcol"></th>{cyc_ths}</tr></thead>')

    out.append('<tbody>')
    for v in METRIC_VARIANTS:
        cls = ' class="is-chosen"' if v["chosen"] else ''
        out.append(f'<tr{cls}><th class="rowhead">{v["name"]}</th>')

        def cells(metric, cols):
            for c in cols:
                val = v[metric][c]
                extra = ' mf' if c == 'MF' else ''
                lo, hi = mins[metric][c], maxs[metric][c]
                t = (val - lo) / (hi - lo) if hi > lo else 0.0
                css, use_light_text = _seq_color(t)
                best = ' best' if val == lo else ''
                text_cls = ' light-text' if use_light_text else ''
                out.append(f'<td class="cell{extra}{best}{text_cls}" style="background:{css}">{_fmt(val)}</td>')

        cells(self_key, METRIC_SELF_DATASETS)
        out.append('<td class="gapcol"></td>')
        cells(cyc_key, METRIC_CYC_DATASETS)
        out.append('</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


METRICS_SECTION = f"""
<section id="metrics" class="metrics">
  <div class="metrics-card">
  <h2 id="metrics-h">Masked MSE: self-retarget vs cyclic <span class="unit">&middot; &times;10&#8315;&#8309; &middot; eval_CBD.py / eval_CBD_cyc.py --use_t_mask</span></h2>
  <div class="mblock">
    <h3 class="mlabel">MSE-in <span>&middot; masked inner region</span></h3>
    <div class="table-scroll">{_build_metric_table('self_in', 'cyc_in')}</div>
  </div>
  <div class="mblock">
    <h3 class="mlabel">MSE-out <span>&middot; masked outer region</span></h3>
    <div class="table-scroll">{_build_metric_table('self_out', 'cyc_out')}</div>
  </div>
  <p class="mnote">cyclic has no ict column (ict-cap is the cyclic pivot identity, not a source dataset). MF = mf_SEN/mf_ROM weighted average by test-frame count (n=6309 / 11649).</p>
  </div>
</section>
"""

HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>mask_cage_consist ablation: original / self / cross / cyclic</title>
<style>
  body { font-family: -apple-system, sans-serif; background:#111; color:#eee; margin:0; padding:24px; }
  h1 { font-size:20px; }
  h2 { font-size:16px; margin-top:40px; border-bottom:1px solid #444; padding-bottom:6px; }
  h3 { font-size:15px; color:#ddd; margin-top:28px; }
  .model-row { display:flex; flex-direction:column; gap:2px; margin-bottom:14px; }
  .model-cell { display:flex; align-items:center; gap:10px; }
  .model-label { width:150px; flex:0 0 150px; font-size:12px; color:#aaa; text-align:right; }
  .model-cell img { flex:1 1 auto; max-width:1800px; width:100%; border:1px solid #333; background:#fff; }
  .missing { flex:1; font-size:12px; color:#666; padding:8px; border:1px dashed #444; }
  .legend { font-size:12px; color:#888; }
  .panel-legend { font-size:11px; color:#777; margin:2px 0 10px 160px; }
  nav { position:sticky; top:0; background:#111; padding:8px 0; border-bottom:1px solid #333; z-index:10; }
  nav a { color:#8cf; margin-right:16px; font-size:12px; text-decoration:none; }

  /* -------- metrics tables (white card, dark page) -------- */
  .metrics { margin-bottom:32px; }
  .metrics-card { background:#fff; color:#111; border-radius:8px; padding:20px 24px; }
  .metrics-card h2 { color:#111; border-bottom:1px solid #ddd; padding-bottom:6px; margin-top:0; }
  .metrics-card .unit { font-size:12px; color:#777; font-weight:normal; }
  .mblock { margin-top:18px; }
  .mlabel { font-size:12px; font-weight:600; color:#444; margin:0 0 6px; text-transform:none; }
  .mlabel span { color:#888; font-weight:normal; }
  .mnote { font-size:11px; color:#888; margin-top:16px; }
  table.mtable { border-collapse:collapse; width:100%; max-width:820px; font-size:13px; font-variant-numeric:tabular-nums; }
  table.mtable th, table.mtable td { padding:6px 12px; text-align:right; white-space:nowrap; }
  table.mtable thead tr.grp th { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#333; border-bottom:1px solid #ddd; padding-bottom:4px; font-weight:600; }
  table.mtable thead tr.grp th.self { color:#1565c0; }
  table.mtable thead tr.grp th.cyc { color:#7b3fa0; }
  table.mtable thead tr.cols th { font-size:10.5px; text-transform:uppercase; letter-spacing:.03em; color:#888; border-bottom:2px solid #ddd; padding-bottom:6px; font-weight:500; }
  table.mtable th.rowhead { text-align:left; color:#111; font-weight:500; }
  table.mtable tr.is-chosen th.rowhead { color:#7b3fa0; }
  table.mtable tr.is-chosen th.rowhead::before { content:""; display:inline-block; width:6px; height:6px; border-radius:50%; background:#7b3fa0; margin-right:7px; }
  table.mtable tbody tr td, table.mtable tbody tr th.rowhead { border-bottom:1px solid #eee; }
  table.mtable tbody tr:last-child td, table.mtable tbody tr:last-child th { border-bottom:none; }
  table.mtable td.cell { color:#0c0c0c; font-weight:500; }
  table.mtable td.cell.light-text { color:#f5f4ef; }
  table.mtable td.cell.mf { font-weight:700; }
  table.mtable td.cell.best { font-weight:800; }
  table.mtable td.gapcol, table.mtable th.gapcol { border:none; padding:0; width:14px; }
</style>
</head><body>
<h1>mask_cage_consist ablation (baseline / cage-consistency+mask / cyclic+mask / cage+cyclic+mask): original / self / cross / cyclic retargeting</h1>
<p class="legend">Grouped by frame so all 4 models are comparable at a glance. Each row = one model's
result for that frame. Each image: original(GT) | self | cross | cyclic. Rendered with
utils.matplotlib_rnd.plot_image_array (size=8, mode=shade, bg_black=False, M_SCALE=0.6, whole mesh, frontal).</p>
""" + METRICS_SECTION + """
<nav>
<a href="#metrics">metrics</a>
"""

rows = [HTML_HEAD]
for direction, desc, _ in SCENARIOS:
    rows.append(f'<a href="#{direction}">{direction}</a>')
rows.append('</nav>')

for direction, desc, frames in SCENARIOS:
    rows.append(f'<h2 id="{direction}">{desc}</h2>')
    for fr in frames:
        rows.append(f'<h3>frame {fr}</h3>')
        rows.append('<p class="panel-legend">original | self | cross | cyclic</p>')
        rows.append('<div class="model-row">')
        for model_name, ckpt_dir in MODELS:
            img_path = f'{ckpt_dir}/{direction}/frame_{fr:06d}.png'
            full_path = OUT_ROOT / img_path
            rows.append('<div class="model-cell">')
            rows.append(f'<div class="model-label">{model_name}</div>')
            if full_path.exists():
                b64 = base64.b64encode(full_path.read_bytes()).decode("ascii")
                rows.append(f'<img src="data:image/png;base64,{b64}" loading="lazy">')
            else:
                rows.append(f'<div class="missing">missing: {img_path}</div>')
            rows.append('</div>')
        rows.append('</div>')

rows.append("</body></html>")

out_path = OUT_ROOT / "vis_ablation_mask.html"
out_path.write_text("\n".join(rows))
print(f"wrote {out_path}")
