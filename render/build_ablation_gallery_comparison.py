"""Build vis_CBD/vis_comparison/vis_comparison.html gallery for
render_ablation_frames_comparison.py output ("comparison" baselines: NC / NFS
/ NFR / PDB).

mf<->ict scenarios only for now (coma/biwi to follow once their eval data
exists -- see render_ablation_frames_comparison.py's own docstring).

Metric source: all four quantitative tables are read directly from
report/quant_sheet_eve-s05_full.tsv's four `tag=comparison` rows (self-
retargeting / cross-identity target-GT / cyclic-retargeting / route-
consistency -- all already measured and cross-checked against their eval_CBD*
result folders), except the deformation-locality table, which isn't in that
sheet and is instead parsed straight from each model's eval_CBD_gc.py
"[Overall] ... Motion-only" summary line (10 real-capture identities x 53
basis, --tau 0.001 --alpha 0.1 --dil 0.2 --area, matching
report/eval_bs_gc_cif_summary.md's protocol). PDB has no eval_CBD_gc result
yet -- shown as "pending" rather than fabricated.
"""
import csv
from pathlib import Path

# qualitative figure section: per-cell PNGs from render/render_comparison_figure.py,
# laid out by build_comparison_figure_html.build_frame (same as vis_ablation_mask.html)
import build_comparison_figure_html as fig_mod
from comparison_frames import MF_FRAMES, ICT_FRAMES_M00, ICT_FRAMES_M02

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison"
TSV_PATH = REPO_ROOT / "report" / "quant_sheet_eve-s05_full.tsv"

MODELS = [
    ("NC", "2025-10-20-00-15-40-CBD"),
    ("NFR", "exp_019_ICT_MF-jacob_NFR"),
    ("NFS", "2024-08-18-23-32-29-all"),
    ("PDB", "2026-09-16-02-19-03-NGBCv5"),
]

SCENARIOS = [
    ("mf_to_ict", "mf -> ict m00 (source: mf_ROM test id, absolute frame idx)",
     [8880, 5840, 8404, 6638]),
    ("ict_to_mf_m00", "ict m00 -> mf (source: ict-cap m00, 20240325_MySlate_924, its own motion)",
     [14, 1139, 336, 1116]),
    ("ict_to_mf_m02", "ict m02 -> mf (source: ict-cap m02, 20240318_MySlate_922, its own motion)",
     [39, 75, 92, 158]),
]

GC_DIRS = {
    "NC": "eval_CBD_gc/2025-10-20-00-15-40-CBD-gc-idm00_m02_m04_m05_m06_w00_w01_w02_w05_w08-tau0.001-alpha0.1-dil0.2-area",
    "NFS": "eval_CBD_gc/2024-08-18-23-32-29-all-gc-idm00_m02_m04_m05_m06_w00_w01_w02_w05_w08-tau0.001-alpha0.1-dil0.2-area",
    "NFR": "eval_CBD_gc/exp_019_ICT_MF-jacob_NFR-gc-idm00_m02_m04_m05_m06_w00_w01_w02_w05_w08-tau0.001-alpha0.1-dil0.2-area",
    "PDB": "eval_CBD_gc/2026-09-16-02-19-03-NGBCv5-gc-idm00_m02_m04_m05_m06_w00_w01_w02_w05_w08-tau0.001-alpha0.1-dil0.2-area",
}


# ---------------------------------------------------------------------------
# read the 4 comparison rows straight from the tsv (tag == "comparison")
# ---------------------------------------------------------------------------
def load_tsv_metrics():
    with open(TSV_PATH, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    blocks = {
        "self": 15, "crossgt": 20, "cyc": 25, "route": 30,
        "self_in": 35, "self_out": 40, "cyc_in": 45, "cyc_out": 50,
        "lap_self": 55, "lap_cyc": 60,
    }
    out = {}
    for r in rows[2:]:
        if len(r) < 65 or r[1] != "comparison":
            continue
        name = r[2]
        out[name] = {
            block: {"MF": r[start + 2], "ICT": r[start + 4]}
            for block, start in blocks.items()
        }
    return out


def parse_gc_overall(path):
    if path is None:
        return None
    log_path = REPO_ROOT / path / "log.txt"
    if not log_path.exists():
        return None
    lines = log_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("[Overall]"):
            motion = lines[i + 2]  # "  Motion-only    | E_in: ... E_out: ... LSR%: ..."
            fields = {}
            for chunk in motion.split("|")[1].split("  "):
                chunk = chunk.strip()
                if not chunk or ":" not in chunk:
                    continue
                key, val = chunk.split(":", 1)
                fields[key.strip()] = val.strip().split("/")[0]  # mean only
            return fields
    return None


def fmt4(s):
    try:
        return f"{float(s):.4f}"
    except (TypeError, ValueError):
        return "-"


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------
def _cell(m, block, dataset):
    v = m[block][dataset]
    return f'<td class="cell">{fmt4(v) if v else "-"}</td>'


def build_retarget_table(tsv_metrics, groups):
    """groups: list of (group title, [(sub label, tsv block, dataset), ...])."""
    out = ['<table class="mtable">']
    head1 = ['<th class="rowhead">model</th>']
    head2 = ['<th></th>']
    for title, cols in groups:
        head1.append(f'<th colspan="{len(cols)}">{title}</th>')
        head2.extend(f'<th>{sub}</th>' for sub, _, _ in cols)
    out.append("<thead><tr>" + "".join(head1) + "</tr>")
    out.append('<tr class="cols">' + "".join(head2) + "</tr></thead>")
    out.append("<tbody>")
    ncols = sum(len(c) for _, c in groups)
    for label, ckpt_name in MODELS:
        m = tsv_metrics.get(label) or tsv_metrics.get(ckpt_name)
        if m is None:
            out.append(f'<tr><th class="rowhead">{label}</th><td colspan="{ncols}">no tsv row found</td></tr>')
            continue
        out.append(f'<tr><th class="rowhead">{label}</th>')
        for _, cols in groups:
            for _, block, dataset in cols:
                out.append(_cell(m, block, dataset))
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def build_self_table(tsv_metrics):
    return build_retarget_table(tsv_metrics, [
        ("MSE", [("MF", "self", "MF"), ("ICT", "self", "ICT")]),
        ("MSE-in", [("MF", "self_in", "MF"), ("ICT", "self_in", "ICT"),
                    ("Laplacian MF", "lap_self", "MF"), ("Laplacian ICT", "lap_self", "ICT")]),
        ("MSE-out", [("MF", "self_out", "MF"), ("ICT", "self_out", "ICT")]),
    ])


def build_cyclic_table(tsv_metrics):
    return build_retarget_table(tsv_metrics, [
        ("MSE", [("MF", "cyc", "MF"), ("ICT", "cyc", "ICT")]),
        ("MSE-in", [("MF", "cyc_in", "MF"), ("ICT", "cyc_in", "ICT"),
                    ("Laplacian MF", "lap_cyc", "MF"), ("Laplacian ICT", "lap_cyc", "ICT")]),
        ("MSE-out", [("MF", "cyc_out", "MF"), ("ICT", "cyc_out", "ICT")]),
    ])


def build_cross_table(tsv_metrics):
    return build_retarget_table(tsv_metrics, [
        ("target-GT MSE", [("ICT", "crossgt", "ICT")]),
        ("route-consistency MSE", [("ICT", "route", "ICT")]),
    ])


def build_locality_table():
    out = ['<table class="mtable">']
    out.append(
        "<thead><tr><th class=\"rowhead\">model</th>"
        "<th>E_in &uarr; (motion-only)</th><th>E_out &darr; (motion-only)</th>"
        "<th>LSR% &darr;</th><th>LER% &darr;</th></tr></thead>"
    )
    out.append("<tbody>")
    for label, ckpt_name in MODELS:
        fields = parse_gc_overall(GC_DIRS.get(label))
        if fields is None:
            out.append(
                f'<tr><th class="rowhead">{label}</th>'
                f'<td class="pending" colspan="4">pending (no eval_CBD_gc result yet)</td></tr>'
            )
            continue
        out.append(
            f'<tr><th class="rowhead">{label}</th>'
            f'<td class="cell">{fields.get("E_in", "-")}</td>'
            f'<td class="cell">{fields.get("E_out", "-")}</td>'
            f'<td class="cell">{fields.get("LSR%", "-")}</td>'
            f'<td class="cell">{fields.get("LER%", "-")}</td></tr>'
        )
    out.append("</tbody></table>")
    return "\n".join(out)


# (scenario key, frames) shown in the figure section
FIG_SCENARIOS = [
    ("mf_to_ict", MF_FRAMES),
    ("ict_to_mf_m00", ICT_FRAMES_M00),
    ("ict_to_mf_m02", ICT_FRAMES_M02),
    ("mf_to_coma", MF_FRAMES),
    ("mf_to_biwi", MF_FRAMES),
    ("ict_to_coma_m00", ICT_FRAMES_M00),
    ("ict_to_coma_m02", ICT_FRAMES_M02),
    ("ict_to_biwi_m00", ICT_FRAMES_M00),
    ("ict_to_biwi_m02", ICT_FRAMES_M02),
]
# figure column order
FIG_COLUMNS = ["NC", "NFR", "NFS", "PDB"]
# figure CSS from build_comparison_figure_html (its page-level body/nav rules excluded)
FIG_CSS = fig_mod.CSS[fig_mod.CSS.index(".figure-card"):]


def main():
    import base64

    tsv_metrics = load_tsv_metrics()

    html = ["""<!doctype html>
<html><head><meta charset="utf-8">
<title>comparison: original / self / cross / cyclic</title>
<style>
  body { font-family: -apple-system, sans-serif; background:#fff; color:#111; margin:0; padding:24px; }
  h1 { font-size:20px; }
  h2 { font-size:16px; margin-top:40px; border-bottom:1px solid #ddd; padding-bottom:6px; }
  h3 { font-size:15px; color:#333; margin-top:28px; }
  .model-row { display:flex; flex-direction:column; gap:2px; margin-bottom:14px; }
  .model-cell { display:flex; align-items:center; gap:10px; }
  .model-label { width:70px; flex:0 0 70px; font-size:12px; color:#666; text-align:right; }
  .model-cell img { flex:0 0 auto; width:auto; max-width:100%; max-height:130px; border:1px solid #ddd; background:#fff; }
  .missing { flex:1; font-size:12px; color:#999; padding:8px; border:1px dashed #ccc; }
  .legend { font-size:12px; color:#666; }
  .panel-legend { font-size:11px; color:#777; margin:2px 0 10px 80px; }
  nav { position:sticky; top:0; background:#fff; padding:8px 0; border-bottom:1px solid #ddd; z-index:10; }
  nav a { color:#06c; margin-right:16px; font-size:12px; text-decoration:none; }

  .metrics { margin-bottom:32px; }
  .metrics .unit { font-size:12px; color:#777; font-weight:normal; }
  .mblock { margin-top:18px; }
  .mlabel { font-size:12px; font-weight:600; color:#444; margin:0 0 6px; }
  table.mtable { border-collapse:collapse; width:100%; max-width:760px; font-size:13px; font-variant-numeric:tabular-nums; }
  table.mtable th, table.mtable td { padding:6px 12px; text-align:right; white-space:nowrap; }
  table.mtable thead th { font-size:10.5px; text-transform:uppercase; letter-spacing:.03em; color:#888; border-bottom:2px solid #ddd; padding-bottom:6px; font-weight:500; }
  table.mtable th.rowhead { text-align:left; color:#111; font-weight:600; }
  table.mtable tbody tr td, table.mtable tbody tr th.rowhead { border-bottom:1px solid #eee; }
  table.mtable tbody tr:last-child td, table.mtable tbody tr:last-child th { border-bottom:none; }
  table.mtable td.cell { color:#0c0c0c; font-weight:500; }
  table.mtable td.pending { color:#aaa; font-style:italic; text-align:center; }

  /* -------- qualitative figure (same as figure.html / vis_ablation_mask.html) -------- */
""" + FIG_CSS + """
</style>
</head><body>
<h1>comparison baselines (NC / NFR / NFS / PDB): original / self / cross / cyclic retargeting</h1>
<p class="legend">Quantitative metrics, then the qualitative figure (mf / ict / coma / biwi scenarios):
per-cell PNGs rendered by render/render_comparison_figure.py (matplotrender plot_mesh_gouraud, size=8, M_SCALE=0.6,
whole mesh, frontal).</p>

<div class="metrics" id="metrics">
<h2>Quantitative metrics</h2>
<p class="legend">Values &times;1e-4 (i.e. table value &times; 1e-4 = actual MSE). MF = mf_SEN/mf_ROM
n-weighted average (n=6309/11649). Source: <code>report/quant_sheet_eve-s05_full.tsv</code>
(tag=comparison rows), cross-checked against each model's eval_CBD*/eval_cyc result folders.</p>
<div class="mblock">
<p class="mlabel">Self-retargeting (MSE / MSE-in / MSE-out, Laplacian in the MSE-in region)</p>
""" + build_self_table(tsv_metrics) + """
</div>
<div class="mblock">
<p class="mlabel">Cross-retargeting (cross-identity target-GT MSE, route-consistency; ICT-only protocol)</p>
""" + build_cross_table(tsv_metrics) + """
</div>
<div class="mblock">
<p class="mlabel">Cyclic-retargeting (MSE / MSE-in / MSE-out, Laplacian in the MSE-in region)</p>
""" + build_cyclic_table(tsv_metrics) + """
</div>
<div class="mblock">
<p class="mlabel">Deformation locality (eval_CBD_gc.py, 10 real-capture identities &times; 53 basis, motion-only)</p>
""" + build_locality_table() + """
</div>
</div>

<nav>
<a href="#metrics">metrics</a>
"""]

    for key, _ in FIG_SCENARIOS:
        html.append(f'<a href="#{key}">{key}</a>')
    html.append("</nav>")

    html.append('<section class="figure-card figure-section"><h2 id="figure">Qualitative figure</h2>'
                '<p class="legend">Two tables per (scenario, frame). Left: Source Neutral Mesh / Source Expression '
                'Mesh (real captured GT deformation) / Target Neutral Mesh. Right: rows = self / cross / cyclic '
                'retargeting, columns = NC / NFR / NFS / PDB. Diff-heatmap colors (YlOrRd) show where each output '
                'deforms the mesh: every cell is normalized on its own, against that row&#39;s diff_base (source template '
                'for self/cyclic, target template for cross). Frames: render/comparison_frames.py. '
                'Generated by render/render_comparison_figure.py + render/build_ablation_gallery_comparison.py.</p>')
    for key, frames in FIG_SCENARIOS:
        html.append(f'<h2 id="{key}">{fig_mod.TITLES[key]}</h2>')
        for fr in frames:
            html.append(f'<h3>frame {fr}</h3>')
            html.append(fig_mod.build_frame(key, fr, columns=FIG_COLUMNS))
    html.append("</section>")

    html.append("</body></html>")

    out_path = OUT_ROOT / "vis_comparison.html"
    out_path.write_text("\n".join(html))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
