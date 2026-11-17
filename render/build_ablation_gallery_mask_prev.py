"""Rebuilds vis_CBD/vis_ablation_mask/vis_ablation_mask_prev.html: keeps the
existing masked-MSE metrics section verbatim (extracted from the current file
on disk -- this script never recomputes those numbers), replaces the old
model-row qualitative image section with the new two-table-per-(scenario,
frame) layout (matches vis_ablation_mask.html / design_study_ablation.html),
using the per-cell PNGs from render_ablation_mask_prev_figure.py under
vis_CBD/vis_ablation_mask/figure_prev/.

5 columns this time (vs. the current gallery's 4): baseline / cage-consistency
/ cyclic / cage+cyclic / PDBplus -- see render_ablation_mask_prev_figure.py's
docstring for the checkpoint identification.

Run from repo root:
    python render/build_ablation_gallery_mask_prev.py
"""
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "vis_CBD" / "vis_ablation_mask" / "vis_ablation_mask_prev.html"
FIG_ROOT = REPO_ROOT / "vis_CBD" / "vis_ablation_mask" / "figure_prev"

LBL_BASELINE = 'Ours<sup>&dagger;</sup>'
LBL_CONSIST = 'w/ <i>&#8466;</i><sub>consist</sub>'
LBL_CYCLIC = 'w/ <i>&#8466;</i><sub>cyclic</sub>'
LBL_BOTH = 'w/ <i>&#8466;</i><sub>consist</sub> + <i>&#8466;</i><sub>cyclic</sub>'
LBL_PDBPLUS = 'PDBplus'

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

REF_ROWS = [
    ("source_neutral", "Source Neutral Mesh", ""),
    ("source_expression", "Source Expression Mesh", "frame"),
    ("target_neutral", "Target Neutral Mesh", ""),
]

FIG_COLUMNS = [
    ("baseline", LBL_BASELINE),
    ("consist", LBL_CONSIST),
    ("cyclic", LBL_CYCLIC),
    ("both", LBL_BOTH),
    ("pdbplus", LBL_PDBPLUS),
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
        for col_key, _ in FIG_COLUMNS:
            out.append(fig_img_cell(frame_dir / f"{row_key}_{col_key}.png"))
        out.append("</tr>")
    out.append("</tbody>")
    out.append("<tfoot><tr>")
    out.append('<th class="rowhead-col"></th>')
    for _, label in FIG_COLUMNS:
        out.append(f"<th>{label}</th>")
    out.append("</tr></tfoot>")
    out.append("</table>")
    return "\n".join(out)


def extract_metrics_section(old_html: str) -> str:
    start = old_html.find('<section id="metrics"')
    end = old_html.find("</section>", start) + len("</section>")
    assert start >= 0 and end > start, "metrics section not found in existing vis_ablation_mask_prev.html"
    return old_html[start:end]


HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>mask_cage_consist ablation (prev/pre-2026-09-21): original / self / cross / cyclic</title>
<style>
  body { font-family: -apple-system, sans-serif; background:#111; color:#eee; margin:0; padding:24px; }
  h1 { font-size:20px; }
  h2 { font-size:16px; margin-top:40px; border-bottom:1px solid #444; padding-bottom:6px; }
  h3 { font-size:15px; color:#ddd; margin-top:28px; }
  .legend { font-size:12px; color:#888; }
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

  /* -------- qualitative figure (2026-09-26 rebuild, white card, dark page) -------- */
  .figure-card { background:#fff; color:#111; border-radius:8px; padding:20px 24px; margin-bottom:32px; }
  .figure-card h2 { color:#111; border-bottom:1px solid #ddd; padding-bottom:6px; margin-top:0; }
  .figure-card .legend { color:#666; }
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
</style>
</head><body>
<h1>mask_cage_consist ablation -- prev/pre-2026-09-21 snapshot (baseline / cage-consistency / cyclic / cage+cyclic / PDBplus)</h1>
<p class="legend">This is the earlier (pre-refresh) checkpoint set: baseline
ckpts_CBD7/2026-08-03-09-09-59-NGBCv5, cage-consistency
ckpts_CBD7/2026-08-02-19-04-17-NGBCv5, cyclic ckpts_CBD_cyclic/2026-08-04-08-48-34-NGBCv5,
cage+cyclic ckpts_CBD7/2026-08-05-00-55-47-NGBCv5 -- the PLAIN (non-masked)
ablation, not the later mask_cage_consist refresh -- plus PDBplus
(ckpts_CBD8/2026-09-15-23-20-58-PDBplusv6, version=6). See
render/render_ablation_mask_prev_figure.py's docstring for how these were
identified. The metrics table below is unchanged from the original version of
this file (not recomputed here).</p>
"""


def main():
    old_html = OUT_PATH.read_text()
    metrics_section = extract_metrics_section(old_html)

    rows = [HTML_HEAD, metrics_section]
    rows.append('<section class="figure-card figure-section">')
    rows.append('<h2 id="figure">Qualitative figure</h2>')
    rows.append("""<p class="legend">Two tables per (scenario, frame). Left: identity reference (vertical row labels,
one column of images), shown once (doesn't vary by self/cross/cyclic) -- Source Neutral Mesh
(plain-shaded) / Source Expression Mesh (real captured GT deformation, own independent
diff-heatmap color scale, captioned with the frame index) / Target Neutral Mesh (plain-shaded).
Right: rows = self / cross / cyclic retargeting, columns = the 5 variants above.
Diff-heatmap colors (YlOrRd) are normalized jointly per row across all 5 variants, against that
row's diff_base (source template for self/cyclic, target template for cross).
Generated by render/render_ablation_mask_prev_figure.py + render/build_ablation_gallery_mask_prev.py.</p>""")
    rows.append("<nav>")
    for key, _desc, _frames in SCENARIOS:
        rows.append(f'<a href="#{key}">{key}</a>')
    rows.append("</nav>")

    for key, desc, frames in SCENARIOS:
        scenario_dir = FIG_ROOT / key
        rows.append(f'<h2 id="{key}">{key} &mdash; {desc}</h2>')
        for fr in frames:
            frame_dir = scenario_dir / f"frame_{fr:06d}"
            rows.append(f"<h3>frame {fr}</h3>")
            rows.append('<div class="figure-row">')
            rows.append(build_fig_ref_table(frame_dir, fr))
            rows.append(build_fig_result_table(frame_dir))
            rows.append("</div>")

    rows.append("</section>")
    rows.append("</body></html>")

    OUT_PATH.write_text("\n".join(rows))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
