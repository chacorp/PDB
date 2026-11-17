"""
Builds the qualitative comparison HTML gallery from the per-cell PNGs produced by
render_design_study_figure.py (vis_CBD/vis_design_study/figure/<scenario>/frame_*/*.png).

Layout per (scenario, frame): TWO tables side by side.
  - Left (reference) table: 3 rows x 1 column -- Source Neutral Mesh, Source
    Expression Mesh (= GT, captioned with the frame index), Target Neutral Mesh,
    stacked vertically, each with a vertical-text row label (same style as the
    right table's self/cross/cyclic row labels). These don't vary by self/cross/
    cyclic, so they're shown once instead of being duplicated per row.
  - Right (results) table: rows = self / cross / cyclic retargeting (row labels
    set vertically), columns = the 8 design_study ablation variants (Ours-dagger,
    no activation, w/softplus, w/ELU, w/softPOU, w/oMask, Equation 15 [delta],
    Equation 16 [matrix]), captioned in a plain (non-bold) footer row below the
    images (matching report/example.png's layout), no rule/divider above it.

Images are embedded as base64 data URIs (same convention as
build_ablation_gallery_design_study.py / build_ablation_gallery_mask.py), so the
output is a single, large, self-contained HTML file.

Run from repo root:
    python render/build_design_study_figure_gallery.py
"""
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_ROOT = REPO_ROOT / "vis_CBD" / "vis_design_study" / "figure"
OUT_PATH = REPO_ROOT / "vis_CBD" / "vis_design_study" / "design_study_figure.html"

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
    # NOTE: build_ablation_gallery_design_study.py's SCENARIOS list has these two
    # entries' frame lists swapped (a pre-existing bug there); the actual frames
    # used/saved on disk (per render_ablation_frames_design_study.py's real
    # SCENARIOS: coma_to_biwi uses COMA_FRAMES, biwi_to_coma uses BIWI_FRAMES) are
    # correct below -- verified against vis_CBD/vis_design_study/figure/*/frame_*.
    ("coma_to_biwi", "coma -> biwi (source: FaceTalk_170809_00138_TA, clip mouth_extreme)",
     [47, 74, 89, 104]),
    ("biwi_to_coma", "biwi -> coma (source: F2, clip F2_e38)",
     [9, 73, 92, 121]),
]

REF_ROWS = [
    ("source_neutral", "Source Neutral Mesh"),
    ("source_expression", "Source Expression Mesh"),
    ("target_neutral", "Target Neutral Mesh"),
]

COLUMNS = [
    ("ours", "Ours<sup>&dagger;</sup>"),
    ("no_activation", "no activation"),
    ("softplus", "w/softplus"),
    ("elu", "w/ELU"),
    ("soft_pou", "w/softPOU"),
    ("wo_mask", "w/oMask"),
    ("eq15_delta", "Equation 15"),
    ("eq16_matrix", "Equation 16"),
]

ROWS = [
    ("self", "Self retargeting"),
    ("cross", "Cross retargeting"),
    ("cyclic", "Cyclic retargeting"),
]

HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>design_study figure: self / cross / cyclic retargeting per-cell comparison</title>
<style>
  body { font-family: "Times New Roman", Times, "Liberation Serif", serif; background:#fff; color:#111; margin:0; padding:24px; }
  h1 { font-size:20px; }
  h2 { font-size:16px; margin-top:40px; border-bottom:1px solid #ddd; padding-bottom:6px; }
  h3 { font-size:15px; color:#333; margin-top:26px; }
  .legend { font-size:12px; color:#666; }
  nav { position:sticky; top:0; background:#fff; padding:8px 0; border-bottom:1px solid #ddd; z-index:10; }
  nav a { color:#06c; margin-right:16px; font-size:12px; text-decoration:none; }

  .figure-row { display:flex; align-items:flex-start; gap:14px; margin-top:10px; margin-bottom:24px; }
  table.grid { border-collapse:collapse; }
  table.grid th, table.grid td { border:none; padding:0 1px; text-align:center; vertical-align:middle; }
  table.grid tfoot th { font-size:12px; color:#333; padding:6px 1px; font-weight:normal; }
  table.grid tfoot th.rowhead-col { border:none; }
  table.grid tbody th.rowhead { font-size:12px; color:#333; font-weight:normal; white-space:nowrap;
    writing-mode:vertical-rl; transform:rotate(180deg); text-align:center; padding:0 4px; }
  table.grid td img { width:120px; height:auto; display:block; margin:0 auto; background:#fff; }
  table.grid td.missing { color:#bbb; font-size:11px; }
  table.grid td .frame-tag { font-size:11px; color:#333; }
  table.ref-grid { border-right:1px solid #ddd; padding-right:14px; }
</style>
</head><body>
<h1>design_study figure: source mesh / GT / self / cross / cyclic retargeting, per-variant comparison</h1>
<p class="legend">Two tables per (scenario, frame). Left: identity reference (vertical row labels,
one column of images), shown once (doesn't vary by self/cross/cyclic) -- Source Neutral Mesh
(plain-shaded) / Source Expression Mesh (real captured GT deformation, own independent
diff-heatmap color scale, captioned with the frame index) / Target Neutral Mesh (plain-shaded).
Right: rows = self / cross / cyclic retargeting, columns = the 8
design_study ablation variants (Ours&dagger; = relu (chosen); Equation 15 = delta/displacement
representation; Equation 16 = matrix/transform representation). Diff-heatmap colors (YlOrRd) are
normalized jointly per row across 7 of the 8 variants, against that row's diff_base (source template
for self/cyclic, target template for cross); w/softplus is excluded from that shared scale and
normalized independently, since its much larger displacement was drowning out the other variants'
color range. Color intensity is therefore comparable within a scale group but not across rows,
frames, or against w/softplus.
Generated by render/render_design_study_figure.py + render/build_design_study_figure_gallery.py.</p>
<nav>
"""


def img_cell(path: Path, caption: str = "") -> str:
    if not path.exists():
        return f'<td class="missing">missing<br>{path.name}</td>'
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    cap = f'<div class="frame-tag">{caption}</div>' if caption else ""
    return f'<td><img src="data:image/png;base64,{b64}" loading="lazy">{cap}</td>'


def build_ref_table(frame_dir: Path, frame_idx: int) -> str:
    out = ['<table class="grid ref-grid">', "<tbody>"]
    for row_key, row_label in REF_ROWS:
        caption = f"Frame {frame_idx}" if row_key == "source_expression" else ""
        cell = img_cell(frame_dir / f"{row_key}.png", caption)
        out.append(f'<tr><th class="rowhead">{row_label}</th>{cell}</tr>')
    out.append("</tbody></table>")
    return "\n".join(out)


def build_result_table(frame_dir: Path) -> str:
    out = ['<table class="grid">', "<tbody>"]
    for row_key, row_label in ROWS:
        out.append(f'<tr><th class="rowhead">{row_label}</th>')
        for col_key, _ in COLUMNS:
            out.append(img_cell(frame_dir / f"{row_key}_{col_key}.png"))
        out.append("</tr>")
    out.append("</tbody>")
    out.append("<tfoot><tr>")
    out.append('<th class="rowhead-col"></th>')
    for _, label in COLUMNS:
        out.append(f"<th>{label}</th>")
    out.append("</tr></tfoot>")
    out.append("</table>")
    return "\n".join(out)


def main():
    rows = [HTML_HEAD]
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
            rows.append(build_ref_table(frame_dir, fr))
            rows.append(build_result_table(frame_dir))
            rows.append("</div>")

    rows.append("</body></html>")
    OUT_PATH.write_text("\n".join(rows))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
