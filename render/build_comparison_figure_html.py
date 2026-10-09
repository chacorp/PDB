"""Assemble vis_CBD/vis_comparison/figure.html from the per-cell PNGs written by
render/render_comparison_figure.py (mf <-> ict scenarios).

Layout per (scenario, frame), same as vis_ablation_mask.html's qualitative figure:
  left table : source neutral / source expression (GT) / target neutral
  right table: rows = self / cross / cyclic retargeting, cols = NC / NFS / NFR / PDB
Missing cells (no cache for that model/row) are shown as "missing".
"""
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# per-cell PNGs are already cached by render/render_comparison_figure.py; this script
# only assembles them, so it must not import the renderer (GPU / eval deps).
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "figure"
COLUMNS = ["NC", "NFS", "NFR", "PDB"]
# (scenario key, frame indices in display order)
SCENARIOS = [
    ("mf_to_ict", [8880, 5840, 8404, 6638]),
    ("ict_to_mf_m00", [14, 1139, 336, 1116]),
    ("ict_to_mf_m02", [39, 75, 92, 158]),
    ("mf_to_coma", [8880, 5840, 8404, 6638]),
    ("mf_to_biwi", [8880, 5840, 8404, 6638]),
    ("ict_to_coma_m00", [14, 1139, 336, 1116]),
    ("ict_to_coma_m02", [39, 75, 92, 158]),
    ("ict_to_biwi_m00", [14, 1139, 336, 1116]),
    ("ict_to_biwi_m02", [39, 75, 92, 158]),
]

OUT_HTML = REPO_ROOT / "vis_CBD" / "vis_comparison" / "figure.html"


def img_tag(path):
    if path.exists():
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<div class="clip"><img src="data:image/png;base64,{b64}" loading="lazy"></div>'
    return '<div class="missing">missing</div>'


def build_frame(key, frame_idx):
    d = OUT_ROOT / key / f"frame_{frame_idx:06d}"
    parts = []
    parts.append('<div class="figure-row">')

    # identity reference: stacked vertically, row labels on the left (as in vis_ablation_mask.html)
    parts.append('<table class="grid ref-grid">')
    parts.append('<tbody>')
    parts.append(f'<tr><th class="rowhead">Source Neutral Mesh</th><td class="imgcell">{img_tag(d / "source_neutral.png")}</td></tr>')
    parts.append(f'<tr><th class="rowhead">Source Expression Mesh</th><td class="imgcell">{img_tag(d / "source_expression.png")}</td></tr>')
    parts.append(f'<tr><th class="rowhead">Target Neutral Mesh</th><td class="imgcell">{img_tag(d / "target_neutral.png")}</td></tr>')
    parts.append('</tbody></table>')

    # results: rows = self / cross / cyclic, columns = models, captions below
    parts.append('<table class="grid"><tbody>')
    for row_key, row_label in [("self", "Self retargeting"), ("cross", "Cross retargeting"), ("cyclic", "Cyclic retargeting")]:
        parts.append(f'<tr><th class="rowhead">{row_label}</th>')
        for col_name in COLUMNS:
            parts.append(f'<td class="imgcell">{img_tag(d / f"{row_key}_{col_name}.png")}</td>')
        parts.append('</tr>')
    parts.append('</tbody><tfoot><tr><th class="rowhead-col"></th>')
    for col_name in COLUMNS:
        parts.append(f'<th>{col_name}</th>')
    parts.append('</tr></tfoot></table>')
    parts.append('</div>')
    return "\n".join(parts)


TITLES = {'mf_to_ict': 'mf_to_ict &mdash; mf -&gt; ict m00 (source: mf_ROM test id, absolute frame idx)', 'ict_to_mf_m00': 'ict_to_mf_m00 &mdash; ict m00 -&gt; mf (source: ict-cap m00, 20240325_MySlate_924, its own motion)', 'ict_to_mf_m02': 'ict_to_mf_m02 &mdash; ict m02 -&gt; mf (source: ict-cap m02, 20240318_MySlate_922, its own motion)',
          'mf_to_coma': 'mf_to_coma &mdash; mf -&gt; coma (source: mf_ROM test id, absolute frame idx)',
          'mf_to_biwi': 'mf_to_biwi &mdash; mf -&gt; biwi (source: mf_ROM test id, absolute frame idx)',
          'ict_to_coma_m00': 'ict_to_coma_m00 &mdash; ict m00 -&gt; coma (source: ict-cap m00, 20240325_MySlate_924, its own motion)',
          'ict_to_coma_m02': 'ict_to_coma_m02 &mdash; ict m02 -&gt; coma (source: ict-cap m02, 20240318_MySlate_922, its own motion)',
          'ict_to_biwi_m00': 'ict_to_biwi_m00 &mdash; ict m00 -&gt; biwi (source: ict-cap m00, 20240325_MySlate_924, its own motion)',
          'ict_to_biwi_m02': 'ict_to_biwi_m02 &mdash; ict m02 -&gt; biwi (source: ict-cap m02, 20240318_MySlate_922, its own motion)'}

CSS = """
body { font-family: -apple-system, sans-serif; background:#111; color:#eee; margin:0; padding:24px; }
h1 { font-size:20px; }
h2 { font-size:16px; margin-top:40px; border-bottom:1px solid #444; padding-bottom:6px; }
h3 { font-size:15px; color:#ddd; margin-top:28px; }
.legend { font-size:12px; color:#888; }
nav { position:sticky; top:0; background:#111; padding:8px 0; border-bottom:1px solid #333; z-index:10; }
nav a { color:#8cf; margin-right:16px; font-size:12px; text-decoration:none; }
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
table.ref-grid { border-right:1px solid #ddd; padding-right:14px; }
"""


def main():
    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>comparison figure: NC / NFS / NFR / PDB (mf / ict / coma / biwi)</title>
<style>{CSS}</style></head><body>
<h1>comparison baselines: qualitative figure (mf / ict / coma / biwi)</h1>
"""]
    parts.append('<section class="figure-card figure-section"><h2 id="figure">Qualitative figure</h2>'
                 '<p class="legend">Two tables per (scenario, frame). Left: identity reference (vertical row labels, '
                 'one column of images): Source Neutral Mesh / Source Expression Mesh (real captured GT deformation)'
                 ' / Target Neutral Mesh. Right: rows = self / cross / cyclic retargeting, '
                 'columns = NC / NFS / NFR / PDB. Diff-heatmap colors (YlOrRd) are normalized jointly per row across the '
                 'four models (NC on its own scale), against that row&#39;s diff_base (source template for self/cyclic, '
                 'target template for cross). "missing" = no cached result.\n'
                 'Generated by render/render_comparison_figure.py + render/build_comparison_figure_html.py.</p>')
    parts.append('<nav>' + "".join(f'<a href="#{sc[0]}">{sc[0]}</a>' for sc in SCENARIOS) + '</nav>')
    for key, idxs in SCENARIOS:
        parts.append(f'<h2 id="{key}">{TITLES[key]}</h2>')
        for frame_idx in idxs:
            parts.append(f"<h3>frame {frame_idx}</h3>")
            parts.append(build_frame(key, frame_idx))
    parts.append("</section>")
    parts.append("</body></html>")
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
