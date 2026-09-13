"""Build vis_CBD/vis_design_study_diff/index.html gallery for
render_ablation_frames_design_study_diff.py output (displacement-from-neutral
heatmap version of the design_study ablation gallery).

Copied from build_ablation_gallery_design_study.py with only OUT_ROOT and the
title/legend text changed to describe the heatmap encoding and the fixed
color-scale value (read from _vmax.txt, written by the render script).
"""
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_design_study_diff"

MODELS = [
    ("no activation",        "2025-09-30-16-28-51-NGBCv5"),
    ("softplus",             "2025-09-28-14-50-34-NGBCv5"),
    ("elu",                  "2025-10-08-02-55-02-NGBCv5"),
    ("relu (chosen)",        "2025-10-23-17-32-49-NGBCv5"),
    ("delta (displacement)", "2025-10-22-16-22-45-NGBCv5"),
    ("matrix (transform)",   "2025-10-22-16-26-26-NGBCv5"),
    ("soft POU",             "2025-10-03-22-40-03-NGBCv5"),
    ("w/o mask",             "2025-09-26-10-16-32-NGBCv5"),
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

vmax_note = ""
vmax_path = OUT_ROOT / "_vmax.txt"
if vmax_path.exists():
    vmax_note = " Fixed color scale: " + vmax_path.read_text().strip() + "."

HTML_HEAD = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>design_study displacement-from-neutral heatmap</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background:#111; color:#eee; margin:0; padding:24px; }}
  h1 {{ font-size:20px; }}
  h2 {{ font-size:16px; margin-top:40px; border-bottom:1px solid #444; padding-bottom:6px; }}
  h3 {{ font-size:15px; color:#ddd; margin-top:28px; }}
  .model-row {{ display:flex; flex-direction:column; gap:2px; margin-bottom:14px; }}
  .model-cell {{ display:flex; align-items:center; gap:10px; }}
  .model-label {{ width:150px; flex:0 0 150px; font-size:12px; color:#aaa; text-align:right; }}
  .model-cell img {{ flex:1 1 auto; max-width:1800px; width:100%; border:1px solid #333; background:#fff; }}
  .missing {{ flex:1; font-size:12px; color:#666; padding:8px; border:1px dashed #444; }}
  .legend {{ font-size:12px; color:#888; }}
  .panel-legend {{ font-size:11px; color:#777; margin:2px 0 10px 160px; }}
  nav {{ position:sticky; top:0; background:#111; padding:8px 0; border-bottom:1px solid #333; z-index:10; }}
  nav a {{ color:#8cf; margin-right:16px; font-size:12px; text-decoration:none; }}
</style>
</head><body>
<h1>design_study displacement-from-neutral heatmap (activation: no activation / softplus / elu / relu(chosen) &mdash; representation: delta / matrix / soft POU &mdash; masking: w/o mask)</h1>
<p class="legend">Companion to vis_design_study/index.html: each panel colors original/self/cross/cyclic by
per-vertex displacement magnitude from a neutral template (original/self/cyclic vs. the SOURCE identity's
neutral; cross vs. the TARGET identity's neutral) instead of shading, YlOrRd colormap, utils.matplotlib_rnd.plot_image_array_diff.{vmax_note}
Same scenarios/frames/identity meshes as vis_ablation_mask/index.html.</p>
<nav>
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

out_path = OUT_ROOT / "index.html"
out_path.write_text("\n".join(rows))
print(f"wrote {out_path}")
