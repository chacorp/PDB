"""Build vis_CBD/vis_ablation_train/index.html gallery for
render_ablation_frames_train.py output.

Grouped by scenario -> sample, with all 4 models' images laid out side by
side so the ablation effect is visible at a glance for each sample.
Train-set counterpart of build_ablation_gallery.py -- samples here are
synthetic PCA/coefficient draws (fixed seeds/indices), not literal test
frame indices, so they're labeled "sample N" instead of a frame number.
"""
import base64
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_ablation_train"

MODELS = [
    ("baseline", "2026-08-03-09-09-59-NGBCv5"),
    ("cage-consistency", "2026-08-02-19-04-17-NGBCv5"),
    ("cyclic", "2026-08-04-08-48-34-NGBCv5"),
    ("cage+cyclic", "2026-08-05-00-55-47-NGBCv5"),
]

N_SAMPLES = 4

SCENARIOS = [
    ("mf_to_ict", "mf -> ict (source: mf_ROM train id m--20171024--0000--002757580--GHS, "
                  "PCA seeds 18/0/3/11; target: ict train id idx 10)"),
    ("ict_to_mf", "ict -> mf (source: ict train id idx 10, expression idx 433/689/1224/1788; "
                  "target: mf_ROM train id m--20171024--0000--002757580--GHS)"),
    ("coma_to_mf", "coma -> mf (source: coma train id FaceTalk_170728_03272_TA, "
                   "PCA seeds 0/11/3/7)"),
    ("mf_to_coma", "mf -> coma (source: mf_ROM train id m--20171024--0000--002757580--GHS, "
                   "PCA seeds 18/0/3/11)"),
    ("biwi_to_mf", "biwi -> mf (source: biwi train id F2, PCA seeds 0/1/7/3)"),
    ("mf_to_biwi", "mf -> biwi (source: mf_ROM train id m--20171024--0000--002757580--GHS, "
                   "PCA seeds 18/0/3/11)"),
    ("coma_to_biwi", "coma -> biwi (source: coma train id FaceTalk_170728_03272_TA, "
                     "PCA seeds 0/11/3/7)"),
    ("biwi_to_coma", "biwi -> coma (source: biwi train id F2, PCA seeds 0/1/7/3)"),
]

HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>4-model ablation (train set): original / self / cross / cyclic</title>
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
</style>
</head><body>
<h1>4-model ablation (baseline / cage-consistency / cyclic / cage+cyclic) -- TRAIN SET: original / self / cross / cyclic retargeting</h1>
<p class="legend">Grouped by sample so all 4 models are comparable at a glance. Each row = one model's
result for that sample. Each image: original(GT) | self | cross | cyclic. Samples are synthetic
draws from the train-time PCA models (mf_ROM/coma/biwi, scale=1.5) or train-split synthetic
coefficient arrays (ict), decoded and retargeted directly via model.predict_coordinate /
model.retarget_animation (mirrors eval_CBD_cyc.py's self-/cross-retargeting exactly), since these
train samples have no EvalDataset-compatible literal frame files. Rendered with
utils.matplotlib_rnd.plot_image_array (size=8, mode=shade, bg_black=False, M_SCALE=0.6, whole mesh, frontal).</p>
<nav>
"""

rows = [HTML_HEAD]
for direction, desc in SCENARIOS:
    rows.append(f'<a href="#{direction}">{direction}</a>')
rows.append('</nav>')

for direction, desc in SCENARIOS:
    rows.append(f'<h2 id="{direction}">{desc}</h2>')
    for i in range(N_SAMPLES):
        rows.append(f'<h3>sample {i}</h3>')
        rows.append('<p class="panel-legend">original | self | cross | cyclic</p>')
        rows.append('<div class="model-row">')
        for model_name, ckpt_dir in MODELS:
            img_path = f'{ckpt_dir}/{direction}/frame_{i:02d}.png'
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
