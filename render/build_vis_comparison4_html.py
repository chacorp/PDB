"""Build vis_CBD/vis_comparison/vis_comparison-4.html: vis_comparison-3.html plus a
"PDBplus-out0" row/column = PDBplus (version 6) out_type 0,
ckpts_PDBplus_ds/out_type0/2026-11-13-08-57-46-PDBplusv6/model_best.pth (epoch 920).

NC / NFR / NFS / PDB: exactly as vis_comparison-3.html. PDBplus-out0: metrics
aggregated from its eval folders with the same rules (build_vis_comparison3_html.
metrics_from_eval), cells from figure_pdbplus_out0/. Evaluation and cell rendering:
eval_pdbplus_out0.sh.

Run from repo root:
    python render/build_vis_comparison4_html.py
"""
import build_ablation_gallery_comparison as gallery
import build_comparison_figure_html as fig_mod
import build_vis_comparison3_html as v3

OUT_HTML = v3.VIS_ROOT / "vis_comparison-4.html"
NEW_NAME = "PDBplus-out0"
NEW_CKPT = "2026-11-13-08-57-46-PDBplusv6"
COLUMNS = v3.COLUMNS + [NEW_NAME]
COL_SRC = dict(v3.COL_SRC)
COL_SRC[NEW_NAME] = (v3.VIS_ROOT / "figure_pdbplus_out0", NEW_NAME)


def main():
    metrics = gallery.load_tsv_metrics()
    metrics["PDB"] = v3.metrics_from_eval(v3.CKPT_NAME)
    metrics[NEW_NAME] = v3.metrics_from_eval(NEW_CKPT)
    gallery.MODELS = list(gallery.MODELS) + [(NEW_NAME, NEW_CKPT)]
    gallery.GC_DIRS["PDB"] = f"eval_CBD_gc/{v3.CKPT_NAME}-{v3.GC_SUFFIX}"
    gallery.GC_DIRS[NEW_NAME] = f"eval_CBD_gc/{NEW_CKPT}-{v3.GC_SUFFIX}"

    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>comparison: NC / NFR / NFS / PDB / PDBplus-out0</title>
<style>{v3.CSS}</style></head><body>
<h1>comparison baselines (NC / NFR / NFS / PDB / PDBplus-out0): self / cross / cyclic retargeting</h1>
<p class="legend"><b>PDB</b> = the --no_col_norm model: ckpts_CBD/{v3.CKPT_NAME}/model_best.pth
(epoch 700, use_data2, relu, no --align_latent).<br>
<b>{NEW_NAME}</b> = PDBplus (version 6), out_type 0 (control displacements):
ckpts_PDBplus_ds/out_type0/{NEW_CKPT}/model_best.pth (epoch 920, use_data4, sqrelu, no_col_norm,
num_cage_v 128, no --align_latent).</p>
<div class="metrics" id="metrics">
<h2>Quantitative metrics</h2>
<p class="legend">Values &times;1e-4 (table value &times; 1e-4 = actual MSE). MF = mf_SEN/mf_ROM weighted by
their number of frames (n=6309/11649). NC / NFR / NFS: report/quant_sheet_eve-s05_full.tsv (tag=comparison rows).
PDB / {NEW_NAME}: aggregated the same way from each model's eval_CBD / eval_cyc / eval_CBD_cif / eval_CBD_gc folders.</p>
<div class="mblock"><p class="mlabel">Self-retargeting (MSE / MSE-in / MSE-out, Laplacian in the MSE-in region)</p>
{gallery.build_self_table(metrics)}</div>
<div class="mblock"><p class="mlabel">Cross-retargeting (cross-identity target-GT MSE, route-consistency; ICT-only protocol)</p>
{gallery.build_cross_table(metrics)}</div>
<div class="mblock"><p class="mlabel">Cyclic-retargeting (MSE / MSE-in / MSE-out, Laplacian in the MSE-in region)</p>
{gallery.build_cyclic_table(metrics)}</div>
<div class="mblock"><p class="mlabel">Deformation locality (eval_CBD_gc.py, 10 real-capture identities &times; 53 basis, motion-only)</p>
{gallery.build_locality_table()}</div>
</div>
"""]
    parts.append('<nav><a href="#metrics">metrics</a>'
                 + "".join(f'<a href="#{k}">{k}</a>' for k, _ in gallery.FIG_SCENARIOS) + '</nav>')
    parts.append('<section class="figure-card figure-section"><h2 id="figure">Qualitative figure</h2>'
                 '<p class="legend">Two tables per (scenario, frame). Left: Source Neutral Mesh / Source Expression '
                 'Mesh (real captured GT deformation) / Target Neutral Mesh. Right: rows = self / cross / cyclic '
                 f'retargeting, columns = NC / NFR / NFS / PDB (the --no_col_norm model) / {NEW_NAME}. '
                 'Diff-heatmap colors (YlOrRd) show where each output deforms the mesh: every cell is normalized '
                 'on its own, against that row&#39;s diff_base (source template for self/cyclic, target template '
                 'for cross). An NC output that leaves the camera view is centered and rescaled on its own; the '
                 'ratio under the cell is its drawn size relative to the other cells (heatmap colors unchanged).</p>')
    for key, frames in gallery.FIG_SCENARIOS:
        parts.append(f'<h2 id="{key}">{fig_mod.TITLES[key]}</h2>')
        for fr in frames:
            parts.append(f"<h3>frame {fr}</h3>")
            parts.append(fig_mod.build_frame(key, fr, columns=COLUMNS, col_src=COL_SRC))
    parts.append("</section></body></html>")
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
