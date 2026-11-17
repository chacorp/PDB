"""Build vis_CBD/vis_comparison/vis_comparison-3.html: vis_comparison.html's metrics
and qualitative figure, with the "PDB" row and column taken from the --no_col_norm
model (ckpts_CBD/2026-11-04-01-05-29-NGBCv5, model_best.pth = epoch 700).

NC / NFR / NFS: same sources as vis_comparison.html (report/quant_sheet_eve-s05_full.tsv
and figure/). PDB: metrics aggregated from the model's eval folders, cells from
figure_no_col_norm/. Evaluation and cell rendering: eval_comparison_no_col_norm.sh.

Run from repo root:
    python render/build_vis_comparison3_html.py
"""
import re
from pathlib import Path

import build_ablation_gallery_comparison as gallery
import build_comparison_figure_html as fig_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
VIS_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison"
OUT_HTML = VIS_ROOT / "vis_comparison-3.html"
CKPT_NAME = "2026-11-04-01-05-29-NGBCv5"
GC_SUFFIX = "gc-idm00_m02_m04_m05_m06_w00_w01_w02_w05_w08-tau0.001-alpha0.1-dil0.2-area"
COLUMNS = ["NC", "NFR", "NFS", "PDB"]
COL_SRC = {"PDB": (VIS_ROOT / "figure_no_col_norm", "PDB")}
SCALE = 1e4  # tables show value x 1e-4

EVAL_RE = re.compile(r"\[Eval\] MSE: (\S+) MSE-in: (\S+) MSE-out: (\S+) Lap: (\S+)")
N_RE = re.compile(r"\[total data\]: (\d+)")
FIELDS = ("mse", "in", "out", "lap")


def read_eval_log(folder):
    """-> (number of frames, {mse, in, out, lap}) from a finished eval folder's log.txt"""
    text = (folder / "log.txt").read_text()
    found = EVAL_RE.findall(text)
    if not found:
        raise RuntimeError(f"no [Eval] line in {folder}/log.txt")
    return int(N_RE.search(text).group(1)), dict(zip(FIELDS, map(float, found[-1])))


def mean_logs(logs, field, weighted=False):
    if weighted:
        return sum(n * m[field] for n, m in logs) / sum(n for n, _ in logs)
    return sum(m[field] for _, m in logs) / len(logs)


def cyc_logs(ckpt_name, pattern, expected):
    folders = sorted((REPO_ROOT / "eval_cyc" / ckpt_name).glob(pattern))
    if len(folders) != expected:
        raise RuntimeError(f"{ckpt_name}/{pattern}: expected {expected} folders, found {len(folders)}")
    return [read_eval_log(d) for d in folders]


def read_cif_mean(ckpt_name, label):
    folders = list((REPO_ROOT / "eval_CBD_cif").glob(f"{ckpt_name}-cif-ict-mf_SEN-mf_ROM-*"))
    if len(folders) != 1:
        raise RuntimeError(f"{ckpt_name}: expected 1 eval_CBD_cif folder, found {len(folders)}")
    return float(re.search(rf"{label}:\s+(\S+) /", (folders[0] / "log.txt").read_text()).group(1))


def metrics_from_eval(ckpt_name):
    """Same aggregation as the tag=comparison rows of the tsv (x1e-4), in
    gallery.load_tsv_metrics()'s format:
      MF  = mf_SEN / mf_ROM weighted by their number of frames
            (cyclic: mean over the 5 ICT targets first)
      ICT cyclic = mean over the 10 ict -> mf folders; Laplacian only is weighted
            by frames (the two motion clips have 183 and 1147 frames)
    """
    self_root = REPO_ROOT / "eval_CBD" / f"{ckpt_name}-eval"
    self_mf = [read_eval_log(self_root / f"{s}-masked-laplacian") for s in ("mf_SEN", "mf_ROM")]
    _, self_ict = read_eval_log(self_root / "ict-masked-laplacian")

    n_sen, n_rom = self_mf[0][0], self_mf[1][0]
    cyc_sen = cyc_logs(ckpt_name, "mf_SEN-src012_test-to-ict-cap_test-ID_*-masked-laplacian", 5)
    cyc_rom = cyc_logs(ckpt_name, "mf_ROM-src012_test-to-ict-cap_test-ID_*-masked-laplacian", 5)
    cyc_ict = cyc_logs(ckpt_name, "ict-cap-ID_*_test-to-mf_ROM-ID_012_test_*-masked-laplacian", 10)

    def cyc_mf(field):
        return (n_sen * mean_logs(cyc_sen, field) + n_rom * mean_logs(cyc_rom, field)) / (n_sen + n_rom)

    def row(mf, ict):
        return {"MF": str(mf * SCALE), "ICT": str(ict * SCALE)}

    out = {}
    for block, field in (("self", "mse"), ("self_in", "in"), ("self_out", "out"), ("lap_self", "lap")):
        out[block] = row(mean_logs(self_mf, field, weighted=True), self_ict[field])
    for block, field in (("cyc", "mse"), ("cyc_in", "in"), ("cyc_out", "out")):
        out[block] = row(cyc_mf(field), mean_logs(cyc_ict, field))
    out["lap_cyc"] = row(cyc_mf("lap"), mean_logs(cyc_ict, "lap", weighted=True))
    out["crossgt"] = {"ICT": str(read_cif_mean(ckpt_name, "Cross-identity target GT MSE") * SCALE)}
    out["route"] = {"ICT": str(read_cif_mean(ckpt_name, "Route-consistency MSE") * SCALE)}
    return out


CSS = fig_mod.CSS.replace("background:#111; color:#eee;", "background:#fff; color:#111;") \
                 .replace("nav { position:sticky; top:0; background:#111;", "nav { position:sticky; top:0; background:#fff;") + """
.metrics { margin-bottom:32px; }
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
"""


def main():
    metrics = gallery.load_tsv_metrics()
    metrics["PDB"] = metrics_from_eval(CKPT_NAME)
    gallery.GC_DIRS["PDB"] = f"eval_CBD_gc/{CKPT_NAME}-{GC_SUFFIX}"

    parts = [f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>comparison: NC / NFR / NFS / PDB (no col-norm)</title>
<style>{CSS}</style></head><body>
<h1>comparison baselines (NC / NFR / NFS / PDB): self / cross / cyclic retargeting</h1>
<p class="legend"><b>PDB on this page = the --no_col_norm model</b>:
ckpts_CBD/{CKPT_NAME}/model_best.pth (epoch 700, use_data2, relu, no --align_latent).</p>
<div class="metrics" id="metrics">
<h2>Quantitative metrics</h2>
<p class="legend">Values &times;1e-4 (table value &times; 1e-4 = actual MSE). MF = mf_SEN/mf_ROM weighted by
their number of frames (n=6309/11649). NC / NFR / NFS: report/quant_sheet_eve-s05_full.tsv (tag=comparison rows).
PDB: aggregated the same way from this model's eval_CBD / eval_cyc / eval_CBD_cif / eval_CBD_gc folders.</p>
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
                 'retargeting, columns = NC / NFR / NFS / PDB (the --no_col_norm model). Diff-heatmap colors (YlOrRd) '
                 'show where each output deforms the mesh: every cell is normalized on its own, against that '
                 'row&#39;s diff_base (source template for self/cyclic, target template for cross). An NC output that '
                 'leaves the camera view is centered and rescaled on its own; the ratio under the cell is its drawn '
                 'size relative to the other cells (heatmap colors unchanged).</p>')
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
