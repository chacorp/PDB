"""
Stage 2 of the stylized retargeting comparison: renders the cached predictions
(preds_cache.npz, written by render_stylized_retarget_comparison.py) as per-cell
PNGs with matplotrender and writes the gallery HTML
vis_CBD/vis_comparison/vis_comparison_stylize_retarget.html.

Layout per (source -> target, frame), same as vis_ablation_mask.html /
vis_comparison.html's qualitative figure:
  left  : Source Neutral / Source Expression (diff) / Target Neutral
  right : rows = self / cross / cyclic retargeting, cols = NC / NFS / PDB
Diff heatmaps (YlOrRd) are normalized per row against that row's diff_base
(source neutral for self/cyclic, target neutral for cross). NC is normalized on
its own scale; NFS/PDB share one scale per row. The source expression uses its
own scale against the source neutral. Front view only.

To show the whole head, meshes on the target identity (target neutral, cross)
share the target neutral's center/scale, and meshes on the source identity
(source neutral/expression, self, cyclic) share the source neutral's.

matplotrender does `from utils import ...`, which clashes with the repo's utils
package, so run it in isolated mode from outside the repo root:
    cd _tmp && python -I ../render/render_stylized_retarget_mpr.py [report ...]
report: keys of REPORTS (default: all). 'main' = NC / NFS / NFR / PDB
(vis_comparison_stylize_retarget.html), 'continue' = PDB vs the continued PDB
run (continue_figure.html), 'ttt' = PDB before vs after test-time training on
bowen (ttt_figure.html).
"""
import sys
import base64
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotrender import plot_mesh_gouraud, calc_face_norm  # noqa: E402
from utils import fix_triangle_widning  # noqa: E402  (matplotrender's site-packages utils.py)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / 'vis_CBD' / 'vis_comparison'
IMG_ROOT = OUT_ROOT / 'stylize_retarget'
CACHE_PATH = IMG_ROOT / 'preds_cache.npz'

# report -> models (columns), cell PNG root, HTML path, title, extra legend.
# Each report has its own cell PNGs since the shared diff scale depends on the
# models shown together.
REPORTS = {
    'main': dict(
        models=['NC', 'NFS', 'NFR', 'PDB'], fig_root=IMG_ROOT / 'figure',
        html=OUT_ROOT / 'vis_comparison_stylize_retarget.html',
        title='stylized mesh retargeting (NC / NFS / NFR / PDB): self / cross / cyclic', note=''),
    'continue': dict(
        models=['PDB', 'PDB-cont', 'PDBplus'], fig_root=IMG_ROOT / 'figure_continue',
        html=OUT_ROOT / 'continue_figure.html',
        title='stylized mesh retargeting: PDB vs continued PDB vs PDBplus',
        note='PDB = ckpts_CBD8/2026-09-16-02-19-03-NGBCv5/model_best.pth (epoch 920, use_data2). '
             'PDB-cont = ckpts_CBD8/2026-10-07-00-58-27-NGBCv5/model_best.pth (epoch 1220, '
             'continued from epoch 1000 with use_data3 via train_continue.sh; run stopped at epoch 1221). '
             'PDBplus = ckpts_CBD/2026-10-06-01-00-32-PDBplusv6/model_best.pth (epoch 430, version 6, '
             'use_data8 via train_PDBplus.sh; run stopped at epoch 463). '),
    'ttt': dict(
        models=['PDB', 'PDB-TTT-bowen'], targets=['bowen'], fig_root=IMG_ROOT / 'figure_ttt',
        html=OUT_ROOT / 'ttt_figure.html',
        title='stylized mesh retargeting: PDB before vs after test-time training (bowen)',
        note='PDB = ckpts_CBD8/2026-09-16-02-19-03-NGBCv5/model_best.pth (epoch 920, before TTT). '
             'PDB-TTT-bowen = ttt_CBD/2026-10-09-00-48-16-NGBCv5-TTT-test-bowen-aligned/model_010.pth '
             '(after 10 epochs of ttt_CBD.py on test-bowen-aligned, from the PDB above; use_data2, lr 1e-4, '
             'ttt-self + ttt-cross, lambda 1). '),
}

SIZE = 8
FIT = 1.8  # largest neutral extent maps to this width in the [-1, 1] view
LIGHT_DIR = np.array([0, 0, 1])
OWN_SCALE = {'NC'}  # diverges, so sharing a scale would wash out the others

REF_ROWS = [
    ('source_neutral', 'Source Neutral Mesh'),
    ('source_expression', 'Source Expression Mesh'),
    ('target_neutral', 'Target Neutral Mesh'),
]
FIG_ROWS = [
    ('self', 'Self retargeting'),
    ('cross', 'Cross retargeting'),
    ('cyclic', 'Cyclic retargeting'),
]
SRC_LABEL = {
    'mf': 'mf test id 12 (mf_ROM)',
    'ict': 'ict m00 (20240325_MySlate_924)',
}


def fit_transform(v):
    """Center/scale from a neutral mesh so it fits inside the [-1, 1] view."""
    center = (v.max(0) + v.min(0)) * 0.5
    scale = FIT / (v.max(0) - v.min(0)).max()
    return lambda x: (x - center) * scale


def joint_colors(Vs, F, diff_base):
    """Heatmap colors for a list of meshes sharing one min/max (diff vs diff_base)."""
    Cdist = np.linalg.norm(np.abs(np.stack(Vs) - diff_base), axis=-1)
    Cnorm = (Cdist - Cdist.min()) / (Cdist.max() - Cdist.min() + 1e-12)
    colors = []
    for k, V in enumerate(Vs):
        shading = np.clip(calc_face_norm(V, F, mode='v') @ LIGHT_DIR, 0, 1)
        vc = shading[..., None].repeat(3, axis=-1) * 0.6 + 0.3
        Sc = plt.get_cmap('YlOrRd')(Cnorm[k])[..., :3]
        mask = Cnorm[k][:, None]
        colors.append(vc * (1 - mask) + Sc * mask)
    return colors


def save_mesh(V, F, out_path, color=None):
    plot_mesh_gouraud(
        [V], [F], Cs=None if color is None else [color], is_color=color is not None,
        backface_culling=True, rot_list=[[0, 0, 0]], size=SIZE, mode='shade', bg_black=False,
        logdir=str(out_path.parent), name=out_path.stem, save=True, show=False,
    )


def render_row(row, vs_by_model, F, diff_base, out_dir):
    shared = [m for m in vs_by_model if m not in OWN_SCALE]
    colors = dict(zip(shared, joint_colors([vs_by_model[m] for m in shared], F, diff_base))) if shared else {}
    for m in vs_by_model:
        if m in OWN_SCALE:
            colors[m] = joint_colors([vs_by_model[m]], F, diff_base)[0]
    for m, V in vs_by_model.items():
        save_mesh(V, F, out_dir / f'{row}_{m}.png', colors[m])


def main():
    c = np.load(CACHE_PATH)
    for key in sys.argv[1:] or list(REPORTS):
        render_report(c, **REPORTS[key])


def render_report(c, models, fig_root, html, title, note, targets=None):
    targets, sources = targets or list(c['targets']), list(c['sources'])
    for sname in sources:
        sv, sf, frames = c[f'src/{sname}/v'], c[f'src/{sname}/f'], c[f'src/{sname}/frames']
        # plot_mesh_gouraud culls by face normal but doesn't fix inverted winding
        # (the stylized meshes have it), so orient faces once per identity
        sf = fix_triangle_widning(sv, sf)
        src_fit = fit_transform(sv)
        sv_n = src_fit(sv)
        for tname in targets:
            tv, tf = c[f'tgt/{tname}/v'], c[f'tgt/{tname}/f']
            tf = fix_triangle_widning(tv, tf)
            tgt_fit = fit_transform(tv)
            tv_n = tgt_fit(tv)
            for k, fidx in enumerate(c[f'src/{sname}/idx']):
                d = fig_root / f'{sname}_to_{tname}' / f'frame_{fidx:06d}'
                d.mkdir(parents=True, exist_ok=True)
                expr = src_fit(frames[k])
                save_mesh(sv_n, sf, d / 'source_neutral.png')
                save_mesh(tv_n, tf, d / 'target_neutral.png')
                save_mesh(expr, sf, d / 'source_expression.png', joint_colors([expr], sf, sv_n)[0])
                rows = [
                    ('self', {m: src_fit(c[f'self/{m}/{sname}'][k]) for m in models}, sf, sv_n),
                    ('cross', {m: tgt_fit(c[f'pred/{m}/{tname}/{sname}'][k]) for m in models}, tf, tv_n),
                    ('cyclic', {m: src_fit(c[f'cyc/{m}/{tname}/{sname}'][k]) for m in models}, sf, sv_n),
                ]
                for row, vs, F, base in rows:
                    render_row(row, vs, F, base, d)
                print(f'  saved {sname} -> {tname} frame {fidx}', flush=True)
    write_html(targets, sources, models, c, fig_root, html, title, note)
    print('wrote', html)


# ---------------------------------------------------------------------------
# HTML: same table layout / CSS as build_ablation_gallery_mask.py's figure
# ---------------------------------------------------------------------------
def img_cell(path):
    if not path.exists():
        return f'<td class="missing">missing<br>{path.name}</td>'
    b64 = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<td class="imgcell"><div class="clip"><img src="data:image/png;base64,{b64}" loading="lazy"></div></td>'


def build_ref_table(d):
    out = ['<table class="grid ref-grid">', '<tbody>']
    for key, label in REF_ROWS:
        out.append(f'<tr><th class="rowhead">{label}</th>{img_cell(d / f"{key}.png")}</tr>')
    out.append('</tbody></table>')
    return '\n'.join(out)


def build_result_table(d, models):
    out = ['<table class="grid">', '<tbody>']
    for row, label in FIG_ROWS:
        out.append(f'<tr><th class="rowhead">{label}</th>')
        out.extend(img_cell(d / f'{row}_{m}.png') for m in models)
        out.append('</tr>')
    out.append('</tbody>')
    out.append('<tfoot><tr><th class="rowhead-col"></th>')
    out.extend(f'<th>{m}</th>' for m in models)
    out.append('</tr></tfoot></table>')
    return '\n'.join(out)


HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body { font-family: -apple-system, sans-serif; background:#fff; color:#111; margin:0; padding:24px; }
  h1 { font-size:20px; }
  h2 { font-size:16px; margin-top:40px; border-bottom:1px solid #ddd; padding-bottom:6px; }
  h3 { font-size:15px; color:#333; margin-top:28px; }
  .legend { font-size:12px; color:#666; }
  nav { position:sticky; top:0; background:#fff; padding:8px 0; border-bottom:1px solid #ddd; z-index:10; }
  nav a { color:#06c; margin-right:16px; font-size:12px; text-decoration:none; }
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
</style>
</head><body>
"""


def write_html(targets, sources, models, c, fig_root, html, title, note):
    parts = [HTML_HEAD.replace('{title}', title),
             f'<h1>{title}</h1>',
             f'<p class="legend">{note}Targets: aligned stylized meshes under test-mesh/. '
             'Left: Source Neutral (plain) / Source Expression (diff vs source neutral, own scale) / '
             'Target Neutral (plain). Right: self = source expression reconstructed on the source neutral, '
             'cross = source expression on the target neutral, cyclic = source &rarr; target &rarr; source. '
             'Diff heatmaps (YlOrRd) are normalized per row against that row\'s diff_base '
             '(source neutral for self/cyclic, target neutral for cross); NC (when shown) on its own scale, '
             'the others share one. Front view, matplotrender plot_mesh_gouraud. '
             'Generated by render/render_stylized_retarget_comparison.py (inference) + '
             'render/render_stylized_retarget_mpr.py.</p>',
             '<nav>']
    scen = [(s, t) for s in sources for t in targets]
    parts += [f'<a href="#{s}_to_{t}">{s}&rarr;{t}</a>' for s, t in scen]
    parts.append('</nav>')
    parts.append('<section class="figure-section">')
    for s, t in scen:
        parts.append(f'<h2 id="{s}_to_{t}">{s} &rarr; {t} &mdash; source: {SRC_LABEL.get(s, s)}, '
                     f'target: {c[f"tgt/{t}/file"]}</h2>')
        for fidx in c[f'src/{s}/idx']:
            d = fig_root / f'{s}_to_{t}' / f'frame_{fidx:06d}'
            parts.append(f'<h3>frame {fidx}</h3>')
            parts.append('<div class="figure-row">')
            parts.append(build_ref_table(d))
            parts.append(build_result_table(d, models))
            parts.append('</div>')
    parts.append('</section></body></html>')
    html.write_text('\n'.join(parts))


if __name__ == '__main__':
    main()
