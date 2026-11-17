"""
Quantitative section of ttt_figure.html: PDB vs PDB-TTT-<target> target self
retargeting (stylized target T -> T) and cyclic (source -> stylized target -> source)
MSE / MSE-in / MSE-out, from _tmp/ttt_eval/metrics_ttt.json (eval_ttt_stylized.py).

Rows = stylized targets (+ mean over targets, + relative change of the mean),
columns = self-retarget {PDB, PDB-TTT} + cyclic x {MF, ict m05} x {PDB, PDB-TTT}.
MF = mf_SEN / mf_ROM (mf test id 12) weighted by their number of frames.
Table style follows build_ablation_gallery_design_study_cbd8.py; values x1e-4.

metrics_section() is also used by render_stylized_retarget_mpr.py when it writes
ttt_figure.html. Run from repo root to splice the section into the existing page
without re-rendering:
    python render/build_ttt_metrics.py
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = REPO_ROOT / '_tmp' / 'ttt_eval' / 'metrics_ttt.json'
HTML_PATH = REPO_ROOT / 'vis_CBD' / 'vis_comparison' / 'ttt_figure.html'
SCALE = 1e4
MODELS = [('PDB', 'PDB'), ('TTT', 'PDB-TTT')]  # (key, column label)
SOURCES = [('MF', 'MF'), ('ict_m05', 'ict m05')]
MODES = [('self', 'self-retarget'), ('cyc', 'cyclic')]
# (mode, source) column groups; target self retargeting has no source
GROUPS = [('self', 'target')] + [('cyc', s) for s, _ in SOURCES]
SRC_LABEL = dict(SOURCES, target='stylized target')
METRICS = [('MSE', 'MSE', 'whole mesh'),
           ('MSE-in', 'MSE-in', 'masked inner region, vs GT'),
           ('MSE-out', 'MSE-out', 'masked outer region, vs neutral')]
START, END = '<!-- ttt-metrics:start -->', '<!-- ttt-metrics:end -->'

CSS = """<style>
  .metrics { margin-bottom:32px; }
  .metrics .unit { font-size:12px; color:#777; font-weight:normal; }
  .mblock { margin-top:18px; }
  .mlabel { font-size:12px; font-weight:600; color:#444; margin:0 0 6px; }
  .mlabel span { color:#888; font-weight:normal; }
  table.mtable { border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
  table.mtable th, table.mtable td { padding:6px 12px; text-align:right; white-space:nowrap; }
  table.mtable thead tr.grp th { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#333; border-bottom:1px solid #ddd; padding-bottom:4px; font-weight:600; }
  table.mtable thead tr.grp th.self { color:#1565c0; }
  table.mtable thead tr.grp th.cyc { color:#7b3fa0; }
  table.mtable thead tr.src th { font-size:11px; color:#555; border-bottom:1px solid #eee; font-weight:600; }
  table.mtable thead tr.cols th { font-size:10.5px; text-transform:uppercase; letter-spacing:.03em; color:#888; border-bottom:2px solid #ddd; padding-bottom:6px; font-weight:500; }
  table.mtable th.rowhead { text-align:left; color:#111; font-weight:500; }
  table.mtable tbody tr td, table.mtable tbody tr th.rowhead { border-bottom:1px solid #eee; }
  table.mtable tbody tr.mean td, table.mtable tbody tr.mean th { border-top:2px solid #ddd; font-weight:700; }
  table.mtable tbody tr.delta td, table.mtable tbody tr.delta th { color:#555; font-size:12px; border-bottom:none; }
  table.mtable td.cell { color:#0c0c0c; font-weight:500; }
  table.mtable td.cell.light-text { color:#f5f4ef; }
  table.mtable td.cell.best { font-weight:800; }
  table.mtable td.better { color:#1b7f3b; }
  table.mtable td.worse { color:#c0392b; }
  table.mtable td.gapcol, table.mtable th.gapcol { border:none; padding:0; width:14px; }
</style>"""


def _lerp(a, b, t):
    return a + (b - a) * t


def _seq_color(t):
    """same light -> deep blue ramp as build_ablation_gallery_design_study_cbd8.py"""
    lo, hi = (0xf3, 0xf8, 0xfe), (0x0d, 0x36, 0x6b)
    t = max(0.0, min(1.0, t))
    r, g, b = (round(_lerp(lo[i], hi[i], t)) for i in range(3))
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return f'rgb({r},{g},{b})', lum < 0.55


def load_values():
    """-> targets, val[(mode, source, model, metric)][target] (x1e-4)"""
    raw = json.loads(METRICS_PATH.read_text())
    targets = [k[len('PDB-TTT-'):] for k in raw if k.startswith('PDB-TTT-')]

    def src_val(res, mode_key, metric):
        def one(s):
            return res[s][mode_key][metric]
        n_sen, n_rom = res['mf_SEN']['n'], res['mf_ROM']['n']
        mf = (n_sen * one('mf_SEN') + n_rom * one('mf_ROM')) / (n_sen + n_rom)
        return {'MF': mf, 'ict_m05': one('ict_m05')}

    val = {}
    for t in targets:
        for model, res in (('PDB', raw['PDB']), ('TTT', raw[f'PDB-TTT-{t}'])):
            for metric, _, _ in METRICS:
                val.setdefault(('self', 'target', model, metric), {})[t] = res['tself'][t][metric] * SCALE
                for s, v in src_val(res, t, metric).items():
                    val.setdefault(('cyc', s, model, metric), {})[t] = v * SCALE
    return targets, val


def build_table(targets, val, metric):
    groups = GROUPS
    out = ['<table class="mtable"><thead><tr class="grp"><th></th>']
    for i, (mode, label) in enumerate(MODES):
        n = sum(g[0] == mode for g in groups) * len(MODELS)
        out.append(('<th class="gapcol"></th>' if i else '') + f'<th class="{mode}" colspan="{n}">{label}</th>')
    out.append('</tr><tr class="src"><th></th>')
    for i, g in enumerate(groups):
        out.append(('<th class="gapcol"></th>' if i and g[0] != groups[i - 1][0] else '')
                   + f'<th colspan="{len(MODELS)}">{SRC_LABEL[g[1]]}</th>')
    out.append('</tr><tr class="cols"><th class="rowhead">target</th>')
    for i, g in enumerate(groups):
        out.append(('<th class="gapcol"></th>' if i and g[0] != groups[i - 1][0] else '')
                   + ''.join(f'<th>{ml}</th>' for _, ml in MODELS))
    out.append('</tr></thead><tbody>')

    rng = {}
    for g in groups:
        vs = [val[(*g, m, metric)][t] for m, _ in MODELS for t in targets]
        rng[g] = (min(vs), max(vs))
    mean = {(*g, m): sum(val[(*g, m, metric)].values()) / len(targets) for g in groups for m, _ in MODELS}

    def row_cells(get):
        cells = []
        for i, g in enumerate(groups):
            if i and g[0] != groups[i - 1][0]:
                cells.append('<td class="gapcol"></td>')
            pair = {m: get(g, m) for m, _ in MODELS}
            lo, hi = rng[g]
            for m, _ in MODELS:
                v = pair[m]
                css, light = _seq_color((v - lo) / (hi - lo) if hi > lo else 0.0)
                cls = 'cell' + (' best' if v == min(pair.values()) else '') + (' light-text' if light else '')
                cells.append(f'<td class="{cls}" style="background:{css}">{v:.4f}</td>')
        return ''.join(cells)

    for t in targets:
        out.append(f'<tr><th class="rowhead">{t}</th>'
                   + row_cells(lambda g, m: val[(*g, m, metric)][t]) + '</tr>')
    out.append('<tr class="mean"><th class="rowhead">mean</th>'
               + row_cells(lambda g, m: mean[(*g, m)]) + '</tr>')
    # relative change of the target-mean, TTT vs PDB
    delta = ['<tr class="delta"><th class="rowhead">TTT vs PDB</th>']
    for i, g in enumerate(groups):
        if i and g[0] != groups[i - 1][0]:
            delta.append('<td class="gapcol"></td>')
        d = (mean[(*g, 'TTT')] - mean[(*g, 'PDB')]) / mean[(*g, 'PDB')] * 100
        delta.append(f'<td></td><td class="{"better" if d < 0 else "worse"}">{d:+.1f}%</td>')
    out.append(''.join(delta) + '</tr></tbody></table>')
    return ''.join(out)


def metrics_section():
    targets, val = load_values()
    raw = json.loads(METRICS_PATH.read_text())['PDB']
    parts = [START, CSS, '<section id="metrics" class="metrics">',
             '<h2 id="metrics-h">Self-retarget vs cyclic MSE, PDB vs PDB-TTT '
             '<span class="unit">&middot; &times;10&#8315;&#8308;</span></h2>',
             '<p class="legend">self = target self retargeting (T &rarr; T): the stylized target neutral '
             'reconstructed on itself, the ttt-self term of NGBC.ttt_loss (key_weight(T) &middot; key_s(T)), '
             'compared with the target neutral (mask r0=1.25, r1=2.5). '
             'cyclic = source &rarr; stylized target &rarr; source '
             '(retarget_animation, as eval_CBD_cyc.py), compared with the source expression. '
             'PDB-TTT is the model adapted to that row\'s target (model_000.pth). '
             f'Sources: MF = mf test id 12, mf_SEN (n={raw["mf_SEN"]["n"]}) and mf_ROM (n={raw["mf_ROM"]["n"]}) '
             'weighted by their number of frames; ict m05 = ICT capture id 5 with the 20240318_MySlate_922 motion '
             f'(n={raw["ict_m05"]["n"]}). MSE-in compares the masked inner region with the GT, MSE-out the masked '
             'outer region with the neutral (plateau_hat_points defaults r0=1.25, r1=2.5). '
             'cyclic values are frame means. '
             'Bold = lower of each PDB / PDB-TTT pair; cell shading shares one scale per (mode, source) pair. '
             'Bottom row: change of the target mean, PDB-TTT vs PDB. Generated by eval_ttt_stylized.py + '
             'render/build_ttt_metrics.py.</p>']
    for metric, label, desc in METRICS:
        parts.append(f'<div class="mblock"><h3 class="mlabel">{label} <span>&middot; {desc}</span></h3>'
                     f'{build_table(targets, val, metric)}</div>')
    parts += ['</section>', END]
    return '\n'.join(parts)


def main():
    html = HTML_PATH.read_text()
    html = re.sub(re.escape(START) + '.*?' + re.escape(END) + r'\n?', '', html, flags=re.S)
    i = html.index('<nav>')
    HTML_PATH.write_text(html[:i] + metrics_section() + '\n' + html[i:])
    print('wrote', HTML_PATH)


if __name__ == '__main__':
    main()
