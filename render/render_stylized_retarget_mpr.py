"""
Stage 2 of the stylized cross retargeting comparison: renders the cached
predictions (preds_cache.npz, written by render_stylized_retarget_comparison.py)
with matplotrender and writes the gallery HTML
vis_CBD/vis_comparison/vis_comparison_stylize_retarget.html.

Each PNG has panels [source expression, target, NC, NFS, PDB]. To show the whole
head, the target and the model outputs share one normalization (center and
scale of the target neutral), and the source uses its own neutral.

matplotrender does `from utils import ...`, which clashes with the repo's utils
package, so run it in isolated mode from outside the repo root:
    cd _tmp && python -I ../render/render_stylized_retarget_mpr.py
"""
import base64
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402
import matplotrender as mpr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / 'vis_CBD' / 'vis_comparison'
IMG_ROOT = OUT_ROOT / 'stylize_retarget'
CACHE_PATH = IMG_ROOT / 'preds_cache.npz'
HTML_PATH = OUT_ROOT / 'vis_comparison_stylize_retarget.html'

VIEWS = [('front', [0, 0, 0]), ('side', [0, 90, 0])]
FIT = 1.8  # largest neutral extent maps to this width in the [-1, 1] view

SRC_LABEL = {
    'mf': 'mf test id 12 (mf_ROM)',
    'ict': 'ict m00 (20240325_MySlate_924)',
}


def fit_transform(v):
    """Center/scale from a neutral mesh so it fits inside the [-1, 1] view."""
    center = (v.max(0) + v.min(0)) * 0.5
    scale = FIT / (v.max(0) - v.min(0)).max()
    return lambda x: (x - center) * scale


def main():
    c = np.load(CACHE_PATH)
    targets, sources, models = list(c['targets']), list(c['sources']), list(c['models'])
    for tname in targets:
        tv, tf = c[f'tgt/{tname}/v'], c[f'tgt/{tname}/f']
        tgt_fit = fit_transform(tv)
        for sname in sources:
            sf, frames = c[f'src/{sname}/f'], c[f'src/{sname}/frames']
            src_fit = fit_transform(c[f'src/{sname}/v'])
            d = IMG_ROOT / tname / sname
            d.mkdir(parents=True, exist_ok=True)
            for k, fidx in enumerate(c[f'src/{sname}/idx']):
                panels = [src_fit(frames[k]), tgt_fit(tv)] + \
                         [tgt_fit(c[f'pred/{m}/{tname}/{sname}'][k]) for m in models]
                faces = [sf, tf] + [tf] * len(models)
                for view, rot in VIEWS:
                    mpr.plot_mesh_image(
                        panels, faces, rot_list=[rot] * len(panels), size=5, norm=False,
                        mode='shade', savedir=str(d), name=f'frame_{fidx:06d}_{view}', save=True,
                    )
                print(f'  saved {tname} {sname} frame {fidx}', flush=True)
    write_html(targets, sources, models, c)
    print('wrote', HTML_PATH)


def write_html(targets, sources, models, c):
    def img(p):
        if not p.exists():
            return f'<div class="missing">missing: {p.name}</div>'
        b64 = base64.b64encode(p.read_bytes()).decode('ascii')
        return f'<img src="data:image/png;base64,{b64}" loading="lazy">'

    cols = ['source expression', 'target (aligned)'] + models
    parts = ['<!doctype html><html><head><meta charset="utf-8">'
             '<title>stylized cross retargeting</title>'
             '<style>body{font-family:-apple-system,sans-serif;margin:24px;color:#111}'
             'h2{border-bottom:1px solid #ddd;padding-bottom:6px}h3{margin-top:24px}'
             '.row{display:flex;gap:8px;align-items:flex-start;margin-bottom:12px;flex-wrap:wrap}'
             '.cell{display:flex;flex-direction:column;align-items:center;font-size:12px}'
             '.cell img{max-width:300px;border:1px solid #ddd}.missing{color:#999;font-size:11px;padding:8px}'
             '</style></head><body>',
             '<h1>stylized mesh cross retargeting (' + ' / '.join(models) + ')</h1>',
             '<p>columns: ' + ' | '.join(cols) + '. front and side per frame.</p>']
    for sname in sources:
        parts.append(f'<h2>source: {SRC_LABEL.get(sname, sname)}</h2>')
        for tname in targets:
            parts.append(f'<h3>target: {tname}</h3>')
            for fidx in c[f'src/{sname}/idx']:
                parts.append(f'<div>frame {fidx}</div>')
                for view, _ in VIEWS:
                    p = IMG_ROOT / tname / sname / f'frame_{fidx:06d}_{view}.png'
                    parts.append(f'<div class="row"><div class="cell">{view}</div>'
                                 f'<div class="cell">{img(p)}</div></div>')
    parts.append('</body></html>')
    HTML_PATH.write_text('\n'.join(parts))


if __name__ == '__main__':
    main()
