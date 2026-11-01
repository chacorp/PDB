"""
render_design_study_figure.py with the columns replaced by the 4 ckpts_CBD9 models
(num_cage_v 512 / 256 / 128 / 64, best checkpoints).

Same scenarios / frames / per-row joint color scale as render_design_study_figure.py.
Output: vis_CBD/vis_ckpts_CBD9/figure/

align_latent must be set via override: eval_CBD_cyc.Options()'s --align_latent
default (False) overrides the train_opts.yml value (True).

Run from repo root:
    python render/render_ckpts_CBD9_figure.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import render_design_study_figure as base  # noqa: E402

CBD9_MODELS = [
    ("cage_v512", "512", "./ckpts_CBD9/2026-10-27-03-49-38-NGBCv5"),
    ("cage_v256", "256", "./ckpts_CBD9/2026-10-27-09-48-28-NGBCv5"),
    ("cage_v128", "128", "./ckpts_CBD9/2026-10-27-15-38-55-NGBCv5"),
    ("cage_v64", "64", "./ckpts_CBD9/2026-10-30-00-43-11-NGBCv5"),
]

base.ds_mod.MODELS[:] = [
    (label, ckpt, {"align_latent": True, "last_activation": "relu"})
    for _key, label, ckpt in CBD9_MODELS
]
base.COLUMN_MODEL.clear()
for key, label, _ckpt in CBD9_MODELS:
    base.COLUMN_MODEL[key] = label
base.SEPARATE_SCALE_COLS = set()

base.OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_ckpts_CBD9" / "figure"

if __name__ == "__main__":
    base.main()
