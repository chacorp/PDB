"""
render_design_study_figure.py + one extra column: the ckpts_CBD8 w/o col-norm model
(train_CBD_align_latent.sh, --no_col_norm --align_latent).

Same scenarios / frames / per-row joint color scale as render_design_study_figure.py,
but with 9 columns instead of 8. Since the per-row color scale is shared across
columns, the existing 8 variants are re-rendered too (verts caches are reused), into
a separate output dir so the original figure/ PNGs stay untouched:
    vis_CBD/vis_design_study/figure_cbd8/

align_latent must be set via override: eval_CBD_cyc.Options()'s --align_latent
default (False) overrides the train_opts.yml value (True), which would build the
model with hid_dim=256 instead of the checkpoint's 128. no_col_norm is not an
argparse option, so the train_opts.yml value (True) is kept as-is.

Run from repo root:
    python render/render_design_study_figure_cbd8.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import render_design_study_figure as base  # noqa: E402

CBD8_LABEL = "w/o col-norm"
CBD8_CKPT = "./ckpts_CBD8/2026-10-27-00-55-40-NGBCv5"

# w/o col-norm without --align_latent: the argparse default (False) already matches the checkpoint
CBD8_NOALIGN_LABEL = "w/o col-norm (no align)"
CBD8_NOALIGN_CKPT = "./ckpts_CBD8/2026-10-30-06-56-58-NGBCv5"

base.ds_mod.MODELS.append((CBD8_LABEL, CBD8_CKPT, {"align_latent": True}))
base.ds_mod.MODELS.append((CBD8_NOALIGN_LABEL, CBD8_NOALIGN_CKPT, {}))

# insert the new columns right after Ours-dagger
_cols = list(base.COLUMN_MODEL.items())
base.COLUMN_MODEL.clear()
for k, v in _cols:
    base.COLUMN_MODEL[k] = v
    if k == "ours":
        base.COLUMN_MODEL["wo_col_norm"] = CBD8_LABEL
        base.COLUMN_MODEL["wo_col_norm_noalign"] = CBD8_NOALIGN_LABEL

base.OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_design_study" / "figure_cbd8"

if __name__ == "__main__":
    base.main()
