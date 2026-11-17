"""
Quick test: Ours-dagger (ckpts_CBD/2026-09-14-22-59-50, trained WITH col-norm) evaluated
with col-norm turned off at inference only (no_col_norm=True, same weights).

Same scenarios / frames as render_design_study_figure.py, but only 2 columns:
    ours              -- as trained
    ours_infer_nocn   -- same checkpoint, no_col_norm=True at inference

The second column loads the checkpoint through a symlink with a different name
(_tmp/ours_infer_nocn/...-infer_nocn) so that get_or_run's verts cache
(eval_cyc/<ckpt name>/...) does not reuse the col-norm results of the first column.

Also writes per-frame full-vertex MSE (x1e4, unmasked; self and cyclic vs. the source
GT frame) to <OUT_ROOT>/mse.json.

Run from repo root:
    python render/render_design_study_figure_ours_nocn.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "render"))

import render_design_study_figure as base  # noqa: E402

NOCN_LABEL = "ours (inference w/o col-norm)"
NOCN_CKPT = "./_tmp/ours_infer_nocn/2026-09-14-22-59-50-NGBCv5-infer_nocn"

base.ds_mod.MODELS.append((NOCN_LABEL, NOCN_CKPT, {"no_col_norm": True}))
base.COLUMN_MODEL.clear()
base.COLUMN_MODEL["ours"] = "relu (chosen)"
base.COLUMN_MODEL["ours_infer_nocn"] = NOCN_LABEL

base.OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_design_study" / "figure_ours_infer_nocn"

MSE = {}  # scenario -> frame -> col -> {"self": .., "cyclic": ..}
_orig_run_row = base.run_row
_current = {}


def run_row_with_mse(row_name, variant_Vs, F, diff_base, out_dir):
    if row_name in ("self", "cyclic"):
        gt = np.asarray(_current["gt_v"])
        d = MSE.setdefault(_current["scenario"], {}).setdefault(_current["frame"], {})
        for ck, V in variant_Vs.items():
            d.setdefault(ck, {})[row_name] = float(((np.asarray(V) - gt) ** 2).sum(-1).mean() * 1e4)
    return _orig_run_row(row_name, variant_Vs, F, diff_base, out_dir)


base.run_row = run_row_with_mse

_orig_get_raw = base.ds_mod.get_raw_vertices


def get_raw_tracking(data_name, id_num, exp_num, frame_idx):
    v = _orig_get_raw(data_name, id_num, exp_num, frame_idx)
    _current["gt_v"] = v
    _current["frame"] = frame_idx
    return v


if __name__ == "__main__":
    ckpt_by_col = base.resolve_column_ckpts()
    pipelines = base.load_pipelines(ckpt_by_col)
    for scenario in base.ds_mod.SCENARIOS:
        _current["scenario"] = scenario[0]
        # track the GT frame only inside the per-frame render loop
        base.ds_mod.get_raw_vertices = get_raw_tracking
        base.run_scenario(ckpt_by_col, pipelines, scenario)
        base.ds_mod.get_raw_vertices = _orig_get_raw
    base.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json.dump(MSE, open(base.OUT_ROOT / "mse.json", "w"), indent=1)
    print("\nALL DONE:", base.OUT_ROOT)
