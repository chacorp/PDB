"""
Does a thresholded (sparse) key_weight keep the retargeting quality?

Model:  ckpts_CBD/2026-04-27-15-36-49-NGBCv5/model_best.pth (out_type=1: output = w @ v)
Source: ict-cap ID_000, 20240325_MySlate_924 (all 1147 frames)
Target: mf_ROM test ID_012

w = key_weight (vertices, 512): per-vertex blend ratios of the control points, from the neutral
v = key_d (512, 3): control point positions, from the source expression, one set per frame
  self   = w_src @ v
  cross  = w_tgt @ v
  cyclic = w_src @ v', v' = control points re-estimated from the cross output

Variants: "dense" and, for each threshold t, entries below t set to 0 on BOTH w_src and w_tgt:
  raw    = only zeroed
  renorm = each row divided by its sum (a row left empty keeps its largest weight)

Writes vis_CBD/sparse_weight_test/:
  weights/w_{src,tgt}.npy, weights/w_{src,tgt}_t{t}_{raw,renorm}.npz (scipy.sparse CSR)
  control_points_src.npy   (frames, 512, 3)
  ref/                     neutrals, faces, source expression of FIG_FRAMES
  verts/{variant}/{self,cross,cyclic}/{frame:06d}.npy   FIG_FRAMES only
  metrics.json             frame-mean metrics (raw values) and sparsity statistics

SPARSE_FORM=delta applies only the control point motion to the neutral, and writes to
vis_CBD/sparse_weight_test_delta/ (also control_points_{src,tgt}_rest.npy):
  self   = src neutral + w_src @ (v  - v_src)
  cross  = tgt neutral + w_tgt @ (v  - v_src)
  cyclic = src neutral + w_src @ (v' - v_tgt)
v_src / v_tgt = control points of that neutral, from a zero-deformation input (key_s in forward()).

SPARSE_LIMIT_FRAMES=3 restricts the run to FIG_FRAMES[:3] (smoke test).

    python render/sparse_weight_infer.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "render"))
sys.argv = [sys.argv[0]]

import igl  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from comparison_frames import ICT_FRAMES_M00 as FIG_FRAMES  # noqa: E402
from dataloader_CBD import EvalDataset  # noqa: E402
from eval_CBD import Trainer  # noqa: E402
from utils.exp_utils import plateau_hat_points  # noqa: E402
from utils.remesh_utils import calc_norm_torch  # noqa: E402
from vis_CBD_retarget_fig import Pipeline  # noqa: E402

CKPT = ROOT / "ckpts_CBD" / "2026-04-27-15-36-49-NGBCv5"
FORM = os.environ.get("SPARSE_FORM", "absolute")  # absolute | delta
OUT = ROOT / "vis_CBD" / ("sparse_weight_test_delta" if FORM == "delta" else "sparse_weight_test")
# same model and source / target pair, computed earlier by save_dynamic_6faces_data.py
PREV = ROOT / "vis_CBD" / "2026-04-27-15-36-49-NGBCv5-best"
PREV_SELF = PREV / "ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked" / "verts"
PREV_CROSS = PREV / "ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01-masked" / "verts"
DEVICE = "cuda:0"
THRESHOLDS = [0.001, 0.005, 0.01, 0.02]
MODES = ["renorm", "raw"]
ROWS = ["self", "cross", "cyclic"]


def load_model():
    opts_yaml = yaml.safe_load(open(CKPT / "train_opts.yml"))
    opts_yaml.update({
        "device": DEVICE, "ckpt": str(CKPT), "version": 5, "batch_size": 1,
        "continue_ckpt": False, "seed": 42, "last_activation": "relu",
        "no_pou": False, "align_latent": True, "is_train": False,
    })
    opts = argparse.Namespace(**opts_yaml)
    trainer = Trainer(opts)  # continue_ckpt False -> model_best.pth
    trainer.model.eval()
    return trainer.model, opts


def th(v, f):
    n = igl.per_vertex_normals(np.asarray(v, np.float64), np.asarray(f))
    return (torch.tensor(np.asarray(v)).float()[None].to(DEVICE),
            torch.tensor(n).float()[None].to(DEVICE))


def sparsify(w, t, mode):
    kept = np.where(w >= t, w, 0.0)
    if mode == "renorm":
        empty = np.flatnonzero(kept.sum(1) == 0)
        kept[empty, w[empty].argmax(1)] = w[empty].max(1)
        kept = kept / kept.sum(1, keepdims=True)
    return sp.csr_matrix(kept.astype(np.float32))


def sparsity_stats(w_dense, w_sp, t):
    nnz = np.diff(w_sp.indptr)
    return {
        "nnz_per_vertex_mean": float(nnz.mean()), "nnz_per_vertex_max": int(nnz.max()),
        "density": float(w_sp.nnz / np.prod(w_sp.shape)), "empty_rows": int((nnz == 0).sum()),
        "kept_mass_before_renorm": float(w_dense[w_dense >= t].sum() / len(w_dense)),
        "row_sum_mean": float(np.asarray(w_sp.sum(1)).mean()),
    }


def laplacian(v_np, f_np):
    L = igl.cotmatrix(np.asarray(v_np, np.float64), np.asarray(f_np)).tocoo()
    return torch.sparse_coo_tensor(np.vstack([L.row, L.col]), L.data, L.shape,
                                   dtype=torch.float32, device=DEVICE).coalesce()


class Meter:
    """Frame-mean metrics. MSE / MSE-in / MSE-out / Lap as stack_mse in eval_CBD_cyc.py
    (masks from plateau_hat_points); to_dense_* compare against the dense output."""

    def __init__(self):
        self.total, self.n = {}, {}

    def add(self, key, value):
        self.total[key] = self.total.get(key, 0.0) + float(value)
        self.n[key] = self.n.get(key, 0) + 1

    def means(self):
        return {k: self.total[k] / self.n[k] for k in self.total}


def lap_mse(L, a, b):
    return (torch.sparse.mm(L, a[0]) - torch.sparse.mm(L, b[0])).square().mean()


def measure_gt(meter, p, gt, tmpl, L, inner):
    meter.add("MSE", (p - gt).square().mean())
    meter.add("MSE-in", ((p - gt) * inner).square().mean())
    meter.add("MSE-out", ((p - tmpl) * (1 - inner)).square().mean())
    meter.add("Lap", lap_mse(L, gt * inner, p * inner))


def main():
    limit = int(os.environ.get("SPARSE_LIMIT_FRAMES", 0))
    for sub in ("weights", "ref/source_expression"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    model, opts = load_model()
    pipe = Pipeline(opts)
    src_ds = EvalDataset(data_name="ict-cap", toggle=False, ict_cap_id_num=0, ict_cap_exp_num=1)
    frames = FIG_FRAMES[:limit] if limit else list(range(len(src_ds)))
    fig_frames = set(FIG_FRAMES)

    item0 = src_ds[0]
    src_v_np = item0[1].numpy()
    src_f = item0[4].numpy() if hasattr(item0[4], "numpy") else np.asarray(item0[4])
    tgt_v_np, tgt_f, _ = pipe.get_mesh("mf_ROM", None, 12)
    tgt_v_np, tgt_f = np.asarray(tgt_v_np, np.float64), np.asarray(tgt_f)
    sv, sn = th(src_v_np, src_f)
    tv, tn = th(tgt_v_np, tgt_f)
    tgt_f_th = torch.tensor(tgt_f.astype(np.int64)).to(DEVICE)
    np.save(OUT / "ref" / "src_neutral.npy", src_v_np)
    np.save(OUT / "ref" / "src_faces.npy", src_f)
    np.save(OUT / "ref" / "tgt_neutral.npy", tgt_v_np)
    np.save(OUT / "ref" / "tgt_faces.npy", tgt_f)

    with torch.no_grad():
        w_src = model.predict_coordinate(sv, sn)
        w_tgt = model.predict_coordinate(tv, tn)
    w_src_np, w_tgt_np = w_src[0].cpu().numpy(), w_tgt[0].cpu().numpy()
    np.save(OUT / "weights" / "w_src.npy", w_src_np)
    np.save(OUT / "weights" / "w_tgt.npy", w_tgt_np)

    # sparse weights: inference below uses the files as saved
    variants = {"dense": (w_src, w_tgt)}
    stats = {}
    for t in THRESHOLDS:
        for mode in MODES:
            name = f"t{t}_{mode}"
            ws, wt = sparsify(w_src_np, t, mode), sparsify(w_tgt_np, t, mode)
            sp.save_npz(OUT / "weights" / f"w_src_{name}.npz", ws)
            sp.save_npz(OUT / "weights" / f"w_tgt_{name}.npz", wt)
            stats[name] = {"src": sparsity_stats(w_src_np, ws, t), "tgt": sparsity_stats(w_tgt_np, wt, t)}
            load = lambda p: torch.tensor(sp.load_npz(p).toarray()).float()[None].to(DEVICE)
            variants[name] = (load(OUT / "weights" / f"w_src_{name}.npz"),
                              load(OUT / "weights" / f"w_tgt_{name}.npz"))
    print(json.dumps(stats, indent=1), flush=True)

    src_inner = plateau_hat_points(sv)
    tgt_inner = plateau_hat_points(tv)
    src_L, tgt_L = laplacian(src_v_np, src_f), laplacian(tgt_v_np, tgt_f)
    meters = {v: {r: Meter() for r in ROWS} for v in variants}
    control = np.zeros((len(frames), w_src_np.shape[1], 3), np.float32)
    nan_frames = {v: 0 for v in variants}
    prev_check = {"self": 0.0, "cross": 0.0}
    w_v_check = 0.0
    with torch.no_grad():
        _, rest_src = model.retarget_animation(sv, sn, sv, sn, w_src, sv)
        _, rest_tgt = model.retarget_animation(tv, tn, tv, tn, w_tgt, tv)
    if FORM == "delta":
        np.save(OUT / "control_points_src_rest.npy", rest_src[0].cpu().numpy())
        np.save(OUT / "control_points_tgt_rest.npy", rest_tgt[0].cpu().numpy())

    for fi, f in enumerate(tqdm(frames, ncols=100)):
        gt_np = src_ds[f][0].numpy()
        ev, en = th(gt_np, src_f)
        with torch.no_grad():
            pred_self, key_d = model.retarget_animation(sv, sn, ev, en, w_src, sv)
            pred_cross, key_d_x = model.retarget_animation(sv, sn, ev, en, w_tgt, tv)
            control[fi] = key_d[0].cpu().numpy()
            # output = w @ v, and v does not depend on the target
            w_v_check = max(w_v_check,
                            (w_src @ key_d - pred_self).abs().max().item(),
                            (w_tgt @ key_d - pred_cross).abs().max().item(),
                            (key_d - key_d_x).abs().max().item())
            dense_out = {}
            for name, (ws, wt) in variants.items():
                if FORM == "delta":
                    self_v = sv + ws @ (key_d - rest_src)
                    cross_v = tv + wt @ (key_d - rest_src)
                else:
                    self_v = ws @ key_d
                    cross_v = wt @ key_d
                cross_n = calc_norm_torch(cross_v, tgt_f_th, at="verts")
                if not torch.isfinite(cross_n).all():
                    nan_frames[name] += 1
                    cross_n = torch.nan_to_num(cross_n)
                cyc_v, key_d_cyc = model.retarget_animation(tv, tn, cross_v, cross_n, ws, sv)
                if FORM == "delta":
                    cyc_v = sv + ws @ (key_d_cyc - rest_tgt)
                out = {"self": self_v, "cross": cross_v, "cyclic": cyc_v}
                if name == "dense":
                    dense_out = out
                m = meters[name]
                # difference from the original model output w_tgt @ v (= the absolute dense cross)
                m["cross"].add("to_abs_dense_MSE", (cross_v - pred_cross).square().mean())
                measure_gt(m["self"], self_v, ev, sv, src_L, src_inner)
                measure_gt(m["cyclic"], cyc_v, ev, sv, src_L, src_inner)
                # no GT for cross: difference from the dense output, on the target's own masks
                m["cross"].add("to_dense_MSE", (cross_v - dense_out["cross"]).square().mean())
                m["cross"].add("to_dense_MSE-in",
                               ((cross_v - dense_out["cross"]) * tgt_inner).square().mean())
                m["cross"].add("to_dense_Lap", lap_mse(tgt_L, cross_v * tgt_inner, dense_out["cross"] * tgt_inner))
                for r in ROWS:
                    L = tgt_L if r == "cross" else src_L
                    dist = (out[r] - dense_out[r]).norm(dim=-1)
                    m[r].add("to_dense_dist_mean", dist.mean())
                    m[r].add("to_dense_dist_p99", torch.quantile(dist, 0.99))
                    if r != "cross":
                        m[r].add("to_dense_MSE", (out[r] - dense_out[r]).square().mean())
                        m[r].add("to_dense_Lap", lap_mse(L, out[r], dense_out[r]))
                if f in fig_frames:
                    for r in ROWS:
                        d = OUT / "verts" / name / r
                        d.mkdir(parents=True, exist_ok=True)
                        np.save(d / f"{f:06d}.npy", out[r][0].cpu().numpy())
        if f in fig_frames:
            np.save(OUT / "ref" / "source_expression" / f"{f:06d}.npy", gt_np)
            prev_check["self"] = max(prev_check["self"], float(
                np.abs(np.load(PREV_SELF / f"{f:06d}.npy") - dense_out["self"][0].cpu().numpy()).max()))
            prev_check["cross"] = max(prev_check["cross"], float(
                np.abs(np.load(PREV_CROSS / f"{f:06d}.npy") - dense_out["cross"][0].cpu().numpy()).max()))

    np.save(OUT / "control_points_src.npy", control)
    result = {
        "form": FORM,
        "frames": len(frames),
        "n_vertices": {"src": len(src_v_np), "tgt": len(tgt_v_np)},
        "checks": {"max_abs_w_at_v_minus_model": w_v_check,
                   "max_abs_dense_minus_previous_run": prev_check,
                   "frames_with_nonfinite_cross_normals": nan_frames},
        "sparsity": stats,
        "metrics": {v: {r: meters[v][r].means() for r in ROWS} for v in variants},
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["checks"], indent=1))
    print(json.dumps(result["metrics"], indent=1))
    print("DONE", OUT)


if __name__ == "__main__":
    main()
