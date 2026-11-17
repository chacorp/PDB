"""Cell PNGs + decimation-attribution numbers for vis_CBD/vis_comparison/vis_robustness.html
(topology robustness of PDB, see eval_CBD_robust.py).

Per dataset (ict / mf / coma), one representative sequence, 2 frames
(max and 75th-percentile GT displacement). For every condition
(clean, hole_s, hole_l, hole_m, dec50, dec25, subdiv1), every identity mesh is
perturbed ("both" side) and these cells are rendered:

    neutral   : perturbed source neutral (plain shading)
    gt        : GT expression on the perturbed mesh (diff vs neutral, own scale)
    self      : self retargeting output            (diff vs neutral, own scale)
    self_err  : |self - GT| per vertex             (shared scale per frame)
    cyc       : cyclic output A -> B -> A           (diff vs neutral, own scale)
    cyc_err   : |cyc - GT|                          (shared scale per frame)
    route     : |direct - via| on the final target c (shared scale per frame)

Shared scale: vmax = 3 x 99th percentile of the clean cell's value; larger values clip.

Attribution (attrib.json): per condition, self MSE with key_weight (kw) and the
encoded expression code (kd) each taken from the clean pipeline (kw projected by
P and row-renormalized) or from the perturbed mesh, over the first N frames.

Run from repo root:
    python render/cache_robust_figure.py [--datasets ict mf coma]
"""
import os
import sys
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# matplotrender imports the site-packages `utils`; hide the repo path while importing it
def _is_repo(p):
    return os.path.abspath(p or ".").startswith(str(REPO_ROOT))


_repo_paths = [p for p in sys.path if _is_repo(p)]
sys.path = [p for p in sys.path if not _is_repo(p)]
try:
    from matplotrender import plot_mesh_gouraud  # noqa: E402
    from matplotrender.utils import calc_face_norm  # noqa: E402
finally:
    sys.path[:0] = _repo_paths
    for _m in [m for m in sys.modules if m == "utils" or m.startswith("utils.")]:
        del sys.modules[_m]

import torch  # noqa: E402
import eval_CBD_robust as R  # noqa: E402

OUT_ROOT = REPO_ROOT / "vis_CBD" / "vis_comparison" / "robust_figure"
M_SCALE = 0.6
SIZE = 8
LIGHT_DIR = np.array([0, 0, 1])
FIT = 1.8
Q_LO, Q_HI = 1, 99
ERR_MULT = 3.0
REP_SEQ = {"ict": "ict-cap_ID000_exp1", "mf": "mf_ROM", "coma": "coma_FaceTalk_170809_00138_TA"}
ROWS = ["neutral", "gt", "self", "self_err", "cyc", "cyc_err", "route"]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def shade_colors(V, F, cnorm):
    vn = calc_face_norm(np.asarray(V) * M_SCALE, F, mode="v")
    shading = np.clip(vn @ LIGHT_DIR, 0, 1)
    vc = shading[..., None].repeat(3, axis=-1) * 0.6 + 0.3
    sc = plt.get_cmap("YlOrRd")(cnorm)[..., :3]
    m = cnorm[:, None]
    return vc * (1 - m) + sc * m


def fit_view(V):
    """Drawn vertices; if their bulk leaves the [-1,1] view, center + refit (returns ratio)."""
    Vd = np.asarray(V) * M_SCALE
    lo, hi = np.percentile(Vd, [Q_LO, Q_HI], axis=0)
    if max(np.abs(lo[:2]).max(), np.abs(hi[:2]).max()) <= 1:
        return Vd, None
    ratio = FIT / (hi - lo).max()
    return (Vd - np.median(Vd, axis=0)) * ratio, ratio


def save_cell(V, F, out_path, values=None, vmax=None):
    """values None -> plain shading. vmax None -> own min-max scale."""
    Vd, ratio = fit_view(V)
    kw = dict(mesh_scale=1.0, mesh_trans=np.zeros(3), backface_culling=True, rot_list=[[0, 0, 0]],
              size=SIZE, mode="shade", bg_black=False, logdir=str(out_path.parent),
              name=out_path.stem, save=True, show=False)
    if values is None:
        plot_mesh_gouraud([Vd], [F], is_diff=False, **kw)
    else:
        if vmax is None:
            c = (values - values.min()) / (values.max() - values.min() + 1e-12)
        else:
            c = np.clip(values / (vmax + 1e-12), 0, 1)
        plot_mesh_gouraud([Vd], [F], Cs=[shade_colors(V, F, c)], is_color=True, **kw)
    plt.close("all")
    txt = out_path.with_suffix(".rescale.txt")
    if ratio is not None:
        txt.write_text(f"{ratio / M_SCALE:.2g}")
    else:
        txt.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def npy(t):
    return t[0].detach().cpu().numpy()


@torch.no_grad()
def frame_outputs(ev, seq, gt0, cond):
    S = ev.ctx(seq["src"], cond)
    B = ev.ctx(seq["cyc_B"], cond)
    Rb = ev.ctx(seq["route_b"], cond)
    Rc = ev.ctx(seq["route_c"], cond)
    gt = ev.apply_P(S["P"], gt0)
    self_out, kd = ev.encode(S, gt)
    cyc = ev.cyclic(S, B, kd)
    direct, via = ev.route(S, Rb, Rc, kd)
    return dict(
        S_faces=S["faces"].cpu().numpy(), C_faces=Rc["faces"].cpu().numpy(),
        neu=npy(S["neu"]), gt=npy(gt), self=npy(self_out), cyc=npy(cyc),
        c_neu=npy(Rc["neu"]), direct=npy(direct), via=npy(via),
    )


@torch.no_grad()
def attribution(ev, seq, conds, n_frames, bs):
    S0 = ev.ctx(seq["src"], "clean")
    frames = seq["frames"][:n_frames]
    out = {}
    for cond in conds:
        Sc = ev.ctx(seq["src"], cond)
        kw_sub = torch.sparse.mm(Sc["P"], S0["kw"][0])[None]
        kw_sub = kw_sub / (kw_sub.sum(-1, keepdim=True) + 1e-12)
        sums = np.zeros(4)
        for st in range(0, len(frames), bs):
            gt0 = frames[st:st + bs].to(ev.device)
            gt = ev.apply_P(Sc["P"], gt0)
            _, kd0 = ev.encode(S0, gt0)
            _, kdc = ev.encode(Sc, gt)
            for i, (kw, kd) in enumerate([(kw_sub, kd0), (Sc["kw"], kd0), (kw_sub, kdc), (Sc["kw"], kdc)]):
                n = kd.shape[0]
                o = ev.model.apply_key_d(kw.expand(n, -1, -1), kd, Sc["neu"].expand(n, -1, -1))
                if not ev.model.use_full_vertex:
                    o = o + Sc["neu"]
                sums[i] += ((o - gt) ** 2).mean(dim=(1, 2)).sum().item()
        out[cond] = dict(zip(["kw_clean+kd_clean", "kw_pert+kd_clean", "kw_clean+kd_pert", "kw_pert+kd_pert"],
                             (sums / len(frames)).tolist()))
        out[cond]["n_frames"] = len(frames)
    return out


def run_dataset(dataset, args):
    sys_argv = sys.argv
    sys.argv = ["x", "--dataset", dataset, "--ckpt", args.ckpt, "--num_workers", "4"]
    if dataset == "ict":
        sys.argv += ["--ict_ids", "0"]
    opts = R.get_opts()
    sys.argv = sys_argv

    ev = R.RobustEval(opts)
    ev.idents, seqs, hole_ref = R.build_sequences(opts)
    V_ref = torch.from_numpy(ev.idents[hole_ref][0]).float()[None]
    ev.hole_candidates = np.nonzero(R.plateau_hat_points(V_ref, **R.SELF_MASK)[0, :, 0].numpy() > 0.999)[0]
    seq = [s for s in seqs if s["name"] == REP_SEQ[dataset]][0]

    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # attribution numbers
    attrib = attribution(ev, seq, R.CONDITION_ORDER[1:], args.attrib_frames, 16)
    meta = dict(seq=seq["name"], src=seq["src"], cyc_B=seq["cyc_B"], route_b=seq["route_b"],
                route_c=seq["route_c"], ckpt=ev.ckpt_file, attribution=attrib)

    # frame selection: GT displacement magnitude
    V0 = torch.from_numpy(ev.idents[seq["src"]][0])
    disp = (seq["frames"] - V0[None]).norm(dim=-1).mean(-1).numpy()
    order = np.argsort(disp)
    frames = [int(order[-1]), int(order[int(0.75 * (len(order) - 1))])]
    meta["frames"] = frames

    for fr in frames:
        fdir = out_dir / f"frame_{fr:06d}"
        fdir.mkdir(parents=True, exist_ok=True)
        gt0 = seq["frames"][fr:fr + 1].to(ev.device)
        res = {c: frame_outputs(ev, seq, gt0, c) for c in R.CONDITION_ORDER}
        r0 = res["clean"]
        save_cell(r0["neu"], r0["S_faces"], fdir / "ref_neutral.png")
        save_cell(r0["gt"], r0["S_faces"], fdir / "ref_expression.png",
                  values=np.linalg.norm(r0["gt"] - r0["neu"], axis=-1))
        err = {c: dict(self_err=np.linalg.norm(r["self"] - r["gt"], axis=-1),
                       cyc_err=np.linalg.norm(r["cyc"] - r["gt"], axis=-1),
                       route=np.linalg.norm(r["direct"] - r["via"], axis=-1)) for c, r in res.items()}
        vmax = {k: ERR_MULT * np.percentile(err["clean"][k], 99) for k in ("self_err", "cyc_err", "route")}
        frame_meta = {"vmax": vmax, "mean_err": {}}
        for c, r in res.items():
            F_s, F_c = r["S_faces"], r["C_faces"]
            save_cell(r["neu"], F_s, fdir / f"neutral_{c}.png")
            save_cell(r["gt"], F_s, fdir / f"gt_{c}.png", values=np.linalg.norm(r["gt"] - r["neu"], axis=-1))
            save_cell(r["self"], F_s, fdir / f"self_{c}.png", values=np.linalg.norm(r["self"] - r["neu"], axis=-1))
            save_cell(r["cyc"], F_s, fdir / f"cyc_{c}.png", values=np.linalg.norm(r["cyc"] - r["neu"], axis=-1))
            save_cell(r["self"], F_s, fdir / f"self_err_{c}.png", values=err[c]["self_err"], vmax=vmax["self_err"])
            save_cell(r["cyc"], F_s, fdir / f"cyc_err_{c}.png", values=err[c]["cyc_err"], vmax=vmax["cyc_err"])
            save_cell(r["direct"], F_c, fdir / f"route_{c}.png", values=err[c]["route"], vmax=vmax["route"])
            frame_meta["mean_err"][c] = {k: float((v ** 2).mean() / 3) for k, v in err[c].items()}
        meta[f"frame_{fr:06d}"] = frame_meta
        print(f"[{dataset}] frame {fr} done")
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    del ev
    torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ict", "mf", "coma"])
    p.add_argument("--ckpt", default="./ckpts_CBD/2026-04-27-15-36-49-NGBCv5")
    p.add_argument("--attrib_frames", type=int, default=256)
    a = p.parse_args()
    for d in a.datasets:
        run_dataset(d, a)
