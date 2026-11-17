"""
Compare weight locality and sparsity of Ours-dagger vs. the w/o col-norm model.

For each test-identity neutral mesh (ICT ID_000-004, COMA test, BIWI test, MF test)
the per-vertex weight matrix W (N, M) = model.predict_coordinate(neutral) is
computed and summarized:

  Locality  sigma_k = sqrt( sum_j w_k^j d_g(j, c_k)^2 / sum_j w_k^j ) / L
            c_k = argmax_j w_k^j, d_g = heat-method geodesic distance (potpourri3d),
            restricted to the connected component of c_k; L = mesh geodesic
            diameter (2-sweep estimate, model independent). Dead columns
            (all-zero) are excluded and counted separately.
  Sparsity  active count per vertex  #{k : w_k^j > 0}
            zero ratio               fraction of exact zeros in W

Outputs go to _tmp/weight_stats/ (metrics.json, summary.md, *.png).

Run from repo root:
    python render/analyze_weight_locality_sparsity.py
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import igl  # noqa: E402
import numpy as np  # noqa: E402
import potpourri3d as pp3d  # noqa: E402
import scipy.sparse.csgraph as csgraph  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_CBD import Trainer  # noqa: E402
from utils.keys import get_data_splits  # noqa: E402
from utils.remesh_utils import ICT_face_model  # noqa: E402

DEVICE = "cuda:0"
OUT_DIR = REPO_ROOT / "_tmp" / "weight_stats"
DATA_ROOT = "/data/sihun/pca"

MODELS = [
    ("Ours†", "./ckpts_CBD/2026-09-14-22-59-50-NGBCv5", "#2a78d6"),
    ("w/o col-norm", "./ckpts_CBD8/2026-10-27-00-55-40-NGBCv5", "#eb6834"),
]
DATASETS = ["ICT", "COMA", "BIWI", "MF"]
ICT_TEST_IDS = range(5)  # ict-cap_test-ID_000-004, same targets as eval_CBD_cyc


def load_model(ckpt):
    opts_yaml = yaml.safe_load(open(os.path.join(ckpt, "train_opts.yml")))
    opts_yaml.update({
        "device": DEVICE,
        "ckpt": ckpt,
        "version": 5,
        "batch_size": 1,
        "continue_ckpt": False,
        "seed": 42,
        "is_train": False,
    })  # architecture options (activation, pou, align_latent, no_col_norm) come from the checkpoint
    trainer = Trainer(argparse.Namespace(**opts_yaml))
    trainer.model.eval()
    return trainer.model


def load_meshes():
    """Return [(dataset, id_name, V (N,3), F (T,3))] for every test-identity neutral mesh."""
    _, voca_split, biwi_split, mf_split, _ = get_data_splits()
    meshes = []

    ict = ICT_face_model()
    iden_vecs = torch.load(REPO_ROOT / "ict_face_pt" / "ict_id_vecs_test.pt").numpy()
    for i in ICT_TEST_IDS:
        _, template, _ = ict.apply_coeffs(iden_vecs[i], np.zeros((1, 53)), return_all=True)  # neutral
        meshes.append(("ICT", f"ID_{i:03d}", np.asarray(template[0], dtype=np.float64), np.asarray(ict.faces)))

    voca_f = np.load(REPO_ROOT / "utils" / "voca" / "standardization.npy", allow_pickle=True).item()["new_f"]
    coma = pickle.load(open(f"{DATA_ROOT}/VOCA-COMA/voca_templates.pkl", "rb"))
    for id_name in voca_split["test"]:
        meshes.append(("COMA", id_name, np.asarray(coma[id_name], dtype=np.float64), np.asarray(voca_f)))

    biwi = pickle.load(open(f"{DATA_ROOT}/BIWI_align_deci/templates_align_deci.pkl", "rb"))
    for id_name in biwi_split["test"]:
        meshes.append(("BIWI", id_name, np.asarray(biwi[id_name], dtype=np.float64), np.asarray(biwi["face"])))

    mf_f = np.load(REPO_ROOT / "utils" / "mf" / "standardization.npy", allow_pickle=True).item()["new_f"]
    mf = pickle.load(open(f"{DATA_ROOT}/multiface_align/mf_templates.pkl", "rb"))
    for id_name in mf_split["test"]:
        meshes.append(("MF", id_name, np.asarray(mf[id_name], dtype=np.float64), np.asarray(mf_f)))
    return meshes


class Geodesics:
    """Heat-method geodesic distances on one mesh, with per-source caching."""

    def __init__(self, V, F):
        self.solver = pp3d.MeshHeatMethodDistanceSolver(V, F.astype(np.int64))
        _, self.comp = csgraph.connected_components(igl.adjacency_matrix(F.astype(np.int64)), directed=False)
        self.cache = {}
        # 2-sweep diameter estimate on the largest component
        big = np.bincount(self.comp).argmax()
        start = int(np.flatnonzero(self.comp == big)[0])
        a = self._masked(start).argmax()
        self.diameter = float(self._masked(a).max())

    def _masked(self, src):
        d = self.solver.compute_distance(src)
        return np.where(self.comp == self.comp[src], d, 0.0)

    def __call__(self, src):
        if src not in self.cache:
            d = self.solver.compute_distance(src)
            self.cache[src] = np.where(self.comp == self.comp[src], d, np.nan)
        return self.cache[src]


def weight_stats(W, geo):
    """W: (N, M) numpy weight matrix."""
    alive = W.sum(axis=0) > 0
    sigmas = []
    for k in np.flatnonzero(alive):
        w = W[:, k]
        d = geo(int(w.argmax()))
        valid = ~np.isnan(d)
        sigmas.append(np.sqrt((w[valid] * d[valid] ** 2).sum() / w[valid].sum()) / geo.diameter)
    active = (W > 0).sum(axis=1)
    return {
        "sigma": np.asarray(sigmas),
        "active": active,
        "zero_ratio": float((W == 0).mean()),
        "dead_cols": int((~alive).sum()),
        "num_cols": int(W.shape[1]),
        "num_verts": int(W.shape[0]),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meshes = load_meshes()
    geos = {}
    for ds, id_name, V, F in meshes:
        print(f"[geodesic] {ds}/{id_name}  N={len(V)}")
        geos[(ds, id_name)] = Geodesics(V, F)

    per_mesh = {}  # (model, ds, id) -> stats
    for label, ckpt, _ in MODELS:
        print(f"[model] {label}: {ckpt}")
        model = load_model(ckpt)
        for ds, id_name, V, F in meshes:
            n = igl.per_vertex_normals(V, F.astype(np.int64))
            v_th = torch.tensor(V).float().unsqueeze(0).to(DEVICE)
            n_th = torch.tensor(n).float().unsqueeze(0).to(DEVICE)
            W = model.predict_coordinate(v_th, n_th)[0].cpu().numpy()
            per_mesh[(label, ds, id_name)] = weight_stats(W, geos[(ds, id_name)])
            s = per_mesh[(label, ds, id_name)]
            print(f"   {ds}/{id_name}: sigma_mean={s['sigma'].mean():.4f} active_mean={s['active'].mean():.2f} "
                  f"zero={s['zero_ratio']:.4f} dead={s['dead_cols']}")
        del model
        torch.cuda.empty_cache()

    # ---- aggregate per dataset: mean over identities of per-mesh statistics ----
    summary = {}
    for label, _, _ in MODELS:
        summary[label] = {}
        for ds in DATASETS:
            rows = [v for (m, d, _), v in per_mesh.items() if m == label and d == ds]
            summary[label][ds] = {
                "num_ids": len(rows),
                "sigma_mean": float(np.mean([r["sigma"].mean() for r in rows])),
                "sigma_median": float(np.mean([np.median(r["sigma"]) for r in rows])),
                "active_mean": float(np.mean([r["active"].mean() for r in rows])),
                "active_median": float(np.mean([np.median(r["active"]) for r in rows])),
                "active_max": int(max(r["active"].max() for r in rows)),
                "zero_ratio": float(np.mean([r["zero_ratio"] for r in rows])),
                "dead_cols": float(np.mean([r["dead_cols"] for r in rows])),
            }
    json.dump({
        "summary": summary,
        "per_mesh": {f"{m}|{d}|{i}": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
                     for (m, d, i), s in per_mesh.items()},
        "diameter": {f"{d}|{i}": g.diameter for (d, i), g in geos.items()},
    }, open(OUT_DIR / "metrics.json", "w"))

    write_summary_md(summary)
    plot_sigma(per_mesh)
    plot_active(per_mesh)
    print(f"wrote {OUT_DIR}")


def write_summary_md(summary):
    labels = [m[0] for m in MODELS]
    lines = ["# Weight locality & sparsity", ""]
    tables = [
        ("Locality: geodesic spread sigma_k / diameter (lower = more local), mean over alive columns",
         "sigma_mean", "{:.4f}"),
        ("Locality: median sigma_k / diameter", "sigma_median", "{:.4f}"),
        ("Sparsity: mean active control points per vertex #{k : w_k^j > 0}", "active_mean", "{:.2f}"),
        ("Sparsity: median active control points per vertex", "active_median", "{:.1f}"),
        ("Sparsity: max active control points per vertex", "active_max", "{:d}"),
        ("Sparsity: zero ratio of W", "zero_ratio", "{:.4f}"),
        ("Dead columns (all-zero control points, of 512)", "dead_cols", "{:.1f}"),
    ]
    for title, key, fmt in tables:
        lines += [f"## {title}", "", "| Model | " + " | ".join(DATASETS) + " |",
                  "|---|" + "---|" * len(DATASETS)]
        for lb in labels:
            lines.append(f"| {lb} | " + " | ".join(fmt.format(summary[lb][ds][key]) for ds in DATASETS) + " |")
        lines.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b8b7b0")
    ax.tick_params(colors="#5f5e5a", labelsize=9)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_sigma(per_mesh):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.2 * len(DATASETS), 3.4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        data = [np.concatenate([s["sigma"] for (m, d, _), s in per_mesh.items() if m == lb and d == ds])
                for lb, _, _ in MODELS]
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for body, (_, _, c) in zip(parts["bodies"], MODELS):
            body.set_facecolor(c)
            body.set_edgecolor("white")
            body.set_alpha(0.85)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[key].set_color("#3d3d3a")
            parts[key].set_linewidth(1)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([m[0] for m in MODELS], fontsize=9)
        ax.set_title(ds, fontsize=11, color="#1f1f1d")
        _style(ax)
    axes[0].set_ylabel("geodesic spread  σ$_k$ / diameter", fontsize=9, color="#5f5e5a")
    fig.suptitle("Weight locality per control point (lower = more local)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "locality_sigma_violin.png", dpi=200)
    plt.close(fig)


def plot_active(per_mesh):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(3.2 * len(DATASETS), 3.4))
    for ax, ds in zip(axes, DATASETS):
        arrs = [np.concatenate([s["active"] for (m, d, _), s in per_mesh.items() if m == lb and d == ds])
                for lb, _, _ in MODELS]
        hi = max(a.max() for a in arrs)
        bins = np.arange(0, hi + 2) - 0.5 if hi <= 60 else np.linspace(0, hi + 1, 50)
        for a, (lb, _, c) in zip(arrs, MODELS):
            ax.hist(a, bins=bins, color=c, alpha=0.55, label=lb, density=True, edgecolor="white", linewidth=0.5)
            ax.axvline(a.mean(), color=c, linewidth=2)
        ax.set_title(ds, fontsize=11, color="#1f1f1d")
        ax.set_xlabel("active control points per vertex", fontsize=9, color="#5f5e5a")
        _style(ax)
    axes[0].set_ylabel("fraction of vertices", fontsize=9, color="#5f5e5a")
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle("Weight sparsity: #{k : w$_k^j$ > 0} per vertex (vertical line = mean)",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sparsity_active_hist.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
