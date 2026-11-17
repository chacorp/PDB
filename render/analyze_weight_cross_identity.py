"""
Cross-identity consistency of blending weights (column k <-> control point k).

For every ICT test identity (ict_data_synth['test'] = data/ICT_live_100/iden_vecs.npy[0:10])
the neutral mesh is fed to the model and W_i (N, M) = model.predict_coordinate(neutral) is
computed. All ICT identities share topology, so columns can be compared vertex by vertex.

For each identity pair (i, j), i < j, and every column k alive in both W_i and W_j:
  same-k   : compare W_i[:, k] with W_j[:, k]
  NN       : compare W_i[:, k] with W_j[:, l], l = spatially nearest other alive column
             (weighted centroid sum_v w_v x_v / sum_v w_v, measured on identity i)
  random   : compare W_i[:, k] with W_j[:, l], l != k drawn uniformly from alive columns
Metrics:
  Pearson r    over vertices
  Support IoU  |S_a & S_b| / |S_a | S_b|,  S = {v : w_v > 0}

Outputs: _tmp/weight_cross_id/{metrics.json, summary.md}

Run from repo root:
    python render/analyze_weight_cross_identity.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import igl  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from eval_CBD import Trainer  # noqa: E402
from utils.keys import get_data_splits  # noqa: E402
from utils.remesh_utils import ICT_face_model  # noqa: E402

DEVICE = "cuda:0"
SEED = 42
CKPT = "./ckpts_CBD/2026-04-27-15-36-49-NGBCv5"
OUT_DIR = REPO_ROOT / "_tmp" / "weight_cross_id"
CONDITIONS = ["same-k", "NN", "random"]
METRICS = ["pearson", "iou"]


def load_model(ckpt):
    opts_yaml = yaml.safe_load(open(os.path.join(ckpt, "train_opts.yml")))
    opts_yaml.update({
        "device": DEVICE, "ckpt": ckpt, "batch_size": 1,
        "continue_ckpt": False, "seed": SEED, "is_train": False,
    })  # version / architecture options come from the checkpoint; loads *_best.pth
    trainer = Trainer(argparse.Namespace(**opts_yaml))
    trainer.model.eval()
    return trainer.model


def ict_test_neutrals():
    """[(id_name, V (N,3))] for ict_data_synth['test'] identities."""
    *_, ict_synth = get_data_splits()
    iden_vecs = np.load(REPO_ROOT / "data" / "ICT_live_100" / "iden_vecs.npy")
    ict = ICT_face_model()
    out = []
    for name in ict_synth["test"]:
        _, template, _ = ict.apply_coeffs(iden_vecs[int(name)], np.zeros((1, 53)), return_all=True)
        out.append((name, np.asarray(template[0], dtype=np.float64)))
    return out, np.asarray(ict.faces)


def pearson_cols(A, B):
    """Column-wise Pearson r between A[:, c] and B[:, c]. A, B: (N, C)."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    return (A * B).sum(0) / (np.linalg.norm(A, axis=0) * np.linalg.norm(B, axis=0))


def iou_cols(A, B):
    Sa, Sb = A > 0, B > 0
    return (Sa & Sb).sum(0) / (Sa | Sb).sum(0)


def centroids(W, V, idx):
    """Weighted centroid c(k) = sum_v W[v,k] x_v / sum_v W[v,k] for columns idx. (K, 3)"""
    return (W[:, idx].T @ V) / W[:, idx].sum(0)[:, None]


def nearest_columns(Wi, Vi, Wj, Vj, alive):
    """k' = argmin_{l != k} || c_j(l) - c_i(k) ||: the column of identity j (other than k)
    whose centroid is nearest to the centroid of column k in identity i."""
    idx = np.flatnonzero(alive)
    d = np.linalg.norm(centroids(Wi, Vi, idx)[:, None] - centroids(Wj, Vj, idx)[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.full(Wi.shape[1], -1)
    nn[idx] = idx[d.argmin(1)]
    return nn, d.min(1)


def nearest_columns_within_i(Wi, Vi, alive):
    """Previous definition (for agreement check only): k' chosen inside identity i."""
    idx = np.flatnonzero(alive)
    c = centroids(Wi, Vi, idx)
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.full(Wi.shape[1], -1)
    nn[idx] = idx[d.argmin(1)]
    return nn


def stats(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std()),
            "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)), "n": int(x.size)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    neutrals, F = ict_test_neutrals()

    model = load_model(CKPT)
    Ws, alives = {}, {}
    with torch.no_grad():
        for name, V in neutrals:
            n = igl.per_vertex_normals(V, F.astype(np.int64))
            v_th = torch.tensor(V).float().unsqueeze(0).to(DEVICE)
            n_th = torch.tensor(n).float().unsqueeze(0).to(DEVICE)
            W = model.predict_coordinate(v_th, n_th)[0].cpu().numpy().astype(np.float64)
            Ws[name], alives[name] = W, W.sum(0) > 0
            print(f"[W] ICT {name}: shape={W.shape} dead_cols={(~alives[name]).sum()} "
                  f"zero_ratio={(W == 0).mean():.4f}")

    names = [nm for nm, _ in neutrals]
    Vs = dict(neutrals)
    vals = {c: {m: [] for m in METRICS} for c in CONDITIONS}
    per_pair = []
    nn_dists, nn_agree = [], []
    # dead-column consistency across identities
    dead = {nm: set(np.flatnonzero(~al).tolist()) for nm, al in alives.items()}
    dead_inter = set.intersection(*dead.values())
    dead_union = set.union(*dead.values())
    dead_pair_iou = [len(dead[p] & dead[q]) / len(dead[p] | dead[q])
                     for x, p in enumerate(names) for q in names[x + 1:]]
    dead_info = {
        "per_identity": {nm: len(s) for nm, s in dead.items()},
        "intersection": len(dead_inter), "union": len(dead_union),
        "pair_iou": stats(dead_pair_iou),
        "all_identical": all(s == dead[names[0]] for s in dead.values()),
        "not_shared": {nm: sorted(s - dead_inter) for nm, s in dead.items()},
        "not_shared_max_colsum": {},  # filled below: how close to dead the column is elsewhere
    }
    for k in sorted(dead_union - dead_inter):
        dead_info["not_shared_max_colsum"][int(k)] = {nm: float(Ws[nm][:, k].sum()) for nm in names}
    print(f"[dead] per identity={dead_info['per_identity']} intersection={len(dead_inter)} "
          f"union={len(dead_union)} identical={dead_info['all_identical']}")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            i, j = names[a], names[b]
            Wi, Wj = Ws[i], Ws[j]
            alive = alives[i] & alives[j]
            ks = np.flatnonzero(alive)
            nn, nd = nearest_columns(Wi, Vs[i], Wj, Vs[j], alive)
            nn_dists.append(nd)
            nn_agree.append(float((nn[ks] == nearest_columns_within_i(Wi, Vs[i], alive)[ks]).mean()))
            # random l != k among alive columns
            rnd = rng.choice(ks, size=ks.size)
            clash = rnd == ks
            while clash.any():
                rnd[clash] = rng.choice(ks, size=clash.sum())
                clash = rnd == ks
            partner = {"same-k": ks, "NN": nn[ks], "random": rnd}
            row = {"pair": f"{i}-{j}", "num_cols": int(ks.size)}
            for c in CONDITIONS:
                A, B = Wi[:, ks], Wj[:, partner[c]]
                r, iou = pearson_cols(A, B), iou_cols(A, B)
                vals[c]["pearson"].append(r)
                vals[c]["iou"].append(iou)
                row[c] = {"pearson_mean": float(r.mean()), "iou_mean": float(iou.mean())}
            per_pair.append(row)
            print(f"[pair {i}-{j}] cols={ks.size} " + " ".join(
                f"{c}: r={row[c]['pearson_mean']:.3f} IoU={row[c]['iou_mean']:.3f}" for c in CONDITIONS))

    summary = {c: {m: stats(np.concatenate(vals[c][m])) for m in METRICS} for c in CONDITIONS}
    # pair-level means (one value per identity pair) and fraction of pairs where same-k beats each baseline
    pair_level = {c: {m: stats([p[c][f"{m}_mean"] for p in per_pair]) for m in METRICS} for c in CONDITIONS}
    win = {f"same-k>{c}": {m: float(np.mean([p["same-k"][f"{m}_mean"] > p[c][f"{m}_mean"] for p in per_pair]))
                           for m in METRICS} for c in ["NN", "random"]}
    meta = {
        "ckpt": CKPT, "identities": names, "num_pairs": len(per_pair),
        "num_verts": int(next(iter(Ws.values())).shape[0]), "num_cols": int(next(iter(Ws.values())).shape[1]),
        "dead_cols": {nm: int((~al).sum()) for nm, al in alives.items()},
        "nn_centroid_dist": stats(np.concatenate(nn_dists)), "seed": SEED,
        "nn_agreement_with_within_i_def": stats(nn_agree),
        "dead": dead_info,
    }
    json.dump({"meta": meta, "summary": summary, "pair_level": pair_level, "win_rate": win,
               "per_pair": per_pair}, open(OUT_DIR / "metrics.json", "w"), indent=1)
    np.savez_compressed(OUT_DIR / "per_column_values.npz",
                        **{f"{c}|{m}": np.concatenate(vals[c][m]) for c in CONDITIONS for m in METRICS})
    write_md(meta, summary, pair_level, win)
    print(f"wrote {OUT_DIR}")


def write_md(meta, summary, pair_level, win):
    L = ["# Cross-identity consistency of blending weights", "",
         f"- ckpt: `{meta['ckpt']}` (model_best.pth)",
         f"- ICT test identities: {', '.join(meta['identities'])} -> {meta['num_pairs']} pairs",
         f"- W: {meta['num_verts']} verts x {meta['num_cols']} columns; "
         f"dead cols per identity: {meta['dead_cols']}",
         f"- NN definition: k' = argmin_(l != k) ||c_j(l) - c_i(k)|| (column of identity j nearest to "
         f"centroid of column k in identity i); mean ||c_j(k') - c_i(k)|| = {meta['nn_centroid_dist']['mean']:.4f}",
         f"- k' agreement with previous within-identity-i NN definition: "
         f"{meta['nn_agreement_with_within_i_def']['mean']:.3f}", "",
         "## Dead columns across identities", "",
         f"- per identity: {meta['dead']['per_identity']}",
         f"- intersection (dead in all 10): {meta['dead']['intersection']}, "
         f"union (dead in any): {meta['dead']['union']}, all identical: {meta['dead']['all_identical']}",
         f"- pairwise IoU of dead sets: mean {meta['dead']['pair_iou']['mean']:.3f}, "
         f"median {meta['dead']['pair_iou']['median']:.3f}", ""]
    for title, tab in [("Per-column values (all pairs x columns pooled)", summary),
                       ("Pair-level (mean over columns, then over pairs)", pair_level)]:
        L += [f"## {title}", "", "| Condition | Pearson r mean | median | std | IoU mean | median | std |",
              "|---|---|---|---|---|---|---|"]
        for c in CONDITIONS:
            p, q = tab[c]["pearson"], tab[c]["iou"]
            L.append(f"| {c} | {p['mean']:.4f} | {p['median']:.4f} | {p['std']:.4f} | "
                     f"{q['mean']:.4f} | {q['median']:.4f} | {q['std']:.4f} |")
        L.append("")
    L += ["## Win rate over pairs (fraction of pairs where same-k mean > baseline mean)", "",
          "| | Pearson r | IoU |", "|---|---|---|"]
    for k, v in win.items():
        L.append(f"| {k} | {v['pearson']:.3f} | {v['iou']:.3f} |")
    (OUT_DIR / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
