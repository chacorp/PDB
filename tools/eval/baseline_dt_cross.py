# tools/eval/baseline_dt_cross.py — Deformation Transfer across topologies.
#
# Correspondence: sparse markers = OUR model's predicted primary-joint bind
# positions (46 non-helper joints, available for any topology) -> gaussian-RBF
# space warp A->B fitted on marker pairs -> per-triangle map (nearest warped
# source-triangle centroid) -> Sumner-style per-triangle deformation-gradient
# transfer + area-weighted Poisson solve on the target (solver cached per pair).
#
# This hands DT the same semantic correspondence our method uses internally —
# a generous (anti-us biased) setting, per the agreed protocol.
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import igl

_PAIR = {}


def _tri_frames(V, F):
    v0 = V[F[:, 0]]; v1 = V[F[:, 1]]; v2 = V[F[:, 2]]
    e1 = v1 - v0; e2 = v2 - v0
    n = np.cross(e1, e2)
    nn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.stack([e1, e2, nn], axis=2)


def _rbf_fit(src_m, dst_m):
    """Gaussian RBF warp R^3->R^3 through marker pairs (+affine term)."""
    n = len(src_m)
    d2 = ((src_m[:, None] - src_m[None]) ** 2).sum(-1)
    sigma2 = np.median(d2[d2 > 0])
    K = np.exp(-d2 / (2 * sigma2))
    P = np.concatenate([src_m, np.ones((n, 1))], axis=1)
    A = np.block([[K + 1e-8 * np.eye(n), P], [P.T, np.zeros((4, 4))]])
    b = np.concatenate([dst_m, np.zeros((4, 3))], axis=0)
    sol = np.linalg.solve(A, b)
    W, aff = sol[:n], sol[n:]

    def warp(X):
        dd2 = ((X[:, None] - src_m[None]) ** 2).sum(-1)
        Kx = np.exp(-dd2 / (2 * sigma2))
        Px = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        return Kx @ W + Px @ aff
    return warp


def _pair_setup(key, src_neu, F_s, src_markers, tgt_neu, F_t, tgt_markers):
    if key in _PAIR:
        return _PAIR[key]
    src_neu = np.asarray(src_neu, np.float64); tgt_neu = np.asarray(tgt_neu, np.float64)
    warp = _rbf_fit(np.asarray(src_markers, np.float64), np.asarray(tgt_markers, np.float64))
    cs_w = warp(src_neu[F_s].mean(1))                 # warped src tri centroids
    tri_map = cKDTree(cs_w).query(tgt_neu[F_t].mean(1))[1]   # tgt tri -> src tri
    G = igl.grad(tgt_neu, F_t.astype(np.int64))
    dbl = igl.doublearea(tgt_neu, F_t.astype(np.int64))
    Aw = sp.diags(np.tile(dbl * 0.5, 3))
    GtA = G.T @ Aw
    solver = spla.factorized((GtA @ G).tocsc())
    Gt_rest = G @ tgt_neu                             # [3F,3]
    _PAIR[key] = (tri_map, GtA, solver, Gt_rest, F_t.shape[0])
    return _PAIR[key]


def dt_cross_transfer(key, src_neu, src_def, F_s, src_markers,
                      tgt_neu, F_t, tgt_markers):
    """Cross-topology DT. Returns predicted target vertices [V_t,3] float32."""
    F_s = np.asarray(F_s, np.int64); F_t = np.asarray(F_t, np.int64)
    tri_map, GtA, solver, Gt_rest, nF = _pair_setup(
        key, src_neu, F_s, src_markers, tgt_neu, F_t, tgt_markers)
    Ms = _tri_frames(np.asarray(src_neu, np.float64), F_s)
    Md = _tri_frames(np.asarray(src_def, np.float64), F_s)
    S = (Md @ np.linalg.inv(Ms))[tri_map]             # [F_t,3,3] mapped gradients
    Gt3 = Gt_rest.reshape(3, nF, 3).transpose(1, 0, 2)
    b = (Gt3 @ S.transpose(0, 2, 1)).transpose(1, 0, 2).reshape(3 * nF, 3)
    rhs = GtA @ b
    x = np.column_stack([solver(rhs[:, c]) for c in range(3)])
    x += (np.asarray(tgt_neu, np.float64).mean(0) - x.mean(0))
    return x.astype(np.float32)
