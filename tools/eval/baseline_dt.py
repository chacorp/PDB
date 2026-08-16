# tools/eval/baseline_dt.py — Deformation Transfer (Sumner & Popovic 2004).
#
# Same-topology path (E1 self-retarget): identity triangle correspondence.
# Per-triangle deformation gradient S_i from (src_neu -> src_def), transferred
# to the target's rest mesh; dense vertices recovered by the standard
# gradient-domain least squares  min_x ||G x - S||^2_A  (area-weighted),
# translation fixed by centroid. The (G^T A G) factorization is cached per
# target mesh so per-frame cost is one back-substitution.
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import igl

_CACHE = {}


def _tri_frames(V, F):
    """Per-triangle 3x3 frame [e1 e2 n_hat] (Sumner's construction)."""
    v0 = V[F[:, 0]]; v1 = V[F[:, 1]]; v2 = V[F[:, 2]]
    e1 = v1 - v0; e2 = v2 - v0
    n = np.cross(e1, e2)
    nn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    M = np.stack([e1, e2, nn], axis=2)   # [F,3,3] columns e1,e2,n
    return M


def _prefactor(key, tgt_neu, F):
    if key in _CACHE:
        return _CACHE[key]
    V = np.asarray(tgt_neu, dtype=np.float64)
    G = igl.grad(V, F.astype(np.int64))            # [3F x V]
    dbl = igl.doublearea(V, F.astype(np.int64))
    A = sp.diags(np.tile(dbl * 0.5, 3))            # area weights per gradient row
    GtA = G.T @ A
    L = (GtA @ G).tocsc()
    solver = spla.factorized(L)
    _CACHE[key] = (G, GtA, solver)
    return _CACHE[key]


def dt_self_transfer(key, src_neu, src_def, tgt_neu, F):
    """Same-topology DT: returns predicted tgt vertices [V,3] (float32).

    key: hashable cache key for the target mesh (e.g. (ds, id)).
    """
    F = np.asarray(F, dtype=np.int64)
    Ms = _tri_frames(np.asarray(src_neu, np.float64), F)      # rest frames
    Md = _tri_frames(np.asarray(src_def, np.float64), F)      # deformed frames
    # S_i = Md_i @ Ms_i^{-1}  (3x3 per triangle)
    S = Md @ np.linalg.inv(Ms)                                # [F,3,3]
    G, GtA, solver = _prefactor(key, tgt_neu, F)
    # igl.grad rows are ordered [Fx; Fy; Fz] blocks: gradient of a scalar field.
    # For coordinate function c, target gradients should equal S^T rows applied
    # to the target's own rest gradients: b = blockstack over xyz of S applied
    # to tgt rest gradient. Standard identity-topology shortcut: G x = S-mapped
    # gradients of tgt rest coordinates -> b = reorder(S) @ (G @ tgt_neu).
    Gt = G @ np.asarray(tgt_neu, np.float64)                  # [3F,3] rest gradients
    nF = F.shape[0]
    Gt3 = Gt.reshape(3, nF, 3).transpose(1, 0, 2)             # [F,3(xyz-block),3(coord)]
    # each triangle: rows are d/dx,d/dy,d/dz of the 3 coord functions -> J^T
    # apply S: new Jacobian J' = S @ J  ->  (J')^T = J^T @ S^T
    B3 = Gt3 @ S.transpose(0, 2, 1)                           # [F,3,3]
    b = B3.transpose(1, 0, 2).reshape(3 * nF, 3)              # back to [3F,3]
    rhs = GtA @ b
    x = np.column_stack([solver(rhs[:, c]) for c in range(3)])
    # translation gauge: match source-deformed centroid (self-retarget GT frame)
    x += (np.asarray(src_def, np.float64).mean(0) - x.mean(0))
    return x.astype(np.float32)
